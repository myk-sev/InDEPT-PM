from __future__ import annotations

import argparse
import copy
import json
import os
import random
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import (
    Normalizer,
    PairDatabase,
    PairWindowDataset,
    build_database,
    file_sha256,
    load_training_data,
    split_series,
)
from .diagnostics import (
    diagnostic_paths,
    reconstruction_snapshot_path,
    write_loss_curve,
    write_metrics,
    write_reconstruction_examples,
)
from .masking import MASK_SENTINEL, STAGES, mask_batch
from .models import (
    DEFAULT_MODEL,
    ModelConfig,
    build_model,
    canonical_model_name,
    model_names,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent
MASKED_INPUTS = REPOSITORY_ROOT / "inputs" / "masked_pretraining"
DEFAULT_TRAINING_DATA = (
    MASKED_INPUTS / "exclusion_aware" / "k12_exclusion_aware_masked_training_data.csv"
)
DEFAULT_CHECKPOINT = PACKAGE_ROOT / "runs" / "checkpoints" / "masked_pretraining.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Masked reconstruction pretraining on school PurpleAir pairs."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="Report pair, history, and window coverage.")
    _add_data_arguments(audit)
    audit.add_argument("--output", type=Path)

    train = commands.add_parser("train", help="Run the masking curriculum.")
    _add_data_arguments(train)
    train.add_argument("--model", choices=model_names(), default=DEFAULT_MODEL)
    train.add_argument("--model-dim", type=int, default=64)
    train.add_argument("--layers", type=int, default=3)
    train.add_argument("--heads", type=int, default=4)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--stages", nargs="+", choices=STAGES)
    train.add_argument(
        "--epochs-per-stage",
        nargs="+",
        type=int,
        default=[20],
        metavar="N",
        help="One uniform epoch count or one count per selected stage.",
    )
    train.add_argument("--patience", type=int, default=3)
    train.add_argument("--minimum-delta", type=float, default=1e-4)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--validation-fraction", type=float, default=0.2)
    train.add_argument("--workers", type=int, default=0)
    train.add_argument(
        "--reconstruction-every-epochs",
        type=int,
        default=0,
        metavar="N",
        help="Refresh validation reconstructions every N stage epochs; 0 disables.",
    )
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="auto")
    train.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    train.add_argument(
        "--resume",
        type=Path,
        metavar="CHECKPOINT",
        help="Resume the checkpoint stage unless --stages is set.",
    )
    return parser


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--training-data", type=Path, default=DEFAULT_TRAINING_DATA)
    parser.add_argument("--history-hours", type=int, default=168)
    parser.add_argument("--minimum-observed-hours", type=int, default=144)
    parser.add_argument("--stride-hours", type=int, default=24)
    parser.add_argument(
        "--responsiveness-tiers",
        nargs="+",
        choices=("high", "moderate", "low"),
        help="Cumulative pair tiers to admit, for example: high moderate.",
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        database, _ = _load_database(args)
        report = _audit_report(database)
        text = json.dumps(report, indent=2)
        print(text)
        if args.output:
            _write_json(args.output, report)
        return
    train(args)


def train(args: argparse.Namespace) -> None:
    _validate_training_arguments(args)
    resume = _load_checkpoint(args.resume) if args.resume else None
    resume_metadata = resume["metadata"] if resume else None
    stages = args.stages or (
        [str(resume_metadata["stage"])] if resume_metadata else list(STAGES)
    )
    stage_epochs = _resolve_stage_epochs(args.epochs_per_stage, stages)
    _seed_everything(args.seed)
    database, training_data_sha256 = _load_database(args)
    train_indices, validation_indices = split_series(
        database, args.validation_fraction, args.seed
    )
    normalizer = Normalizer.fit(database.series, train_indices)
    train_data = PairWindowDataset(database, train_indices, normalizer)
    validation_data = PairWindowDataset(database, validation_indices, normalizer)
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=args.workers,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    config = ModelConfig(
        history_hours=args.history_hours,
        model_dim=args.model_dim,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
    )
    device = _resolve_device(args.device)
    model = build_model(args.model, config).to(device)
    _validate_model_contract(model, config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    optimizer_resumed = False
    if resume:
        _validate_resume_checkpoint(
            resume,
            args.model,
            config,
            normalizer,
            database,
            train_indices,
            validation_indices,
            training_data_sha256,
        )
        model.load_state_dict(resume["model_state"])
        if "optimizer_state" in resume:
            optimizer.load_state_dict(resume["optimizer_state"])
            optimizer_resumed = True
        print(
            f"resumed_from={args.resume} "
            f"optimizer_state={'restored' if optimizer_resumed else 'fresh'}"
        )
    completed = (
        list(dict.fromkeys(resume_metadata.get("completed_stages", ())))
        if resume_metadata
        else []
    )
    metrics: list[dict[str, object]] = []
    diagnostics = diagnostic_paths(args.checkpoint)
    write_metrics(diagnostics.metrics, metrics)
    audit = _audit_report(database)
    print(
        f"device={device} model={args.model} train_sensors={len(train_indices)} "
        f"validation_sensors={len(validation_indices)} train_windows={len(train_data)} "
        f"validation_windows={len(validation_data)}"
    )
    started = time.perf_counter()
    estimated_total_epochs = sum(stage_epochs)
    for stage_index, (stage, epoch_count) in enumerate(zip(stages, stage_epochs)):
        continuing_stage = bool(
            resume_metadata and stage_index == 0 and stage == resume_metadata["stage"]
        )
        best_loss = (
            float(resume_metadata["validation_metrics"]["loss"])
            if continuing_stage
            else float("inf")
        )
        best_state = _cpu_state(model) if continuing_stage else None
        best_optimizer_state = (
            copy.deepcopy(optimizer.state_dict()) if continuing_stage else None
        )
        stale_epochs = 0
        training_masks = torch.Generator().manual_seed(
            args.seed + (stage_index + 1) * 10_000
        )
        best_metrics = (
            dict(resume_metadata["validation_metrics"])
            if continuing_stage
            else None
        )
        for epoch in range(1, epoch_count + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                stage,
                normalizer,
                device,
                training_masks,
                optimizer,
            )
            validation_masks = torch.Generator().manual_seed(
                args.seed + (stage_index + 1) * 1_000_000
            )
            validation_metrics = run_epoch(
                model,
                validation_loader,
                stage,
                normalizer,
                device,
                validation_masks,
            )
            improved = validation_metrics["loss"] < best_loss - args.minimum_delta
            if improved:
                best_loss = validation_metrics["loss"]
                best_state = _cpu_state(model)
                best_optimizer_state = copy.deepcopy(optimizer.state_dict())
                best_metrics = validation_metrics
                stale_epochs = 0
            else:
                stale_epochs += 1
            stop_early = stale_epochs >= args.patience and not continuing_stage
            metrics.append(
                {
                    "global_epoch": len(metrics) + 1,
                    "stage": stage,
                    "stage_epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "validation_loss": validation_metrics["loss"],
                    "validation_indoor_rmse": validation_metrics["indoor_rmse"],
                    "validation_outdoor_rmse": validation_metrics["outdoor_rmse"],
                    "train_target_count": int(train_metrics["target_count"]),
                    "validation_target_count": int(
                        validation_metrics["target_count"]
                    ),
                    "improved_checkpoint": improved,
                }
            )
            if stop_early:
                estimated_total_epochs -= epoch_count - epoch
            elapsed = time.perf_counter() - started
            estimated_remaining = elapsed / len(metrics) * (
                estimated_total_epochs - len(metrics)
            )
            print(
                f"stage={stage} epoch={epoch}/{epoch_count} "
                f"time_taken={format_duration(elapsed)} "
                f"ETA={format_duration(estimated_remaining)} "
                f"train_loss={train_metrics['loss']:.6f} "
                f"validation_loss={validation_metrics['loss']:.6f} "
                f"indoor_rmse={validation_metrics['indoor_rmse']:.3f} "
                f"outdoor_rmse={validation_metrics['outdoor_rmse']:.3f}"
            )
            write_metrics(diagnostics.metrics, metrics)
            write_loss_curve(diagnostics.loss_curve, metrics)
            if (
                args.reconstruction_every_epochs
                and epoch % args.reconstruction_every_epochs == 0
            ):
                write_reconstruction_examples(
                    reconstruction_snapshot_path(
                        diagnostics.reconstructions, stage, epoch
                    ),
                    model,
                    validation_data,
                    stage,
                    normalizer,
                    device,
                    args.seed + (stage_index + 1) * 1_000_000 + 1,
                )
            if stop_early:
                break
        if best_state is None or best_metrics is None:
            raise RuntimeError(f"stage {stage} produced no valid checkpoint")
        model.load_state_dict(best_state)
        if stage not in completed:
            completed.append(stage)
        write_reconstruction_examples(
            diagnostics.reconstructions,
            model,
            validation_data,
            stage,
            normalizer,
            device,
            args.seed + (stage_index + 1) * 1_000_000 + 1,
        )
        metadata = {
            "format_version": 3,
            "model_name": args.model,
            "model_config": asdict(config),
            "completed_stages": completed.copy(),
            "stage": stage,
            "validation_metrics": best_metrics,
            "normalizer": asdict(normalizer),
            "train_indoor_sensor_ids": [
                database.series[index].indoor_id for index in train_indices
            ],
            "validation_indoor_sensor_ids": [
                database.series[index].indoor_id for index in validation_indices
            ],
            "train_assignment_ids": [
                assignment
                for index in train_indices
                for assignment in database.series[index].assignment_ids
            ],
            "validation_assignment_ids": [
                assignment
                for index in validation_indices
                for assignment in database.series[index].assignment_ids
            ],
            "training_data_sha256": training_data_sha256,
            "data_audit": audit,
            "seed": args.seed,
            "masking": {
                "sentinel": MASK_SENTINEL,
                "natural_missing_uses_sentinel": True,
                "target_mask_is_model_input": False,
            },
            "diagnostics": {
                "metrics_csv": str(diagnostics.metrics.resolve()),
                "loss_curve_png": str(diagnostics.loss_curve.resolve()),
                "reconstruction_examples_png": str(
                    diagnostics.reconstructions.resolve()
                ),
                "reconstruction_every_epochs": args.reconstruction_every_epochs,
            },
            "training": {
                "resumed_from": str(args.resume.resolve()) if args.resume else None,
                "optimizer_state_resumed": optimizer_resumed,
                "epochs_completed_this_run": len(metrics),
            },
            "transfer": {
                "retain_parameter_prefixes": ["input_projection.", "position", "encoder."],
                "discard_parameter_prefixes": ["reconstruction_head."],
                "input_feature_order": [
                    "outdoor_value",
                    "indoor_value",
                    "daily_sin",
                    "daily_cos",
                    "weekly_sin",
                    "weekly_cos",
                    "annual_sin",
                    "annual_cos",
                ],
            },
        }
        stage_checkpoint = _stage_checkpoint(args.checkpoint, stage)
        _save_checkpoint(stage_checkpoint, best_state, metadata, best_optimizer_state)
        _save_checkpoint(args.checkpoint, best_state, metadata, best_optimizer_state)
        print(f"saved={stage_checkpoint}")
    print(f"final_checkpoint={args.checkpoint}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    stage: str,
    normalizer: Normalizer,
    device: torch.device,
    mask_generator: torch.Generator,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    model.train(optimizer is not None)
    total_loss = total_targets = 0.0
    channel_squares = [0.0, 0.0]
    channel_counts = [0.0, 0.0]
    deviations = torch.tensor(
        normalizer.standard_deviation, dtype=torch.float32, device=device
    )
    for batch in loader:
        masked = mask_batch(
            batch["values"],
            batch["observed"],
            batch["time_features"],
            stage,
            mask_generator,
        )
        features = masked.features.to(device)
        target = masked.target.to(device)
        target_mask = masked.target_mask.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            prediction = model(features)
            squared = torch.square(prediction - target)
            loss = squared[target_mask].mean()
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        count = target_mask.sum().item()
        total_loss += loss.item() * count
        total_targets += count
        raw_squared = squared * torch.square(deviations.view(1, 1, 2))
        for channel in range(2):
            selected = target_mask[..., channel]
            channel_squares[channel] += raw_squared[..., channel][selected].sum().item()
            channel_counts[channel] += selected.sum().item()
    if not total_targets or min(channel_counts) == 0:
        raise RuntimeError("masking stage produced no reconstruction targets")
    return {
        "loss": total_loss / total_targets,
        "outdoor_rmse": math_sqrt(channel_squares[0] / channel_counts[0]),
        "indoor_rmse": math_sqrt(channel_squares[1] / channel_counts[1]),
        "target_count": total_targets,
    }


def _load_database(
    args: argparse.Namespace,
) -> tuple[PairDatabase, str]:
    tiers = set(args.responsiveness_tiers or ())
    selection, history = load_training_data(args.training_data, tiers)
    database = build_database(
        selection,
        history,
        args.history_hours,
        args.minimum_observed_hours,
        args.stride_hours,
    )
    return database, file_sha256(args.training_data)


def _audit_report(database: PairDatabase) -> dict[str, Any]:
    available_indoor = sum(
        int(summary["indoor_observations"] > 0) for summary in database.sensor_summaries
    )
    available_outdoor = sum(
        int(summary["outdoor_observations"] > 0) for summary in database.sensor_summaries
    )
    return {
        "input_indoor_sensors": database.selection.indoor_sensor_count,
        "selected_indoor_sensors": len(database.selection.indoor_sensor_ids),
        "selected_training_intervals": len(database.selection.intervals),
        "intervals_removed_by_responsiveness": (
            database.selection.responsiveness_filtered_interval_count
        ),
        "sensors_with_indoor_history": available_indoor,
        "sensors_with_outdoor_history": available_outdoor,
        "sensors_with_eligible_windows": len(database.series),
        "eligible_windows": database.window_count,
        "outdoor_handoffs": database.outdoor_handoff_count,
        "windows_crossing_outdoor_handoffs": database.windows_crossing_handoffs,
        "hard_gap_hours": sum(
            int(summary["hard_gap_hours"]) for summary in database.sensor_summaries
        ),
        "history_rows_loaded": database.history.row_count,
        "history_hours": database.history_hours,
        "minimum_observed_hours_per_channel": database.minimum_observed_hours,
        "stride_hours": database.stride_hours,
        "sensors": list(database.sensor_summaries),
    }


def _validate_training_arguments(args: argparse.Namespace) -> None:
    positive = (
        "model_dim",
        "layers",
        "heads",
        "patience",
        "batch_size",
    )
    for name in positive:
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if not 0 <= args.dropout < 1:
        raise ValueError("dropout must be between zero and one")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.minimum_delta < 0:
        raise ValueError("learning_rate, weight_decay, and minimum_delta are invalid")
    if args.reconstruction_every_epochs < 0:
        raise ValueError("reconstruction_every_epochs cannot be negative")


def _resolve_stage_epochs(values: list[int], stages: list[str]) -> list[int]:
    if any(value < 1 for value in values):
        raise ValueError("epochs_per_stage values must be positive")
    if len(values) == 1:
        return values * len(stages)
    if len(values) != len(stages):
        raise ValueError(
            "--epochs-per-stage requires one value or one value per selected stage "
            f"({len(stages)} stages selected; {len(values)} values supplied)"
        )
    return values


def _validate_resume_checkpoint(
    checkpoint: dict[str, Any],
    model_name: str,
    config: ModelConfig,
    normalizer: Normalizer,
    database: PairDatabase,
    train_indices: list[int],
    validation_indices: list[int],
    training_data_sha256: str,
) -> None:
    metadata = checkpoint["metadata"]
    expected = {
        "model_name": model_name,
        "model_config": asdict(config),
        "normalizer": asdict(normalizer),
        "train_indoor_sensor_ids": [
            database.series[index].indoor_id for index in train_indices
        ],
        "validation_indoor_sensor_ids": [
            database.series[index].indoor_id for index in validation_indices
        ],
        "train_assignment_ids": [
            assignment
            for index in train_indices
            for assignment in database.series[index].assignment_ids
        ],
        "validation_assignment_ids": [
            assignment
            for index in validation_indices
            for assignment in database.series[index].assignment_ids
        ],
        "training_data_sha256": training_data_sha256,
    }
    mismatches = []
    for name, value in expected.items():
        actual = metadata.get(name)
        if name == "model_name" and isinstance(actual, str):
            actual = canonical_model_name(actual)
        if actual != value:
            mismatches.append(name)
    if mismatches:
        raise ValueError(
            "resume checkpoint does not match the current training configuration: "
            + ", ".join(mismatches)
        )


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("model_state"), dict
    ) or not isinstance(checkpoint.get("metadata"), dict):
        raise ValueError(f"invalid masked pretraining checkpoint: {path}")
    metadata = checkpoint["metadata"]
    if metadata.get("stage") not in STAGES or "loss" not in metadata.get(
        "validation_metrics", {}
    ):
        raise ValueError(f"masked pretraining checkpoint is incomplete: {path}")
    return checkpoint


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(value)
    if device.type == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("requested XPU is unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _validate_model_contract(
    model: nn.Module, config: ModelConfig, device: torch.device
) -> None:
    with torch.no_grad():
        result = model(
            torch.zeros(2, config.history_hours, config.input_features, device=device)
        )
    expected = (2, config.history_hours, config.output_channels)
    if tuple(result.shape) != expected:
        raise ValueError(f"model returned {list(result.shape)}; expected {list(expected)}")


def _cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _stage_checkpoint(path: Path, stage: str) -> Path:
    return path.with_name(f"{path.stem}.{stage}{path.suffix or '.pt'}")


def _save_checkpoint(
    path: Path,
    state: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    optimizer_state: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        checkpoint = {"model_state": state, "metadata": metadata}
        if optimizer_state is not None:
            checkpoint["optimizer_state"] = optimizer_state
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    _write_json(path.with_suffix(".json"), metadata)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def math_sqrt(value: float) -> float:
    return float(np.sqrt(value))


def format_duration(seconds: float) -> str:
    hours, seconds = divmod(max(0, round(seconds)), 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


if __name__ == "__main__":
    main()
