from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import (
    HistoryData,
    Normalizer,
    PairDatabase,
    PairWindowDataset,
    build_database,
    file_sha256,
    history_inventory_sha256,
    load_school_pairs,
    read_purpleair_history,
    split_series,
)
from .masking import STAGES, mask_batch
from .models import ModelConfig, build_model, model_names
from purpleair_pair_exclusions.outdoor_quality import (
    exclude_outdoor_readings,
    read_indoor_exclusions,
    read_outdoor_exclusions,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent
PURPLEAIR_ROOT = REPOSITORY_ROOT.parent / "purple-air-pull"
DEFAULT_PAIRS = PURPLEAIR_ROOT / "purpleair_continental_us_pairs.csv"
DEFAULT_HISTORIES = (
    PURPLEAIR_ROOT / "purpleair_hourly_pm25_atm",
    PURPLEAIR_ROOT / "purpleair_hourly_pm25_atm_indoor_schools_5y",
)
DEFAULT_SCHOOLS = (
    REPOSITORY_ROOT
    / "school_indoor_pm25"
    / "data"
    / "purpleair_indoor_school_sensors.csv"
)
DEFAULT_EXCLUSIONS = (
    REPOSITORY_ROOT / "permanently_excluded_indoor_sensors.csv",
    REPOSITORY_ROOT / "excluded_indoor_sensors_pm25_gt1000.csv",
    REPOSITORY_ROOT
    / "school_indoor_pm25"
    / "data"
    / "excluded_indoor_schools_pm25_gt1000.csv",
)
DEFAULT_OUTDOOR_EXCLUSIONS = (
    REPOSITORY_ROOT / "excluded_outdoor_purpleair_ranges.csv"
)
DEFAULT_INDOOR_RANGE_EXCLUSIONS = (
    REPOSITORY_ROOT / "excluded_indoor_purpleair_ranges.csv"
)
DEFAULT_RESPONSIVENESS = (
    REPOSITORY_ROOT / "pair_responsiveness" / "results" / "pair_responsiveness.csv"
)
DEFAULT_CHECKPOINT = PACKAGE_ROOT / "runs" / "masked_pretraining.pt"


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
    train.add_argument("--model", choices=model_names(), default="transformer")
    train.add_argument("--model-dim", type=int, default=64)
    train.add_argument("--layers", type=int, default=3)
    train.add_argument("--heads", type=int, default=4)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    train.add_argument("--epochs-per-stage", type=int, default=20)
    train.add_argument("--patience", type=int, default=3)
    train.add_argument("--minimum-delta", type=float, default=1e-4)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--validation-fraction", type=float, default=0.2)
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="auto")
    train.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    return parser


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--school-sensors", type=Path, default=DEFAULT_SCHOOLS)
    parser.add_argument(
        "--excluded-sensors",
        action="append",
        type=Path,
        help="Additional repeatable exclusion CSV; the validated school list is always used.",
    )
    parser.add_argument(
        "--history",
        action="append",
        type=Path,
        help="Repeatable PurpleAir hourly CSV or directory. TEMPO files are unsupported.",
    )
    parser.add_argument("--history-hours", type=int, default=168)
    parser.add_argument("--minimum-observed-hours", type=int, default=144)
    parser.add_argument("--stride-hours", type=int, default=24)
    parser.add_argument(
        "--responsiveness",
        type=Path,
        default=DEFAULT_RESPONSIVENESS,
        help="Pair responsiveness CSV used when --responsiveness-tiers is set.",
    )
    parser.add_argument(
        "--responsiveness-tiers",
        nargs="+",
        choices=("high", "moderate", "low"),
        help="Cumulative pair tiers to admit, for example: high moderate.",
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        database, sources = _load_database(args)
        report = _audit_report(database, sources)
        text = json.dumps(report, indent=2)
        print(text)
        if args.output:
            _write_json(args.output, report)
        return
    train(args)


def train(args: argparse.Namespace) -> None:
    _validate_training_arguments(args)
    _seed_everything(args.seed)
    database, sources = _load_database(args)
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
    completed: list[str] = []
    audit = _audit_report(database, sources)
    print(
        f"device={device} model={args.model} train_pairs={len(train_indices)} "
        f"validation_pairs={len(validation_indices)} train_windows={len(train_data)} "
        f"validation_windows={len(validation_data)}"
    )
    for stage_index, stage in enumerate(args.stages):
        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        stale_epochs = 0
        training_masks = torch.Generator().manual_seed(
            args.seed + (stage_index + 1) * 10_000
        )
        best_metrics: dict[str, float] | None = None
        for epoch in range(1, args.epochs_per_stage + 1):
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
            print(
                f"stage={stage} epoch={epoch}/{args.epochs_per_stage} "
                f"train_loss={train_metrics['loss']:.6f} "
                f"validation_loss={validation_metrics['loss']:.6f} "
                f"indoor_rmse={validation_metrics['indoor_rmse']:.3f} "
                f"outdoor_rmse={validation_metrics['outdoor_rmse']:.3f}"
            )
            if validation_metrics["loss"] < best_loss - args.minimum_delta:
                best_loss = validation_metrics["loss"]
                best_state = _cpu_state(model)
                best_metrics = validation_metrics
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= args.patience:
                    break
        if best_state is None or best_metrics is None:
            raise RuntimeError(f"stage {stage} produced no valid checkpoint")
        model.load_state_dict(best_state)
        completed.append(stage)
        metadata = {
            "format_version": 1,
            "model_name": args.model,
            "model_config": asdict(config),
            "completed_stages": completed.copy(),
            "stage": stage,
            "validation_metrics": best_metrics,
            "normalizer": asdict(normalizer),
            "train_pair_ids": [database.series[index].pair.pair_id for index in train_indices],
            "validation_pair_ids": [
                database.series[index].pair.pair_id for index in validation_indices
            ],
            "sources": sources,
            "data_audit": audit,
            "seed": args.seed,
            "transfer": {
                "retain_parameter_prefixes": ["input_projection.", "position", "encoder."],
                "discard_parameter_prefixes": ["reconstruction_head."],
                "input_feature_order": [
                    "outdoor_value",
                    "indoor_value",
                    "outdoor_visible",
                    "indoor_visible",
                    "outdoor_artificial_mask",
                    "indoor_artificial_mask",
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
        _save_checkpoint(stage_checkpoint, best_state, metadata)
        _save_checkpoint(args.checkpoint, best_state, metadata)
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
) -> tuple[PairDatabase, dict[str, Any]]:
    exclusions = list(
        dict.fromkeys([*DEFAULT_EXCLUSIONS, *(args.excluded_sensors or [])])
    )
    histories = args.history or list(DEFAULT_HISTORIES)
    tiers = set(args.responsiveness_tiers or ())
    selection = load_school_pairs(
        args.pairs,
        args.school_sensors,
        exclusions,
        args.responsiveness,
        tiers,
    )
    sensor_ids = {
        sensor_id
        for pair in selection.pairs
        for sensor_id in (pair.indoor_id, pair.outdoor_id)
    }
    history = read_purpleair_history(histories, sensor_ids)
    indoor_exclusions = read_indoor_exclusions(DEFAULT_INDOOR_RANGE_EXCLUSIONS)
    values, excluded_indoor_hours = exclude_outdoor_readings(
        history.values, indoor_exclusions
    )
    outdoor_exclusions = read_outdoor_exclusions(DEFAULT_OUTDOOR_EXCLUSIONS)
    values, excluded_outdoor_hours = exclude_outdoor_readings(
        values, outdoor_exclusions
    )
    history = HistoryData(
        values,
        history.files,
        history.row_count - excluded_indoor_hours - excluded_outdoor_hours,
    )
    database = build_database(
        selection,
        history,
        args.history_hours,
        args.minimum_observed_hours,
        args.stride_hours,
    )
    sources = {
        "pairs": _source_record(args.pairs),
        "school_sensors": _source_record(args.school_sensors),
        "excluded_sensors": [_source_record(path) for path in exclusions],
        "excluded_indoor_ranges": _source_record(DEFAULT_INDOOR_RANGE_EXCLUSIONS),
        "excluded_indoor_hours": excluded_indoor_hours,
        "excluded_outdoor_ranges": _source_record(DEFAULT_OUTDOOR_EXCLUSIONS),
        "excluded_outdoor_hours": excluded_outdoor_hours,
        "responsiveness": (
            {
                **_source_record(args.responsiveness),
                "included_tiers": sorted(tiers),
            }
            if tiers
            else None
        ),
        "purpleair_history_roots": [str(path.resolve()) for path in histories],
        "purpleair_history_file_count": len(history.files),
        "purpleair_history_inventory_sha256": history_inventory_sha256(history.files),
        "tempo_used": False,
    }
    return database, sources


def _audit_report(
    database: PairDatabase, sources: dict[str, Any]
) -> dict[str, Any]:
    available_indoor = sum(
        int(summary["indoor_observations"] > 0) for summary in database.pair_summaries
    )
    available_outdoor = sum(
        int(summary["outdoor_observations"] > 0) for summary in database.pair_summaries
    )
    return {
        "school_cohort_sensors": database.selection.school_sensor_count,
        "selected_genuine_pairs": len(database.selection.pairs),
        "exclusion_sensor_ids": list(database.selection.excluded_sensor_ids),
        "pairs_removed_by_exclusions": database.selection.excluded_pair_count,
        "pairs_removed_by_responsiveness": database.selection.responsiveness_filtered_pair_count,
        "pairs_with_indoor_history": available_indoor,
        "pairs_with_outdoor_history": available_outdoor,
        "pairs_with_eligible_windows": len(database.series),
        "eligible_windows": database.window_count,
        "history_rows_loaded": database.history.row_count,
        "history_hours": database.history_hours,
        "minimum_observed_hours_per_channel": database.minimum_observed_hours,
        "stride_hours": database.stride_hours,
        "sources": sources,
        "pairs": list(database.pair_summaries),
    }


def _source_record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def _validate_training_arguments(args: argparse.Namespace) -> None:
    positive = (
        "model_dim",
        "layers",
        "heads",
        "epochs_per_stage",
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
    path: Path, state: dict[str, torch.Tensor], metadata: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        torch.save({"model_state": state, "metadata": metadata}, temporary)
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


if __name__ == "__main__":
    main()
