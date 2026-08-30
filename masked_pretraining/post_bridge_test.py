"""Measure reconstruction loss before and after the synthetic bridge."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from inference.reporting import write_csv_report
from pm25_models import validate_bridge_checkpoint

from .data import (
    Normalizer,
    PairWindowDataset,
    build_database,
    file_sha256,
    load_training_data,
)
from .diagnostics import read_metrics
from .masking import ALL_STAGES, STAGES
from .models import ModelConfig, build_model, canonical_model_name
from .train import (
    _load_checkpoint,
    _resolve_device,
    _validate_resume_checkpoint,
    run_epoch,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAINING_DATA = (
    ROOT
    / "inputs"
    / "reconstruction"
    / "all_sensors_exclusion_informed_finetuned_masked_training_data.csv"
)
DEFAULT_CHECKPOINT_ROOT = ROOT / "inference" / "checkpoints"
DEFAULT_METRICS_ROOT = ROOT / "inference" / "metrics"
DEFAULT_OUTPUT = (
    ROOT
    / "inference"
    / "reports"
    / "all_excl_fine_t_hp_post_bridge_reconstruction_loss.csv"
)
MODELS = (
    "gru",
    "single-self-attention-encoder",
    "dual-encoder-cross-fusion",
    "dual-encoder-cross-fusion-outdoor-availability-recency",
)
CONFIGURATIONS = (
    ("learning-rate", "2e-4", "2e-4", 64, 3, 4),
    ("learning-rate", "1e-4", "1e-4", 64, 3, 4),
    *(
        ("model-dim", str(value), "3e-4", value, 3, 4)
        for value in (16, 32, 64, 128, 256)
    ),
    *(
        ("transformer-depth", str(value), "3e-4", 64, value, 4)
        for value in range(1, 7)
    ),
    *(
        ("head-size", str(value), "3e-4", 64, 3, value)
        for value in (1, 2, 4, 8, 16)
    ),
)
ROW_FIELDS = (
    "model_number",
    "artifact_name",
    "model_name",
    "sweep",
    "sweep_value",
    "learning_rate",
    "model_dim",
    "layers",
    "heads",
    "comparison",
    "split",
    *STAGES,
)


@dataclass(frozen=True)
class SweepRun:
    number: int
    sweep: str
    sweep_value: str
    learning_rate: str
    model_dim: int
    layers: int
    heads: int
    model_name: str

    @property
    def artifact_name(self) -> str:
        return (
            f"all_excl_fine_t_hp_{self.sweep}_{self.sweep_value}_"
            f"lr{self.learning_rate}_dim{self.model_dim}_depth{self.layers}_"
            f"heads{self.heads}_{self.model_name}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the recorded reconstruction-stage losses with inference loss "
            "from all 72 final bridge checkpoints. Each model produces six CSV rows."
        )
    )
    parser.add_argument("--training-data", type=Path, default=DEFAULT_TRAINING_DATA)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--metrics-root", type=Path, default=DEFAULT_METRICS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the 72 expected checkpoint and metrics pairs without opening them.",
    )
    return parser


def expected_runs() -> list[SweepRun]:
    runs = []
    for configuration in CONFIGURATIONS:
        for model_name in MODELS:
            runs.append(SweepRun(len(runs) + 1, *configuration, model_name))
    return runs


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1 or args.workers < 0 or args.torch_threads < 0:
        raise ValueError(
            "batch_size must be positive; workers and torch_threads cannot be negative"
        )
    runs = expected_runs()
    paths = [
        (
            run,
            args.checkpoint_root / f"{run.artifact_name}.pt",
            args.metrics_root / f"{run.artifact_name}.csv",
        )
        for run in runs
    ]
    if args.dry_run:
        for run, checkpoint_path, metrics_path in paths:
            print(
                f"model={run.number}/72 checkpoint={checkpoint_path} "
                f"metrics={metrics_path}"
            )
        print(f"output={args.output} rows={len(runs) * 6}")
        return

    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    device = _resolve_device(args.device)
    datasets = _preflight(paths, args.training_data)
    print(
        f"verified=72 training_windows={len(datasets[runs[0].artifact_name][0])} "
        f"validation_windows={len(datasets[runs[0].artifact_name][1])} device={device}"
    )
    write_csv_report(args.output, ROW_FIELDS, [])
    rows: list[dict[str, object]] = []
    for run, checkpoint_path, metrics_path in paths:
        checkpoint = _load_checkpoint(checkpoint_path)
        train_data, validation_data = datasets[run.artifact_name]
        metadata = checkpoint["metadata"]
        normalizer = Normalizer(**metadata["normalizer"])
        loaders = {
            "training": DataLoader(
                train_data,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
            ),
            "validation": DataLoader(
                validation_data,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
            ),
        }
        config = ModelConfig(**metadata["model_config"])
        model = build_model(run.model_name, config).to(device)
        model.load_state_dict(checkpoint["model_state"])
        baseline = _recorded_stage_losses(read_metrics(metrics_path))
        post_bridge = {
            split: _inference_losses(
                model,
                loader,
                normalizer,
                device,
                int(metadata["seed"]),
                split,
            )
            for split, loader in loaders.items()
        }
        rows.extend(_comparison_rows(run, baseline, post_bridge))
        write_csv_report(args.output, ROW_FIELDS, rows)
        print(f"completed={run.number}/72 artifact={run.artifact_name}")
        del model, checkpoint
        if device.type in {"cuda", "xpu"}:
            getattr(torch, device.type).empty_cache()
    print(f"output={args.output.resolve()} models={len(runs)} rows={len(rows)}")


def _preflight(
    paths: list[tuple[SweepRun, Path, Path]], training_data: Path
) -> dict[str, tuple[PairWindowDataset, PairWindowDataset]]:
    missing = [
        path
        for _, checkpoint, metrics in paths
        for path in (checkpoint, metrics)
        if not path.is_file()
    ]
    if not training_data.is_file():
        missing.insert(0, training_data)
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        suffix = f"\n... and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"required post-bridge test inputs are missing:\n{preview}{suffix}"
        )

    digest = file_sha256(training_data)
    first = _load_checkpoint(paths[0][1])
    audit = first["metadata"].get("data_audit", {})
    selection, history = load_training_data(training_data)
    database = build_database(
        selection,
        history,
        int(first["metadata"]["model_config"]["history_hours"]),
        int(audit["minimum_observed_hours_per_channel"]),
        int(audit["stride_hours"]),
    )
    by_sensor = {series.indoor_id: index for index, series in enumerate(database.series)}
    datasets: dict[tuple[Any, ...], tuple[PairWindowDataset, PairWindowDataset]] = {}
    run_datasets = {}
    for run, checkpoint_path, metrics_path in paths:
        checkpoint = (
            first if checkpoint_path == paths[0][1] else _load_checkpoint(checkpoint_path)
        )
        metadata = validate_bridge_checkpoint(checkpoint, run.model_name)
        config = ModelConfig(**metadata["model_config"])
        _validate_sweep_configuration(checkpoint, run, config)
        train_indices = _series_indices(metadata, "train_indoor_sensor_ids", by_sensor)
        validation_indices = _series_indices(
            metadata, "validation_indoor_sensor_ids", by_sensor
        )
        normalizer = Normalizer.fit(database.series, train_indices)
        _validate_resume_checkpoint(
            checkpoint,
            run.model_name,
            config,
            normalizer,
            database,
            train_indices,
            validation_indices,
            digest,
        )
        _recorded_stage_losses(read_metrics(metrics_path))
        key = (
            tuple(train_indices),
            tuple(validation_indices),
            *normalizer.mean,
            *normalizer.standard_deviation,
        )
        if key not in datasets:
            datasets[key] = (
                PairWindowDataset(database, train_indices, normalizer),
                PairWindowDataset(database, validation_indices, normalizer),
            )
        run_datasets[run.artifact_name] = datasets[key]
    return run_datasets


def _validate_sweep_configuration(
    checkpoint: dict[str, Any], run: SweepRun, config: ModelConfig
) -> None:
    actual_model = canonical_model_name(str(checkpoint["metadata"]["model_name"]))
    expected_config = ModelConfig(
        model_dim=run.model_dim,
        layers=run.layers,
        heads=run.heads,
    )
    if actual_model != run.model_name or config != expected_config:
        raise ValueError(f"sweep configuration mismatch for {run.artifact_name}")
    optimizer = checkpoint.get("optimizer_state", {})
    groups = optimizer.get("param_groups", [])
    if not groups or not math.isclose(float(groups[0]["lr"]), float(run.learning_rate)):
        raise ValueError(f"learning-rate mismatch for {run.artifact_name}")


def _series_indices(
    metadata: dict[str, Any], field: str, by_sensor: dict[int, int]
) -> list[int]:
    sensor_ids = metadata.get(field)
    if (
        not isinstance(sensor_ids, list)
        or not sensor_ids
        or len(sensor_ids) != len(set(sensor_ids))
    ):
        raise ValueError(f"checkpoint has invalid {field}")
    missing = [sensor_id for sensor_id in sensor_ids if sensor_id not in by_sensor]
    if missing:
        raise ValueError(f"checkpoint {field} contains unavailable sensors: {missing[:5]}")
    return [by_sensor[sensor_id] for sensor_id in sensor_ids]


def _recorded_stage_losses(
    metrics: list[dict[str, object]],
) -> dict[str, dict[str, float]]:
    result = {"training": {}, "validation": {}}
    for stage in STAGES:
        candidates = [row for row in metrics if row["stage"] == stage]
        if not candidates:
            raise ValueError(f"metrics CSV has no recorded reconstruction stage {stage!r}")
        selected = [row for row in candidates if row["improved_checkpoint"]]
        stage_end = selected[-1] if selected else min(
            candidates, key=lambda row: float(row["validation_loss"])
        )
        result["training"][stage] = float(stage_end["train_loss"])
        result["validation"][stage] = float(stage_end["validation_loss"])
    return result


def _inference_losses(
    model: torch.nn.Module,
    loader: DataLoader,
    normalizer: Normalizer,
    device: torch.device,
    seed: int,
    split: str,
) -> dict[str, float]:
    seed_multiplier = 10_000 if split == "training" else 1_000_000
    with torch.inference_mode():
        return {
            stage: run_epoch(
                model,
                loader,
                stage,
                normalizer,
                device,
                torch.Generator().manual_seed(
                    seed + (ALL_STAGES.index(stage) + 1) * seed_multiplier
                ),
            )["loss"]
            for stage in STAGES
        }


def _comparison_rows(
    run: SweepRun,
    baseline: dict[str, dict[str, float]],
    post_bridge: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    common = {
        "model_number": run.number,
        "artifact_name": run.artifact_name,
        "model_name": run.model_name,
        "sweep": run.sweep,
        "sweep_value": run.sweep_value,
        "learning_rate": run.learning_rate,
        "model_dim": run.model_dim,
        "layers": run.layers,
        "heads": run.heads,
    }
    rows = []
    for comparison, values in (
        ("pre_bridge_recorded", baseline),
        ("post_bridge_inference", post_bridge),
        (
            "difference_post_minus_pre",
            {
                split: {
                    stage: post_bridge[split][stage] - baseline[split][stage]
                    for stage in STAGES
                }
                for split in ("training", "validation")
            },
        ),
    ):
        for split in ("training", "validation"):
            rows.append(
                {
                    **common,
                    "comparison": comparison,
                    "split": split,
                    **{stage: round(values[split][stage], 10) for stage in STAGES},
                }
            )
    return rows


if __name__ == "__main__":
    main()
