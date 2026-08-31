r"""Run reconstruction, bridge, and paired 18-hour forecast training.

Usage:
    .venv\Scripts\python.exe scripts\run_four_core_18h_bridge_degradation_pipeline.py
    .venv\Scripts\python.exe scripts\run_four_core_18h_bridge_degradation_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from functools import cache
import math
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from masked_pretraining.data import file_sha256
from masked_pretraining.masking import STAGES, TEMPO_BRIDGE_STAGES
from masked_pretraining.models import model_names
from pm25_models import validate_bridge_checkpoint


MODELS = (
    "gru",
    "single-self-attention-encoder",
    "dual-encoder-cross-fusion",
    "dual-encoder-cross-fusion-outdoor-availability-recency",
)
ALL_HISTORY_STAGES = STAGES + TEMPO_BRIDGE_STAGES
HYPERPARAMETER_STEM = "all_excl_fine_t_hp_model-dim_64_lr3e-4_dim64_depth3_heads4"
RECONSTRUCTION_DATA = (
    REPOSITORY_ROOT
    / "inputs/reconstruction/all_sensors_exclusion_informed_finetuned_masked_training_data.csv"
)
FORECAST_DATA = (
    REPOSITORY_ROOT
    / "inputs/forecasting/all_old_training_data_exclusion_informed_finetuned_cyclical.csv"
)
PYTHON = REPOSITORY_ROOT / ".venv/Scripts/python.exe"
CHECKPOINT_ROOT = REPOSITORY_ROOT / "inference/checkpoints"
METRICS_ROOT = REPOSITORY_ROOT / "inference/metrics"
GRAPH_ROOT = REPOSITORY_ROOT / "inference/graphs"
REPORT_ROOT = REPOSITORY_ROOT / "inference/reports"
RECONSTRUCTION_ROOT = REPOSITORY_ROOT / "inference/reconstructions"
SUMMARY = REPORT_ROOT / "four_core_18h_bridge_degradation.csv"
FORECAST_EPOCHS = 50
FORECAST_HORIZONS = (3, 6, 12, 18)
FORECAST_STAGE_EPOCHS = (5, 5, 10, 30)


@dataclass(frozen=True)
class Artifacts:
    source_model: str

    @property
    def source_name(self) -> str:
        return f"{HYPERPARAMETER_STEM}_{self.source_model}"

    @property
    def forecast_model(self) -> str:
        return f"bridge-forecast-{self.source_model}"

    @property
    def reconstruction_checkpoint(self) -> Path:
        return CHECKPOINT_ROOT / f"{self.source_name}.pt"

    @property
    def bridge_checkpoint(self) -> Path:
        return CHECKPOINT_ROOT / f"{self.source_name}_bridge.pt"

    @property
    def history_metrics(self) -> Path:
        return METRICS_ROOT / f"{self.source_name}.csv"

    def forecast_name(self, source_stage: str) -> str:
        return f"{HYPERPARAMETER_STEM}_{self.forecast_model}-from-{source_stage}-18h"

    def forecast_checkpoint(self, source_stage: str) -> Path:
        return CHECKPOINT_ROOT / f"{self.forecast_name(source_stage)}.pt"

    def forecast_recovery(self, source_stage: str) -> Path:
        checkpoint = self.forecast_checkpoint(source_stage)
        return checkpoint.with_name(f"{checkpoint.stem}.last.pt")

    def forecast_metrics(self, source_stage: str) -> Path:
        return METRICS_ROOT / f"{self.forecast_name(source_stage)}.csv"

    def forecast_graph(self, source_stage: str) -> Path:
        return GRAPH_ROOT / f"{self.forecast_name(source_stage)}.png"

    def forecast_report(self, source_stage: str) -> Path:
        return REPORT_ROOT / f"{self.forecast_name(source_stage)}.csv"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--device", default="auto")
    result.add_argument("--epochs-per-stage", type=int, default=20)
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--workers", type=int, default=0)
    result.add_argument("--torch-threads", type=int, default=0)
    return result


def run(command: list[str], quiet: bool = False) -> None:
    print(subprocess.list2cmdline(command), flush=True)
    subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.DEVNULL if quiet else None,
    )


@cache
def training_data_hash(path: Path) -> str:
    return file_sha256(path)


def load_history(path: Path, source_model: str) -> tuple[dict, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("metadata"), dict
    ):
        raise ValueError(f"invalid history checkpoint: {path}")
    metadata = checkpoint["metadata"]
    config = metadata.get("model_config", {})
    if metadata.get("model_name") != source_model:
        raise ValueError(f"history model mismatch: {path}")
    if metadata.get("training_data_sha256") != training_data_hash(RECONSTRUCTION_DATA):
        raise ValueError(f"history training-data hash mismatch: {path}")
    expected = {"model_dim": 64, "layers": 3, "heads": 4}
    if any(config.get(name) != value for name, value in expected.items()):
        raise ValueError(f"history configuration mismatch: {path}")
    groups = checkpoint.get("optimizer_state", {}).get("param_groups", ())
    if not groups or not math.isclose(groups[0].get("lr", 0), 3e-4):
        raise ValueError(f"history optimizer learning-rate mismatch: {path}")
    return checkpoint, metadata


def remaining_history_stages(
    path: Path, source_model: str, curriculum: tuple[str, ...]
) -> tuple[str, ...]:
    _, metadata = load_history(path, source_model)
    completed = set(metadata.get("completed_stages", ()))
    return tuple(stage for stage in curriculum if stage not in completed)


def reconstruction_command(run_paths: Artifacts, args: argparse.Namespace) -> list[str]:
    command = [
        str(PYTHON),
        "-m",
        "masked_pretraining",
        "train",
        "--training-data",
        str(RECONSTRUCTION_DATA),
        "--model",
        run_paths.source_model,
        "--learning-rate",
        "3e-4",
        "--model-dim",
        "64",
        "--layers",
        "3",
        "--heads",
        "4",
    ]
    checkpoint = run_paths.reconstruction_checkpoint
    if checkpoint.is_file():
        remaining = remaining_history_stages(checkpoint, run_paths.source_model, STAGES)
        if not remaining:
            return []
        if not run_paths.history_metrics.is_file():
            raise FileNotFoundError(f"resume metrics not found: {run_paths.history_metrics}")
        command.extend(("--resume", str(checkpoint)))
    else:
        remaining = STAGES
    command.extend(
        (
            "--stages",
            *remaining,
            "--epochs-per-stage",
            str(args.epochs_per_stage),
            "--patience",
            str(args.epochs_per_stage),
            "--reconstruction-every-epochs",
            "5",
            "--reconstruction-output",
            str(RECONSTRUCTION_ROOT / run_paths.source_name / "run.reconstruction_examples.png"),
            "--loss-curve-output",
            str(GRAPH_ROOT / f"{run_paths.source_name}.png"),
            "--metrics-output",
            str(run_paths.history_metrics),
            "--report-output",
            str(REPORT_ROOT / f"{run_paths.source_name}.csv"),
            "--final-checkpoint-only",
            "--batch-size",
            str(args.batch_size),
            "--workers",
            str(args.workers),
            "--torch-threads",
            str(args.torch_threads),
            "--device",
            args.device,
            "--checkpoint",
            str(checkpoint),
        )
    )
    return command


def bridge_command(run_paths: Artifacts, args: argparse.Namespace) -> list[str]:
    bridge = run_paths.bridge_checkpoint
    if bridge.is_file():
        remaining = remaining_history_stages(
            bridge, run_paths.source_model, ALL_HISTORY_STAGES
        )
        if not remaining:
            return []
        resume = bridge
    else:
        remaining = TEMPO_BRIDGE_STAGES
        resume = run_paths.reconstruction_checkpoint
    if not run_paths.history_metrics.is_file():
        raise FileNotFoundError(f"resume metrics not found: {run_paths.history_metrics}")
    return [
        str(PYTHON),
        "-m",
        "masked_pretraining",
        "train",
        "--training-data",
        str(RECONSTRUCTION_DATA),
        "--model",
        run_paths.source_model,
        "--learning-rate",
        "3e-4",
        "--model-dim",
        "64",
        "--layers",
        "3",
        "--heads",
        "4",
        "--resume",
        str(resume),
        "--stages",
        *remaining,
        "--tempo-missingness-bridge",
        "--epochs-per-stage",
        str(args.epochs_per_stage),
        "--patience",
        str(args.epochs_per_stage),
        "--reconstruction-every-epochs",
        "5",
        "--reconstruction-output",
        str(RECONSTRUCTION_ROOT / run_paths.source_name / "run.reconstruction_examples.png"),
        "--loss-curve-output",
        str(GRAPH_ROOT / f"{run_paths.source_name}.png"),
        "--metrics-output",
        str(run_paths.history_metrics),
        "--report-output",
        str(REPORT_ROOT / f"{run_paths.source_name}.csv"),
        "--resume-metrics",
        "--final-checkpoint-only",
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--torch-threads",
        str(args.torch_threads),
        "--device",
        args.device,
        "--checkpoint",
        str(bridge),
    ]


def forecast_source(run_paths: Artifacts, source_stage: str) -> Path:
    return (
        run_paths.bridge_checkpoint
        if source_stage == "bridge"
        else run_paths.reconstruction_checkpoint
    )


def validate_forecast_checkpoint(
    run_paths: Artifacts, source_stage: str, checkpoint_path: Path
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_source = forecast_source(run_paths, source_stage).resolve()
    training = checkpoint.get("training_config", {})
    pretraining = checkpoint.get("pretraining", {})
    if checkpoint.get("model_name") != run_paths.forecast_model:
        raise ValueError(f"forecast model mismatch: {checkpoint_path}")
    if checkpoint.get("model_config", {}).get("prediction_hours") != 18:
        raise ValueError(f"forecast horizon mismatch: {checkpoint_path}")
    if training.get("forecast_horizons") != list(FORECAST_HORIZONS):
        raise ValueError(f"forecast curriculum mismatch: {checkpoint_path}")
    if training.get("horizon_stage_epochs") != list(FORECAST_STAGE_EPOCHS):
        raise ValueError(f"forecast stage epochs mismatch: {checkpoint_path}")
    if training.get("training_data_sha256") != training_data_hash(FORECAST_DATA):
        raise ValueError(f"forecast training-data hash mismatch: {checkpoint_path}")
    if pretraining.get("checkpoint") != str(expected_source):
        raise ValueError(f"forecast source path mismatch: {checkpoint_path}")
    if pretraining.get("checkpoint_sha256") != file_sha256(expected_source):
        raise ValueError(f"forecast source hash mismatch: {checkpoint_path}")
    expected_stage = (
        TEMPO_BRIDGE_STAGES[-1] if source_stage == "bridge" else STAGES[-1]
    )
    if pretraining.get("source_stage") != expected_stage or not pretraining.get(
        "weights_loaded"
    ):
        raise ValueError(f"forecast source stage mismatch: {checkpoint_path}")
    return checkpoint


def forecast_complete(run_paths: Artifacts, source_stage: str) -> bool:
    checkpoint_path = run_paths.forecast_checkpoint(source_stage)
    recovery_path = run_paths.forecast_recovery(source_stage)
    metrics_path = run_paths.forecast_metrics(source_stage)
    report_path = run_paths.forecast_report(source_stage)
    if not all(path.is_file() for path in (checkpoint_path, recovery_path, metrics_path, report_path)):
        return False
    validate_forecast_checkpoint(run_paths, source_stage, checkpoint_path)
    recovery = validate_forecast_checkpoint(run_paths, source_stage, recovery_path)
    with metrics_path.open(encoding="utf-8", newline="") as source:
        metrics = list(csv.DictReader(source))
    with report_path.open(encoding="utf-8", newline="") as source:
        report = list(csv.DictReader(source))
    expected_report_rows = {
        (split, str(horizon))
        for split in ("validation", "temporal_test", "location_test")
        for horizon in FORECAST_HORIZONS
    }
    checkpoint_hash = file_sha256(checkpoint_path)
    return (
        recovery.get("epoch") == FORECAST_EPOCHS
        and [int(row["epoch"]) for row in metrics] == list(range(1, FORECAST_EPOCHS + 1))
        and {(row["split"], row["horizon_hours"]) for row in report}
        == expected_report_rows
        and all(row["checkpoint_sha256"] == checkpoint_hash for row in report)
    )


def forecast_command(
    run_paths: Artifacts, source_stage: str, args: argparse.Namespace
) -> list[str]:
    checkpoint = run_paths.forecast_checkpoint(source_stage)
    command = [
        str(PYTHON),
        "pm25_transformer.py",
        "train",
        "--model",
        run_paths.forecast_model,
        "--pretrained-checkpoint",
        str(forecast_source(run_paths, source_stage)),
        "--pretrained-checkpoint-stage",
        source_stage,
        "--history-initialization",
        "pretrained",
        "--training-data",
        str(FORECAST_DATA),
        "--prediction-hours",
        "18",
        "--epochs",
        str(FORECAST_EPOCHS),
        "--freeze-history-epochs",
        "3",
        "--forecast-horizons",
        *map(str, FORECAST_HORIZONS),
        "--horizon-stage-epochs",
        *map(str, FORECAST_STAGE_EPOCHS),
        "--early-stopping-patience",
        "0",
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.workers),
        "--device",
        args.device,
        "--metrics-output",
        str(run_paths.forecast_metrics(source_stage)),
        "--loss-plot",
        str(run_paths.forecast_graph(source_stage)),
        "--report-output",
        str(run_paths.forecast_report(source_stage)),
        "--checkpoint",
        str(checkpoint),
    ]
    if checkpoint.is_file() or run_paths.forecast_recovery(source_stage).is_file():
        command.append("--resume")
    return command


def train_model(run_paths: Artifacts, args: argparse.Namespace) -> None:
    print(f"\n=== {run_paths.source_model} ===", flush=True)
    command = reconstruction_command(run_paths, args)
    if command:
        run(command)
    validate_bridge_checkpoint(
        load_history(run_paths.reconstruction_checkpoint, run_paths.source_model)[0],
        run_paths.source_model,
        source_stage="reconstruction",
    )
    print(f"verified reconstruction checkpoint: {run_paths.reconstruction_checkpoint}")

    command = bridge_command(run_paths, args)
    if command:
        run(command)
    validate_bridge_checkpoint(
        load_history(run_paths.bridge_checkpoint, run_paths.source_model)[0],
        run_paths.source_model,
        source_stage="bridge",
    )
    print(f"verified bridge checkpoint: {run_paths.bridge_checkpoint}")

    for source_stage in ("bridge", "reconstruction"):
        if not forecast_complete(run_paths, source_stage):
            run(forecast_command(run_paths, source_stage, args))
        if not forecast_complete(run_paths, source_stage):
            raise RuntimeError(
                f"forecast training did not complete from {source_stage}: "
                f"{run_paths.forecast_checkpoint(source_stage)}"
            )
        print(
            f"verified forecast-from-{source_stage} checkpoint: "
            f"{run_paths.forecast_checkpoint(source_stage)}"
        )


def report_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    return {(row["split"], row["horizon_hours"]): row for row in rows}


def write_summary(runs: list[Artifacts]) -> None:
    rows = []
    for run_paths in runs:
        bridge = report_rows(run_paths.forecast_report("bridge"))
        reconstruction = report_rows(run_paths.forecast_report("reconstruction"))
        if bridge.keys() != reconstruction.keys():
            raise ValueError(f"forecast report rows do not align: {run_paths.source_model}")
        for split, horizon in bridge:
            bridge_row = bridge[(split, horizon)]
            reconstruction_row = reconstruction[(split, horizon)]
            bridge_rmse = float(bridge_row["model_rmse_1_to_h_ug_m3"])
            reconstruction_rmse = float(
                reconstruction_row["model_rmse_1_to_h_ug_m3"]
            )
            bridge_mae = float(bridge_row["model_mae_1_to_h_ug_m3"])
            reconstruction_mae = float(
                reconstruction_row["model_mae_1_to_h_ug_m3"]
            )
            rows.append(
                {
                    "source_model": run_paths.source_model,
                    "split": split,
                    "horizon_hours": horizon,
                    "bridge_rmse_ug_m3": bridge_rmse,
                    "reconstruction_rmse_ug_m3": reconstruction_rmse,
                    "rmse_delta_bridge_minus_reconstruction_ug_m3": bridge_rmse
                    - reconstruction_rmse,
                    "bridge_degrades_rmse": bridge_rmse > reconstruction_rmse,
                    "bridge_mae_ug_m3": bridge_mae,
                    "reconstruction_mae_ug_m3": reconstruction_mae,
                    "mae_delta_bridge_minus_reconstruction_ug_m3": bridge_mae
                    - reconstruction_mae,
                    "bridge_degrades_mae": bridge_mae > reconstruction_mae,
                    "bridge_checkpoint": str(
                        run_paths.forecast_checkpoint("bridge").resolve()
                    ),
                    "reconstruction_checkpoint": str(
                        run_paths.forecast_checkpoint("reconstruction").resolve()
                    ),
                }
            )
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    temporary = SUMMARY.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(SUMMARY)


def print_dry_run(runs: list[Artifacts]) -> None:
    print(f"reconstruction_data={RECONSTRUCTION_DATA}")
    print(f"forecast_data={FORECAST_DATA}")
    for run_paths in runs:
        print(f"\nmodel={run_paths.source_model}")
        print(f"reconstruction_checkpoint={run_paths.reconstruction_checkpoint}")
        print(f"bridge_checkpoint={run_paths.bridge_checkpoint}")
        print(
            "forecast_from_bridge_checkpoint="
            f"{run_paths.forecast_checkpoint('bridge')}"
        )
        print(
            "forecast_from_reconstruction_checkpoint="
            f"{run_paths.forecast_checkpoint('reconstruction')}"
        )
    print(f"\ncomparison_report={SUMMARY}")


def main() -> None:
    args = parser().parse_args()
    if args.epochs_per_stage < 1 or args.batch_size < 1:
        raise ValueError("epochs-per-stage and batch-size must be positive")
    if args.workers < 0 or args.torch_threads < 0:
        raise ValueError("workers and torch-threads cannot be negative")
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError(f"run this script with the project environment: {PYTHON}")
    missing_models = set(MODELS) - set(model_names())
    if missing_models:
        raise RuntimeError(f"unregistered models: {', '.join(sorted(missing_models))}")
    runs = [Artifacts(model) for model in MODELS]
    if args.dry_run:
        print_dry_run(runs)
        return
    for path in (RECONSTRUCTION_DATA, FORECAST_DATA):
        if not path.is_file():
            raise FileNotFoundError(f"required training data not found: {path}")
    run(
        [
            str(PYTHON),
            "-m",
            "masked_pretraining",
            "audit",
            "--training-data",
            str(RECONSTRUCTION_DATA),
        ],
        quiet=True,
    )
    print("reconstruction training-data audit passed")
    for run_paths in runs:
        train_model(run_paths, args)
    write_summary(runs)
    print(f"\nVerified 16 checkpoint routes. Comparison report: {SUMMARY}")


if __name__ == "__main__":
    main()
