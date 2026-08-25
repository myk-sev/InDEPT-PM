from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import random
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from inference.artifacts import artifact_paths, dataset_model_stem
from inference.reporting import write_csv_report

from data_loader import (
    PERMANENT_EXCLUSIONS_PATH,
    DualEncoderDataset,
    DualEncoderLoaders,
    SingularTrainingDataset,
    create_data_loaders,
    create_singular_data_loaders,
)
from pm25_models import (
    DEFAULT_MODEL,
    DualEncoderPatchTransformer,
    ModelConfig,
    PatchEmbedding,
    _missing_aware_history,
    bridge_config_values,
    bridge_forecast_name,
    bridge_history_model_name,
    build_config,
    build_model,
    load_bridge_checkpoint,
    model_names,
    validate_bridge_checkpoint,
)


CHECKPOINT_FORMAT_VERSION = 2
FORECAST_METRIC_FIELDS = (
    "epoch",
    "forecast_horizon",
    "train_loss",
    "validation_loss",
    "train_mae",
    "train_mse",
    "train_rmse",
    "validation_mae",
    "validation_mse",
    "validation_rmse",
    "pipeline_elapsed_seconds",
)
FINAL_REPORT_HORIZONS = (3, 6, 12, 24, 36)


@dataclass(frozen=True)
class ZScores:
    indoor_mean: float
    indoor_std: float
    history_outdoor_mean: float
    history_outdoor_std: float
    forecast_mean: float
    forecast_std: float
    encoder_indoor_mean: float | None = None
    encoder_indoor_std: float | None = None
    encoder_outdoor_mean: float | None = None
    encoder_outdoor_std: float | None = None

    def denormalize_indoor(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.indoor_std + self.indoor_mean


def fit_zscores(batches: Iterable[dict[str, torch.Tensor]]) -> ZScores:
    indoor = [0.0, 0.0, 0]
    history_outdoor = [0.0, 0.0, 0]
    forecast = [0.0, 0.0, 0]
    for batch in batches:
        _accumulate(indoor, batch["history"][..., 1])
        _accumulate(indoor, batch["target"])
        _accumulate(history_outdoor, batch["history"][..., 0])
        _accumulate(forecast, batch["forecast"][..., 0])
    indoor_mean, indoor_std = _mean_std(indoor, "indoor")
    history_mean, history_std = _mean_std(history_outdoor, "historical outdoor")
    forecast_mean, forecast_std = _mean_std(forecast, "forecast")
    return ZScores(
        indoor_mean,
        indoor_std,
        history_mean,
        history_std,
        forecast_mean,
        forecast_std,
    )


def _accumulate(total: list[float | int], values: torch.Tensor) -> None:
    values = values.detach().double()
    values = values[torch.isfinite(values)]
    total[0] += values.sum().item()
    total[1] += values.square().sum().item()
    total[2] += values.numel()


def _mean_std(total: list[float | int], label: str) -> tuple[float, float]:
    count = int(total[2])
    if not count:
        raise ValueError(f"cannot fit {label} z-score without training values")
    mean = float(total[0]) / count
    variance = float(total[1]) / count - mean * mean
    if variance <= 0:
        raise ValueError(f"{label} training values must have non-zero variance")
    return mean, math.sqrt(variance)


def normalize_batch(
    batch: dict[str, torch.Tensor],
    zscores: ZScores,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    history = batch["history"].to(device, non_blocking=True).clone()
    forecast = batch["forecast"].to(device, non_blocking=True).clone()
    target = batch["target"].to(device, non_blocking=True).clone()
    history[..., 0].sub_(
        zscores.encoder_outdoor_mean
        if zscores.encoder_outdoor_mean is not None
        else zscores.history_outdoor_mean
    ).div_(
        zscores.encoder_outdoor_std
        if zscores.encoder_outdoor_std is not None
        else zscores.history_outdoor_std
    )
    history[..., 1].sub_(
        zscores.encoder_indoor_mean
        if zscores.encoder_indoor_mean is not None
        else zscores.indoor_mean
    ).div_(
        zscores.encoder_indoor_std
        if zscores.encoder_indoor_std is not None
        else zscores.indoor_std
    )
    forecast[..., 0].sub_(zscores.forecast_mean).div_(zscores.forecast_std)
    target.sub_(zscores.indoor_mean).div_(zscores.indoor_std)
    return history, forecast, target


def make_loss(name: str, huber_delta: float = 1.0) -> nn.Module:
    if name == "mae":
        return nn.L1Loss()
    if name == "mse":
        return nn.MSELoss()
    if name == "huber":
        if huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        return nn.HuberLoss(delta=huber_delta)
    raise ValueError("loss must be mae, mse, or huber")


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_function: nn.Module,
    zscores: ZScores,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip: float = 0.0,
    horizon: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    frozen_history = getattr(model, "history", None)
    if training and frozen_history is not None and not any(
        parameter.requires_grad for parameter in frozen_history.parameters()
    ):
        frozen_history.eval()
    loss_total = absolute_total = squared_total = 0.0
    count = 0

    for batch in loader:
        history, forecast, target = normalize_batch(batch, zscores, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            prediction = model(history, forecast)
            if horizon is not None:
                prediction = prediction[..., :horizon]
                target = target[..., :horizon]
            loss = loss_function(prediction, target)
        if training:
            loss.backward()
            if gradient_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

        physical_prediction = zscores.denormalize_indoor(prediction.detach())
        physical_target = zscores.denormalize_indoor(target.detach())
        error = physical_prediction - physical_target
        elements = target.numel()
        loss_total += loss.item() * elements
        absolute_total += error.abs().sum().item()
        squared_total += error.square().sum().item()
        count += elements

    if not count:
        raise ValueError("data loader contains no samples")
    mse = squared_total / count
    return {
        "loss": loss_total / count,
        "mae": absolute_total / count,
        "mse": mse,
        "rmse": math.sqrt(mse),
    }


def evaluate_forecast_split(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    zscores: ZScores,
    device: torch.device,
    horizons: list[int],
) -> dict:
    model.eval()
    totals = {
        method: {
            metric: torch.zeros(horizons[-1], dtype=torch.float64, device=device)
            for metric in ("error", "absolute", "squared")
        }
        for method in ("model", "persistence")
    }
    samples = 0
    with torch.inference_mode():
        for batch in loader:
            history, forecast, _ = normalize_batch(batch, zscores, device)
            target = batch["target"].to(device).double()[..., : horizons[-1]]
            predictions = {
                "model": zscores.denormalize_indoor(
                    model(history, forecast)
                ).double()[..., : horizons[-1]],
                "persistence": batch["history"][:, -1, 1]
                .to(device)
                .double()[:, None]
                .expand_as(target),
            }
            for method, prediction in predictions.items():
                error = prediction - target
                totals[method]["error"] += error.sum(dim=0)
                totals[method]["absolute"] += error.abs().sum(dim=0)
                totals[method]["squared"] += error.square().sum(dim=0)
            samples += target.shape[0]
    if not samples:
        raise ValueError("data loader contains no samples")

    report = {
        method: _forecast_metrics(values, samples, horizons)
        for method, values in totals.items()
    }
    for horizon in horizons:
        model_rmse = report["model"]["by_horizon"][str(horizon)]["rmse"]
        persistence_rmse = report["persistence"]["by_horizon"][str(horizon)][
            "rmse"
        ]
        report["model"]["by_horizon"][str(horizon)][
            "rmse_skill_vs_persistence"
        ] = 1 - model_rmse / persistence_rmse if persistence_rmse else None
    return {"samples": samples, **report}


def _forecast_metrics(
    totals: dict[str, torch.Tensor], samples: int, horizons: list[int]
) -> dict:
    by_horizon = {}
    for horizon in horizons:
        values = samples * horizon
        mse = totals["squared"][:horizon].sum().item() / values
        by_horizon[str(horizon)] = {
            "mae": totals["absolute"][:horizon].sum().item() / values,
            "mse": mse,
            "rmse": math.sqrt(mse),
            "bias": totals["error"][:horizon].sum().item() / values,
            "values": values,
        }
    by_lead = []
    for lead in range(horizons[-1]):
        mse = totals["squared"][lead].item() / samples
        by_lead.append(
            {
                "forecast_hour": lead + 1,
                "mae": totals["absolute"][lead].item() / samples,
                "mse": mse,
                "rmse": math.sqrt(mse),
                "bias": totals["error"][lead].item() / samples,
            }
        )
    return {"by_horizon": by_horizon, "by_lead": by_lead}


def write_forecast_final_report(
    path: Path,
    checkpoint_path: Path,
    loaders: DualEncoderLoaders,
    device: torch.device,
) -> dict:
    model, config, zscores, checkpoint = load_checkpoint(checkpoint_path, device)
    horizons = [
        horizon
        for horizon in FINAL_REPORT_HORIZONS
        if horizon <= config.prediction_hours
    ]
    if not horizons or horizons[-1] != config.prediction_hours:
        horizons.append(config.prediction_hours)
    splits = {
        name: evaluate_forecast_split(
            model, getattr(loaders, name), zscores, device, horizons
        )
        for name in ("validation", "temporal_test", "location_test")
    }
    training_config = checkpoint["training_config"]
    generated_at = datetime.now(timezone.utc).isoformat()
    checkpoint_sha256 = file_sha256(checkpoint_path)
    rows = []
    for split, statistics in splits.items():
        for horizon in horizons:
            model_metrics = statistics["model"]["by_horizon"][str(horizon)]
            persistence = statistics["persistence"]["by_horizon"][str(horizon)]
            rows.append(
                {
                    "generated_at_utc": generated_at,
                    "model_name": checkpoint["model_name"],
                    "split": split,
                    "horizon_hours": horizon,
                    "samples": statistics["samples"],
                    "values": model_metrics["values"],
                    "model_rmse_1_to_h_ug_m3": model_metrics["rmse"],
                    "model_rmse_at_h_ug_m3": statistics["model"]["by_lead"][
                        horizon - 1
                    ]["rmse"],
                    "model_mae_1_to_h_ug_m3": model_metrics["mae"],
                    "model_bias_1_to_h_ug_m3": model_metrics["bias"],
                    "persistence_rmse_1_to_h_ug_m3": persistence["rmse"],
                    "persistence_rmse_at_h_ug_m3": statistics["persistence"][
                        "by_lead"
                    ][horizon - 1]["rmse"],
                    "rmse_skill_vs_persistence_pct": (
                        model_metrics["rmse_skill_vs_persistence"] * 100
                        if model_metrics["rmse_skill_vs_persistence"] is not None
                        else ""
                    ),
                    "selected_epoch": checkpoint["epoch"],
                    "selection_split": "validation",
                    "selection_metric": training_config["loss"],
                    "normalized_validation_loss": checkpoint["validation_loss"],
                    "history_initialization": training_config.get(
                        "history_initialization"
                    ),
                    "checkpoint_path": str(checkpoint_path.resolve()),
                    "checkpoint_sha256": checkpoint_sha256,
                    "training_data_path": training_config.get("training_data"),
                    "training_data_sha256": training_config.get(
                        "training_data_sha256"
                    ),
                }
            )
    write_csv_report(path, tuple(rows[0]), rows)
    return {"reported_horizons_hours": horizons, "final_model_metrics": splits}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: ModelConfig,
    zscores: ZScores,
    training_config: dict,
    epoch: int,
    validation_loss: float,
    model_name: str = DEFAULT_MODEL,
    optimizer: torch.optim.Optimizer | None = None,
    training_state: dict | None = None,
    pretraining: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_name": model_name,
        "model_config": asdict(config),
        "normalization": asdict(zscores),
        "training_config": training_config,
        "epoch": epoch,
        "validation_loss": validation_loss,
        "model_state": model.state_dict(),
    }
    if pretraining is not None:
        checkpoint["pretraining"] = pretraining
    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    if training_state is not None:
        checkpoint["training_state"] = training_state
    torch.save(checkpoint, temporary)
    for attempt in range(11):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 10:
                raise
            time.sleep(1)


def recovery_checkpoint_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.last{path.suffix}")


def output_path(value: Path | None, default: Path) -> Path:
    if value is None:
        return default
    return default.with_name(value.name) if value.parent == Path(".") else value


def write_forecast_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=FORECAST_METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[nn.Module, ModelConfig, ZScores, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "checkpoint predates the missing-aware history architecture; retrain it"
        )
    model_name = checkpoint.get("model_name", DEFAULT_MODEL)
    config = build_config(model_name, checkpoint["model_config"])
    zscores = ZScores(**checkpoint["normalization"])
    model = build_model(model_name, config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, config, zscores, checkpoint


def plot_prediction(
    output: Path,
    sample: dict[str, torch.Tensor],
    prediction: torch.Tensor,
    config: ModelConfig,
    location_id: str,
    split: str,
    sample_index: int,
    *,
    model_name: str | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    import seaborn as sns

    history_hours = np.arange(-config.history_hours + 1, 1)
    future_hours = np.arange(1, config.prediction_hours + 1)
    history = sample["history"].cpu().numpy()
    forecast = sample["forecast"][:, 0].cpu().numpy()
    target = sample["target"].cpu().numpy()
    predicted = prediction.detach().cpu().numpy()
    error = predicted - target
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    displayed = np.concatenate(
        (history[:, :2].ravel(), forecast, target, predicted)
    )
    displayed = displayed[np.isfinite(displayed)]
    padding = max(float(np.ptp(displayed)) * 0.08, 0.5)
    y_limits = (
        max(0, float(displayed.min()) - padding),
        float(displayed.max()) + padding,
    )
    history_title = (
        f"PAST {config.history_hours // 24} DAYS"
        if config.history_hours % 24 == 0
        else f"PAST {config.history_hours} HOURS"
    )
    anchor = datetime.fromtimestamp(
        int(sample["anchor_time_utc"]), timezone.utc
    ).strftime("%Y-%m-%d %H:%M UTC")

    sns.set_theme(style="ticks")
    figure, (history_axis, forecast_axis) = plt.subplots(
        1,
        2,
        figsize=(12, 6),
        sharey=True,
        gridspec_kw={"width_ratios": (1, 2), "wspace": 0.05},
    )
    history_lines = (
        (
            history_hours,
            history[:, 0],
            "Outdoor observed",
            "-",
            "#9C6B30",
        ),
        (
            history_hours,
            history[:, 1],
            "Indoor observed",
            "-",
            "#40566F",
        ),
    )
    forecast_lines = (
        (future_hours, forecast, "Outdoor forecast", "--", "#9C6B30"),
        (future_hours, target, "Indoor measured", "-", "#40566F"),
        (
            future_hours,
            predicted,
            "Model prediction",
            "--",
            "#267873",
        ),
    )
    for axis, lines in (
        (history_axis, history_lines),
        (forecast_axis, forecast_lines),
    ):
        for hours, values, label, style, color in lines:
            sns.lineplot(
                x=hours,
                y=values,
                label=label,
                linestyle=style,
                color=color,
                linewidth=(
                    2.25
                    if label in {"Indoor measured", "Model prediction"}
                    else 1.6
                ),
                ax=axis,
            )

    forecast_axis.axvline(
        0, color="#555A60", linestyle=":", linewidth=1.5
    )
    forecast_axis.annotate(
        "Forecast begins",
        xy=(0, y_limits[1]),
        xytext=(5, -4),
        textcoords="offset points",
        ha="left",
        va="top",
        color="#555A60",
        fontsize=9,
    )
    history_axis.set(
        xlabel="Days before forecast",
        ylabel="PM2.5 (µg/m³)",
        xlim=(-config.history_hours + 1, 0),
        ylim=y_limits,
    )
    forecast_axis.set(
        xlabel="Forecast lead hour",
        ylabel="",
        xlim=(0, config.prediction_hours),
    )
    history_axis.spines["right"].set_visible(False)
    forecast_axis.spines["left"].set_visible(False)
    forecast_axis.tick_params(axis="y", left=False)
    history_axis.set_title(history_title, loc="left", fontsize=10, color="#555A60")
    forecast_axis.set_title(
        f"{config.prediction_hours}-HOUR HOLDOUT EVALUATION",
        loc="left",
        fontsize=10,
        color="#555A60",
    )
    if config.history_hours == 168:
        history_axis.set_xticks(
            (-168, -120, -72, -24, 0),
            ("−7 d", "−5 d", "−3 d", "−1 d", "0"),
        )
    forecast_axis.set_xticks(
        np.arange(0, config.prediction_hours + 1, 6)
    )
    for axis in (history_axis, forecast_axis):
        axis.grid(axis="y", color="#DFE3E7", linewidth=0.8)
        axis.grid(axis="x", visible=False)

    handles = []
    labels = []
    for axis in (history_axis, forecast_axis):
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        handles.extend(axis_handles)
        labels.extend(axis_labels)
        axis.get_legend().remove()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.86),
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    title_prefix = f"{model_name}: " if model_name else ""
    figure.suptitle(
        f"{title_prefix}Model error averaged {mae:.2f} µg/m³ "
        f"over {config.prediction_hours} hours",
        y=0.99,
        fontsize=16,
    )
    figure.text(
        0.5,
        0.925,
        f"{location_id} · {anchor} · retrospective {split} sample {sample_index} · RMSE {rmse:.2f} µg/m³",
        ha="center",
        color="#555A60",
        fontsize=10,
    )
    figure.text(0.35, 0.14, "//", ha="center", color="#555A60", fontsize=13)
    figure.subplots_adjust(top=0.77, bottom=0.14, left=0.09, right=0.98)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def plot_training_losses(
    output: Path,
    training_losses: list[float],
    validation_losses: list[float],
    training_metrics: dict[str, float],
    validation_metrics: dict[str, float],
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    epochs = range(1, len(training_losses) + 1)
    figure, axis = plt.subplots(figsize=(8, 5))
    for name, losses, marker, metrics in (
        ("Training", training_losses, "o", training_metrics),
        ("Validation", validation_losses, "s", validation_metrics),
    ):
        label = (
            f"{name} — final loss={metrics['loss']:.4g}, "
            f"MAE={metrics['mae']:.4g}, MSE={metrics['mse']:.4g}, "
            f"RMSE={metrics['rmse']:.4g}"
        )
        axis.plot(epochs, losses, marker=marker, label=label)
    axis.set(xlabel="Epoch", ylabel="Loss", title="Training and validation loss")
    axis.set_xticks(list(epochs))
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.xpu.is_available():
            name = "xpu"
        else:
            name = "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if device.type == "xpu" and not torch.xpu.is_available():
        raise ValueError("XPU was requested but is not available")
    return device


def file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_duration(seconds: float) -> str:
    hours, seconds = divmod(max(0, round(seconds)), 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def build_loaders(
    pairs: Path,
    indoor_history: list[Path],
    outdoor_history: Path,
    forecast_root: Path,
    config: ModelConfig,
    training_config: dict,
    device: torch.device,
    balanced_training_index: Path | None = None,
    excluded_sensors: Path | None = None,
) -> tuple[DualEncoderDataset, DualEncoderLoaders]:
    dataset = DualEncoderDataset(
        pairs,
        indoor_history,
        outdoor_history,
        forecast_root,
        history_hours=config.history_hours,
        forecast_hours=config.prediction_hours,
        minimum_outdoor_history_hours=training_config.get(
            "minimum_outdoor_history_hours", 24
        ),
        excluded_sensors_path=excluded_sensors,
        cyclical_time=getattr(config, "cyclical_time", False),
    )
    loaders = create_data_loaders(
        dataset,
        batch_size=training_config["batch_size"],
        train_fraction=training_config["train_fraction"],
        validation_fraction=training_config["validation_fraction"],
        location_holdout_fraction=training_config["location_holdout_fraction"],
        seed=training_config["seed"],
        num_workers=training_config["num_workers"],
        pin_memory=device.type in {"cuda", "xpu"},
        balanced_training_index=balanced_training_index,
    )
    return dataset, loaders


def build_singular_loaders(
    training_data: Path,
    config: ModelConfig,
    training_config: dict,
    device: torch.device,
) -> tuple[SingularTrainingDataset, DualEncoderLoaders]:
    dataset = SingularTrainingDataset(
        training_data,
        history_hours=config.history_hours,
        forecast_hours=config.prediction_hours,
        cyclical_time=getattr(config, "cyclical_time", False),
    )
    loaders = create_singular_data_loaders(
        dataset,
        batch_size=training_config["batch_size"],
        seed=training_config["seed"],
        num_workers=training_config["num_workers"],
        pin_memory=device.type in {"cuda", "xpu"},
    )
    return dataset, loaders


def training_data_path(
    args: argparse.Namespace, recorded: dict | None = None
) -> Path | None:
    legacy = (
        args.pairs,
        args.indoor_history,
        args.outdoor_history,
        args.forecast_root,
    )
    training_data = args.training_data
    if training_data is None and not any(legacy) and recorded and recorded.get(
        "training_data"
    ):
        training_data = Path(recorded["training_data"])
    if training_data is not None:
        if any(legacy) or args.balanced_training_index or args.excluded_sensors:
            raise ValueError(
                "--training-data cannot be combined with legacy data source, "
                "balance, or exclusion arguments"
            )
        return training_data
    if not all(legacy):
        raise ValueError(
            "provide --training-data or all of --pairs, --indoor-history, "
            "--outdoor-history, and --forecast-root"
        )
    return None


def bridge_history_zscores(zscores: ZScores, checkpoint: dict) -> ZScores:
    normalizer = validate_bridge_checkpoint(checkpoint)["normalizer"]
    means = normalizer["mean"]
    deviations = normalizer["standard_deviation"]
    return replace(
        zscores,
        encoder_outdoor_mean=float(means[0]),
        encoder_outdoor_std=float(deviations[0]),
        encoder_indoor_mean=float(means[1]),
        encoder_indoor_std=float(deviations[1]),
    )


def resolve_horizon_schedule(
    prediction_hours: int,
    epochs: int,
    horizons: list[int],
    stage_epochs: list[int] | None,
) -> list[int]:
    if (
        not horizons
        or horizons != sorted(set(horizons))
        or horizons[-1] != prediction_hours
        or any(not 1 <= horizon <= prediction_hours for horizon in horizons)
    ):
        raise ValueError(
            "forecast horizons must be unique, increasing, positive, and end at "
            "prediction_hours"
        )
    if stage_epochs is None:
        if horizons != [prediction_hours]:
            raise ValueError(
                "--horizon-stage-epochs is required for a multi-stage forecast curriculum"
            )
        return [prediction_hours] * epochs
    if len(stage_epochs) != len(horizons) or any(value < 1 for value in stage_epochs):
        raise ValueError(
            "horizon stage epochs must be positive and match the forecast horizons"
        )
    if sum(stage_epochs) != epochs:
        raise ValueError("horizon stage epochs must sum to --epochs")
    return [
        horizon
        for horizon, count in zip(horizons, stage_epochs)
        for _ in range(count)
    ]


def history_parameters(model: nn.Module) -> tuple[nn.Parameter, ...]:
    getter = getattr(model, "history_parameters", None)
    return tuple(getter()) if getter else ()


def set_history_trainable(model: nn.Module, trainable: bool) -> int:
    parameters = history_parameters(model)
    for parameter in parameters:
        parameter.requires_grad_(trainable)
    return sum(parameter.numel() for parameter in parameters)


def bridge_provenance(
    path: Path,
    checkpoint: dict,
    weights_loaded: bool,
    transferred_names: tuple[str, ...] = (),
) -> dict:
    metadata = validate_bridge_checkpoint(checkpoint)
    return {
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": file_sha256(path),
        "source_model_name": metadata["model_name"],
        "source_stage": metadata["stage"],
        "source_completed_stages": metadata["completed_stages"],
        "source_training_data_sha256": metadata["training_data_sha256"],
        "weights_loaded": weights_loaded,
        "transferred_tensor_count": len(transferred_names),
        "transferred_parameter_count": sum(
            checkpoint["model_state"][name].numel() for name in transferred_names
        ),
    }


def train(args: argparse.Namespace) -> None:
    training_data = training_data_path(args)
    bridge_path = args.pretrained_checkpoint
    bridge_checkpoint = load_bridge_checkpoint(bridge_path) if bridge_path else None
    initialization = args.history_initialization or (
        "pretrained" if bridge_checkpoint else "random"
    )
    if initialization == "pretrained" and bridge_checkpoint is None:
        raise ValueError("pretrained history initialization requires a bridge checkpoint")
    config_values = dict(vars(args))
    if bridge_checkpoint:
        source_name = bridge_checkpoint["metadata"]["model_name"]
        required_model = bridge_forecast_name(source_name)
        if args.model == DEFAULT_MODEL:
            args.model = required_model
        elif args.model != required_model:
            raise ValueError(
                f"bridge checkpoint requires --model {required_model}, got {args.model}"
            )
        config_values.update(bridge_config_values(bridge_checkpoint))
    config = build_config(args.model, config_values)
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    if args.gradient_clip < 0 or args.early_stopping_patience < 0:
        raise ValueError(
            "gradient_clip and early_stopping_patience cannot be negative"
        )
    if args.checkpoint_every < 0:
        raise ValueError("checkpoint_every cannot be negative")
    if args.freeze_history_epochs < 0:
        raise ValueError("freeze_history_epochs cannot be negative")
    forecast_horizons = args.forecast_horizons or [config.prediction_hours]
    horizon_schedule = resolve_horizon_schedule(
        config.prediction_hours,
        args.epochs,
        forecast_horizons,
        args.horizon_stage_epochs,
    )

    set_seed(args.seed)
    device = resolve_device(args.device)
    source = training_data or args.pairs
    stem = (
        args.checkpoint.stem
        if args.checkpoint
        else dataset_model_stem(source, args.model)
    )
    paths = artifact_paths(stem)
    checkpoint_path = output_path(args.checkpoint, paths.checkpoint)
    metrics_path = output_path(args.metrics_output, paths.metrics)
    loss_plot = output_path(args.loss_plot, paths.graph)
    report_path = output_path(args.report_output, paths.report)
    recovery_path = recovery_checkpoint_path(checkpoint_path)
    training_config = {
        "model": args.model,
        "batch_size": args.batch_size,
        "train_fraction": args.train_fraction,
        "validation_fraction": args.validation_fraction,
        "location_holdout_fraction": args.location_holdout_fraction,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "loss": args.loss,
        "huber_delta": args.huber_delta,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "early_stopping_patience": args.early_stopping_patience,
        "epochs": args.epochs,
        "device": args.device,
        "minimum_outdoor_history_hours": args.minimum_outdoor_history_hours,
        "history_initialization": initialization,
        "freeze_history_epochs": args.freeze_history_epochs,
        "forecast_horizons": forecast_horizons,
        "horizon_stage_epochs": args.horizon_stage_epochs,
        "bridge_history_normalization": bool(
            bridge_checkpoint and args.bridge_history_normalization
        ),
    }
    if training_data is not None:
        training_config["training_data"] = str(training_data.resolve())
        training_config["training_data_sha256"] = file_sha256(training_data)
    else:
        training_config.update(
            {
                "pairs": str(args.pairs.resolve()),
                "indoor_history": [
                    str(path.resolve()) for path in args.indoor_history
                ],
                "outdoor_history": str(args.outdoor_history.resolve()),
                "forecast_root": str(args.forecast_root.resolve()),
                "permanent_excluded_sensors": str(PERMANENT_EXCLUSIONS_PATH),
                "permanent_excluded_sensors_sha256": file_sha256(
                    PERMANENT_EXCLUSIONS_PATH
                ),
            }
        )
    if training_data is None and args.balanced_training_index is not None:
        training_config["balanced_training_index"] = str(
            args.balanced_training_index.resolve()
        )
        training_config["balanced_training_index_sha256"] = file_sha256(
            args.balanced_training_index
        )
    if training_data is None and args.excluded_sensors is not None:
        training_config["excluded_sensors"] = str(args.excluded_sensors.resolve())
        training_config["excluded_sensors_sha256"] = file_sha256(
            args.excluded_sensors
        )
    if training_data is not None:
        dataset, loaders = build_singular_loaders(
            training_data, config, training_config, device
        )
    else:
        dataset, loaders = build_loaders(
            args.pairs,
            args.indoor_history,
            args.outdoor_history,
            args.forecast_root,
            config,
            training_config,
            device,
            args.balanced_training_index,
            args.excluded_sensors,
        )
    if loaders.balance_report is not None:
        training_config["balance_report"] = loaders.balance_report
        report = loaders.balance_report
        print(
            "balanced training "
            f"requested={report['requested_anchors']} "
            f"valid={report['valid_anchors']} "
            f"eligible={report['training_eligible_anchors']} "
            f"selected={report['selected_training_anchors']} "
            f"quota_per_cell={report['quota_per_cell']}"
        )
    loss_function = make_loss(args.loss, args.huber_delta)
    pretraining = None
    has_training_state = False
    if args.resume:
        resume_path = recovery_path if recovery_path.is_file() else checkpoint_path
        model, resumed_config, zscores, checkpoint = load_checkpoint(
            resume_path, device
        )
        saved_training_config = dict(checkpoint["training_config"])
        for key in ("device", "epochs"):
            saved_training_config.pop(key, None)
            training_config.pop(key, None)
        if resumed_config != config or saved_training_config != training_config:
            raise ValueError("resume arguments do not match the recovery checkpoint")
        training_config["device"] = args.device
        training_config["epochs"] = args.epochs
        pretraining = checkpoint.get("pretraining")
    else:
        zscores = fit_zscores(loaders.train)
        model = build_model(args.model, config)
        transferred_names: tuple[str, ...] = ()
        if bridge_checkpoint:
            validate_bridge_checkpoint(
                bridge_checkpoint,
                bridge_history_model_name(args.model),
                config,
            )
            if args.bridge_history_normalization:
                zscores = bridge_history_zscores(zscores, bridge_checkpoint)
            if initialization == "pretrained":
                transferred_names = model.load_pretrained_history(bridge_checkpoint)
            pretraining = bridge_provenance(
                bridge_path,
                bridge_checkpoint,
                initialization == "pretrained",
                transferred_names,
            )
        model = model.to(device)
    if args.freeze_history_epochs and not history_parameters(model):
        raise ValueError("freeze_history_epochs requires a bridge forecast model")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        foreach=False,
    )
    if args.resume:
        has_training_state = (
            "optimizer_state" in checkpoint and "training_state" in checkpoint
        )
        if has_training_state:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
    counts = {
        name: len(getattr(loaders, name).dataset)
        for name in ("train", "validation", "temporal_test", "location_test")
    }
    exclusions = "embedded_in_csv" if training_data is not None else len(
        dataset.excluded_sensor_ids
    )
    print(f"device={device} excluded_sensors={exclusions} samples={counts}")
    print(
        f"indoor z-score mean={zscores.indoor_mean:.6g} "
        f"std={zscores.indoor_std:.6g}; "
        f"historical outdoor mean={zscores.history_outdoor_mean:.6g} "
        f"std={zscores.history_outdoor_std:.6g}; "
        f"forecast mean={zscores.forecast_mean:.6g} "
        f"std={zscores.forecast_std:.6g}"
    )
    if zscores.encoder_indoor_mean is not None:
        print(
            "bridge history z-score "
            f"indoor mean={zscores.encoder_indoor_mean:.6g} "
            f"std={zscores.encoder_indoor_std:.6g}; "
            f"outdoor mean={zscores.encoder_outdoor_mean:.6g} "
            f"std={zscores.encoder_outdoor_std:.6g}"
        )

    if args.resume and has_training_state:
        state = checkpoint["training_state"]
        start_epoch = checkpoint["epoch"]
        if start_epoch >= args.epochs:
            raise ValueError(
                f"recovery checkpoint already completed {start_epoch} epochs; "
                "--epochs must be greater"
            )
        best_loss = state["best_loss"]
        stale_epochs = state["stale_epochs"]
        training_losses = state["training_losses"]
        validation_losses = state["validation_losses"]
        metrics = state.get("metrics", [])
        random.setstate(state["python_random_state"])
        np.random.set_state(state["numpy_random_state"])
        torch.set_rng_state(state["torch_random_state"].cpu())
        loaders.train.generator.set_state(state["loader_random_state"].cpu())
        if state["device_type"] == device.type and device.type != "cpu":
            getattr(torch, device.type).set_rng_state(
                state["device_random_state"].cpu(), device
            )
        print(f"resuming={resume_path} completed_epochs={start_epoch}")
    elif args.resume:
        start_epoch = checkpoint["epoch"]
        if start_epoch >= args.epochs:
            raise ValueError(
                f"checkpoint already completed {start_epoch} epochs; "
                "--epochs must be greater"
            )
        best_loss = checkpoint["validation_loss"]
        stale_epochs = 0
        training_losses = [math.nan] * start_epoch
        validation_losses = [math.nan] * start_epoch
        validation_losses[-1] = best_loss
        metrics = []
        set_seed(args.seed)
        print(
            f"resuming_weights={resume_path} completed_epochs={start_epoch} "
            "optimizer=fresh"
        )
    else:
        start_epoch = 0
        best_loss = math.inf
        stale_epochs = 0
        training_losses = []
        validation_losses = []
        metrics = []
    if len(metrics) != start_epoch:
        metrics = [
            {
                "epoch": epoch,
                "forecast_horizon": horizon_schedule[epoch - 1],
                "train_loss": training_losses[epoch - 1],
                "validation_loss": validation_losses[epoch - 1],
                "train_mae": math.nan,
                "train_mse": math.nan,
                "train_rmse": math.nan,
                "validation_mae": math.nan,
                "validation_mse": math.nan,
                "validation_rmse": math.nan,
                "pipeline_elapsed_seconds": 0,
            }
            for epoch in range(1, start_epoch + 1)
        ]
    elapsed_offset = (
        float(metrics[-1]["pipeline_elapsed_seconds"]) if metrics else 0.0
    )
    previous_horizon = (
        state.get("forecast_horizon")
        if args.resume and has_training_state
        else horizon_schedule[start_epoch - 1] if start_epoch else None
    )
    started = time.perf_counter()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        horizon = horizon_schedule[epoch - 1]
        if horizon != previous_horizon:
            best_loss = math.inf
            stale_epochs = 0
            previous_horizon = horizon
            print(f"forecast_horizon={horizon} hours")
        history_frozen = epoch <= args.freeze_history_epochs
        transferred_parameter_count = set_history_trainable(model, not history_frozen)
        training = run_epoch(
            model,
            loaders.train,
            loss_function,
            zscores,
            device,
            optimizer,
            args.gradient_clip,
            horizon,
        )
        validation = run_epoch(
            model,
            loaders.validation,
            loss_function,
            zscores,
            device,
            horizon=horizon,
        )
        training_losses.append(training["loss"])
        validation_losses.append(validation["loss"])
        elapsed = time.perf_counter() - started
        pipeline_elapsed = elapsed_offset + elapsed
        estimated_remaining = (
            elapsed / (epoch - start_epoch) * (args.epochs - epoch)
        )
        print(
            f"epoch={epoch}/{args.epochs} "
            f"horizon={horizon} "
            f"history_frozen={history_frozen and bool(transferred_parameter_count)} "
            f"time_taken={format_duration(elapsed)} "
            f"ETA={format_duration(estimated_remaining)} "
            f"train_loss={training['loss']:.6g} "
            f"val_loss={validation['loss']:.6g} "
            f"train_mae={training['mae']:.6g} "
            f"train_mse={training['mse']:.6g} "
            f"train_rmse={training['rmse']:.6g} "
            f"val_mae={validation['mae']:.6g} "
            f"val_mse={validation['mse']:.6g} "
            f"val_rmse={validation['rmse']:.6g}"
        )
        metrics.append(
            {
                "epoch": epoch,
                "forecast_horizon": horizon,
                "train_loss": training["loss"],
                "validation_loss": validation["loss"],
                "train_mae": training["mae"],
                "train_mse": training["mse"],
                "train_rmse": training["rmse"],
                "validation_mae": validation["mae"],
                "validation_mse": validation["mse"],
                "validation_rmse": validation["rmse"],
                "pipeline_elapsed_seconds": pipeline_elapsed,
            }
        )
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            stale_epochs = 0
            save_checkpoint(
                checkpoint_path,
                model,
                config,
                zscores,
                training_config,
                epoch,
                best_loss,
                args.model,
                pretraining=pretraining,
            )
        else:
            stale_epochs += 1
        state = {
            "best_loss": best_loss,
            "stale_epochs": stale_epochs,
            "training_losses": training_losses,
            "validation_losses": validation_losses,
            "metrics": metrics,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "loader_random_state": loaders.train.generator.get_state(),
            "device_type": device.type,
            "forecast_horizon": horizon,
            "history_frozen": history_frozen and bool(transferred_parameter_count),
        }
        if device.type != "cpu":
            state["device_random_state"] = getattr(
                torch, device.type
            ).get_rng_state(device)
        save_checkpoint(
            recovery_path,
            model,
            config,
            zscores,
            training_config,
            epoch,
            validation["loss"],
            args.model,
            optimizer,
            state,
            pretraining,
        )
        write_forecast_metrics(metrics_path, metrics)
        plot_training_losses(
            loss_plot, training_losses, validation_losses, training, validation
        )
        if args.checkpoint_every and epoch % args.checkpoint_every == 0:
            periodic_path = checkpoint_path.with_name(
                f"{checkpoint_path.stem}.epoch-{epoch:04d}{checkpoint_path.suffix}"
            )
            save_checkpoint(
                periodic_path,
                model,
                config,
                zscores,
                training_config,
                epoch,
                validation["loss"],
                args.model,
                optimizer,
                state,
                pretraining,
            )
            print(f"periodic_checkpoint={periodic_path}")
        if (
            args.early_stopping_patience
            and horizon == config.prediction_hours
            and stale_epochs >= args.early_stopping_patience
        ):
            print(f"early stopping after {epoch} epochs")
            break
    del model, optimizer
    if args.resume:
        del checkpoint
    final_report = write_forecast_final_report(
        report_path, checkpoint_path, loaders, device
    )
    for split, statistics in final_report["final_model_metrics"].items():
        by_horizon = statistics["model"]["by_horizon"]
        print(
            f"final_statistics split={split} "
            + " ".join(
                f"rmse_{horizon}h={by_horizon[str(horizon)]['rmse']:.3f}"
                for horizon in final_report["reported_horizons_hours"]
            )
        )
    print(
        f"best_{args.loss}={best_loss:.6g} checkpoint={checkpoint_path} "
        f"recovery_checkpoint={recovery_path} "
        f"metrics={metrics_path} loss_plot={loss_plot} report={report_path} "
        f"time_taken={format_duration(time.perf_counter() - started)}"
    )


def infer(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    paths = artifact_paths(args.checkpoint.stem)
    checkpoint_path = output_path(args.checkpoint, paths.checkpoint)
    model, config, zscores, checkpoint = load_checkpoint(
        checkpoint_path, device
    )
    training_config = dict(checkpoint["training_config"])
    training_config["batch_size"] = 1
    training_config["num_workers"] = 0
    training_data = training_data_path(args, training_config)
    if training_data is not None:
        expected = training_config.get("training_data_sha256")
        if expected is not None and file_sha256(training_data) != expected:
            raise ValueError("singular training CSV does not match the checkpoint")
        dataset, loaders = build_singular_loaders(
            training_data, config, training_config, device
        )
    else:
        expected_permanent = training_config.get(
            "permanent_excluded_sensors_sha256"
        )
        if expected_permanent is not None and file_sha256(
            PERMANENT_EXCLUSIONS_PATH
        ) != expected_permanent:
            raise ValueError(
                "permanent excluded sensor list does not match the checkpoint"
            )
        balanced_training_index = args.balanced_training_index
        if balanced_training_index is None and training_config.get(
            "balanced_training_index"
        ):
            balanced_training_index = Path(
                training_config["balanced_training_index"]
            )
        if balanced_training_index is not None:
            expected = training_config.get("balanced_training_index_sha256")
            actual = file_sha256(balanced_training_index)
            if expected is not None and actual != expected:
                raise ValueError(
                    "balanced training index does not match the checkpoint"
                )
        excluded_sensors = args.excluded_sensors
        if excluded_sensors is None and training_config.get("excluded_sensors"):
            excluded_sensors = Path(training_config["excluded_sensors"])
        if excluded_sensors is not None:
            expected = training_config.get("excluded_sensors_sha256")
            actual = file_sha256(excluded_sensors)
            if expected is not None and actual != expected:
                raise ValueError("excluded sensor list does not match the checkpoint")
        dataset, loaders = build_loaders(
            args.pairs,
            args.indoor_history,
            args.outdoor_history,
            args.forecast_root,
            config,
            training_config,
            device,
            balanced_training_index,
            excluded_sensors,
        )
    loader = {
        "train": loaders.train,
        "validation": loaders.validation,
        "temporal-test": loaders.temporal_test,
        "location-test": loaders.location_test,
    }[args.split]
    if not 0 <= args.sample_index < len(loader.dataset):
        raise IndexError(
            f"sample_index must be between 0 and {len(loader.dataset) - 1}"
        )
    sample = loader.dataset[args.sample_index]
    batch = {
        name: sample[name].unsqueeze(0)
        for name in ("history", "forecast", "target")
    }
    history, forecast, _ = normalize_batch(batch, zscores, device)
    model.eval()
    with torch.no_grad():
        prediction = zscores.denormalize_indoor(model(history, forecast))[0]
    location_index = int(sample["location_index"])
    name = checkpoint.get("model_name", DEFAULT_MODEL)
    default_output = (
        paths.forecasts / f"{name}_{args.split}_sample_{args.sample_index}.png"
    )
    output = output_path(args.output, default_output)
    plot_prediction(
        output,
        sample,
        prediction,
        config,
        dataset.location_ids[location_index],
        args.split,
        args.sample_index,
    )
    print(f"saved diagnostic graph: {output}")


def _add_stream_arguments(
    parser: argparse.ArgumentParser, stream: str, patches: bool
) -> None:
    defaults = build_config(DEFAULT_MODEL, {})
    prefix = stream.replace("_", "-")
    if patches:
        parser.add_argument(
            f"--{prefix}-patch-size",
            type=int,
            default=getattr(defaults, f"{stream}_patch_size"),
        )
        parser.add_argument(
            f"--{prefix}-patch-stride",
            type=int,
            default=getattr(defaults, f"{stream}_patch_stride"),
        )
    parser.add_argument(
        f"--{prefix}-embedding-dim",
        type=int,
        default=getattr(defaults, f"{stream}_embedding_dim"),
    )
    parser.add_argument(
        f"--{prefix}-heads",
        type=int,
        default=getattr(defaults, f"{stream}_heads"),
    )
    parser.add_argument(
        f"--{prefix}-head-dim",
        type=int,
        default=getattr(defaults, f"{stream}_head_dim"),
    )
    parser.add_argument(
        f"--{prefix}-layers",
        type=int,
        default=getattr(defaults, f"{stream}_layers"),
    )
    parser.add_argument(
        f"--{prefix}-feedforward-dim",
        type=int,
        default=getattr(defaults, f"{stream}_feedforward_dim"),
    )
    parser.add_argument(
        f"--{prefix}-dropout",
        type=float,
        default=getattr(defaults, f"{stream}_dropout"),
    )
    parser.add_argument(
        f"--{prefix}-activation",
        choices=("relu", "gelu"),
        default=getattr(defaults, f"{stream}_activation"),
    )
    parser.add_argument(
        f"--{prefix}-norm-first",
        action=argparse.BooleanOptionalAction,
        default=getattr(defaults, f"{stream}_norm_first"),
    )
    parser.add_argument(
        f"--{prefix}-layer-norm-eps",
        type=float,
        default=getattr(defaults, f"{stream}_layer_norm_eps"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and inspect PM2.5 forecasting models.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser(
        "train", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    train_parser.add_argument(
        "--model", choices=model_names(), default=DEFAULT_MODEL
    )
    train_parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        help=(
            "completed tempo_bridge_86 checkpoint; the matching bridge forecast "
            "model is selected automatically when --model is omitted"
        ),
    )
    train_parser.add_argument(
        "--history-initialization",
        choices=("pretrained", "random"),
        help=(
            "load bridge history weights or retain random history weights; the "
            "latter supports an architecture-matched control"
        ),
    )
    train_parser.add_argument(
        "--bridge-history-normalization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply the bridge checkpoint's indoor/outdoor history z-scores",
    )
    train_parser.add_argument(
        "--training-data",
        "--training-csv",
        dest="training_data",
        type=Path,
        help="singular CSV containing materialized windows and split labels",
    )
    train_parser.add_argument("--pairs", type=Path)
    train_parser.add_argument(
        "--indoor-history", type=Path, action="append"
    )
    train_parser.add_argument("--outdoor-history", type=Path)
    train_parser.add_argument("--forecast-root", type=Path)
    train_parser.add_argument(
        "--balanced-training-index",
        type=Path,
        help="balancer intersection CSV used to select only training anchors",
    )
    train_parser.add_argument(
        "--excluded-sensors",
        type=Path,
        help="CSV with a sensor_id column; removes sensors from every split",
    )
    train_parser.add_argument(
        "--checkpoint",
        type=Path,
        help="checkpoint path; defaults to inference/checkpoints/DATASET_MODEL.pt",
    )
    train_parser.add_argument(
        "--loss-plot",
        type=Path,
        help="loss graph path; defaults to inference/graphs/DATASET_MODEL.png",
    )
    train_parser.add_argument(
        "--metrics-output",
        type=Path,
        help="metrics CSV path; defaults to inference/metrics/DATASET_MODEL.csv",
    )
    train_parser.add_argument(
        "--report-output",
        type=Path,
        help="final statistics CSV; defaults to inference/reports/DATASET_MODEL.csv",
    )
    train_parser.add_argument("--history-hours", type=int, default=168)
    train_parser.add_argument("--prediction-hours", type=int, default=36)
    train_parser.add_argument("--minimum-outdoor-history-hours", type=int, default=24)
    _add_stream_arguments(train_parser, "history", True)
    _add_stream_arguments(train_parser, "forecast", True)
    _add_stream_arguments(train_parser, "decoder", False)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument(
        "--freeze-history-epochs",
        type=int,
        default=0,
        help="freeze the bridge history encoder for the first N forecast epochs",
    )
    train_parser.add_argument(
        "--forecast-horizons",
        type=int,
        nargs="+",
        metavar="H",
        help="increasing prefix-loss horizons ending at prediction_hours",
    )
    train_parser.add_argument(
        "--horizon-stage-epochs",
        type=int,
        nargs="+",
        metavar="N",
        help="epoch count per forecast horizon; values must sum to --epochs",
    )
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--gradient-clip", type=float, default=1.0)
    train_parser.add_argument("--loss", choices=("mae", "mse", "huber"), default="mse")
    train_parser.add_argument("--huber-delta", type=float, default=1.0)
    train_parser.add_argument("--train-fraction", type=float, default=0.75)
    train_parser.add_argument("--validation-fraction", type=float, default=0.15)
    train_parser.add_argument(
        "--location-holdout-fraction", type=float, default=0.20
    )
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="save a resumable checkpoint every N epochs; 0 disables",
    )
    train_parser.add_argument("--early-stopping-patience", type=int, default=10)
    train_parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from the last completed epoch in CHECKPOINT_STEM.last.pt",
    )
    train_parser.set_defaults(function=train)

    infer_parser = commands.add_parser(
        "infer", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    infer_parser.add_argument(
        "--training-data",
        "--training-csv",
        dest="training_data",
        type=Path,
        help="singular CSV; defaults to the file recorded in the checkpoint",
    )
    infer_parser.add_argument("--pairs", type=Path)
    infer_parser.add_argument(
        "--indoor-history", type=Path, action="append"
    )
    infer_parser.add_argument("--outdoor-history", type=Path)
    infer_parser.add_argument("--forecast-root", type=Path)
    infer_parser.add_argument(
        "--balanced-training-index",
        type=Path,
        help="override the balancer intersection CSV recorded in the checkpoint",
    )
    infer_parser.add_argument(
        "--excluded-sensors",
        type=Path,
        help="override the excluded sensor CSV recorded in the checkpoint",
    )
    infer_parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="checkpoint path or file name under inference/checkpoints/",
    )
    infer_parser.add_argument(
        "--split",
        choices=("train", "validation", "temporal-test", "location-test"),
        default="temporal-test",
    )
    infer_parser.add_argument("--sample-index", type=int, default=0)
    infer_parser.add_argument(
        "--output",
        type=Path,
        help="forecast graph path; defaults under inference/forecasts/DATASET_MODEL/",
    )
    infer_parser.add_argument("--device", default="auto")
    infer_parser.set_defaults(function=infer)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
