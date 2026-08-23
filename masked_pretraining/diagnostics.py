"""Small, deterministic training diagnostics."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.dates import ConciseDateFormatter, AutoDateLocator
import numpy as np
import torch
from torch import nn
from torch.utils.data import default_collate

from .data import Normalizer, PairWindowDataset
from .masking import mask_batch


METRIC_FIELDS = (
    "global_epoch",
    "stage",
    "stage_epoch",
    "train_loss",
    "validation_loss",
    "validation_indoor_rmse",
    "validation_outdoor_rmse",
    "train_target_count",
    "validation_target_count",
    "improved_checkpoint",
)


@dataclass(frozen=True)
class DiagnosticPaths:
    metrics: Path
    loss_curve: Path
    reconstructions: Path


def diagnostic_paths(checkpoint: Path) -> DiagnosticPaths:
    base = checkpoint.with_suffix("") if checkpoint.suffix else checkpoint
    return DiagnosticPaths(
        base.with_name(base.name + ".metrics.csv"),
        base.with_name(base.name + ".loss_curve.png"),
        base.with_name(base.name + ".reconstruction_examples.png"),
    )


def write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_loss_curve(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["global_epoch"]) for row in rows]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(epochs, [row["train_loss"] for row in rows], label="Training")
    axis.plot(epochs, [row["validation_loss"] for row in rows], label="Validation")
    previous = str(rows[0]["stage"])
    stage_start = epochs[0]
    for row in rows:
        stage = str(row["stage"])
        if stage != previous:
            boundary = int(row["global_epoch"])
            axis.axvline(boundary - 0.5, color="0.75", linewidth=1)
            _label_stage(axis, previous, stage_start, boundary - 1)
            previous, stage_start = stage, boundary
    _label_stage(axis, previous, stage_start, epochs[-1])
    axis.set(title="Masked reconstruction loss", xlabel="Epoch", ylabel="MSE loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, path)


def write_reconstruction_examples(
    path: Path,
    model: nn.Module,
    dataset: PairWindowDataset,
    stage: str,
    normalizer: Normalizer,
    device: torch.device,
    seed: int,
    maximum_examples: int = 4,
) -> None:
    indices = _example_indices(dataset, maximum_examples)
    count = len(indices)
    batch = default_collate([dataset[index] for index in indices])
    masked = mask_batch(
        batch["values"],
        batch["observed"],
        batch["time_features"],
        stage,
        torch.Generator().manual_seed(seed),
    )
    model.eval()
    with torch.no_grad():
        prediction = model(masked.features.to(device)).cpu()
    mean = torch.tensor(normalizer.mean).view(1, 1, 2)
    deviation = torch.tensor(normalizer.standard_deviation).view(1, 1, 2)
    actual = (masked.target * deviation + mean).numpy()
    prediction = (prediction * deviation + mean).numpy()
    observed = batch["observed"].numpy()
    artificial = masked.target_mask.numpy()

    figure, axes = plt.subplots(count, 2, figsize=(14, 3 * count), squeeze=False)
    for item in range(count):
        timestamps = _timestamps(
            int(batch["start_time_utc"][item]), actual.shape[1]
        )
        indoor_id = int(batch["indoor_sensor_id"][item])
        outdoor_ids = sorted(
            set(int(value) for value in batch["outdoor_sensor_ids"][item] if value)
        )
        for channel, axis in enumerate(axes[item]):
            full = np.where(observed[item, :, channel], actual[item, :, channel], np.nan)
            visible = np.where(
                observed[item, :, channel] & ~artificial[item, :, channel],
                actual[item, :, channel],
                np.nan,
            )
            selected = artificial[item, :, channel]
            axis.plot(timestamps, full, color="0.75", linewidth=1, label="Full label")
            axis.plot(timestamps, visible, color="0.15", linewidth=1, label="Visible history")
            axis.scatter(
                np.asarray(timestamps)[selected],
                actual[item, selected, channel],
                color="#d62728",
                s=18,
                label="Masked label",
                zorder=3,
            )
            axis.scatter(
                np.asarray(timestamps)[selected],
                prediction[item, selected, channel],
                color="#1f77b4",
                marker="x",
                s=24,
                label="Prediction",
                zorder=4,
            )
            name = (
                f"Outdoor sensors {', '.join(map(str, outdoor_ids))}"
                if channel == 0
                else f"Indoor sensor {indoor_id}"
            )
            axis.set(title=name, ylabel="PM2.5 (µg/m³)")
            axis.grid(alpha=0.2)
            locator = AutoDateLocator(minticks=4, maxticks=8)
            axis.xaxis.set_major_locator(locator)
            axis.xaxis.set_major_formatter(ConciseDateFormatter(locator))
    axes[0, 0].legend(ncol=4, fontsize=8, loc="upper left")
    figure.suptitle(
        f"Fixed validation reconstructions — {stage}\nNatural missing observations appear as gaps",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    _save_figure(figure, path)


def _timestamps(start: int, hours: int) -> list[datetime]:
    return [
        datetime.fromtimestamp(start + hour * 3600, timezone.utc)
        for hour in range(hours)
    ]


def _label_stage(axis: plt.Axes, stage: str, start: int, end: int) -> None:
    axis.text(
        (start + end) / 2,
        0.98,
        stage,
        color="0.4",
        fontsize=8,
        ha="center",
        va="top",
        transform=axis.get_xaxis_transform(),
    )


def _example_indices(
    dataset: PairWindowDataset, maximum_examples: int
) -> list[int]:
    selected, seen_series = [], set()
    for index, (series_index, _) in enumerate(dataset.windows):
        if series_index not in seen_series:
            selected.append(index)
            seen_series.add(series_index)
        if len(selected) == maximum_examples:
            return selected
    for index in range(len(dataset)):
        if index not in selected:
            selected.append(index)
        if len(selected) == maximum_examples:
            break
    return selected


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    figure.savefig(temporary, format="png", dpi=150)
    plt.close(figure)
    os.replace(temporary, path)
