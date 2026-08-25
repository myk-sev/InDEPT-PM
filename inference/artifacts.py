"""Canonical paths for training and inference artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


INFERENCE_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ArtifactPaths:
    checkpoint: Path
    cache: Path
    metrics: Path
    graph: Path
    report: Path
    reconstructions: Path
    forecasts: Path
    evaluation: Path


def dataset_model_stem(source: Path | None, model: str) -> str:
    dataset = source.stem if source else "legacy_sources"
    for suffix in ("_masked_training_data", "_training_data"):
        if dataset.endswith(suffix):
            dataset = dataset.removesuffix(suffix)
            break
    return f"{dataset}_{model}"


def artifact_paths(stem: str, root: Path = INFERENCE_ROOT) -> ArtifactPaths:
    return ArtifactPaths(
        checkpoint=root / "checkpoints" / f"{stem}.pt",
        cache=root / "caches" / f"{stem}.pt",
        metrics=root / "metrics" / f"{stem}.csv",
        graph=root / "graphs" / f"{stem}.png",
        report=root / "reports" / f"{stem}.csv",
        reconstructions=root / "reconstructions" / stem,
        forecasts=root / "forecasts" / stem,
        evaluation=root / "evaluations" / f"{stem}.json",
    )
