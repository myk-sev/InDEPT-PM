"""Evaluate a bridge forecast against persistence and fitted linear baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pm25_transformer import (
    build_singular_loaders,
    file_sha256,
    load_checkpoint,
    normalize_batch,
    resolve_device,
)


def collect_raw(loader) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indoor, forecast, target = [], [], []
    for batch in loader:
        indoor.append(batch["history"][:, -1, 1].double())
        forecast.append(batch["forecast"][..., 0].double())
        target.append(batch["target"].double())
    return torch.cat(indoor), torch.cat(forecast), torch.cat(target)


def fit_linear_baseline(loader) -> torch.Tensor:
    indoor, forecast, target = collect_raw(loader)
    coefficients = []
    ones = torch.ones_like(indoor)
    for lead in range(target.shape[1]):
        design = torch.stack((ones, indoor, forecast[:, lead]), dim=1)
        coefficients.append(
            torch.linalg.lstsq(design, target[:, lead, None]).solution[:, 0]
        )
    return torch.stack(coefficients)


def linear_prediction(
    indoor: torch.Tensor, forecast: torch.Tensor, coefficients: torch.Tensor
) -> torch.Tensor:
    return (
        coefficients[:, 0]
        + indoor[:, None] * coefficients[:, 1]
        + forecast * coefficients[:, 2]
    )


def model_prediction(model, loader, zscores, device) -> torch.Tensor:
    predictions = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            history, forecast, _ = normalize_batch(batch, zscores, device)
            predictions.append(
                zscores.denormalize_indoor(model(history, forecast)).cpu().double()
            )
    return torch.cat(predictions)


def metric_report(
    prediction: torch.Tensor, target: torch.Tensor, horizons: list[int]
) -> dict:
    by_horizon = {}
    for horizon in horizons:
        error = prediction[:, :horizon] - target[:, :horizon]
        mse = error.square().mean().item()
        by_horizon[str(horizon)] = {
            "mae": error.abs().mean().item(),
            "mse": mse,
            "rmse": mse**0.5,
            "values": error.numel(),
        }
    by_lead = []
    for lead in range(target.shape[1]):
        error = prediction[:, lead] - target[:, lead]
        mse = error.square().mean().item()
        by_lead.append(
            {
                "forecast_hour": lead + 1,
                "mae": error.abs().mean().item(),
                "mse": mse,
                "rmse": mse**0.5,
            }
        )
    return {"by_horizon": by_horizon, "by_lead": by_lead}


def evaluate(checkpoint_path: Path, output: Path, device_name: str = "auto") -> dict:
    device = resolve_device(device_name)
    model, config, zscores, checkpoint = load_checkpoint(checkpoint_path, device)
    training_config = checkpoint["training_config"]
    if "training_data" not in training_config:
        raise ValueError("bridge forecast evaluation requires a singular training CSV")
    training_data = Path(training_config["training_data"])
    if file_sha256(training_data) != training_config["training_data_sha256"]:
        raise ValueError("forecast training CSV no longer matches the checkpoint")
    _, loaders = build_singular_loaders(
        training_data, config, training_config, device
    )
    coefficients = fit_linear_baseline(loaders.train)
    horizons = training_config.get("forecast_horizons", [config.prediction_hours])
    splits = {}
    for name in ("validation", "temporal_test", "location_test"):
        loader = getattr(loaders, name)
        indoor, forecast, target = collect_raw(loader)
        predictions = {
            "model": model_prediction(model, loader, zscores, device),
            "persistence": indoor[:, None].expand_as(target),
            "linear": linear_prediction(indoor, forecast, coefficients),
        }
        splits[name] = {
            method: metric_report(prediction, target, horizons)
            for method, prediction in predictions.items()
        }
    report = {
        "format_version": 1,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "model_name": checkpoint["model_name"],
        "training_data": str(training_data.resolve()),
        "training_data_sha256": training_config["training_data_sha256"],
        "forecast_horizons": horizons,
        "linear_coefficients_by_lead": coefficients.tolist(),
        "splits": splits,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a bridge forecast with persistence and linear baselines."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    report = evaluate(args.checkpoint, args.output, args.device)
    print(f"evaluated_model={report['model_name']}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
