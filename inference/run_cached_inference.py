"""Run a trained model and create graphs from a compact inference cache."""

import argparse
import re
import time
from pathlib import Path

import torch

from pm25_transformer import (
    load_checkpoint,
    normalize_batch,
    plot_prediction,
    resolve_device,
)


CACHE_FORMAT_VERSION = 2
SUPPORTED_CACHE_FORMAT_VERSIONS = {1, CACHE_FORMAT_VERSION}
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run cached inference and generate diagnostic forecast graphs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        help="cached sample indices to graph; omit to graph every cached sample",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("inference_graphs"))
    parser.add_argument("--device", default="auto")
    return parser


def validate_data_contract(cache: dict, records: list[dict], config: object) -> None:
    cyclical = getattr(config, "cyclical_time", False)
    expected = {
        "history_shape": (config.history_hours, 8 if cyclical else 6),
        "forecast_shape": (config.prediction_hours, 7 if cyclical else 1),
        "target_shape": (config.prediction_hours,),
        "cyclical_time": cyclical,
    }
    contract = cache.get("data_contract")
    if contract is not None and contract != expected:
        raise ValueError("cache data contract is incompatible with the checkpoint")

    for record in records:
        sample = record.get("sample")
        if not isinstance(sample, dict):
            raise ValueError("inference cache contains an invalid sample")
        for name in ("history", "forecast", "target"):
            value = sample.get(name)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected[
                f"{name}_shape"
            ]:
                raise ValueError(
                    f"cached sample {record.get('sample_index')} has an "
                    f"incompatible {name} shape"
                )


def main() -> None:
    args = build_parser().parse_args()
    started = time.perf_counter()
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    if cache.get("format_version") not in SUPPORTED_CACHE_FORMAT_VERSIONS:
        raise ValueError("unsupported inference cache format")

    records = cache.get("samples")
    if not isinstance(records, list) or not records:
        raise ValueError("inference cache contains no samples")
    by_index = {record["sample_index"]: record for record in records}
    if len(by_index) != len(records):
        raise ValueError("inference cache contains duplicate sample indices")
    names = [record.get("name") for record in records if record.get("name")]
    if len(set(names)) != len(names) or any(
        not isinstance(name, str) or not NAME_PATTERN.fullmatch(name)
        for name in names
    ):
        raise ValueError("inference cache contains invalid or duplicate sample names")

    requested = args.indices if args.indices is not None else list(by_index)
    if len(set(requested)) != len(requested):
        raise ValueError("indices must be unique")
    missing = [index for index in requested if index not in by_index]
    if missing:
        raise ValueError(f"indices are not present in the cache: {missing}")
    selected = [by_index[index] for index in requested]

    device = resolve_device(args.device)
    model, config, zscores, _ = load_checkpoint(args.checkpoint, device)
    validate_data_contract(cache, records, config)

    samples = [record["sample"] for record in selected]
    batch = {
        name: torch.stack([sample[name] for sample in samples])
        for name in ("history", "forecast", "target")
    }
    history, forecast, _ = normalize_batch(batch, zscores, device)
    model.eval()
    inference_started = time.perf_counter()
    with torch.inference_mode():
        predictions = zscores.denormalize_indoor(model(history, forecast))
    inference_seconds = time.perf_counter() - inference_started

    split = cache["split"]
    for record, prediction in zip(selected, predictions):
        index = record["sample_index"]
        name = record.get("name")
        filename = (
            f"{name}.png"
            if name
            else f"{split.replace('-', '_')}_sample_{index}.png"
        )
        output = args.output_dir / filename
        plot_prediction(
            output,
            record["sample"],
            prediction,
            config,
            record["location_id"],
            split,
            index,
        )
        label = f"{name}={index}" if name else str(index)
        print(f"saved sample {label}: {output}")

    print(
        f"device={device} samples={len(selected)} "
        f"inference={inference_seconds:.3f}s "
        f"total={time.perf_counter() - started:.1f}s"
    )


if __name__ == "__main__":
    main()
