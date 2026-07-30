"""Run a trained model and create graphs from a compact inference cache."""

import argparse
import time
from pathlib import Path

import torch

from pm25_transformer import (
    file_sha256,
    load_checkpoint,
    normalize_batch,
    plot_prediction,
    resolve_device,
)


CACHE_FORMAT_VERSION = 1


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


def main() -> None:
    args = build_parser().parse_args()
    started = time.perf_counter()
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    if cache.get("format_version") != CACHE_FORMAT_VERSION:
        raise ValueError("unsupported inference cache format")
    if cache.get("checkpoint_sha256") != file_sha256(args.checkpoint):
        raise ValueError("cache was built for a different checkpoint")

    records = cache.get("samples")
    if not isinstance(records, list) or not records:
        raise ValueError("inference cache contains no samples")
    by_index = {record["sample_index"]: record for record in records}
    if len(by_index) != len(records):
        raise ValueError("inference cache contains duplicate sample indices")

    requested = args.indices if args.indices is not None else list(by_index)
    if len(set(requested)) != len(requested):
        raise ValueError("indices must be unique")
    missing = [index for index in requested if index not in by_index]
    if missing:
        raise ValueError(f"indices are not present in the cache: {missing}")
    selected = [by_index[index] for index in requested]

    device = resolve_device(args.device)
    model, config, zscores, checkpoint = load_checkpoint(args.checkpoint, device)
    if cache.get("model_config") != checkpoint["model_config"]:
        raise ValueError("cache model configuration does not match the checkpoint")

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
        output = (
            args.output_dir
            / f"{split.replace('-', '_')}_sample_{index}.png"
        )
        plot_prediction(
            output,
            record["sample"],
            prediction,
            config,
            record["location_id"],
            split,
            index,
        )
        print(f"saved sample {index}: {output}")

    print(
        f"device={device} samples={len(selected)} "
        f"inference={inference_seconds:.3f}s "
        f"total={time.perf_counter() - started:.1f}s"
    )


if __name__ == "__main__":
    main()
