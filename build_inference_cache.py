"""Cache selected retrospective inference samples from a trained checkpoint."""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from pm25_transformer import build_loaders, file_sha256, load_checkpoint


CACHE_FORMAT_VERSION = 1
SPLITS = ("train", "validation", "temporal-test", "location-test")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a compact cache for selected inference sample indices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=SPLITS, default="temporal-test")
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if any(index < 0 for index in args.indices):
        raise ValueError("indices cannot be negative")
    if len(set(args.indices)) != len(args.indices):
        raise ValueError("indices must be unique")

    started = time.perf_counter()
    device = torch.device("cpu")
    _, config, _, checkpoint = load_checkpoint(args.checkpoint, device)
    training_config = dict(checkpoint["training_config"])
    source_config = dict(training_config)
    training_config["batch_size"] = 1
    training_config["num_workers"] = 0

    required = ("pairs", "indoor_history", "outdoor_history", "forecast_root")
    missing = [name for name in required if name not in training_config]
    if missing:
        raise ValueError(
            "checkpoint does not record required data paths: " + ", ".join(missing)
        )

    balanced_index = training_config.get("balanced_training_index")
    balanced_path = Path(balanced_index) if balanced_index else None
    if balanced_path is not None:
        expected = training_config.get("balanced_training_index_sha256")
        if expected is not None and file_sha256(balanced_path) != expected:
            raise ValueError("balanced training index does not match the checkpoint")

    print(
        f"loading {args.split} data for {len(args.indices)} requested samples..."
    )
    dataset, loaders = build_loaders(
        Path(training_config["pairs"]),
        [Path(path) for path in training_config["indoor_history"]],
        Path(training_config["outdoor_history"]),
        Path(training_config["forecast_root"]),
        config,
        training_config,
        device,
        balanced_path,
    )
    loader = getattr(loaders, args.split.replace("-", "_"))
    invalid = [index for index in args.indices if index >= len(loader.dataset)]
    if invalid:
        raise IndexError(
            f"indices exceed the {args.split} maximum of "
            f"{len(loader.dataset) - 1}: {invalid}"
        )

    records = []
    for completed, index in enumerate(args.indices, 1):
        sample = loader.dataset[index]
        location = int(sample["location_index"])
        records.append(
            {
                "sample_index": index,
                "location_id": dataset.location_ids[location],
                "sensor_id": dataset.sensor_ids[location],
                "sample": {
                    name: sample[name].detach().cpu().clone()
                    for name in (
                        "history",
                        "forecast",
                        "target",
                        "anchor_time_utc",
                    )
                },
            }
        )
        print(f"cached {completed}/{len(args.indices)}: sample {index}")

    cache = {
        "format_version": CACHE_FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "model_config": checkpoint["model_config"],
        "training_config": source_config,
        "split": args.split,
        "samples": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f"{args.output.name}.tmp")
    torch.save(cache, temporary)
    temporary.replace(args.output)
    print(
        f"saved {len(records)} samples to {args.output} "
        f"in {time.perf_counter() - started:.1f}s"
    )


if __name__ == "__main__":
    main()
