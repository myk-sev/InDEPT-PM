"""Run a trained model and create graphs from a compact inference cache."""

import argparse
import re
import shutil
import time
from pathlib import Path

import torch

from pm25_models import DEFAULT_MODEL
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
    parser.add_argument(
        "--loss-plot",
        type=Path,
        help="training/validation loss graph to copy into the output directory",
    )
    parser.add_argument("--device", default="auto")
    return parser


def stack_graphs(output: Path, graphs: list[tuple[Path, str]]) -> None:
    from PIL import Image, ImageDraw, ImageFont

    images = []
    for path, label in graphs:
        with Image.open(path) as image:
            images.append((image.convert("RGB"), label))

    label_height = 52
    width = max(image.width for image, _ in images)
    stacked = Image.new(
        "RGB",
        (width, sum(image.height + label_height for image, _ in images)),
        "white",
    )
    draw = ImageDraw.Draw(stacked)
    font = ImageFont.load_default(size=26)
    top = 0
    for image, label in images:
        draw.text((20, top + 10), label, fill="black", font=font)
        top += label_height
        stacked.paste(image, (0, top))
        top += image.height
    output.parent.mkdir(parents=True, exist_ok=True)
    stacked.save(output, dpi=(150, 150))


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
    if args.loss_plot is not None and not args.loss_plot.is_file():
        raise FileNotFoundError(f"loss plot not found: {args.loss_plot}")
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
    model, config, zscores, checkpoint = load_checkpoint(args.checkpoint, device)
    model_name = checkpoint.get("model_name", DEFAULT_MODEL)
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
    graphs: list[tuple[Path, str]] = []
    for record, prediction in zip(selected, predictions):
        index = record["sample_index"]
        name = record.get("name")
        filename = (
            f"{name}.png"
            if name
            else f"{split.replace('-', '_')}_sample_{index}.png"
        )
        output = args.output_dir / f"{model_name}_{filename}"
        plot_prediction(
            output,
            record["sample"],
            prediction,
            config,
            record["location_id"],
            split,
            index,
            model_name=model_name,
        )
        description = re.sub(r"[_-]+", " ", name).title() if name else "Inference"
        split_label = split.replace("-", " ").title()
        graphs.append((output, f"{description} - {split_label} Sample {index}"))
        label = f"{name}={index}" if name else str(index)
        print(f"saved sample {label}: {output}")

    stacked_output = args.output_dir / "stacked_inference_graphs.png"
    stack_graphs(stacked_output, graphs)
    print(f"saved stacked inference graphs: {stacked_output}")

    if args.loss_plot is not None:
        loss_output = args.output_dir / "training_validation_loss.png"
        shutil.copy2(args.loss_plot, loss_output)
        print(f"copied training/validation loss graph: {loss_output}")

    print(
        f"device={device} samples={len(selected)} "
        f"inference={inference_seconds:.3f}s "
        f"total={time.perf_counter() - started:.1f}s"
    )


if __name__ == "__main__":
    main()
