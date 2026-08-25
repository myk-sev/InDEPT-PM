from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from data_loader import DualEncoderDataset, create_data_loaders
from pm25_models import DEFAULT_MODEL, build_config, model_names


ROOT = Path(__file__).resolve().parent
INPUT_ROOT = ROOT / "inputs"
DATA_ROOT = ROOT / "data"
PURPLEAIR_ROOT = ROOT.parent / "purple-air-pull"
DEFAULT_PAIRS = DATA_ROOT / "legacy" / "purpleair_continental_us_pairs_thinned_20km.csv"
DEFAULT_INDOOR = PURPLEAIR_ROOT / "purpleair_hourly_pm25_atm"
DEFAULT_TEMPO = (
    PURPLEAIR_ROOT / "tempo_pm25_sensor_match" / "tempo_pm25_indoor_sensors.csv"
)
DEFAULT_NAQFC = ROOT / "naqfc_output"
DEFAULT_EXCLUSIONS = DATA_ROOT / "exclusions" / "excluded_indoor_sensors_pm25_gt1000.csv"
SPLITS = ("train", "validation", "temporal_test", "location_test")
LINEAR_TIME_FEATURES = ("hour", "weekday", "month", "day")
CYCLICAL_TIME_FEATURES = (
    "daily_sin",
    "daily_cos",
    "weekly_sin",
    "weekly_cos",
    "annual_sin",
    "annual_cos",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the older non-masked pipeline's eligible windows and splits "
            "to one flat CSV."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--indoor-history", type=Path, action="append")
    parser.add_argument("--outdoor-history", type=Path, default=DEFAULT_TEMPO)
    parser.add_argument("--forecast-root", type=Path, default=DEFAULT_NAQFC)
    parser.add_argument("--excluded-sensors", type=Path, default=DEFAULT_EXCLUSIONS)
    parser.add_argument("--balanced-training-index", type=Path)
    parser.add_argument("--output", type=Path, default=INPUT_ROOT / "old_training_data.csv")
    parser.add_argument("--model", choices=model_names(), default=DEFAULT_MODEL)
    parser.add_argument("--history-hours", type=int, default=168)
    parser.add_argument("--prediction-hours", type=int, default=36)
    parser.add_argument("--minimum-outdoor-history-hours", type=int, default=24)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--location-holdout-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def isoformat(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def columns(
    history_hours: int,
    prediction_hours: int,
    time_features: tuple[str, ...],
) -> tuple[str, ...]:
    metadata = (
        "sample_index",
        "split",
        "location_id",
        "sensor_id",
        "model_name",
        "history_hours",
        "prediction_hours",
        "minimum_tempo_history_hours",
        "train_fraction",
        "validation_fraction",
        "location_holdout_fraction",
        "split_seed",
        "history_start_utc",
        "anchor_time_utc",
        "forecast_start_utc",
        "forecast_end_utc",
        "naqfc_cycle_time_utc",
        "naqfc_first_forecast_hour",
        "naqfc_last_forecast_hour",
        "pairs_source",
        "indoor_history_sources",
        "tempo_history_source",
        "naqfc_forecast_source",
        "excluded_sensors_source",
        "balanced_training_index_source",
    )
    history = tuple(
        f"history_{hour:03d}_{feature}"
        for hour in range(history_hours)
        for feature in ("tempo_pm25_ug_m3", "indoor_pm25_ug_m3", *time_features)
    )
    future_features = (
        ("naqfc_pm25_ug_m3", *time_features)
        if len(time_features) == 6
        else ("naqfc_pm25_ug_m3",)
    )
    forecast = tuple(
        f"forecast_{hour:03d}_{feature}"
        for hour in range(1, prediction_hours + 1)
        for feature in future_features
    )
    target = tuple(
        f"target_{hour:03d}_indoor_pm25_ug_m3"
        for hour in range(1, prediction_hours + 1)
    )
    return metadata + history + forecast + target


def split_indices(loaders) -> dict[int, str]:
    result: dict[int, str] = {}
    for name in SPLITS:
        for index in getattr(loaders, name).dataset.indices:
            index = int(index)
            if index in result:
                raise RuntimeError(f"sample {index} appears in more than one split")
            result[index] = name
    return result


def export(args: argparse.Namespace) -> tuple[int, dict[str, int]]:
    indoor_history = [
        path.resolve() for path in (args.indoor_history or [DEFAULT_INDOOR])
    ]
    model_config = build_config(args.model, {})
    cyclical_time = getattr(model_config, "cyclical_time", False)
    time_features = CYCLICAL_TIME_FEATURES if cyclical_time else LINEAR_TIME_FEATURES
    excluded_sensors = args.excluded_sensors.resolve() if args.excluded_sensors else None

    print("Building eligible windows with DualEncoderDataset...")
    dataset = DualEncoderDataset(
        args.pairs.resolve(),
        indoor_history,
        args.outdoor_history.resolve(),
        args.forecast_root.resolve(),
        history_hours=args.history_hours,
        forecast_hours=args.prediction_hours,
        minimum_outdoor_history_hours=args.minimum_outdoor_history_hours,
        excluded_sensors_path=excluded_sensors,
        cyclical_time=cyclical_time,
    )
    loaders = create_data_loaders(
        dataset,
        batch_size=1,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        location_holdout_fraction=args.location_holdout_fraction,
        seed=args.seed,
        balanced_training_index=args.balanced_training_index,
    )
    selected = split_indices(loaders)
    counts = {
        name: sum(split_name == name for split_name in selected.values())
        for name in SPLITS
    }

    output = args.output.resolve()
    inputs = {
        path.resolve()
        for path in (
            args.pairs,
            args.outdoor_history,
            args.forecast_root,
            *indoor_history,
            *([excluded_sensors] if excluded_sensors else []),
            *(
                [args.balanced_training_index]
                if args.balanced_training_index
                else []
            ),
        )
    }
    if output.suffix.lower() != ".csv":
        raise ValueError(f"output must have a .csv extension: {output}")
    if output in inputs:
        raise ValueError(f"output cannot replace an input: {output}")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; use --overwrite to replace it: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    sources = (
        str(args.pairs.resolve()),
        ";".join(map(str, indoor_history)),
        str(args.outdoor_history.resolve()),
        str(args.forecast_root.resolve()),
        str(excluded_sensors or ""),
        str(
            args.balanced_training_index.resolve()
            if args.balanced_training_index
            else ""
        ),
    )

    try:
        with partial.open("w", encoding="utf-8", newline="") as target:
            writer = csv.writer(target)
            writer.writerow(
                columns(args.history_hours, args.prediction_hours, time_features)
            )
            for written, sample_index in enumerate(sorted(selected), 1):
                sample = dataset[sample_index]
                location = int(sample["location_index"])
                anchor = int(sample["anchor_time_utc"])
                code = int(dataset._sample_codes[sample_index])
                anchor_index = code % dataset._steps
                cycle_index = int(dataset._anchor_cycles[anchor_index])
                lead = int(dataset._anchor_leads[anchor_index])
                history = sample["history"].tolist()
                forecast = sample["forecast"].tolist()
                expected_history_features = 2 + len(time_features)
                expected_forecast_features = 1 + (
                    len(time_features) if cyclical_time else 0
                )
                if (
                    len(history[0]) != expected_history_features
                    or len(forecast[0]) != expected_forecast_features
                ):
                    raise RuntimeError("export schema no longer matches DualEncoderDataset")
                metadata = (
                    sample_index,
                    selected[sample_index],
                    dataset.location_ids[location],
                    dataset.sensor_ids[location],
                    args.model,
                    args.history_hours,
                    args.prediction_hours,
                    args.minimum_outdoor_history_hours,
                    args.train_fraction,
                    args.validation_fraction,
                    args.location_holdout_fraction,
                    args.seed,
                    isoformat(anchor - (args.history_hours - 1) * 3600),
                    isoformat(anchor),
                    isoformat(anchor + 3600),
                    isoformat(anchor + args.prediction_hours * 3600),
                    isoformat(int(dataset._cycles[cycle_index])),
                    lead + 1,
                    lead + args.prediction_hours,
                    *sources,
                )
                writer.writerow(
                    (
                        *metadata,
                        *(value for hour in history for value in hour),
                        *(value for hour in forecast for value in hour),
                        *sample["target"].tolist(),
                    )
                )
                if written % 500 == 0:
                    print(f"Exported {written:,}/{len(selected):,} windows")
        partial.replace(output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    with output.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    print(f"output={output}")
    column_count = len(columns(args.history_hours, args.prediction_hours, time_features))
    print(f"rows={len(selected):,} columns={column_count:,}")
    print("splits=" + ", ".join(f"{name}:{counts[name]:,}" for name in SPLITS))
    print(f"sha256={digest}")
    return len(selected), counts


if __name__ == "__main__":
    export(parse_args())
