"""Extract the indoor PurpleAir observations used by an old training CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "inputs" / "old_training_data.csv"
DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "purple air"
    / "old_non_masked_purpleair"
    / "indoor_pm25.csv"
)


def timestamp(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"timestamp is not UTC: {value}")
    result = int(parsed.timestamp())
    if result % 3600:
        raise ValueError(f"timestamp is not an exact hour: {value}")
    return result


def add(values: dict[tuple[int, int], float], key: tuple[int, int], text: str) -> None:
    value = float(text)
    if value < 0 or not math.isfinite(value):
        raise ValueError(f"invalid PurpleAir value for sensor-hour {key}: {text}")
    previous = values.get(key)
    if previous is not None and not math.isclose(previous, value, abs_tol=1e-6):
        raise ValueError(f"conflicting PurpleAir value for sensor-hour {key}")
    values[key] = value


def extract(
    source: Path,
    output: Path,
    selected_split: str = "train",
    overwrite: bool = False,
) -> None:
    source, output = source.resolve(), output.resolve()
    if source == output:
        raise ValueError("input and output must be different files")
    if output.exists() and not overwrite:
        raise FileExistsError(f"output exists; use --overwrite to replace it: {output}")

    values: dict[tuple[int, int], float] = {}
    with source.open(encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        required = {
            "sensor_id",
            "history_hours",
            "prediction_hours",
            "history_start_utc",
            "forecast_start_utc",
            "split",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing input columns: {', '.join(sorted(missing))}")
        selected_windows = 0
        for number, row in enumerate(reader, 2):
            if selected_split != "all" and row["split"] != selected_split:
                continue
            selected_windows += 1
            sensor = int(row["sensor_id"])
            history_hours = int(row["history_hours"])
            prediction_hours = int(row["prediction_hours"])
            history_start = timestamp(row["history_start_utc"])
            forecast_start = timestamp(row["forecast_start_utc"])
            for hour in range(history_hours):
                add(
                    values,
                    (sensor, history_start + hour * 3600),
                    row[f"history_{hour:03d}_indoor_pm25_ug_m3"],
                )
            for hour in range(prediction_hours):
                add(
                    values,
                    (sensor, forecast_start + hour * 3600),
                    row[f"target_{hour + 1:03d}_indoor_pm25_ug_m3"],
                )
            if selected_windows % 500 == 0:
                print(f"Read {selected_windows:,} selected windows")

    if not values:
        raise ValueError(f"no rows matched split: {selected_split}")

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    try:
        with partial.open("w", encoding="utf-8", newline="") as target:
            writer = csv.writer(target, lineterminator="\n")
            writer.writerow(("time_stamp", "sensor_index", "pm2.5_atm"))
            for (sensor, time), value in sorted(values.items()):
                writer.writerow((time, sensor, value))
        partial.replace(output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    with output.open("rb") as target:
        digest = hashlib.file_digest(target, "sha256").hexdigest()
    print(f"output={output}")
    print(
        f"split={selected_split} windows={selected_windows:,} "
        f"rows={len(values):,} sensors={len({key[0] for key in values}):,}"
    )
    print(f"sha256={digest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=("train", "all"), default="train")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    extract(args.input, args.output, args.split, args.overwrite)


if __name__ == "__main__":
    main()
