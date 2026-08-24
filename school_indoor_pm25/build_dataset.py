"""Copy and combine hourly PM2.5 for the 119 indoor school sensors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_COLUMNS = {"time_stamp", "sensor_index", "pm2.5_atm"}
COLLECTIONS = (
    ("history_tempo_overlap", Path("purpleair_hourly_pm25_atm_tempo_overlap")),
    ("current_fixed48_trend", Path("purpleair_pm25_download/data/fixed48_trend")),
    ("current_tempo_overlap", Path("purpleair_pm25_download/data/tempo_overlap")),
)


def arguments() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-repo",
        type=Path,
        default=repo.parent / "purple-air-pull",
        help="purple-air-pull checkout containing the source downloads",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "data" / "purple air",
    )
    return parser.parse_args()


def sensor_from_filename(path: Path) -> int | None:
    prefix = path.stem.split("_", 1)[0]
    return int(prefix) if prefix.isdigit() else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cohort(path: Path) -> set[int]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    sensors = {
        int(row["sensor_index"])
        for row in rows
        if row["location_type"].strip().lower() == "inside"
        and row["k12_status"].strip().lower() == "school"
        and row["is_k12"].strip().lower() == "true"
    }
    if len(rows) != 119 or len(sensors) != 119:
        raise ValueError(f"expected 119 validated indoor school sensors, found {len(sensors)}")
    return sensors


def copy_history(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and not path.name.endswith(".part"):
            shutil.copy2(path, destination / path.name)


def data_files(directory: Path, cohort: set[int]) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return sorted(
        path
        for path in directory.glob("*.csv")
        if sensor_from_filename(path) in cohort
    )


def read_collection(
    name: str,
    files: list[Path],
    cohort: set[int],
    values: dict[tuple[int, str], str],
) -> dict[str, object]:
    sensors: set[int] = set()
    file_sensors = {sensor_from_filename(path) for path in files}
    rows = duplicates = conflicts = 0
    for path in files:
        filename_sensor = sensor_from_filename(path)
        with path.open(newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            if not REQUIRED_COLUMNS.issubset(reader.fieldnames or ()):
                raise ValueError(f"unexpected columns in {path}")
            for line_number, row in enumerate(reader, 2):
                sensor = int(row["sensor_index"])
                timestamp = row["time_stamp"].strip()
                pm25 = row["pm2.5_atm"].strip()
                if sensor != filename_sensor or sensor not in cohort:
                    raise ValueError(f"unexpected sensor at {path}:{line_number}")
                if not timestamp or not pm25:
                    continue
                timestamp_value = int(timestamp)
                pm25_value = float(pm25)
                if timestamp_value % 3600 or pm25_value < 0 or not math.isfinite(pm25_value):
                    raise ValueError(f"invalid PM2.5 row at {path}:{line_number}")
                key = sensor, timestamp
                previous = values.get(key)
                if previous is not None:
                    duplicates += 1
                    if float(previous) != pm25_value:
                        conflicts += 1
                values[key] = pm25
                sensors.add(sensor)
                rows += 1
    return {
        "collection": name,
        "input_files": len(files),
        "input_rows": rows,
        "sensors_with_download_files": len(file_sensors),
        "sensors_with_rows": len(sensors),
        "duplicate_sensor_hours": duplicates,
        "changed_sensor_hours": conflicts,
    }


def write_dataset(path: Path, values: dict[tuple[int, str], str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(("time_stamp", "sensor_index", "pm2.5_atm"))
        for (sensor, timestamp), pm25 in sorted(
            values.items(), key=lambda item: (item[0][0], int(item[0][1]))
        ):
            writer.writerow((timestamp, sensor, pm25))


def main() -> None:
    args = arguments()
    source_repo = args.source_repo.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cohort_source = source_repo / "purpleair_indoor_school_sensors.csv"
    cohort = load_cohort(cohort_source)
    cohort_copy = output_dir / cohort_source.name
    shutil.copy2(cohort_source, cohort_copy)

    history_source = source_repo / COLLECTIONS[0][1]
    history_copy = output_dir / "source_history" / history_source.name
    copy_history(history_source, history_copy)

    values: dict[tuple[int, str], str] = {}
    summaries = []
    for name, relative_path in COLLECTIONS:
        directory = history_copy if name == "history_tempo_overlap" else source_repo / relative_path
        files = data_files(directory, cohort)
        if name == "history_tempo_overlap" and {
            sensor_from_filename(path) for path in files
        } != cohort:
            raise ValueError("history downloads do not cover all 119 cohort sensors")
        summaries.append(read_collection(name, files, cohort, values))

    dataset = output_dir / "school_indoor_pm25.csv"
    write_dataset(dataset, values)
    rows_by_sensor = Counter(sensor for sensor, _ in values)
    timestamps = [int(timestamp) for _, timestamp in values]
    without_rows = sorted(cohort - rows_by_sensor.keys())
    as_utc = lambda timestamp: datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
    summary = {
        "cohort_definition": {
            "location_type": "inside",
            "k12_status": "school",
            "is_k12": "true",
            "sensor_count": len(cohort),
        },
        "source_repo": str(source_repo),
        "collections": summaries,
        "output": {
            "path": str(dataset),
            "rows": len(values),
            "sensors_with_rows": len(rows_by_sensor),
            "sensors_without_rows": len(without_rows),
            "sensor_ids_without_rows": without_rows,
            "first_time_utc": as_utc(min(timestamps)),
            "last_time_utc": as_utc(max(timestamps)),
            "minimum_rows_per_sensor": min(rows_by_sensor.values()),
            "maximum_rows_per_sensor": max(rows_by_sensor.values()),
            "sha256": sha256(dataset),
        },
        "copied_history": {
            "path": str(history_copy),
            "files": len(list(history_copy.iterdir())),
        },
        "cohort_sha256": sha256(cohort_copy),
    }
    with (output_dir / "build_summary.json").open("w", encoding="utf-8") as target:
        json.dump(summary, target, indent=2)
        target.write("\n")

    print(json.dumps(summary["output"], indent=2))


if __name__ == "__main__":
    main()
