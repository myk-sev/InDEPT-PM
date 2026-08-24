"""Safely merge staged 1 km PurpleAir downloads into the main CSV archives."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PURPLEAIR_DATA = ROOT / "data" / "purple air"
COLUMNS = ("time_stamp", "sensor_index", "pm2.5_atm")
MAX_REQUEST = timedelta(days=180)


@dataclass(frozen=True)
class Archive:
    name: str
    manifest: Path
    source: Path
    destination: Path


def expected_files(manifest: Path) -> set[str]:
    names = set()
    with manifest.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            sensor = int(row["sensor_index"])
            start = datetime.fromisoformat(row["start_time_utc"].replace("Z", "+00:00"))
            final_end = datetime.fromisoformat(row["end_time_utc"].replace("Z", "+00:00"))
            while start < final_end:
                end = min(start + MAX_REQUEST, final_end)
                names.add(f"{sensor}_{start:%Y%m%dT%H%M%SZ}_{end:%Y%m%dT%H%M%SZ}.csv")
                start = end
    if not names:
        raise ValueError(f"empty download manifest: {manifest}")
    return names


def source_files(archive: Archive) -> list[Path]:
    if not archive.source.is_dir():
        raise FileNotFoundError(f"{archive.name} staging directory not found: {archive.source}")
    partials = list(archive.source.rglob("*.part"))
    if partials:
        raise ValueError(f"{archive.name} staging directory contains partial downloads")
    expected = expected_files(archive.manifest)
    files = {
        path.name: path
        for path in archive.source.rglob("*.csv")
        if path.stem.split("_", 1)[0].isdigit()
    }
    missing, unexpected = expected - files.keys(), files.keys() - expected
    if missing or unexpected:
        raise ValueError(
            f"{archive.name} staging files do not match the manifest: "
            f"{len(missing)} missing, {len(unexpected)} unexpected"
        )
    return [files[name] for name in sorted(expected)]


def read_values(
    paths: list[Path], check_filename: bool
) -> tuple[dict[tuple[int, int], tuple[float, str]], int]:
    values, missing_rows = {}, 0
    for path in paths:
        expected_sensor = (
            int(path.stem.split("_", 1)[0]) if check_filename else None
        )
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            missing = set(COLUMNS) - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"missing columns {sorted(missing)} in {path}")
            for line, row in enumerate(reader, 2):
                text = (row["pm2.5_atm"] or "").strip()
                if text.lower() in {"", "null", "nan"}:
                    missing_rows += 1
                    continue
                try:
                    timestamp = int(row["time_stamp"])
                    sensor = int(row["sensor_index"])
                    number = float(text)
                    if timestamp % 3600 or sensor < 1 or number < 0 or not math.isfinite(number):
                        raise ValueError
                    if expected_sensor is not None and sensor != expected_sensor:
                        raise ValueError
                except (TypeError, ValueError) as error:
                    raise ValueError(f"invalid PurpleAir row at {path}:{line}") from error
                key = sensor, timestamp
                previous = values.setdefault(key, (number, text))
                if previous[0] != number:
                    raise ValueError(f"conflicting sensor-hour at {path}:{line}")
    return values, missing_rows


def merge_archive(archive: Archive, files: list[Path], execute: bool) -> dict[str, int]:
    if not archive.destination.is_file():
        raise FileNotFoundError(f"{archive.name} destination not found: {archive.destination}")
    merged, _ = read_values([archive.destination], False)
    existing_rows = len(merged)
    staged, missing_rows = read_values(files, True)
    for key, value in staged.items():
        previous = merged.setdefault(key, value)
        if previous[0] != value[0]:
            raise ValueError(
                f"{archive.name} conflicts with the existing archive at "
                f"sensor {key[0]}, timestamp {key[1]}"
            )
    added_rows = len(merged) - existing_rows
    if execute and added_rows:
        part = archive.destination.with_suffix(archive.destination.suffix + ".part")
        if part.exists():
            raise FileExistsError(f"refusing to overwrite partial file: {part}")
        try:
            with part.open("x", encoding="utf-8", newline="") as target:
                writer = csv.writer(target, lineterminator="\n")
                writer.writerow(COLUMNS)
                writer.writerows(
                    (timestamp, sensor, merged[(sensor, timestamp)][1])
                    for sensor, timestamp in sorted(merged)
                )
            verified, _ = read_values([part], False)
            if verified != merged:
                raise ValueError(f"verification failed for {archive.name} merged archive")
            part.replace(archive.destination)
        except Exception:
            part.unlink(missing_ok=True)
            raise
    return {
        "existing_rows": existing_rows,
        "staged_rows": len(staged),
        "missing_staged_rows": missing_rows,
        "added_rows": added_rows,
        "merged_rows": len(merged),
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--indoor-source",
        type=Path,
        default=PURPLEAIR_DATA / "missing_1km_indoor_history",
    )
    parser.add_argument(
        "--outdoor-source",
        type=Path,
        default=PURPLEAIR_DATA / "missing_1km_outdoor_history",
    )
    parser.add_argument("--execute", action="store_true", help="replace the main archives")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    archives = (
        Archive(
            "indoor",
            ROOT / "pair_movement_correlation" / "download_lists" / "missing_1km_indoor_history.csv",
            args.indoor_source,
            PURPLEAIR_DATA / "school_indoor_pm25.csv",
        ),
        Archive(
            "outdoor",
            ROOT / "pair_movement_correlation" / "download_lists" / "missing_1km_outdoor_history.csv",
            args.outdoor_source,
            PURPLEAIR_DATA / "school_outdoor_pm25.csv",
        ),
    )
    files = {archive.name: source_files(archive) for archive in archives}
    for archive in archives:
        summary = merge_archive(archive, files[archive.name], args.execute)
        print(f"{archive.name}: " + ", ".join(f"{key}={value:,}" for key, value in summary.items()))
    if not args.execute:
        print("Dry run only. Add --execute to replace both main archives.")


if __name__ == "__main__":
    main()
