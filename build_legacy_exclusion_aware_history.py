"""Build the bounded-exclusion PurpleAir history used by legacy CSV exports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        yield from csv.DictReader(source)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc(value: str, default: int) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()) if value else default


def write_exclusions(path: Path, sensors: set[int], criterion: str) -> None:
    partial = path.with_suffix(path.suffix + ".part")
    with partial.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=("sensor_id", "criterion"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {"sensor_id": sensor, "criterion": criterion} for sensor in sorted(sensors)
        )
    partial.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=ROOT / "data/legacy/masked_cohort_tempo_naqfc_locations.csv")
    parser.add_argument("--input", type=Path, default=ROOT / "data/purple air/all_indoor_pm25.csv")
    parser.add_argument("--ranges", type=Path, default=ROOT / "data/exclusions/excluded_indoor_purpleair_ranges.csv")
    parser.add_argument(
        "--school-intervals",
        type=Path,
        default=(
            ROOT
            / "inputs/masked_pretraining/exclusion_aware"
            / "k12_exclusion_aware_masked_training_data.csv"
        ),
    )
    parser.add_argument("--whole-sensor-exclusions", type=Path, action="append")
    parser.add_argument("--output", type=Path, default=ROOT / "data/legacy/masked_cohort_indoor_pm25_exclusion_aware.csv")
    parser.add_argument("--exclusions-output", type=Path, default=ROOT / "data/legacy/masked_cohort_excluded_indoor_sensors.csv")
    parser.add_argument("--school-exclusions-output", type=Path, default=ROOT / "data/legacy/masked_school_scope_excluded_indoor_sensors.csv")
    parser.add_argument("--non-school-exclusions-output", type=Path, default=ROOT / "data/legacy/masked_non_school_scope_excluded_indoor_sensors.csv")
    args = parser.parse_args()

    exclusion_paths = args.whole_sensor_exclusions or [
        ROOT / "data/exclusions/permanently_excluded_indoor_sensors.csv",
        ROOT / "data/exclusions/excluded_indoor_sensors_pm25_gt1000.csv",
        ROOT / "data/exclusions/excluded_indoor_schools_pm25_gt1000.csv",
    ]
    sensors = {int(row["sensor_id"]) for row in rows(args.pairs)}
    school = {int(row["indoor_sensor_id"]) for row in rows(args.school_intervals)} & sensors
    whole = {
        int(row.get("sensor_id") or row["indoor_sensor_id"])
        for path in exclusion_paths
        for row in rows(path)
    } & sensors
    ranges: dict[int, list[tuple[int, int]]] = {}
    for row in rows(args.ranges):
        sensor = int(row["indoor_sensor_id"])
        if sensor in sensors:
            ranges.setdefault(sensor, []).append(
                (utc(row["start_utc"], -(2**63)), utc(row["end_utc"], 2**63 - 1))
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".part")
    kept = removed = 0
    with args.input.open(encoding="utf-8-sig", newline="") as source, partial.open(
        "w", encoding="utf-8", newline=""
    ) as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=reader.fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in reader:
            sensor = int(row["sensor_index"])
            if sensor not in sensors or sensor in whole:
                continue
            timestamp = int(row["time_stamp"])
            if any(start <= timestamp < end for start, end in ranges.get(sensor, ())):
                removed += 1
                continue
            writer.writerow(row)
            kept += 1
    partial.replace(args.output)

    write_exclusions(args.exclusions_output, whole, "reviewed whole-sensor exclusion")
    write_exclusions(
        args.school_exclusions_output,
        whole | (sensors - school),
        "outside school cohort or reviewed whole-sensor exclusion",
    )
    write_exclusions(
        args.non_school_exclusions_output,
        whole | school,
        "outside non-school cohort or reviewed whole-sensor exclusion",
    )

    metadata = {
        "pairs": str(args.pairs.resolve()),
        "pairs_sha256": sha256(args.pairs),
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "range_exclusions": str(args.ranges.resolve()),
        "range_exclusions_sha256": sha256(args.ranges),
        "whole_sensor_exclusions": [str(path.resolve()) for path in exclusion_paths],
        "whole_sensor_exclusion_sha256": {
            str(path.resolve()): sha256(path) for path in exclusion_paths
        },
        "cohort_sensors": len(sensors),
        "school_sensors": len(school),
        "whole_sensors_removed": len(whole),
        "range_sensors": len(ranges),
        "rows_kept": kept,
        "range_rows_removed": removed,
    }
    args.output.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Complete: kept={kept:,}, range_rows_removed={removed:,}, whole_sensors={len(whole):,}")


if __name__ == "__main__":
    main()
