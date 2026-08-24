"""Build legacy-pipeline pair manifests and NAQFC data for the masked cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data_loader import (
    _align_forecast_cycles,
    _observations,
    _read_indoor_history,
    _read_outdoor_history,
)
from pull_naqfc import SCHEMA, extract


ROOT = Path(__file__).resolve().parent
AIRGUARD = ROOT.parent
PAIR_FIELDS = (
    "sensor_id",
    "latitude",
    "longitude",
    "indoor_name",
    "outdoor_sensor_id",
    "outdoor_name",
    "distance_meters",
    "cohort_sources",
)
CYCLE = re.compile(r"naqfc_(\d{8})T(\d{2})\.parquet$")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def atomic_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    with partial.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    partial.replace(path)


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    partial.replace(path)


def epoch(day: str, cycle: str) -> int:
    return int(
        datetime.strptime(day + cycle, "%Y%m%d%H")
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )


def cohort_rows(args: argparse.Namespace) -> tuple[list[dict[str, object]], set[int]]:
    inventory = {
        int(row["sensor_index"]): row for row in read_rows(args.sensor_inventory)
    }
    school = {
        int(row["indoor_sensor_id"])
        for row in read_rows(args.school_intervals)
    }
    rows = []
    for row in read_rows(args.selected_pairs):
        indoor = int(row["indoor_sensor_id"])
        outdoor = int(row["outdoor_sensor_id"])
        sensor = inventory[outdoor]
        rows.append(
            {
                "sensor_id": indoor,
                "latitude": float(sensor["latitude"]),
                "longitude": float(sensor["longitude"]),
                "indoor_name": row["indoor_name"],
                "outdoor_sensor_id": outdoor,
                "outdoor_name": row["outdoor_name"],
                "distance_meters": row["distance_meters"],
                "cohort_sources": row["cohort_sources"],
            }
        )
    if not rows or len({int(row["sensor_id"]) for row in rows}) != len(rows):
        raise ValueError("selected pair input must contain unique indoor sensors")
    school &= {int(row["sensor_id"]) for row in rows}
    return rows, school


def write_pair_manifests(
    output: Path, rows: list[dict[str, object]], school: set[int]
) -> dict[str, Path]:
    paths = {
        "all": output / "masked_cohort_tempo_naqfc_locations.csv",
        "school": output / "masked_school_tempo_naqfc_locations.csv",
        "non_school": output / "masked_non_school_tempo_naqfc_locations.csv",
    }
    atomic_csv(paths["all"], PAIR_FIELDS, rows)
    atomic_csv(
        paths["school"],
        PAIR_FIELDS,
        [row for row in rows if int(row["sensor_id"]) in school],
    )
    atomic_csv(
        paths["non_school"],
        PAIR_FIELDS,
        [row for row in rows if int(row["sensor_id"]) not in school],
    )
    return paths


def source_cycles(root: Path) -> dict[int, Path]:
    cycles = {}
    for path in root.rglob("naqfc_*.parquet"):
        match = CYCLE.fullmatch(path.name)
        if match:
            cycles[epoch(match.group(1), match.group(2))] = path
    if len(cycles) != 3638:
        raise ValueError(f"expected 3,638 validated NAQFC cycles; found {len(cycles):,}")
    return dict(sorted(cycles.items()))


def location_contract(
    rows: list[dict[str, object]], source_root: Path
) -> tuple[list[dict[str, object]], list[int], list[str], dict[int, dict[str, object]]]:
    source = pq.read_table(
        source_root / "locations.parquet",
        columns=["location_id", "latitude", "longitude"],
    )
    coordinate_to_source = {
        (float(latitude), float(longitude)): location
        for location, latitude, longitude in zip(
            source["location_id"].to_pylist(),
            source["latitude"].to_pylist(),
            source["longitude"].to_pylist(),
        )
    }
    locations = []
    covered_source_ids = []
    covered_new_ids = []
    missing = {}
    for number, row in enumerate(rows, 1):
        location_id = f"location_{number:06d}"
        latitude = float(row["latitude"])
        longitude = float(row["longitude"])
        location = {
            "location_id": location_id,
            "latitude": latitude,
            "longitude": longitude,
        }
        locations.append(location)
        source_id = coordinate_to_source.get((latitude, longitude))
        if source_id is None:
            missing[int(row["sensor_id"])] = location
        else:
            covered_source_ids.append(source_id)
            covered_new_ids.append(location_id)
    if len(covered_new_ids) + len(missing) != len(rows):
        raise ValueError("every selected pair must map to a remapped or raw location")
    return locations, covered_source_ids, covered_new_ids, missing


def required_raw_locations(
    args: argparse.Namespace,
    rows: list[dict[str, object]],
    missing: dict[int, dict[str, object]],
    cycles: np.ndarray,
) -> dict[int, list[dict[str, object]]]:
    if args.all_raw_cycles:
        locations = [missing[sensor] for sensor in sorted(missing)]
        return {int(cycle): locations for cycle in cycles}
    sensors = [int(row["sensor_id"]) for row in rows]
    indoor = _read_indoor_history([args.indoor_history], sensors)
    outdoor = _read_outdoor_history(args.tempo_history, sensors)
    timestamps, observations = _observations(sensors, indoor, outdoor)
    aligned, _ = _align_forecast_cycles(timestamps, cycles, 72, 36)
    outdoor_valid = np.isfinite(observations[:, :, 0])
    indoor_valid = np.isfinite(observations[:, :, 1])
    outdoor_count = np.pad(
        outdoor_valid.cumsum(axis=1, dtype=np.int32), ((0, 0), (1, 0))
    )
    indoor_missing = np.pad(
        (~indoor_valid).cumsum(axis=1, dtype=np.int32), ((0, 0), (1, 0))
    )
    required: dict[int, set[int]] = defaultdict(set)
    for anchor in range(167, len(timestamps) - 36):
        cycle_index = int(aligned[anchor])
        if cycle_index < 0:
            continue
        valid = (
            (indoor_missing[:, anchor + 1] - indoor_missing[:, anchor - 167] == 0)
            & (outdoor_count[:, anchor + 1] - outdoor_count[:, anchor - 167] >= 24)
            & (outdoor_count[:, anchor + 1] - outdoor_count[:, anchor - 2] == 3)
            & (indoor_missing[:, anchor + 37] - indoor_missing[:, anchor + 1] == 0)
        )
        for index in np.flatnonzero(valid):
            sensor = sensors[int(index)]
            if sensor in missing:
                required[int(cycles[cycle_index])].add(sensor)
    sample = epoch("20250916", "12")
    if sample not in set(cycles):
        raise ValueError("required NAQFC sample cycle is unavailable")
    required[sample].update(missing)
    return {
        cycle: [missing[sensor] for sensor in sorted(sensors)]
        for cycle, sensors in sorted(required.items())
    }


def write_locations(path: Path, locations: list[dict[str, object]]) -> None:
    table = pa.table(
        {
            "location_id": [row["location_id"] for row in locations],
            "latitude": [row["latitude"] for row in locations],
            "longitude": [row["longitude"] for row in locations],
        }
    )
    partial = path.with_suffix(".parquet.part")
    pq.write_table(table, partial, compression="zstd")
    partial.replace(path)


def raw_rows(
    args: argparse.Namespace, cycle: int, locations: list[dict[str, object]]
) -> list[dict[str, object]]:
    moment = datetime.fromtimestamp(cycle, timezone.utc)
    grib = args.grib_root / f"naqfc_{moment:%Y%m%dT%H}.grib2"
    return extract(args.wgrib2, grib, locations, moment.date(), moment.hour)


def raw_table(rows: list[dict[str, object]]) -> pa.Table:
    return pa.table(
        {field.name: [row[field.name] for row in rows] for field in SCHEMA},
        schema=SCHEMA,
    )


def remap_table(
    source: Path,
    source_ids: list[str],
    new_ids: list[str],
    extracted: list[dict[str, object]],
) -> pa.Table:
    table = pq.read_table(source, columns=SCHEMA.names).cast(SCHEMA)
    positions: dict[str, list[int]] = defaultdict(list)
    for index, location_id in enumerate(table["location_id"].to_pylist()):
        positions[location_id].append(index)
    take = []
    replacement = []
    for source_id, new_id in zip(source_ids, new_ids):
        indices = positions.get(source_id)
        if not indices or len(indices) != 72:
            raise ValueError(f"unexpected source rows for {source_id} in {source}")
        take.extend(indices)
        replacement.extend([new_id] * len(indices))
    selected = table.take(pa.array(take, type=pa.int64()))
    selected = selected.set_column(
        selected.schema.get_field_index("location_id"),
        "location_id",
        pa.array(replacement),
    )
    return pa.concat_tables([selected, raw_table(extracted)]) if extracted else selected


def output_path(output: Path, source: Path) -> Path:
    return output.joinpath(*source.parts[-4:])


def valid_output(path: Path, rows: int) -> bool:
    try:
        return path.is_file() and pq.read_metadata(path).num_rows == rows
    except (OSError, pa.ArrowException):
        return False


def write_cycle(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".parquet.part")
    pq.write_table(table, partial, compression="zstd")
    partial.replace(path)


def build_forecasts(
    args: argparse.Namespace,
    rows: list[dict[str, object]],
    locations: list[dict[str, object]],
    source_ids: list[str],
    new_ids: list[str],
    missing: dict[int, dict[str, object]],
    cycles: dict[int, Path],
) -> None:
    args.forecast_output.mkdir(parents=True, exist_ok=True)
    write_locations(args.forecast_output / "locations.parquet", locations)
    required = required_raw_locations(
        args, rows, missing, np.array(list(cycles), dtype=np.int64)
    )
    print(
        f"NAQFC plan: {len(cycles):,} cycles, {len(new_ids):,} remapped locations, "
        f"{len(missing):,} missing locations, {len(required):,} raw cycles",
        flush=True,
    )
    futures: dict[int, Future[list[dict[str, object]]]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for cycle, requested in required.items():
            destination = output_path(args.forecast_output, cycles[cycle])
            expected = 72 * (len(new_ids) + len(requested))
            if not valid_output(destination, expected):
                futures[cycle] = executor.submit(raw_rows, args, cycle, requested)
        started = time.monotonic()
        last_update = started
        manifest = []
        for number, (cycle, source) in enumerate(cycles.items(), 1):
            requested = required.get(cycle, [])
            destination = output_path(args.forecast_output, source)
            expected = 72 * (len(new_ids) + len(requested))
            if not valid_output(destination, expected):
                extracted = futures[cycle].result() if requested else []
                write_cycle(
                    destination,
                    remap_table(source, source_ids, new_ids, extracted),
                )
            manifest.append(
                {
                    "cycle_time_utc": datetime.fromtimestamp(
                        cycle, timezone.utc
                    ).isoformat().replace("+00:00", "Z"),
                    "remapped_locations": len(new_ids),
                    "raw_locations": len(requested),
                    "rows": expected,
                    "source_parquet": str(source.resolve()),
                }
            )
            now = time.monotonic()
            if number == len(cycles) or now - last_update >= 30:
                rate = number / max(now - started, 0.001)
                eta = (len(cycles) - number) / rate
                print(
                    f"NAQFC progress: {number:,}/{len(cycles):,} cycles "
                    f"({number / len(cycles):.1%}), ETA {eta / 60:.1f} minutes",
                    flush=True,
                )
                last_update = now
    atomic_csv(
        args.forecast_output / "build_manifest.csv",
        (
            "cycle_time_utc",
            "remapped_locations",
            "raw_locations",
            "rows",
            "source_parquet",
        ),
        manifest,
    )


def build_view(
    source_root: Path,
    output_root: Path,
    locations: list[dict[str, object]],
    selected: list[int],
) -> None:
    mapping = {
        f"location_{source + 1:06d}": f"location_{target + 1:06d}"
        for target, source in enumerate(selected)
    }
    write_locations(
        output_root / "locations.parquet",
        [
            {**locations[source], "location_id": f"location_{target + 1:06d}"}
            for target, source in enumerate(selected)
        ],
    )
    paths = sorted(source_root.rglob("naqfc_*.parquet"))
    started = last_update = time.monotonic()
    for number, source in enumerate(paths, 1):
        table = pq.read_table(source, columns=SCHEMA.names).cast(SCHEMA)
        replacements = [mapping.get(value) for value in table["location_id"].to_pylist()]
        indices = [index for index, value in enumerate(replacements) if value]
        destination = output_root / source.relative_to(source_root)
        if not valid_output(destination, len(indices)):
            selected_table = table.take(pa.array(indices, type=pa.int64()))
            selected_table = selected_table.set_column(
                selected_table.schema.get_field_index("location_id"),
                "location_id",
                pa.array([replacements[index] for index in indices]),
            )
            write_cycle(destination, selected_table)
        now = time.monotonic()
        if number == len(paths) or now - last_update >= 30:
            rate = number / max(now - started, 0.001)
            eta = (len(paths) - number) / rate
            print(
                f"{output_root.name}: {number:,}/{len(paths):,} cycles "
                f"({number / len(paths):.1%}), ETA {eta / 60:.1f} minutes",
                flush=True,
            )
            last_update = now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selected-pairs",
        type=Path,
        default=ROOT / "purpleair_pair_exclusions" / "results" / "selected_pairs.csv",
    )
    parser.add_argument(
        "--school-intervals",
        type=Path,
        default=(
            ROOT
            / "inputs"
            / "masked_pretraining"
            / "exclusion_aware"
            / "k12_exclusion_aware_masked_training_data.csv"
        ),
    )
    parser.add_argument(
        "--sensor-inventory",
        type=Path,
        default=AIRGUARD / "purple-air-pull" / "purpleair_continental_us_sensors.csv",
    )
    parser.add_argument(
        "--indoor-history",
        type=Path,
        default=ROOT / "data" / "purple air" / "all_indoor_pm25.csv",
    )
    parser.add_argument(
        "--tempo-history",
        type=Path,
        default=AIRGUARD / "purple-air-pull" / "tempo_pm25_sensor_match" / "tempo_pm25_indoor_sensors.csv",
    )
    parser.add_argument("--source-forecast-root", type=Path, default=ROOT / "naqfc_output")
    parser.add_argument(
        "--grib-root",
        type=Path,
        default=Path("D:/AirGuard-data/naqfc_output/naqfc_gribs"),
    )
    parser.add_argument(
        "--pair-output", type=Path, default=ROOT / "data" / "legacy"
    )
    parser.add_argument(
        "--forecast-output", type=Path, default=ROOT / "naqfc_output_masked_cohort"
    )
    parser.add_argument(
        "--school-forecast-output",
        type=Path,
        default=ROOT / "naqfc_output_masked_school",
    )
    parser.add_argument(
        "--non-school-forecast-output",
        type=Path,
        default=ROOT / "naqfc_output_masked_non_school",
    )
    parser.add_argument(
        "--wgrib2", type=Path, default=ROOT / ".tools" / "wgrib2" / "wgrib2.exe"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--all-raw-cycles",
        action="store_true",
        help="extract every missing cohort location for every validated NAQFC cycle",
    )
    parser.add_argument(
        "--skip-cohort-views",
        action="store_true",
        help="do not create duplicate school and non-school forecast views",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError("workers must be between 1 and 16")
    rows, school = cohort_rows(args)
    manifests = write_pair_manifests(args.pair_output, rows, school)
    cycles = source_cycles(args.source_forecast_root)
    locations, source_ids, new_ids, missing = location_contract(
        rows, args.source_forecast_root
    )
    config = {
        "selected_pairs_sha256": sha256(args.selected_pairs),
        "school_intervals_sha256": sha256(args.school_intervals),
        "sensor_inventory_sha256": sha256(args.sensor_inventory),
        "pair_manifests": {name: str(path.resolve()) for name, path in manifests.items()},
        "locations": len(locations),
        "remapped_locations": len(new_ids),
        "raw_locations": len(missing),
        "source_cycles": len(cycles),
    }
    if args.all_raw_cycles:
        config["raw_cycle_scope"] = "all"
    config_path = args.forecast_output / "build_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("build inputs changed; use a new forecast output directory")
    atomic_json(config_path, config)
    build_forecasts(args, rows, locations, source_ids, new_ids, missing, cycles)
    if not args.skip_cohort_views:
        school_indices = [
            index for index, row in enumerate(rows) if int(row["sensor_id"]) in school
        ]
        build_view(
            args.forecast_output,
            args.school_forecast_output,
            locations,
            school_indices,
        )
        build_view(
            args.forecast_output,
            args.non_school_forecast_output,
            locations,
            [index for index in range(len(rows)) if index not in set(school_indices)],
        )
    print(
        f"Complete: pairs={len(rows):,}, school={len(school):,}, "
        f"non_school={len(rows) - len(school):,}, forecasts={args.forecast_output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
