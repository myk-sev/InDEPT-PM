"""Generate paired-PurpleAir training contracts for every retrieved indoor sensor."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from purpleair_pair_exclusions.detect_pair_exclusions import (
    INDOOR_EXCLUSION_PATHS,
    INDOOR_RANGE_EXCLUSIONS_PATH,
    OUTDOOR_EXCLUSIONS_PATH,
    read_excluded_sensor_ids,
    read_fema_school_ids,
    read_histories,
    read_overlap_indoor_ids,
    read_pairs,
)
from purpleair_pair_exclusions.outdoor_quality import (
    read_indoor_exclusions,
    read_outdoor_exclusions,
)
from purpleair_pair_exclusions.training_intervals import (
    build_training_intervals,
    read_ranked_candidates,
    source_record,
    write_training_contract,
)


ROOT = Path(__file__).resolve().parent.parent
AIR = ROOT / "data" / "purple air"
DEFAULT_INDOOR = (AIR / "school_indoor_pm25.csv", AIR / "general_non_school_indoor_pm25.csv")
DEFAULT_OUTDOOR = (
    AIR / "outdoor_school" / "school_outdoor_pm25.csv",
    AIR / "outdoor_non_school" / "non_school_outdoor_pm25.csv",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--pairs", type=Path, default=ROOT.parent / "purple-air-pull" / "purpleair_continental_us_pairs.csv")
    result.add_argument("--selected-pairs", type=Path, default=ROOT / "purpleair_pair_exclusions" / "results" / "selected_pairs.csv")
    result.add_argument("--sensor-inventory", type=Path, default=ROOT.parent / "purple-air-pull" / "purpleair_continental_us_sensors.csv")
    result.add_argument("--school-smoke-overlap", type=Path, default=ROOT.parent / "purple-air-pull" / "smoke_plume_intersection" / "results" / "purpleair_indoor_school_sparse_wildfire_ranges.csv")
    result.add_argument("--fema-school-sensors", type=Path, default=ROOT.parent / "purple-air-pull" / "purpleair_indoor_school_sensors.csv")
    result.add_argument("--indoor-history", action="append", type=Path)
    result.add_argument("--outdoor-history", action="append", type=Path)
    result.add_argument("--school-pair-distance", type=float, default=1000.0)
    result.add_argument("--output-root", type=Path, default=ROOT / "masked_pretraining" / "inputs" / "all_sensors")
    return result


def _pair(row: dict[str, object], rank: int = 1) -> dict[str, object]:
    return {
        "indoor_sensor_id": int(row["indoor_sensor_id"]),
        "indoor_name": row["indoor_name"],
        "outdoor_sensor_id": int(row["outdoor_sensor_id"]),
        "outdoor_name": row["outdoor_name"],
        "distance_meters": float(row["distance_meters"]),
        "candidate_rank": rank,
    }


def _selected_pairs(path: Path) -> dict[int, dict[str, object]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return {int(row["indoor_sensor_id"]): row for row in csv.DictReader(source)}


def main() -> None:
    args = parser().parse_args()
    indoor_paths = args.indoor_history or list(DEFAULT_INDOOR)
    outdoor_paths = args.outdoor_history or list(DEFAULT_OUTDOOR)
    indoor = read_histories(indoor_paths)
    retrieved_ids = set(indoor)
    overlap = read_overlap_indoor_ids(args.school_smoke_overlap) & retrieved_ids
    fema = read_fema_school_ids(args.fema_school_sensors) & retrieved_ids
    school_ids = overlap | fema
    candidates = read_ranked_candidates(
        args.sensor_inventory, school_ids, args.school_pair_distance
    )
    snapshot = {int(row["indoor_sensor_id"]): row for row in read_pairs(args.pairs)}
    replacements = _selected_pairs(args.selected_pairs)
    for indoor_id in sorted(retrieved_ids - school_ids):
        original = snapshot.get(indoor_id)
        if original is None:
            continue
        choices = [_pair(original)]
        replacement = replacements.get(indoor_id)
        if replacement and int(replacement["outdoor_sensor_id"]) != int(original["outdoor_sensor_id"]):
            choices.append(_pair(replacement, 2))
        candidates[indoor_id] = choices

    cohorts = {
        "smoke_overlap_school": overlap,
        "fema_school": fema,
        "downloaded_history": retrieved_ids,
    }
    common_sources = {
        "cohort_scope": "all retrieved indoor PurpleAir histories; school pairs use the 1 km inventory match and non-school pairs preserve the downloaded pair snapshot",
        "purpleair_pair_snapshot": source_record(args.pairs),
        "selected_pair_replacements": source_record(args.selected_pairs),
        "sensor_inventory": source_record(args.sensor_inventory),
        "school_cohorts": [
            source_record(args.school_smoke_overlap),
            source_record(args.fema_school_sensors),
        ],
        "indoor_history": [source_record(path) for path in indoor_paths],
        "outdoor_history": [source_record(path) for path in outdoor_paths],
    }
    exclusion_paths = (*INDOOR_EXCLUSION_PATHS, INDOOR_RANGE_EXCLUSIONS_PATH, OUTDOOR_EXCLUSIONS_PATH)
    excluded_indoor = set().union(*(read_excluded_sensor_ids(path) for path in INDOOR_EXCLUSION_PATHS))
    variants = (
        (
            "exclusion_aware",
            excluded_indoor,
            read_indoor_exclusions(INDOOR_RANGE_EXCLUSIONS_PATH),
            read_outdoor_exclusions(OUTDOOR_EXCLUSIONS_PATH),
            common_sources,
            [source_record(path) for path in exclusion_paths],
        ),
        (
            "no_exclusions",
            set(),
            (),
            (),
            common_sources | {"exclusion_policy": "none; all reviewed whole-sensor and bounded-range exclusions are intentionally ignored"},
            [],
        ),
    )
    for name, excluded, indoor_ranges, outdoor_ranges, sources, exclusion_records in variants:
        intervals, unresolved = build_training_intervals(
            candidates, indoor, cohorts, excluded, indoor_ranges, outdoor_ranges
        )
        metadata = write_training_contract(
            args.output_root / name,
            intervals,
            unresolved,
            args.school_pair_distance,
            sources,
            exclusion_records,
            len(retrieved_ids),
        )
        counts = metadata["counts"]
        print(f"{name}: {counts['assigned_indoor_sensors']} sensors, {counts['training_intervals']} intervals")


if __name__ == "__main__":
    main()
