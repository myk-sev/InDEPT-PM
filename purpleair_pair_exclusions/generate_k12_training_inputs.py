"""Generate self-contained K-12 paired-PurpleAir masked-training inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from purpleair_pair_exclusions.detect_pair_exclusions import (
    HISTORY_ROOT,
    INDOOR_EXCLUSION_PATHS,
    INDOOR_RANGE_EXCLUSIONS_PATH,
    OUTDOOR_EXCLUSIONS_PATH,
    ROOT,
    read_excluded_sensor_ids,
    read_fema_school_ids,
    read_histories,
    read_overlap_indoor_ids,
)
from purpleair_pair_exclusions.outdoor_quality import (
    read_indoor_exclusions,
    read_outdoor_exclusions,
)
from purpleair_pair_exclusions.training_intervals import (
    build_training_intervals,
    read_ranked_candidates,
    read_responsiveness,
    write_training_data,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--sensor-inventory",
        type=Path,
        default=ROOT.parent / "purple-air-pull" / "purpleair_continental_us_sensors.csv",
    )
    result.add_argument(
        "--school-smoke-overlap",
        type=Path,
        default=(
            ROOT.parent
            / "purple-air-pull"
            / "smoke_plume_intersection"
            / "results"
            / "purpleair_indoor_school_sparse_wildfire_ranges.csv"
        ),
    )
    result.add_argument(
        "--fema-school-sensors",
        type=Path,
        default=ROOT.parent / "purple-air-pull" / "purpleair_indoor_school_sensors.csv",
    )
    result.add_argument("--indoor-history", action="append", type=Path)
    result.add_argument("--outdoor-history", action="append", type=Path)
    result.add_argument("--school-pair-distance", type=float, default=1000.0)
    result.add_argument(
        "--responsiveness",
        type=Path,
        default=(
            ROOT
            / "inputs"
            / "masked_pretraining"
            / "responsiveness"
            / "pair_responsiveness.csv"
        ),
    )
    result.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "inputs" / "masked_pretraining",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    indoor_paths = args.indoor_history or [HISTORY_ROOT / "all_indoor_pm25.csv"]
    outdoor_paths = args.outdoor_history or [HISTORY_ROOT / "all_outdoor_pm25.csv"]
    indoor = read_histories(indoor_paths)
    outdoor = read_histories(outdoor_paths)
    retrieved_ids = set(indoor)
    overlap = read_overlap_indoor_ids(args.school_smoke_overlap) & retrieved_ids
    fema = read_fema_school_ids(args.fema_school_sensors) & retrieved_ids
    school_ids = overlap | fema
    candidates = read_ranked_candidates(
        args.sensor_inventory, school_ids, args.school_pair_distance
    )
    cohorts = {"smoke_overlap_school": overlap, "fema_school": fema}
    responsiveness = read_responsiveness(args.responsiveness)
    excluded_indoor = set().union(
        *(read_excluded_sensor_ids(path) for path in INDOOR_EXCLUSION_PATHS)
    )
    variants = (
        (
            "exclusion_aware",
            excluded_indoor,
            read_indoor_exclusions(INDOOR_RANGE_EXCLUSIONS_PATH),
            read_outdoor_exclusions(OUTDOOR_EXCLUSIONS_PATH),
        ),
        (
            "no_exclusions",
            set(),
            (),
            (),
        ),
    )
    for name, excluded, indoor_ranges, outdoor_ranges in variants:
        intervals, _ = build_training_intervals(
            candidates, indoor, cohorts, excluded, indoor_ranges, outdoor_ranges
        )
        counts = write_training_data(
            args.output_root / name,
            intervals,
            indoor,
            outdoor,
            responsiveness,
            filename=f"k12_{name}_masked_training_data.csv",
        )
        print(
            f"{name}: {counts['assigned_indoor_sensors']} sensors, "
            f"{counts['training_intervals']} intervals, "
            f"{counts['training_readings']} PM2.5 readings"
        )


if __name__ == "__main__":
    main()
