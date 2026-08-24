"""Test PurpleAir pair roles against cached EPA AQS hourly PM2.5.

This is a validation pilot, not an automatic correction writer. It evaluates
confirmed and exploratory cases, permits both/neither sensors to look ambient,
and leaves weak evidence unclassified.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pair_movement_correlation.analyze_movement_correlation import (
    pearson,
    ranks,
    winsorize,
)
from pair_responsiveness.classify import (
    DEFAULT_INDOOR,
    DEFAULT_OUTDOOR,
    DEFAULT_PAIRS,
    read_pairs,
)
from purpleair_pair_exclusions.detect_pair_exclusions import read_histories
from purpleair_pair_exclusions.outdoor_quality import (
    exclude_outdoor_readings,
    read_indoor_exclusions,
    read_outdoor_exclusions,
)


HOUR = 3600
ROOT = Path(__file__).resolve().parent
AIRGUARD = ROOT.parent
DEFAULT_CASES = ROOT / "sensor_role_test_cases.json"
DEFAULT_OUTPUT = ROOT / "sensor_role_aqs_pilot_results"
DEFAULT_METADATA = AIRGUARD / "purple-air-pull" / "purpleair_continental_us_sensors.csv"
DEFAULT_AQS = AIRGUARD / "purple-air-pull" / "epa_pm25_monitor_data" / "aqs"
DEFAULT_INTERPOLATOR = (
    AIRGUARD / "purple-air-pull" / "pm25_interpolation" / "interpolate_pm25.py"
)
EXCLUSION_ROOT = ROOT / "data" / "exclusions"
DEFAULT_INDOOR_RANGES = EXCLUSION_ROOT / "excluded_indoor_purpleair_ranges.csv"
DEFAULT_OUTDOOR_RANGES = EXCLUSION_ROOT / "excluded_outdoor_purpleair_ranges.csv"
DEFAULT_PERMANENT_INDOOR = EXCLUSION_ROOT / "permanently_excluded_indoor_sensors.csv"
CASE_FIELDS = (
    "indoor_sensor_id",
    "outdoor_sensor_id",
    "case_type",
    "expected_assessment",
    "note",
)
REFERENCE_FIELDS = (
    "sensor_index",
    "latitude",
    "longitude",
    "time_stamp",
    "status",
    "estimated_pm2_5_ug_m3",
    "monitor_count",
    "available_monitor_count",
    "nearest_monitor_km",
    "farthest_monitor_km",
    "reason",
)
OUTPUT_FIELDS = (
    "pair_id",
    *CASE_FIELDS,
    "indoor_name",
    "outdoor_name",
    "assessment",
    "validation_status",
    "reason",
    "requested_hours",
    "failed_reference_hours",
    "reference_success_fraction",
    "reference_hours",
    "requested_changes",
    "reference_changes",
    "requested_qualifying_months",
    "qualifying_months",
    "indoor_month_wins",
    "outdoor_month_wins",
    "period_consistency",
    "indoor_ambient_score",
    "outdoor_ambient_score",
    "score_advantage_indoor",
    "indoor_level_spearman",
    "outdoor_level_spearman",
    "indoor_change_pearson",
    "outdoor_change_pearson",
    "indoor_quality_warning",
    "outdoor_quality_warning",
    "median_monitor_count",
    "median_nearest_monitor_km",
    "maximum_nearest_monitor_km",
)


@dataclass(frozen=True)
class Criteria:
    minimum_reference_hours: int = 120
    minimum_changes: int = 60
    minimum_month_hours: int = 24
    minimum_month_changes: int = 8
    minimum_months: int = 3
    minimum_score_advantage: float = 0.08
    minimum_consistency: float = 0.67
    ambient_like_score: float = 0.35
    non_ambient_score: float = 0.15
    maximum_median_nearest_km: float = 100.0


def read_cases(path: Path) -> list[dict[str, object]]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, list):
        raise ValueError("test cases must be a JSON list")
    cases, seen = [], set()
    for number, row in enumerate(raw, 1):
        if not isinstance(row, dict):
            raise ValueError(f"invalid test case {number}")
        missing = set(CASE_FIELDS) - set(row)
        if missing:
            raise ValueError(f"missing test-case columns: {', '.join(sorted(missing))}")
        try:
            indoor = int(row["indoor_sensor_id"])
            outdoor = int(row["outdoor_sensor_id"])
            key = indoor, outdoor
            if min(key) < 1 or key in seen:
                raise ValueError
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid test case {number}") from error
        if row["case_type"] not in {"validation", "exploratory"}:
            raise ValueError(f"invalid case_type in test case {number}")
        seen.add(key)
        cases.append(
            {
                "indoor_sensor_id": indoor,
                "outdoor_sensor_id": outdoor,
                "case_type": row["case_type"],
                "expected_assessment": row["expected_assessment"],
                "note": row["note"],
            }
        )
    return cases


def read_metadata(path: Path) -> dict[int, tuple[float, float]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"sensor_index", "latitude", "longitude"}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError(f"sensor metadata is missing: {', '.join(sorted(required))}")
        locations = {}
        for row in reader:
            try:
                sensor = int(row["sensor_index"])
                latitude = float(row["latitude"])
                longitude = float(row["longitude"])
            except (TypeError, ValueError):
                continue
            if math.isfinite(latitude) and math.isfinite(longitude):
                locations[sensor] = latitude, longitude
    return locations


def read_sensor_ids(path: Path) -> set[int]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if "sensor_id" not in (reader.fieldnames or ()):
            raise ValueError(f"{path} is missing sensor_id")
        return {int(row["sensor_id"]) for row in reader}


def select_cases(
    cases: list[dict[str, object]], pairs: list[dict[str, object]]
) -> list[dict[str, object]]:
    by_ids = {
        (int(pair["indoor_sensor_id"]), int(pair["outdoor_sensor_id"])): pair
        for pair in pairs
    }
    selected = []
    for case in cases:
        key = int(case["indoor_sensor_id"]), int(case["outdoor_sensor_id"])
        if key not in by_ids:
            raise ValueError(f"test pair {key[0]}-{key[1]} is not in the pair CSV")
        selected.append({**by_ids[key], **case, "pair_id": f"{key[0]}-{key[1]}"})
    return selected


def all_cases(pairs: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            **pair,
            "pair_id": f"{pair['indoor_sensor_id']}-{pair['outdoor_sensor_id']}",
            "case_type": "exploratory",
            "expected_assessment": "",
            "note": "all-pair AQS coverage audit",
        }
        for pair in pairs
    ]


def available_aqs_years(path: Path) -> set[int]:
    years = set()
    for archive in path.glob("hourly_*.zip"):
        try:
            parameter, year = archive.stem.removeprefix("hourly_").split("_")
            if parameter in {"88101", "88502"}:
                years.add(int(year))
        except ValueError:
            continue
    if not years:
        raise FileNotFoundError(f"no hourly AQS archives found in {path}")
    return years


def _evenly_spaced(values: list[int], count: int) -> list[int]:
    if len(values) <= count:
        return values
    if count == 1:
        return [values[len(values) // 2]]
    return [values[round(index * (len(values) - 1) / (count - 1))] for index in range(count)]


def sample_consecutive_hours(
    timestamps: set[int], available_years: set[int], maximum_month_hours: int
) -> list[int]:
    """Select evenly distributed consecutive pairs for level and movement tests."""
    by_month: dict[str, list[int]] = defaultdict(list)
    eligible = {
        timestamp
        for timestamp in timestamps
        if datetime.fromtimestamp(timestamp, timezone.utc).year in available_years
    }
    for timestamp in sorted(eligible):
        month = datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m")
        by_month[month].append(timestamp)
    sampled = set()
    for month_times in by_month.values():
        month_set = set(month_times)
        starts = [timestamp for timestamp in month_times if timestamp + HOUR in month_set]
        if not starts:
            sampled.update(_evenly_spaced(month_times, maximum_month_hours))
            continue
        for start in _evenly_spaced(starts, max(1, maximum_month_hours // 2)):
            sampled.update((start, start + HOUR))
    return sorted(sampled)


def build_requests(
    pairs: list[dict[str, object]],
    locations: dict[int, tuple[float, float]],
    indoor: dict[int, dict[int, float]],
    outdoor: dict[int, dict[int, float]],
    years: set[int],
    maximum_month_hours: int,
) -> list[dict[str, object]]:
    requests = []
    for pair in pairs:
        indoor_id = int(pair["indoor_sensor_id"])
        outdoor_id = int(pair["outdoor_sensor_id"])
        if indoor_id not in locations or outdoor_id not in locations:
            raise ValueError(f"missing coordinates for pair {pair['pair_id']}")
        first, second = locations[indoor_id], locations[outdoor_id]
        latitude = (first[0] + second[0]) / 2
        longitude = (first[1] + second[1]) / 2
        common = set(indoor[indoor_id]) & set(outdoor[outdoor_id])
        for timestamp in sample_consecutive_hours(common, years, maximum_month_hours):
            requests.append(
                {
                    "sensor_index": str(pair["pair_id"]),
                    "latitude": latitude,
                    "longitude": longitude,
                    "time_stamp": timestamp,
                }
            )
    return requests


def _similarity(
    values: dict[int, float], reference: dict[int, float], times: list[int]
) -> dict[str, float | int | None]:
    selected = sorted(set(times) & values.keys() & reference.keys())
    sensor = np.log1p([values[timestamp] for timestamp in selected])
    ambient = np.log1p([reference[timestamp] for timestamp in selected])
    level = pearson(ranks(sensor), ranks(ambient)) if len(selected) >= 2 else None
    changes = [
        timestamp
        for timestamp in selected
        if timestamp - HOUR in selected
    ]
    sensor_change = np.asarray(
        [math.log1p(values[t]) - math.log1p(values[t - HOUR]) for t in changes]
    )
    ambient_change = np.asarray(
        [math.log1p(reference[t]) - math.log1p(reference[t - HOUR]) for t in changes]
    )
    change = (
        pearson(winsorize(sensor_change, 1), winsorize(ambient_change, 1))
        if len(changes) >= 2
        else None
    )
    score = None if level is None or change is None else 0.65 * level + 0.35 * change
    return {"hours": len(selected), "changes": len(changes), "level": level, "change": change, "score": score}


def sensor_metrics(
    values: dict[int, float], reference: dict[int, float], criteria: Criteria
) -> dict[str, object]:
    times = sorted(values.keys() & reference.keys())
    overall = _similarity(values, reference, times)
    months: dict[str, list[int]] = defaultdict(list)
    for timestamp in times:
        months[datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m")].append(timestamp)
    monthly_scores = {}
    for month, month_times in months.items():
        metric = _similarity(values, reference, month_times)
        if (
            metric["hours"] >= criteria.minimum_month_hours
            and metric["changes"] >= criteria.minimum_month_changes
            and metric["score"] is not None
        ):
            monthly_scores[month] = float(metric["score"])
    sufficient = (
        overall["hours"] >= criteria.minimum_reference_hours
        and overall["changes"] >= criteria.minimum_changes
        and overall["score"] is not None
    )
    return {**overall, "monthly_scores": monthly_scores, "sufficient": sufficient}


def sampled_coverage(times: set[int], criteria: Criteria) -> tuple[int, int, int]:
    changes = sum(timestamp - HOUR in times for timestamp in times)
    months: dict[str, set[int]] = defaultdict(set)
    for timestamp in times:
        months[datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m")].add(timestamp)
    qualifying = sum(
        len(month_times) >= criteria.minimum_month_hours
        and sum(timestamp - HOUR in month_times for timestamp in month_times)
        >= criteria.minimum_month_changes
        for month_times in months.values()
    )
    return len(times), changes, qualifying


def quality_warning(values: dict[int, float], permanently_excluded: bool = False) -> tuple[str, bool]:
    if permanently_excluded:
        return "permanently_excluded", True
    if not values:
        return "missing_history", True
    data = np.asarray(list(values.values()))
    if np.ptp(data) < 0.1 and float(np.median(data)) >= 1000:
        return "stuck_extreme_error_level", True
    if float(np.mean(data >= 1000)) >= 0.01:
        return "repeated_extreme_readings", False
    if float(np.mean(data < 0.1)) >= 0.90:
        return "persistent_near_zero", False
    if np.ptp(data) < 0.1:
        return "flatline", False
    return "none", False


def assess_pair(
    pair: dict[str, object],
    indoor: dict[int, float],
    outdoor: dict[int, float],
    reference_rows: list[dict[str, object]],
    criteria: Criteria,
    permanent_indoor: set[int] | None = None,
) -> dict[str, object]:
    reference = {
        int(row["time_stamp"]): float(row["estimated_pm2_5_ug_m3"])
        for row in reference_rows
        if row["status"] == "ok"
    }
    requested_times = {int(row["time_stamp"]) for row in reference_rows}
    requested_hours, requested_changes, requested_months = sampled_coverage(
        requested_times, criteria
    )
    indoor_id = int(pair["indoor_sensor_id"])
    permanent_indoor = permanent_indoor or set()
    inside_quality, inside_blocked = quality_warning(indoor, indoor_id in permanent_indoor)
    outside_quality, outside_blocked = quality_warning(outdoor)
    inside = sensor_metrics(indoor, reference, criteria)
    outside = sensor_metrics(outdoor, reference, criteria)
    nearest = [float(row["nearest_monitor_km"]) for row in reference_rows if row["status"] == "ok"]
    monitor_counts = [float(row["monitor_count"]) for row in reference_rows if row["status"] == "ok"]
    common_months = sorted(set(inside["monthly_scores"]) & set(outside["monthly_scores"]))
    monthly_advantages = [
        float(inside["monthly_scores"][month]) - float(outside["monthly_scores"][month])
        for month in common_months
    ]
    inside_wins = sum(value > 0 for value in monthly_advantages)
    outside_wins = sum(value < 0 for value in monthly_advantages)
    consistency = max(inside_wins, outside_wins) / len(common_months) if common_months else 0.0
    advantage = (
        float(inside["score"]) - float(outside["score"])
        if inside["score"] is not None and outside["score"] is not None
        else math.nan
    )
    median_nearest = float(np.median(nearest)) if nearest else math.nan

    if inside_blocked or outside_blocked:
        assessment, reason = "sensor_quality_review", "blocking_sensor_quality_warning"
    elif (
        requested_hours < criteria.minimum_reference_hours
        or requested_changes < criteria.minimum_changes
    ):
        assessment, reason = "insufficient_reference", "too_few_paired_PurpleAir_samples"
    elif not inside["sufficient"] or not outside["sufficient"]:
        assessment, reason = "insufficient_reference", "too_few_AQS_PM2.5_samples"
    elif median_nearest > criteria.maximum_median_nearest_km:
        assessment, reason = "insufficient_reference", "AQS_monitors_too_distant"
    elif len(common_months) < criteria.minimum_months:
        assessment, reason = (
            "insufficient_reference",
            "too_few_paired_PurpleAir_periods"
            if requested_months < criteria.minimum_months
            else "too_few_AQS_PM2.5_periods",
        )
    elif min(float(inside["score"]), float(outside["score"])) >= criteria.ambient_like_score:
        assessment, reason = "both_ambient_like", "both_sensors_track_AQS_without_a_decisive_advantage"
    elif max(float(inside["score"]), float(outside["score"])) <= criteria.non_ambient_score:
        assessment, reason = "neither_ambient_like", "neither_sensor_tracks_AQS"
    elif advantage >= criteria.minimum_score_advantage and inside_wins / len(common_months) >= criteria.minimum_consistency:
        assessment, reason = "supports_reversal", "declared_indoor_is_consistently_more_ambient_like"
    elif advantage <= -criteria.minimum_score_advantage and outside_wins / len(common_months) >= criteria.minimum_consistency:
        assessment, reason = "supports_declared_roles", "declared_outdoor_is_consistently_more_ambient_like"
    else:
        assessment, reason = "ambiguous", "AQS_evidence_does_not_separate_sensor_roles"

    expected = str(pair["expected_assessment"])
    if pair["case_type"] != "validation":
        validation = "not_scored"
    elif expected == "not_reversed":
        validation = (
            "pass"
            if assessment
            in {"supports_declared_roles", "both_ambient_like", "neither_ambient_like", "ambiguous"}
            else "fail"
        )
    else:
        validation = "pass" if assessment == expected else "fail"
    return {
        **pair,
        "assessment": assessment,
        "validation_status": validation,
        "reason": reason,
        "requested_hours": requested_hours,
        "failed_reference_hours": requested_hours - len(reference),
        "reference_success_fraction": len(reference) / requested_hours if requested_hours else 0.0,
        "reference_hours": min(int(inside["hours"]), int(outside["hours"])),
        "requested_changes": requested_changes,
        "reference_changes": min(int(inside["changes"]), int(outside["changes"])),
        "requested_qualifying_months": requested_months,
        "qualifying_months": len(common_months),
        "indoor_month_wins": inside_wins,
        "outdoor_month_wins": outside_wins,
        "period_consistency": consistency,
        "indoor_ambient_score": inside["score"],
        "outdoor_ambient_score": outside["score"],
        "score_advantage_indoor": advantage,
        "indoor_level_spearman": inside["level"],
        "outdoor_level_spearman": outside["level"],
        "indoor_change_pearson": inside["change"],
        "outdoor_change_pearson": outside["change"],
        "indoor_quality_warning": inside_quality,
        "outdoor_quality_warning": outside_quality,
        "median_monitor_count": float(np.median(monitor_counts)) if monitor_counts else math.nan,
        "median_nearest_monitor_km": median_nearest,
        "maximum_nearest_monitor_km": max(nearest) if nearest else math.nan,
    }


def load_interpolator(path: Path):
    spec = importlib.util.spec_from_file_location("airguard_aqs_interpolator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load AQS interpolator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.interpolate_requests


def write_csv(path: Path, rows: list[dict[str, object]], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_reference(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = set(REFERENCE_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"reference cache is missing: {', '.join(sorted(missing))}")
        return list(reader)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--test-cases", type=Path, default=DEFAULT_CASES)
    result.add_argument("--all-pairs", action="store_true")
    result.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    result.add_argument("--sensor-metadata", type=Path, default=DEFAULT_METADATA)
    result.add_argument("--indoor-history", type=Path, action="append")
    result.add_argument("--outdoor-history", type=Path, action="append")
    result.add_argument("--indoor-ranges", type=Path, default=DEFAULT_INDOOR_RANGES)
    result.add_argument("--outdoor-ranges", type=Path, default=DEFAULT_OUTDOOR_RANGES)
    result.add_argument("--permanent-indoor", type=Path, default=DEFAULT_PERMANENT_INDOOR)
    result.add_argument("--aqs-dir", type=Path, default=DEFAULT_AQS)
    result.add_argument("--interpolator", type=Path, default=DEFAULT_INTERPOLATOR)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--maximum-month-hours", type=int, default=48)
    result.add_argument("--reuse-reference", action="store_true")
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.maximum_month_hours < 2:
        raise SystemExit("--maximum-month-hours must be at least 2")
    pairs = read_pairs(args.pairs)
    cases = all_cases(pairs) if args.all_pairs else select_cases(read_cases(args.test_cases), pairs)
    locations = read_metadata(args.sensor_metadata)
    indoor_ids = {int(pair["indoor_sensor_id"]) for pair in cases}
    outdoor_ids = {int(pair["outdoor_sensor_id"]) for pair in cases}
    indoor_paths = args.indoor_history or list(DEFAULT_INDOOR)
    outdoor_paths = args.outdoor_history or list(DEFAULT_OUTDOOR)
    indoor = read_histories(indoor_paths, indoor_ids)
    outdoor = read_histories(outdoor_paths, outdoor_ids)
    indoor, removed_indoor = exclude_outdoor_readings(indoor, read_indoor_exclusions(args.indoor_ranges))
    outdoor, removed_outdoor = exclude_outdoor_readings(outdoor, read_outdoor_exclusions(args.outdoor_ranges))
    permanent_indoor = read_sensor_ids(args.permanent_indoor)
    requests = build_requests(
        cases,
        locations,
        indoor,
        outdoor,
        available_aqs_years(args.aqs_dir),
        args.maximum_month_hours,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    request_fields = ("sensor_index", "latitude", "longitude", "time_stamp")
    write_csv(args.output_dir / "sampled_requests.csv", requests, request_fields)
    reference_path = args.output_dir / "aqs_reference.csv"
    if args.reuse_reference:
        reference_rows = read_reference(reference_path)
        expected = {(row["sensor_index"], str(row["time_stamp"])) for row in requests}
        observed = {(row["sensor_index"], str(row["time_stamp"])) for row in reference_rows}
        if expected != observed:
            raise ValueError("cached AQS reference does not match sampled requests")
    else:
        reference_rows = load_interpolator(args.interpolator)(requests, args.aqs_dir)
        write_csv(reference_path, reference_rows, REFERENCE_FIELDS)
    by_pair: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in reference_rows:
        by_pair[str(row["sensor_index"])].append(row)
    criteria = Criteria()
    rows = [
        assess_pair(
            pair,
            indoor[int(pair["indoor_sensor_id"])],
            outdoor[int(pair["outdoor_sensor_id"])],
            by_pair[str(pair["pair_id"])],
            criteria,
            permanent_indoor,
        )
        for pair in cases
    ]
    write_csv(args.output_dir / "pair_assessments.csv", rows, OUTPUT_FIELDS)
    counts = Counter(str(row["assessment"]) for row in rows)
    validation = Counter(str(row["validation_status"]) for row in rows)
    archives = [
        {"path": str(path.resolve()), "bytes": path.stat().st_size}
        for path in sorted(args.aqs_dir.glob("hourly_*.zip"))
        if path.stem.rsplit("_", 1)[-1] in {str(year) for year in available_aqs_years(args.aqs_dir)}
    ]
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": "cached EPA AQS hourly 88101 and 88502 with nearest-10 IDW2 interpolation",
        "scope": "validation pilot only; no sensor correction is written",
        "counts": dict(sorted(counts.items())),
        "validation": dict(sorted(validation.items())),
        "sampled_requests": len(requests),
        "successful_references": sum(row["status"] == "ok" for row in reference_rows),
        "failed_references": sum(row["status"] != "ok" for row in reference_rows),
        "removed_known_bad_indoor_hours": removed_indoor,
        "removed_known_bad_outdoor_hours": removed_outdoor,
        "criteria": asdict(criteria),
        "aqs_archives": archives,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"counts": summary["counts"], "validation": summary["validation"]}, indent=2))


if __name__ == "__main__":
    main()
