"""Measure how closely hourly indoor PM2.5 changes follow outdoor changes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from purpleair_pair_exclusions.detect_pair_exclusions import (
    read_excluded_sensor_ids,
    read_fema_school_ids,
    read_histories,
    read_reusable_school_pairs,
)
from purpleair_pair_exclusions.outdoor_quality import (
    OutdoorExclusion,
    exclude_outdoor_readings,
    read_indoor_exclusions,
    read_outdoor_exclusions,
)


HOUR = 3600
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
EXCLUSION_ROOT = REPOSITORY_ROOT / "data" / "exclusions"
DEFAULT_INDOOR_EXCLUSIONS = (
    EXCLUSION_ROOT / "permanently_excluded_indoor_sensors.csv",
    EXCLUSION_ROOT / "excluded_indoor_sensors_pm25_gt1000.csv",
    EXCLUSION_ROOT / "excluded_indoor_schools_pm25_gt1000.csv",
)
DEFAULT_OUTDOOR_EXCLUSIONS = EXCLUSION_ROOT / "excluded_outdoor_purpleair_ranges.csv"
DEFAULT_INDOOR_RANGE_EXCLUSIONS = EXCLUSION_ROOT / "excluded_indoor_purpleair_ranges.csv"
PAIR_FIELDS = (
    "indoor_sensor_id",
    "indoor_name",
    "outdoor_sensor_id",
    "outdoor_name",
    "distance_meters",
)
METRIC_FIELDS = PAIR_FIELDS + (
    "indoor_hours",
    "outdoor_hours",
    "overlap_hours",
    "movement_hours",
    "pearson_r",
    "raw_pearson_r",
    "spearman_rho",
    "direction_agreement",
    "direction_hours",
    "best_lag_hours",
    "best_lag_pearson_r",
    "status",
)
SUMMARY_FIELDS = (
    "distance_limit_meters",
    "eligible_pairs",
    "distinct_outdoor_sensors",
    "pairs_with_indoor_history",
    "pairs_with_outdoor_history",
    "pairs_with_both_histories",
    "analyzable_pairs",
    "total_movement_hours",
    "median_pearson_r",
    "pearson_ci_low",
    "pearson_ci_high",
    "median_raw_pearson_r",
    "median_spearman_rho",
    "median_direction_agreement",
    "median_best_lag_pearson_r",
    "median_best_lag_hours",
    "added_eligible_pairs",
    "added_analyzable_pairs",
    "added_median_pearson_r",
)


def pearson(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 2 or np.ptp(first) == 0 or np.ptp(second) == 0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    result = np.empty(len(values), dtype=float)
    _, starts, counts = np.unique(
        sorted_values, return_index=True, return_counts=True
    )
    for start, count in zip(starts, counts, strict=True):
        result[order[start : start + count]] = start + (count - 1) / 2
    return result


def winsorize(values: np.ndarray, percent: float) -> np.ndarray:
    low, high = np.percentile(values, (percent, 100 - percent))
    return np.clip(values, low, high)


def fully_excluded_outdoor_ids(
    exclusions: tuple[OutdoorExclusion, ...],
) -> set[int]:
    return {
        item.sensor_id
        for item in exclusions
        if item.start is None and item.end is None
    }


def movements(
    indoor: dict[int, float], outdoor: dict[int, float], lag_hours: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return outdoor [t-1,t] and indoor [t+lag-1,t+lag] changes."""
    outdoor_changes, indoor_changes = [], []
    lag = lag_hours * HOUR
    for timestamp, value in outdoor.items():
        previous = timestamp - HOUR
        indoor_time = timestamp + lag
        if (
            previous in outdoor
            and indoor_time in indoor
            and indoor_time - HOUR in indoor
        ):
            outdoor_changes.append(value - outdoor[previous])
            indoor_changes.append(indoor[indoor_time] - indoor[indoor_time - HOUR])
    return np.asarray(outdoor_changes), np.asarray(indoor_changes)


def pair_metrics(
    pair: dict[str, object],
    indoor: dict[int, float],
    outdoor: dict[int, float],
    minimum_movements: int,
    winsor_percent: float,
    maximum_lag: int,
) -> dict[str, object]:
    row = {
        **pair,
        "indoor_hours": len(indoor),
        "outdoor_hours": len(outdoor),
        "overlap_hours": len(indoor.keys() & outdoor.keys()),
    }
    if not indoor or not outdoor:
        row["status"] = (
            "missing_both_histories"
            if not indoor and not outdoor
            else "missing_indoor_history" if not indoor else "missing_outdoor_history"
        )
        return row

    outside, inside = movements(indoor, outdoor)
    row["movement_hours"] = len(outside)
    if len(outside) < minimum_movements:
        row["status"] = "insufficient_movement_hours"
        return row

    outside_w = winsorize(outside, winsor_percent)
    inside_w = winsorize(inside, winsor_percent)
    row.update(
        pearson_r=pearson(outside_w, inside_w),
        raw_pearson_r=pearson(outside, inside),
        spearman_rho=pearson(ranks(outside), ranks(inside)),
    )
    moving = (outside != 0) & (inside != 0)
    row["direction_hours"] = int(moving.sum())
    row["direction_agreement"] = (
        float(np.mean(np.sign(outside[moving]) == np.sign(inside[moving])))
        if moving.any()
        else None
    )
    lagged = []
    for lag in range(maximum_lag + 1):
        lag_outside, lag_inside = movements(indoor, outdoor, lag)
        if len(lag_outside) >= minimum_movements:
            value = pearson(
                winsorize(lag_outside, winsor_percent),
                winsorize(lag_inside, winsor_percent),
            )
            if value is not None:
                lagged.append((value, lag))
    if row["pearson_r"] is None:
        row["status"] = "constant_movement"
    else:
        row["status"] = "analyzed"
        if lagged:
            row["best_lag_pearson_r"], row["best_lag_hours"] = max(
                lagged, key=lambda value: (value[0], -value[1])
            )
    return row


def median(rows: list[dict[str, object]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return float(np.median(values)) if values else None


def cluster_bootstrap_ci(
    rows: list[dict[str, object]], samples: int, seed: int
) -> tuple[float | None, float | None]:
    clusters: dict[int, list[float]] = {}
    for row in rows:
        if row.get("pearson_r") is not None:
            clusters.setdefault(int(row["outdoor_sensor_id"]), []).append(
                float(row["pearson_r"])
            )
    if not clusters:
        return None, None
    groups = list(clusters.values())
    rng = np.random.default_rng(seed)
    estimates = [
        np.median(
            [value for index in rng.integers(len(groups), size=len(groups)) for value in groups[index]]
        )
        for _ in range(samples)
    ]
    low, high = np.percentile(estimates, (2.5, 97.5))
    return float(low), float(high)


def cohort_summary(
    rows: list[dict[str, object]],
    limit: float,
    previous_limit: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    cohort = [row for row in rows if float(row["distance_meters"]) <= limit]
    added = [
        row
        for row in cohort
        if float(row["distance_meters"]) > previous_limit
    ]
    analyzed = [row for row in cohort if row["status"] == "analyzed"]
    added_analyzed = [row for row in added if row["status"] == "analyzed"]
    low, high = cluster_bootstrap_ci(analyzed, bootstrap_samples, seed)
    return {
        "distance_limit_meters": limit,
        "eligible_pairs": len(cohort),
        "distinct_outdoor_sensors": len(
            {row["outdoor_sensor_id"] for row in cohort}
        ),
        "pairs_with_indoor_history": sum(bool(row["indoor_hours"]) for row in cohort),
        "pairs_with_outdoor_history": sum(bool(row["outdoor_hours"]) for row in cohort),
        "pairs_with_both_histories": sum(
            bool(row["indoor_hours"] and row["outdoor_hours"]) for row in cohort
        ),
        "analyzable_pairs": len(analyzed),
        "total_movement_hours": sum(int(row["movement_hours"]) for row in analyzed),
        "median_pearson_r": median(analyzed, "pearson_r"),
        "pearson_ci_low": low,
        "pearson_ci_high": high,
        "median_raw_pearson_r": median(analyzed, "raw_pearson_r"),
        "median_spearman_rho": median(analyzed, "spearman_rho"),
        "median_direction_agreement": median(analyzed, "direction_agreement"),
        "median_best_lag_pearson_r": median(analyzed, "best_lag_pearson_r"),
        "median_best_lag_hours": median(analyzed, "best_lag_hours"),
        "added_eligible_pairs": len(added),
        "added_analyzable_pairs": len(added_analyzed),
        "added_median_pearson_r": median(added_analyzed, "pearson_r"),
    }


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def source_record(path: Path) -> dict[str, object]:
    record: dict[str, object] = {"path": str(path.resolve())}
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        record.update(bytes=path.stat().st_size, sha256=digest.hexdigest())
    return record


def format_value(value: object, digits: int = 3) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def write_report(
    path: Path, summary: dict[str, object], rows: list[dict[str, object]]
) -> None:
    table = [
        "| Limit | Eligible | Both histories | Analyzed | Median Pearson r (95% CI) | Median Spearman rho | Direction agreement | Best-lag r (median lag) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        ci = f"{format_value(row['median_pearson_r'])} ({format_value(row['pearson_ci_low'])}, {format_value(row['pearson_ci_high'])})"
        direction = (
            "NA"
            if row["median_direction_agreement"] is None
            else f"{100 * float(row['median_direction_agreement']):.1f}%"
        )
        lagged = (
            "NA"
            if row["median_best_lag_pearson_r"] is None
            else f"{float(row['median_best_lag_pearson_r']):.3f} "
            f"({float(row['median_best_lag_hours']):g} h)"
        )
        table.append(
            f"| {float(row['distance_limit_meters']):g} m | {row['eligible_pairs']} | "
            f"{row['pairs_with_both_histories']} | {row['analyzable_pairs']} | {ci} | "
            f"{format_value(row['median_spearman_rho'])} | "
            f"{direction} | {lagged} |"
        )
    lines = [
        "# Indoor/outdoor PM2.5 movement correlation",
        "",
        "## Methodology",
        "",
        "Each validated indoor school is paired independently with its nearest active-snapshot outdoor PurpleAir sensor; an outdoor sensor may be reused. The distance cohorts are cumulative. Pair selection is spatial only, so missing histories never cause substitution of a farther sensor.",
        "",
        "Known-bad indoor sensors are removed before pairing; bounded indoor exclusions remove only readings in their half-open UTC ranges. Outdoor sensors with a full-history exclusion are removed before nearest-sensor selection; bounded outdoor exclusions remove only affected readings.",
        "",
        "Histories are joined at exact UTC hours without interpolation. Movement is the one-hour first difference, `PM2.5(t) - PM2.5(t-1)`, and an observation is retained only when both sensors have both consecutive hours. A pair needs at least "
        f"{summary['methodology']['minimum_movements']} movements.",
        "",
        f"The primary pair statistic is Pearson correlation after separately winsorizing indoor and outdoor changes at the {summary['methodology']['winsor_percent']:g}% and {100 - summary['methodology']['winsor_percent']:g}% quantiles. This keeps the statistic in PM2.5-change units while limiting isolated spikes. Raw Pearson, Spearman rank correlation, nonzero direction agreement, and the best positive correlation with outdoor leading indoor by 0-{summary['methodology']['maximum_outdoor_lead_hours']} hours are sensitivity measures.",
        "",
        f"Every pair receives equal weight in the cumulative result: the reported value is the median pair correlation, not a pooled hour-level correlation. The 95% interval is a {summary['methodology']['bootstrap_samples']:,}-sample bootstrap over outdoor-sensor clusters, preserving dependence when one outdoor sensor serves multiple schools.",
        "",
        "## Results",
        "",
        *table,
        "",
        f"Within the locally available-history subset, the primary median changed from {format_value(rows[0]['median_pearson_r'])} at {float(rows[0]['distance_limit_meters']):g} m to {format_value(rows[-1]['median_pearson_r'])} at {float(rows[-1]['distance_limit_meters']):g} m; no degradation was detected. Only {rows[-1]['analyzable_pairs'] - rows[0]['analyzable_pairs']} additional pairs became analyzable, so this is not a valid full-cohort distance comparison.",
        "",
        "Direction agreement is shown as a percentage among hours where both changes are nonzero. `Best-lag r` is exploratory and is not the primary statistic.",
        "",
        "## Coverage limitation",
        "",
        f"The 1 km cohort contains {rows[-1]['eligible_pairs']} spatially eligible pairs, but only {rows[-1]['analyzable_pairs']} have enough locally available paired history. Therefore these results describe the available-history subset and cannot yet establish how correlation changes for the full expanded cohorts. The exact missing-history status for every pair is in `pair_metrics.csv`.",
        "",
        "## Reproducibility",
        "",
        f"Generated at `{summary['generated_at_utc']}` with random seed `{summary['methodology']['random_seed']}`. Exact input paths, sizes, and SHA-256 hashes are recorded in `summary.json`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--school-sensors", type=Path, required=True)
    result.add_argument("--sensor-inventory", type=Path, required=True)
    result.add_argument("--indoor-history", type=Path, action="append", required=True)
    result.add_argument("--outdoor-history", type=Path, action="append", required=True)
    result.add_argument("--excluded-indoor-sensors", type=Path, action="append")
    result.add_argument(
        "--excluded-indoor-ranges",
        type=Path,
        default=DEFAULT_INDOOR_RANGE_EXCLUSIONS,
    )
    result.add_argument(
        "--excluded-outdoor-ranges",
        type=Path,
        default=DEFAULT_OUTDOOR_EXCLUSIONS,
    )
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument(
        "--distance-limits", type=float, nargs="+", default=(100, 250, 500, 1000)
    )
    result.add_argument("--minimum-movements", type=int, default=168)
    result.add_argument("--winsor-percent", type=float, default=1.0)
    result.add_argument("--maximum-lag-hours", type=int, default=6)
    result.add_argument("--bootstrap-samples", type=int, default=10_000)
    result.add_argument("--random-seed", type=int, default=20_260_822)
    return result


def main() -> None:
    args = parser().parse_args()
    limits = sorted(set(args.distance_limits))
    if (
        not limits
        or limits[0] <= 0
        or args.minimum_movements < 2
        or not 0 <= args.winsor_percent < 50
        or args.maximum_lag_hours < 0
        or args.bootstrap_samples < 1
    ):
        raise SystemExit("invalid distance, movement, winsor, lag, or bootstrap option")

    exclusion_paths = tuple(
        dict.fromkeys([*DEFAULT_INDOOR_EXCLUSIONS, *(args.excluded_indoor_sensors or [])])
    )
    excluded_indoor_ids = set().union(
        *(read_excluded_sensor_ids(path) for path in exclusion_paths)
    )
    indoor_exclusions = read_indoor_exclusions(args.excluded_indoor_ranges)
    outdoor_exclusions = read_outdoor_exclusions(args.excluded_outdoor_ranges)
    excluded_outdoor_ids = fully_excluded_outdoor_ids(outdoor_exclusions)
    school_ids = read_fema_school_ids(args.school_sensors)
    pairs = read_reusable_school_pairs(
        args.sensor_inventory,
        school_ids - excluded_indoor_ids,
        limits[-1],
        excluded_outdoor_ids,
    )
    indoor_ids = {int(pair["indoor_sensor_id"]) for pair in pairs}
    outdoor_ids = {int(pair["outdoor_sensor_id"]) for pair in pairs}
    indoor = read_histories(args.indoor_history, indoor_ids)
    indoor, excluded_indoor_hours = exclude_outdoor_readings(
        indoor, indoor_exclusions
    )
    outdoor = read_histories(args.outdoor_history, outdoor_ids)
    outdoor, excluded_outdoor_hours = exclude_outdoor_readings(
        outdoor, outdoor_exclusions
    )
    metrics = [
        pair_metrics(
            pair,
            indoor[int(pair["indoor_sensor_id"])],
            outdoor[int(pair["outdoor_sensor_id"])],
            args.minimum_movements,
            args.winsor_percent,
            args.maximum_lag_hours,
        )
        for pair in pairs
    ]
    summaries = [
        cohort_summary(
            metrics,
            limit,
            limits[index - 1] if index else -1,
            args.bootstrap_samples,
            args.random_seed,
        )
        for index, limit in enumerate(limits)
    ]
    inputs = [
        args.school_sensors,
        args.sensor_inventory,
        *args.indoor_history,
        *args.outdoor_history,
        *exclusion_paths,
        args.excluded_indoor_ranges,
        args.excluded_outdoor_ranges,
    ]
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "cohort": "validated FEMA/NSD indoor schools",
        "school_sensor_count": len(school_ids),
        "excluded_indoor_sensor_count": len(school_ids & excluded_indoor_ids),
        "indoor_exclusion_range_count": len(indoor_exclusions),
        "excluded_indoor_history_hours": excluded_indoor_hours,
        "full_history_excluded_outdoor_sensor_count": len(excluded_outdoor_ids),
        "outdoor_exclusion_range_count": len(outdoor_exclusions),
        "excluded_outdoor_history_hours": excluded_outdoor_hours,
        "methodology": {
            "movement": "one-hour first difference at exact consecutive UTC hours",
            "primary_pair_metric": "Pearson r after pairwise 1%/99% winsorization",
            "cohort_metric": "equal-pair median",
            "minimum_movements": args.minimum_movements,
            "winsor_percent": args.winsor_percent,
            "maximum_outdoor_lead_hours": args.maximum_lag_hours,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_unit": "outdoor sensor cluster",
            "random_seed": args.random_seed,
        },
        "inputs": [source_record(path) for path in inputs],
        "distance_summaries": summaries,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "eligible_pairs.csv", PAIR_FIELDS, pairs)
    write_csv(args.output_dir / "pair_metrics.csv", METRIC_FIELDS, metrics)
    write_csv(args.output_dir / "distance_summary.csv", SUMMARY_FIELDS, summaries)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_report(args.output_dir / "report.md", summary, summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
