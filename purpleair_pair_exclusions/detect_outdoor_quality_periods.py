"""Find outdoor PurpleAir periods resembling reviewed sensor failures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from purpleair_pair_exclusions.detect_pair_exclusions import (
    EVENT_FIELDS,
    Criteria,
    analyze_pairs,
    iso_utc,
    read_histories,
    write_csv,
)
from purpleair_pair_exclusions.outdoor_quality import (
    OutdoorExclusion,
    read_outdoor_exclusions,
)
from purpleair_pair_exclusions.outdoor_sensor_error import (
    ErrorLevelCriteria,
    ErrorLevelPeriod,
    detect_error_level_periods,
)


ROOT = Path(__file__).resolve().parent.parent
KNOWN_EXCLUSIONS = ROOT / "excluded_outdoor_purpleair_ranges.csv"
PERIOD_FIELDS = (
    "indoor_sensor_id",
    "indoor_name",
    "outdoor_sensor_id",
    "outdoor_name",
    "period_start_utc",
    "period_end_utc",
    "analyzed_events",
    "low_response_events",
    "low_response_rate",
    "median_indoor_rise_pm25",
    "median_peak_response_ratio",
    "median_area_response_ratio",
    "error_level_periods",
    "error_level_readings",
    "first_error_level_utc",
    "last_error_level_utc",
    "candidate_signals",
    "known_exclusion_overlap",
    "selected_for_review",
    "selection_reason",
)
PERIOD_EVENT_FIELDS = EVENT_FIELDS + ("selected_for_period_review",)
ERROR_LEVEL_FIELDS = (
    "outdoor_sensor_id",
    "outdoor_name",
    "period_start_utc",
    "period_end_utc",
    "readings",
    "minimum_pm25",
    "maximum_pm25",
)


@dataclass(frozen=True)
class PeriodCriteria:
    minimum_events: int = 3
    minimum_low_response_rate: float = 0.55
    maximum_peak_response_ratio: float = 0.10
    maximum_area_response_ratio: float = 0.10

    def validate(self) -> None:
        if self.minimum_events < 1:
            raise ValueError("minimum_events must be at least one")
        rates = (
            self.minimum_low_response_rate,
            self.maximum_peak_response_ratio,
            self.maximum_area_response_ratio,
        )
        if any(not 0 <= value <= 1 for value in rates):
            raise ValueError("rates and ratios must be between zero and one")


def is_period_low_response(
    event: dict[str, object], criteria: PeriodCriteria
) -> bool:
    return "outdoor_rise_pm25" in event and (
        float(event["peak_response_ratio"])
        <= criteria.maximum_peak_response_ratio
        and float(event["area_response_ratio"])
        <= criteria.maximum_area_response_ratio
    )


def read_pairs(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = set(reader.fieldnames or ())
        selected = {"indoor_sensor_id", "outdoor_sensor_id"} <= fields
        snapshot = {"indoor_sensor_index", "outdoor_sensor_index"} <= fields
        if not selected and not snapshot:
            raise ValueError("pair CSV is missing indoor/outdoor sensor IDs")
        indoor_key = "indoor_sensor_id" if selected else "indoor_sensor_index"
        outdoor_key = "outdoor_sensor_id" if selected else "outdoor_sensor_index"
        rows, seen = [], set()
        for number, row in enumerate(reader, 2):
            try:
                indoor, outdoor = int(row[indoor_key]), int(row[outdoor_key])
                distance = float(row.get("distance_meters", 0) or 0)
                key = (indoor, outdoor)
                if (
                    min(key) < 1
                    or key in seen
                    or distance < 0
                    or not math.isfinite(distance)
                ):
                    raise ValueError
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid or duplicate pair row {number}") from error
            seen.add(key)
            rows.append(
                {
                    "indoor_sensor_id": indoor,
                    "indoor_name": row.get("indoor_name", ""),
                    "outdoor_sensor_id": outdoor,
                    "outdoor_name": row.get("outdoor_name", ""),
                    "distance_meters": distance,
                    "cohort_sources": row.get("cohort_sources", ""),
                }
            )
    if not rows:
        raise ValueError("pair CSV contains no rows")
    return rows


def _intersects(exclusion: OutdoorExclusion, start: int, end: int) -> bool:
    return (exclusion.start is None or exclusion.start < end) and (
        exclusion.end is None or exclusion.end > start
    )


def period_rows(
    events: list[dict[str, object]],
    exclusions: tuple[OutdoorExclusion, ...],
    criteria: PeriodCriteria,
    pairs: list[dict[str, object]] | None = None,
    error_periods: dict[tuple[int, int], tuple[ErrorLevelPeriod, ...]] | None = None,
) -> list[dict[str, object]]:
    criteria.validate()
    pairs, error_periods = pairs or [], error_periods or {}
    pair_by_outdoor = {int(row["outdoor_sensor_id"]): row for row in pairs}
    grouped: dict[tuple[int, int, int], list[dict[str, object]]] = defaultdict(list)
    for event in events:
        if "outdoor_rise_pm25" not in event:
            continue
        year = datetime.fromisoformat(
            str(event["event_start_utc"]).replace("Z", "+00:00")
        ).year
        grouped[
            (int(event["indoor_sensor_id"]), int(event["outdoor_sensor_id"]), year)
        ].append(event)
    for outdoor, year in error_periods:
        if not any(key[1:] == (outdoor, year) for key in grouped):
            pair = pair_by_outdoor[outdoor]
            grouped[(int(pair["indoor_sensor_id"]), outdoor, year)] = []

    rows = []
    for (indoor_id, outdoor_id, year), group in sorted(grouped.items()):
        pair = group[0] if group else pair_by_outdoor[outdoor_id]
        errors = error_periods.get((outdoor_id, year), ())
        start = int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp())
        end = int(datetime(year + 1, 1, 1, tzinfo=timezone.utc).timestamp())
        selected_events = sum(is_period_low_response(row, criteria) for row in group)
        rate = selected_events / len(group) if group else 0.0
        response_selected = (
            selected_events >= criteria.minimum_events
            and rate >= criteria.minimum_low_response_rate
        )
        selected = response_selected or bool(errors)
        known = any(
            item.sensor_id == outdoor_id and _intersects(item, start, end)
            for item in exclusions
        )
        rows.append(
            {
                "indoor_sensor_id": indoor_id,
                "indoor_name": pair["indoor_name"],
                "outdoor_sensor_id": outdoor_id,
                "outdoor_name": pair["outdoor_name"],
                "period_start_utc": iso_utc(start),
                "period_end_utc": iso_utc(end),
                "analyzed_events": len(group),
                "low_response_events": selected_events,
                "low_response_rate": rate,
                "median_indoor_rise_pm25": statistics.median(
                    float(row["indoor_rise_pm25"]) for row in group
                ) if group else "",
                "median_peak_response_ratio": statistics.median(
                    float(row["peak_response_ratio"]) for row in group
                ) if group else "",
                "median_area_response_ratio": statistics.median(
                    float(row["area_response_ratio"]) for row in group
                ) if group else "",
                "error_level_periods": len(errors),
                "error_level_readings": sum(item.readings for item in errors),
                "first_error_level_utc": iso_utc(min(item.start for item in errors))
                if errors else "",
                "last_error_level_utc": iso_utc(max(item.end for item in errors))
                if errors else "",
                "candidate_signals": ";".join(
                    name
                    for present, name in (
                        (response_selected, "repeated_low_response"),
                        (bool(errors), "recurring_error_level"),
                    )
                    if present
                ),
                "known_exclusion_overlap": known,
                "selected_for_review": selected,
                "selection_reason": (
                    "known_period_recovered"
                    if selected and known
                    else "new_period_candidate"
                    if selected
                    else "below_repeated_event_threshold"
                ),
            }
        )
    return rows


def _reviewed_range_count(
    exclusions: tuple[OutdoorExclusion, ...],
    rows: list[dict[str, object]],
    selected_only: bool,
) -> int:
    count = 0
    for exclusion in exclusions:
        matches = [
            row
            for row in rows
            if int(row["outdoor_sensor_id"]) == exclusion.sensor_id
            and _intersects(
                exclusion,
                int(
                    datetime.fromisoformat(
                        str(row["period_start_utc"]).replace("Z", "+00:00")
                    ).timestamp()
                ),
                int(
                    datetime.fromisoformat(
                        str(row["period_end_utc"]).replace("Z", "+00:00")
                    ).timestamp()
                ),
            )
        ]
        if matches and (
            not selected_only or any(row["selected_for_review"] for row in matches)
        ):
            count += 1
    return count


def write_plot(path: Path, rows: list[dict[str, object]], criteria: PeriodCriteria) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 6))
    for known, selected, label, color, marker in (
        (False, False, "Below threshold", "#2878b5", "o"),
        (False, True, "New review candidate", "#c43c39", "x"),
        (True, False, "Known period below threshold", "#d39c22", "s"),
        (True, True, "Recovered known period", "#2b8a3e", "s"),
    ):
        group = [
            row
            for row in rows
            if row["known_exclusion_overlap"] is known
            and row["selected_for_review"] is selected
        ]
        if group:
            axis.scatter(
                [row["low_response_events"] for row in group],
                [row["low_response_rate"] for row in group],
                color=color,
                marker=marker,
                alpha=0.8,
                label=label,
            )
    axis.axvline(criteria.minimum_events, color="#666", linestyle="--", linewidth=1)
    axis.axhline(
        criteria.minimum_low_response_rate,
        color="#666",
        linestyle="--",
        linewidth=1,
    )
    axis.set(
        title="Outdoor PurpleAir failure-period screening",
        xlabel="Low indoor-response events in sensor-year",
        ylabel="Low-response event rate",
        ylim=(-0.02, 1.02),
    )
    axis.grid(alpha=0.2)
    if rows:
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_outputs(
    output: Path,
    events: list[dict[str, object]],
    periods: list[dict[str, object]],
    exclusions: tuple[OutdoorExclusion, ...],
    event_criteria: Criteria,
    period_criteria: PeriodCriteria,
    inputs: dict[str, object],
    complete_pairs: int,
    error_rows: list[dict[str, object]],
    error_criteria: ErrorLevelCriteria,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    candidates = [row for row in periods if row["selected_for_review"]]
    new_candidates = [row for row in candidates if not row["known_exclusion_overlap"]]
    known_periods = [row for row in periods if row["known_exclusion_overlap"]]
    summary = {
        "source": "paired indoor/outdoor PurpleAir pm2.5_atm only",
        "scope": "review-only outdoor failure-period candidates; no exclusions are applied",
        "complete_pairs": complete_pairs,
        "analyzed_events": sum("outdoor_rise_pm25" in row for row in events),
        "strict_low_response_events": sum(
            bool(row["selected_for_exclusion"]) for row in events
        ),
        "period_low_response_events": sum(
            is_period_low_response(row, period_criteria) for row in events
        ),
        "additional_proportional_low_response_events": sum(
            is_period_low_response(row, period_criteria)
            and not bool(row["selected_for_exclusion"])
            for row in events
        ),
        "error_level_periods": len(error_rows),
        "error_level_sensor_years": len(
            {
                (row["outdoor_sensor_id"], str(row["period_start_utc"])[:4])
                for row in error_rows
            }
        ),
        "sensor_years": len(periods),
        "candidate_sensor_years": len(candidates),
        "recovered_known_sensor_years": sum(
            bool(row["selected_for_review"]) for row in known_periods
        ),
        "known_sensor_years_with_event_evidence": len(known_periods),
        "new_candidate_sensor_years": len(new_candidates),
        "reviewed_ranges_with_event_evidence": _reviewed_range_count(
            exclusions, periods, False
        ),
        "reviewed_ranges_recovered": _reviewed_range_count(exclusions, periods, True),
        "reviewed_ranges": len(exclusions),
        "weakest_known_low_response_events": min(
            (int(row["low_response_events"]) for row in known_periods), default=None
        ),
        "weakest_known_low_response_rate": min(
            (float(row["low_response_rate"]) for row in known_periods), default=None
        ),
        "event_criteria": asdict(event_criteria),
        "period_criteria": asdict(period_criteria),
        "error_level_criteria": asdict(error_criteria),
        "inputs": inputs,
    }
    write_csv(
        output / "event_scores.csv",
        PERIOD_EVENT_FIELDS,
        [
            row
            | {"selected_for_period_review": is_period_low_response(row, period_criteria)}
            for row in events
        ],
    )
    write_csv(output / "period_scores.csv", PERIOD_FIELDS, periods)
    write_csv(output / "candidate_periods.csv", PERIOD_FIELDS, candidates)
    write_csv(output / "new_candidate_periods.csv", PERIOD_FIELDS, new_candidates)
    write_csv(output / "error_level_periods.csv", ERROR_LEVEL_FIELDS, error_rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_plot(output / "outdoor_quality_period_summary.png", periods, period_criteria)
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Find repeated low-response periods resembling known outdoor failures."
    )
    result.add_argument(
        "--pairs",
        type=Path,
        default=Path("purpleair_pair_exclusions/results/selected_pairs.csv"),
    )
    result.add_argument("--indoor-history", type=Path, action="append", required=True)
    result.add_argument("--outdoor-history", type=Path, action="append", required=True)
    result.add_argument("--known-exclusions", type=Path, default=KNOWN_EXCLUSIONS)
    result.add_argument(
        "--output-dir",
        type=Path,
        default=Path("purpleair_pair_exclusions/outdoor_quality_results"),
    )
    result.add_argument("--minimum-period-events", type=int, default=3)
    result.add_argument("--minimum-low-response-rate", type=float, default=0.55)
    result.add_argument("--maximum-peak-response-ratio", type=float, default=0.10)
    result.add_argument("--maximum-area-response-ratio", type=float, default=0.10)
    return result


def main() -> None:
    args = parser().parse_args()
    event_criteria = Criteria()
    period_criteria = PeriodCriteria(
        minimum_events=args.minimum_period_events,
        minimum_low_response_rate=args.minimum_low_response_rate,
        maximum_peak_response_ratio=args.maximum_peak_response_ratio,
        maximum_area_response_ratio=args.maximum_area_response_ratio,
    )
    period_criteria.validate()
    pairs = read_pairs(args.pairs)
    indoor = read_histories(
        args.indoor_history, {int(row["indoor_sensor_id"]) for row in pairs}
    )
    outdoor = read_histories(
        args.outdoor_history, {int(row["outdoor_sensor_id"]) for row in pairs}
    )
    events, coverage = analyze_pairs(pairs, indoor, outdoor, event_criteria)
    error_criteria = ErrorLevelCriteria()
    error_periods: dict[tuple[int, int], tuple[ErrorLevelPeriod, ...]] = {}
    error_rows = []
    outdoor_names = {
        int(row["outdoor_sensor_id"]): row["outdoor_name"] for row in pairs
    }
    for sensor_id, readings in outdoor.items():
        annual: dict[int, dict[int, float]] = defaultdict(dict)
        for timestamp, value in readings.items():
            annual[datetime.fromtimestamp(timestamp, timezone.utc).year][timestamp] = value
        for year, values in annual.items():
            found = detect_error_level_periods(values, error_criteria)
            if not found:
                continue
            error_periods[(sensor_id, year)] = found
            error_rows.extend(
                {
                    "outdoor_sensor_id": sensor_id,
                    "outdoor_name": outdoor_names[sensor_id],
                    "period_start_utc": iso_utc(item.start),
                    "period_end_utc": iso_utc(item.end),
                    "readings": item.readings,
                    "minimum_pm25": item.minimum_pm25,
                    "maximum_pm25": item.maximum_pm25,
                }
                for item in found
            )
    exclusions = read_outdoor_exclusions(args.known_exclusions)
    periods = period_rows(events, exclusions, period_criteria, pairs, error_periods)
    inputs = {
        "pairs": str(args.pairs.resolve()),
        "indoor_history": [str(path.resolve()) for path in args.indoor_history],
        "outdoor_history": [str(path.resolve()) for path in args.outdoor_history],
        "known_exclusions": str(args.known_exclusions.resolve()),
    }
    summary = write_outputs(
        args.output_dir,
        events,
        periods,
        exclusions,
        event_criteria,
        period_criteria,
        inputs,
        sum(row["status"] == "complete_pair" for row in coverage),
        error_rows,
        error_criteria,
    )
    omitted = {"event_criteria", "period_criteria", "inputs"}
    print(" ".join(f"{key}={value}" for key, value in summary.items() if key not in omitted))
    print(f"output={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
