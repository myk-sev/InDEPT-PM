"""Detect weak indoor responses to elevated paired PurpleAir outdoor events."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from bisect import bisect_left
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from purpleair_pair_exclusions.location_history_explorer import (
    write_location_history_explorer,
)
from purpleair_pair_exclusions.outdoor_quality import (
    exclude_outdoor_readings,
    read_indoor_exclusions,
    read_outdoor_exclusions,
)
from purpleair_pair_exclusions.training_intervals import (
    build_training_intervals,
    read_ranked_candidates,
    source_record,
    write_training_contract,
)


HOUR = 3600
EARTH_RADIUS_METERS = 6_371_008.8
PAIR_COLUMNS = (
    "indoor_sensor_index",
    "indoor_name",
    "outdoor_sensor_index",
    "outdoor_name",
    "distance_meters",
)
HISTORY_COLUMNS = ("time_stamp", "sensor_index", "pm2.5_atm")
OVERLAP_COLUMN = "sensor_index"
SCHOOL_COLUMNS = ("sensor_index", "location_type", "k12_status", "is_k12")
SENSOR_COLUMNS = ("sensor_index", "name", "location_type", "latitude", "longitude")
EVENT_FIELDS = (
    "event_id",
    "indoor_sensor_id",
    "indoor_name",
    "outdoor_sensor_id",
    "outdoor_name",
    "distance_meters",
    "cohort_sources",
    "event_start_utc",
    "event_end_utc",
    "response_end_utc",
    "trigger_hours",
    "baseline_overlap_hours",
    "response_overlap_hours",
    "response_coverage",
    "outdoor_baseline_pm25",
    "indoor_baseline_pm25",
    "outdoor_peak_pm25",
    "indoor_peak_pm25",
    "outdoor_rise_pm25",
    "indoor_rise_pm25",
    "peak_response_ratio",
    "area_response_ratio",
    "peak_lag_hours",
    "selected_for_exclusion",
    "selection_reason",
)
COVERAGE_FIELDS = (
    "indoor_sensor_id",
    "indoor_name",
    "outdoor_sensor_id",
    "outdoor_name",
    "distance_meters",
    "cohort_sources",
    "indoor_hours",
    "outdoor_hours",
    "overlap_hours",
    "status",
)
SELECTED_PAIR_FIELDS = COVERAGE_FIELDS[:6]
COHORT_SELECTION_FIELDS = (
    "indoor_sensor_id",
    "cohort_sources",
    "outdoor_sensor_id",
    "selection_status",
)
SENSOR_FIELDS = (
    "sensor_id",
    "outdoor_sensor_id",
    "analyzed_events",
    "excluded_events",
    "exclusion_rate",
    "selected_for_exclusion",
    "criterion",
)
ROOT = Path(__file__).resolve().parent.parent
INPUT_ROOT = ROOT / "inputs"
DATA_ROOT = ROOT / "data"
HISTORY_ROOT = DATA_ROOT / "purple air"
EXCLUSION_ROOT = DATA_ROOT / "exclusions"
DEFAULT_TRAINING_OUTPUT = INPUT_ROOT / "masked_pretraining" / "exclusion_aware"
PERMANENT_EXCLUSIONS_PATH = EXCLUSION_ROOT / "permanently_excluded_indoor_sensors.csv"
INDOOR_EXCLUSION_PATHS = (
    PERMANENT_EXCLUSIONS_PATH,
    EXCLUSION_ROOT / "excluded_indoor_sensors_pm25_gt1000.csv",
    EXCLUSION_ROOT / "excluded_indoor_schools_pm25_gt1000.csv",
)
INDOOR_RANGE_EXCLUSIONS_PATH = EXCLUSION_ROOT / "excluded_indoor_purpleair_ranges.csv"
OUTDOOR_EXCLUSIONS_PATH = EXCLUSION_ROOT / "excluded_outdoor_purpleair_ranges.csv"


@dataclass(frozen=True)
class Criteria:
    baseline_hours: int = 24
    minimum_baseline_hours: int = 12
    minimum_outdoor_pm25: float = 55.5
    minimum_outdoor_rise: float = 25.0
    minimum_event_hours: int = 3
    merge_gap_hours: int = 2
    response_hours: int = 24
    minimum_response_coverage: float = 0.80
    maximum_indoor_rise: float = 5.0
    maximum_peak_response_ratio: float = 0.10
    maximum_area_response_ratio: float = 0.10
    minimum_sensor_events: int = 2
    minimum_sensor_exclusion_rate: float = 0.75

    def validate(self) -> None:
        positive = (
            self.baseline_hours,
            self.minimum_baseline_hours,
            self.minimum_outdoor_pm25,
            self.minimum_outdoor_rise,
            self.minimum_event_hours,
            self.response_hours,
            self.maximum_indoor_rise,
            self.minimum_sensor_events,
        )
        if any(value <= 0 or not math.isfinite(value) for value in positive):
            raise ValueError("positive criteria must be finite and greater than zero")
        if self.minimum_baseline_hours > self.baseline_hours:
            raise ValueError("minimum_baseline_hours cannot exceed baseline_hours")
        if self.merge_gap_hours < 0:
            raise ValueError("merge_gap_hours cannot be negative")
        rates = (
            self.minimum_response_coverage,
            self.maximum_peak_response_ratio,
            self.maximum_area_response_ratio,
            self.minimum_sensor_exclusion_rate,
        )
        if any(not 0 <= value <= 1 for value in rates):
            raise ValueError("coverage and ratio criteria must be between zero and one")


def iso_utc(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def read_pairs(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = set(PAIR_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing pair columns: {', '.join(sorted(missing))}")
        rows = []
        indoor_ids, outdoor_ids = set(), set()
        for number, row in enumerate(reader, 2):
            try:
                indoor = int(row["indoor_sensor_index"])
                outdoor = int(row["outdoor_sensor_index"])
                distance = float(row["distance_meters"])
                if (
                    indoor < 1
                    or outdoor < 1
                    or indoor in indoor_ids
                    or outdoor in outdoor_ids
                    or distance < 0
                    or not math.isfinite(distance)
                ):
                    raise ValueError
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid or duplicate pair row {number}") from error
            indoor_ids.add(indoor)
            outdoor_ids.add(outdoor)
            rows.append(
                {
                    "indoor_sensor_id": indoor,
                    "indoor_name": row["indoor_name"],
                    "outdoor_sensor_id": outdoor,
                    "outdoor_name": row["outdoor_name"],
                    "distance_meters": distance,
                }
            )
    if not rows:
        raise ValueError("pair CSV contains no rows")
    return rows


def read_overlap_indoor_ids(path: Path) -> set[int]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if OVERLAP_COLUMN not in (reader.fieldnames or ()):
            raise ValueError(f"missing school smoke-overlap column: {OVERLAP_COLUMN}")
        try:
            sensor_ids = {int(row[OVERLAP_COLUMN]) for row in reader}
        except (TypeError, ValueError) as error:
            raise ValueError("invalid school smoke-overlap sensor ID") from error
    if not sensor_ids or min(sensor_ids) < 1:
        raise ValueError("school smoke-overlap CSV contains no valid sensor IDs")
    return sensor_ids


def read_fema_school_ids(path: Path) -> set[int]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = set(SCHOOL_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing FEMA school columns: {', '.join(sorted(missing))}")
        sensor_ids = set()
        for number, row in enumerate(reader, 2):
            if not (
                row["location_type"].strip().lower() == "inside"
                and row["k12_status"].strip().lower() == "school"
                and row["is_k12"].strip().lower() == "true"
            ):
                continue
            try:
                sensor = int(row["sensor_index"])
                if sensor < 1 or sensor in sensor_ids:
                    raise ValueError
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid or duplicate FEMA school row {number}") from error
            sensor_ids.add(sensor)
    if not sensor_ids:
        raise ValueError("FEMA school CSV contains no validated indoor schools")
    return sensor_ids


def read_excluded_sensor_ids(path: Path) -> set[int]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if "sensor_id" not in (reader.fieldnames or ()):
            raise ValueError("permanent exclusion CSV is missing sensor_id")
        try:
            sensor_ids = {int(row["sensor_id"]) for row in reader}
        except (TypeError, ValueError) as error:
            raise ValueError("invalid permanent exclusion sensor ID") from error
    if not sensor_ids or min(sensor_ids) < 1:
        raise ValueError("permanent exclusion CSV contains no valid sensor IDs")
    return sensor_ids


def read_permanent_exclusions(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"sensor_id", "sensor_name", "reason"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"missing permanent exclusion columns: {', '.join(sorted(missing))}"
            )
        rows, seen = [], set()
        for number, row in enumerate(reader, 2):
            try:
                sensor_id = int(row["sensor_id"])
                name, reason = row["sensor_name"].strip(), row["reason"].strip()
                if sensor_id < 1 or sensor_id in seen or not name or not reason:
                    raise ValueError
            except (AttributeError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid permanent exclusion row {number}"
                ) from error
            seen.add(sensor_id)
            rows.append(
                {"sensor_id": sensor_id, "sensor_name": name, "reason": reason}
            )
    if not rows:
        raise ValueError("permanent exclusion CSV contains no rows")
    return rows


def read_sensor_names(path: Path) -> dict[int, str]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not {"sensor_index", "name"} <= set(reader.fieldnames or ()):
            raise ValueError("sensor inventory is missing sensor_index or name")
        return {int(row["sensor_index"]): row["name"] for row in reader}


def distance_meters(first: dict[str, object], second: dict[str, object]) -> float:
    latitude_1, latitude_2 = map(
        math.radians, (float(first["latitude"]), float(second["latitude"]))
    )
    latitude_delta = latitude_2 - latitude_1
    longitude_delta = math.radians(
        (float(second["longitude"]) - float(first["longitude"]) + 180) % 360 - 180
    )
    value = (
        math.sin(latitude_delta / 2) ** 2
        + math.cos(latitude_1)
        * math.cos(latitude_2)
        * math.sin(longitude_delta / 2) ** 2
    )
    return 2 * EARTH_RADIUS_METERS * math.asin(math.sqrt(value))


def read_reusable_school_pairs(
    path: Path,
    school_ids: set[int],
    maximum_distance: float,
    excluded_outdoor_ids: set[int] | None = None,
) -> list[dict[str, object]]:
    excluded_outdoor_ids = excluded_outdoor_ids or set()
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = set(SENSOR_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing sensor inventory columns: {', '.join(sorted(missing))}")
        schools, outdoors = {}, []
        for number, row in enumerate(reader, 2):
            try:
                sensor = {
                    "sensor_index": int(row["sensor_index"]),
                    "name": row["name"],
                    "location_type": row["location_type"].strip().lower(),
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                }
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid sensor inventory row {number}") from error
            if (
                sensor["location_type"] == "outside"
                and sensor["sensor_index"] not in excluded_outdoor_ids
            ):
                outdoors.append(sensor)
            elif sensor["sensor_index"] in school_ids:
                schools[int(sensor["sensor_index"])] = sensor
    if not outdoors:
        raise ValueError("sensor inventory contains no outdoor sensors")

    pairs = []
    for sensor_id in sorted(school_ids & schools.keys()):
        indoor = schools[sensor_id]
        distance, outdoor = min(
            ((distance_meters(indoor, item), item) for item in outdoors),
            key=lambda value: (value[0], value[1]["sensor_index"]),
        )
        if distance <= maximum_distance:
            pairs.append(
                {
                    "indoor_sensor_id": sensor_id,
                    "indoor_name": indoor["name"],
                    "outdoor_sensor_id": outdoor["sensor_index"],
                    "outdoor_name": outdoor["name"],
                    "distance_meters": round(distance, 2),
                }
            )
    return pairs


def select_pairs(
    pairs: list[dict[str, object]],
    cohorts: dict[str, set[int]],
    excluded_ids: set[int] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    excluded_ids = excluded_ids or set()
    pair_by_indoor = {int(row["indoor_sensor_id"]): row for row in pairs}
    all_ids = set().union(*cohorts.values())
    selected, audit = [], []
    for indoor in sorted(all_ids):
        sources = ";".join(name for name, ids in cohorts.items() if indoor in ids)
        pair = pair_by_indoor.get(indoor)
        excluded = indoor in excluded_ids
        audit.append(
            {
                "indoor_sensor_id": indoor,
                "cohort_sources": sources,
                "outdoor_sensor_id": pair["outdoor_sensor_id"] if pair else "",
                "selection_status": (
                    "permanently_excluded_indoor_sensor"
                    if excluded
                    else "selected_outdoor_purpleair_pair"
                    if pair
                    else "no_outdoor_purpleair_pair"
                ),
            }
        )
        if pair and not excluded:
            selected.append(pair | {"cohort_sources": sources})
    if not selected:
        raise ValueError("selected cohorts contain no outdoor PurpleAir sensor pairs")
    return selected, audit


def history_files(
    paths: list[Path], requested_sensors: set[int] | None
) -> list[Path]:
    files = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            for file in path.rglob("*.csv"):
                prefix = file.stem.split("_", 1)[0]
                if prefix.isdigit() and (
                    requested_sensors is None or int(prefix) in requested_sensors
                ):
                    files.append(file)
        else:
            raise FileNotFoundError(f"PurpleAir history path not found: {path}")
    return sorted(set(files))


def read_histories(
    paths: list[Path], requested_sensors: set[int] | None = None
) -> dict[int, dict[int, float]]:
    values: dict[int, dict[int, float]] = (
        {} if requested_sensors is None else {sensor: {} for sensor in requested_sensors}
    )
    for path in history_files(paths, requested_sensors):
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            missing = set(HISTORY_COLUMNS) - set(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"missing PurpleAir columns in {path}: {', '.join(sorted(missing))}"
                )
            for number, row in enumerate(reader, 2):
                try:
                    sensor = int(row["sensor_index"])
                    if requested_sensors is not None and sensor not in requested_sensors:
                        continue
                    timestamp = int(row["time_stamp"])
                    text = (row["pm2.5_atm"] or "").strip().lower()
                    if text in {"", "null", "nan"}:
                        continue
                    value = float(text)
                    if timestamp % HOUR or value < 0 or not math.isfinite(value):
                        raise ValueError
                except (TypeError, ValueError) as error:
                    raise ValueError(f"invalid PurpleAir row {number} in {path}") from error
                sensor_values = values.setdefault(sensor, {})
                previous = sensor_values.get(timestamp)
                if previous is not None and not math.isclose(
                    previous, value, abs_tol=1e-6
                ):
                    raise ValueError(f"conflicting duplicate sensor-hour in {path}")
                sensor_values[timestamp] = value
    return values


def trigger_groups(outdoor: dict[int, float], criteria: Criteria) -> list[list[int]]:
    times = sorted(outdoor)
    triggers = []
    for timestamp in times:
        start = bisect_left(times, timestamp - criteria.baseline_hours * HOUR)
        end = bisect_left(times, timestamp)
        baseline = [outdoor[value] for value in times[start:end]]
        if len(baseline) < criteria.minimum_baseline_hours:
            continue
        median = statistics.median(baseline)
        if (
            outdoor[timestamp] >= criteria.minimum_outdoor_pm25
            and outdoor[timestamp] - median >= criteria.minimum_outdoor_rise
        ):
            triggers.append(timestamp)

    groups: list[list[int]] = []
    for timestamp in triggers:
        if not groups or timestamp - groups[-1][-1] > (
            criteria.merge_gap_hours + 1
        ) * HOUR:
            groups.append([])
        groups[-1].append(timestamp)
    return [group for group in groups if len(group) >= criteria.minimum_event_hours]


def _series_rows(
    indoor: dict[int, float],
    outdoor: dict[int, float],
    start: int,
    end: int,
) -> list[dict[str, object]]:
    return [
        {
            "timestamp_utc": iso_utc(timestamp),
            "indoor_pm25": indoor.get(timestamp),
            "outdoor_pm25": outdoor.get(timestamp),
        }
        for timestamp in range(start, end + HOUR, HOUR)
        if timestamp in indoor or timestamp in outdoor
    ]


def analyze_event(
    pair: dict[str, object],
    indoor: dict[int, float],
    outdoor: dict[int, float],
    triggers: list[int],
    criteria: Criteria,
) -> dict[str, object]:
    start, end = triggers[0], triggers[-1]
    response_end = end + criteria.response_hours * HOUR
    baseline_start = start - criteria.baseline_hours * HOUR
    common = set(indoor) & set(outdoor)
    baseline_times = sorted(time for time in common if baseline_start <= time < start)
    response_times = sorted(time for time in common if start <= time <= response_end)
    expected_response_hours = (response_end - start) // HOUR + 1
    coverage = len(response_times) / expected_response_hours
    event_id = (
        f"pair_{pair['indoor_sensor_id']}_{pair['outdoor_sensor_id']}_"
        f"{datetime.fromtimestamp(start, timezone.utc):%Y%m%dT%H%M%SZ}"
    )
    row: dict[str, object] = {
        "event_id": event_id,
        **pair,
        "event_start_utc": iso_utc(start),
        "event_end_utc": iso_utc(end),
        "response_end_utc": iso_utc(response_end),
        "trigger_hours": len(triggers),
        "baseline_overlap_hours": len(baseline_times),
        "response_overlap_hours": len(response_times),
        "response_coverage": coverage,
        "selected_for_exclusion": False,
        "selection_reason": "",
        "_series": _series_rows(indoor, outdoor, baseline_start, response_end),
    }
    if len(baseline_times) < criteria.minimum_baseline_hours:
        row["selection_reason"] = "insufficient_paired_baseline"
        return row
    if coverage < criteria.minimum_response_coverage:
        row["selection_reason"] = "insufficient_paired_response_coverage"
        return row

    outdoor_baseline = statistics.median(outdoor[time] for time in baseline_times)
    indoor_baseline = statistics.median(indoor[time] for time in baseline_times)
    event_times = [time for time in outdoor if start <= time <= end]
    outdoor_peak_time = max(event_times, key=outdoor.__getitem__)
    indoor_peak_time = max(response_times, key=indoor.__getitem__)
    outdoor_peak = outdoor[outdoor_peak_time]
    indoor_peak = indoor[indoor_peak_time]
    outdoor_rise = max(0.0, outdoor_peak - outdoor_baseline)
    indoor_rise = max(0.0, indoor_peak - indoor_baseline)
    peak_ratio = indoor_rise / outdoor_rise if outdoor_rise else math.inf
    outdoor_area = sum(max(0.0, outdoor[time] - outdoor_baseline) for time in response_times)
    indoor_area = sum(max(0.0, indoor[time] - indoor_baseline) for time in response_times)
    area_ratio = indoor_area / outdoor_area if outdoor_area else math.inf
    selected = (
        indoor_rise <= criteria.maximum_indoor_rise
        and peak_ratio <= criteria.maximum_peak_response_ratio
        and area_ratio <= criteria.maximum_area_response_ratio
    )
    row.update(
        {
            "outdoor_baseline_pm25": outdoor_baseline,
            "indoor_baseline_pm25": indoor_baseline,
            "outdoor_peak_pm25": outdoor_peak,
            "indoor_peak_pm25": indoor_peak,
            "outdoor_rise_pm25": outdoor_rise,
            "indoor_rise_pm25": indoor_rise,
            "peak_response_ratio": peak_ratio,
            "area_response_ratio": area_ratio,
            "peak_lag_hours": (indoor_peak_time - outdoor_peak_time) / HOUR,
            "selected_for_exclusion": selected,
            "selection_reason": (
                "low_indoor_peak_and_exposure_response"
                if selected
                else "measurable_indoor_response"
            ),
        }
    )
    return row


def analyze_pairs(
    pairs: list[dict[str, object]],
    indoor_histories: dict[int, dict[int, float]],
    outdoor_histories: dict[int, dict[int, float]],
    criteria: Criteria,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    criteria.validate()
    events, coverage = [], []
    for pair in pairs:
        indoor = indoor_histories[int(pair["indoor_sensor_id"])]
        outdoor = outdoor_histories[int(pair["outdoor_sensor_id"])]
        overlap = len(set(indoor) & set(outdoor))
        status = (
            "complete_pair"
            if indoor and outdoor
            else "missing_both"
            if not indoor and not outdoor
            else "missing_indoor"
            if not indoor
            else "missing_outdoor"
        )
        coverage.append(
            {
                **pair,
                "indoor_hours": len(indoor),
                "outdoor_hours": len(outdoor),
                "overlap_hours": overlap,
                "status": status,
            }
        )
        if not indoor or not outdoor:
            continue
        events.extend(
            analyze_event(pair, indoor, outdoor, group, criteria)
            for group in trigger_groups(outdoor, criteria)
        )
    return events, coverage


def sensor_rows(
    events: list[dict[str, object]], criteria: Criteria
) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for event in events:
        if event["selection_reason"].startswith("insufficient_"):
            continue
        grouped[
            (int(event["indoor_sensor_id"]), int(event["outdoor_sensor_id"]))
        ].append(event)
    rows = []
    for (indoor, outdoor), group in sorted(grouped.items()):
        excluded = sum(bool(event["selected_for_exclusion"]) for event in group)
        rate = excluded / len(group)
        selected = (
            excluded >= criteria.minimum_sensor_events
            and rate >= criteria.minimum_sensor_exclusion_rate
        )
        rows.append(
            {
                "sensor_id": indoor,
                "outdoor_sensor_id": outdoor,
                "analyzed_events": len(group),
                "excluded_events": excluded,
                "exclusion_rate": rate,
                "selected_for_exclusion": selected,
                "criterion": (
                    f">={criteria.minimum_sensor_events} low-response events and "
                    f">={criteria.minimum_sensor_exclusion_rate:g} exclusion rate"
                ),
            }
        )
    return rows


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: str(value).lower() if isinstance(value, bool) else value
                    for key, value in row.items()
                    if key in fields
                }
            )
    temporary.replace(path)


def write_plot(
    path: Path,
    events: list[dict[str, object]],
    coverage: list[dict[str, object]],
    criteria: Criteria,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    analyzed = [row for row in events if "outdoor_rise_pm25" in row]
    selected = [row for row in analyzed if row["selected_for_exclusion"]]
    figure, axis = plt.subplots(figsize=(9, 6))
    if analyzed:
        for flag, label, color, marker in (
            (False, "Retained event", "#2878b5", "o"),
            (True, "Excluded event", "#c43c39", "x"),
        ):
            rows = [row for row in analyzed if row["selected_for_exclusion"] is flag]
            if rows:
                axis.scatter(
                    [row["outdoor_rise_pm25"] for row in rows],
                    [row["indoor_rise_pm25"] for row in rows],
                    label=label,
                    color=color,
                    marker=marker,
                    alpha=0.8,
                )
        maximum = max(float(row["outdoor_rise_pm25"]) for row in analyzed)
        axis.axhline(
            criteria.maximum_indoor_rise,
            color="#666666",
            linestyle="--",
            linewidth=1,
            label="Indoor-rise limit",
        )
        axis.plot(
            [0, maximum],
            [0, maximum * criteria.maximum_peak_response_ratio],
            color="#777777",
            linestyle=":",
            linewidth=1,
            label="Peak-ratio limit",
        )
        axis.legend()
        axis.set(
            title=f"Paired PurpleAir event responses ({len(selected)} exclusions)",
            xlabel="Outdoor PM2.5 rise above baseline (µg/m³)",
            ylabel="Indoor PM2.5 rise above baseline (µg/m³)",
        )
    else:
        labels = ("Complete", "Missing outdoor", "Missing indoor", "Missing both")
        statuses = ("complete_pair", "missing_outdoor", "missing_indoor", "missing_both")
        counts = [sum(row["status"] == status for row in coverage) for status in statuses]
        bars = axis.bar(labels, counts, color="#2878b5")
        axis.bar_label(bars, padding=3)
        axis.set(
            title="Paired PurpleAir history coverage (no analyzable events)",
            xlabel="Pair-history status",
            ylabel="Number of sensor pairs",
        )
        axis.set_ylim(0, max(counts, default=1) * 1.12)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PurpleAir event exclusion explorer</title>
<style>
:root{color-scheme:light dark;--bg:#fff;--fg:#202124;--muted:#666;--line:#ccd2d8;--in:#2878b5;--out:#c43c39;--shade:#f1c75b55} @media(prefers-color-scheme:dark){:root{--bg:#17191c;--fg:#eee;--muted:#aaa;--line:#59616b;--in:#6db7e8;--out:#f08078;--shade:#b9873255}} *{box-sizing:border-box} body{margin:0 auto;max-width:1100px;padding:1rem;font:15px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--fg)} h1{font-size:1.4rem;margin:.2rem 0} .muted{color:var(--muted)} .controls{display:flex;gap:.6rem;align-items:end;flex-wrap:wrap;margin:1rem 0} label{display:grid;gap:.25rem} select,button{font:inherit;padding:.45rem .6rem} select{max-width:38rem} dl{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.7rem;margin:1rem 0} dt{color:var(--muted)} dd{margin:0;font-variant-numeric:tabular-nums} svg{display:block;width:100%;height:auto;border:1px solid var(--line);touch-action:none} svg text{fill:var(--fg);font:12px system-ui,sans-serif}.axis{stroke:var(--line)} .indoor{stroke:var(--in)} .outdoor{stroke:var(--out)} .series{fill:none;stroke-width:2} .event{fill:var(--shade)} .legend{display:flex;gap:1rem;margin:.6rem 0;flex-wrap:wrap}.reading{display:flex;gap:1.2rem;flex-wrap:wrap;align-items:baseline;margin:.5rem 0;font-variant-numeric:tabular-nums}.reading strong{font-weight:500}.reading .indoor-value{color:var(--in)}.reading .outdoor-value{color:var(--out)}.hit{fill:transparent;cursor:crosshair}.guide{stroke:var(--fg);stroke-width:1;stroke-dasharray:4 3;pointer-events:none}.marker{stroke:var(--bg);stroke-width:2;pointer-events:none}.marker.indoor{fill:var(--in)}.marker.outdoor{fill:var(--out)}.indoor-axis{fill:var(--in)}.empty{padding:4rem 1rem;text-align:center;border:1px solid var(--line)} table{width:100%;border-collapse:collapse;margin-top:1rem} th,td{text-align:left;padding:.4rem;border-bottom:1px solid var(--line)} tr[data-index]{cursor:pointer} tr[data-index]:hover{background:var(--shade)} @media(max-width:600px){body{padding:.7rem} table{font-size:.8rem}.optional{display:none}}
</style>
</head>
<body>
<h1>PurpleAir event exclusion explorer</h1>
<p id="summary" class="muted"></p>
<div id="empty" class="empty" hidden>No events were selected for exclusion.</div>
<main id="content" hidden>
  <div class="controls">
    <label>Location<select id="location"></select></label>
    <label>Excluded event<select id="event"></select></label>
    <button id="previous" type="button">Previous</button>
    <button id="next" type="button">Next</button>
  </div>
  <div class="legend"><label><input id="showIndoor" type="checkbox" checked> Indoor PurpleAir</label><label><input id="showOutdoor" type="checkbox" checked> Outdoor PurpleAir</label><label><input id="magnifyIndoor" type="checkbox" checked> Magnify indoor scale</label></div>
  <dl id="metrics"></dl>
  <div id="reading" class="reading" aria-live="polite">Click the chart to inspect an hourly reading.</div>
  <svg id="chart" viewBox="0 0 1000 500" role="img" aria-labelledby="chartTitle chartDescription"><title id="chartTitle">Indoor and outdoor PM2.5 event</title><desc id="chartDescription">Hourly paired PurpleAir PM2.5 before, during, and after the selected event.</desc></svg>
  <p id="indoorDiagnostics" class="muted"></p>
  <table><thead><tr><th>Event</th><th>Outdoor rise</th><th>Indoor rise</th><th class="optional">Peak ratio</th><th class="optional">Area ratio</th></tr></thead><tbody id="rows"></tbody></table>
</main>
<script>
const report=__REPORT__;
const events=report.events;
const summary=document.getElementById('summary');
summary.textContent=`${report.summary.selected_events} excluded events; ${report.summary.excluded_sensors} excluded sensors; ${report.summary.complete_pairs}/${report.summary.total_pairs} pairs have both PurpleAir histories.`;
const empty=document.getElementById('empty'),content=document.getElementById('content'),locationSelect=document.getElementById('location'),select=document.getElementById('event');let pickedTime=null,visibleIndices=[];
if(!events.length){empty.hidden=false}else{content.hidden=false;renderLocations();filterEvents()}
document.getElementById('previous').onclick=()=>{select.selectedIndex=(select.selectedIndex-1+select.options.length)%select.options.length;pickedTime=null;draw()};
document.getElementById('next').onclick=()=>{select.selectedIndex=(select.selectedIndex+1)%select.options.length;pickedTime=null;draw()};
locationSelect.onchange=filterEvents;
select.onchange=()=>{pickedTime=null;draw()};document.getElementById('showIndoor').onchange=draw;document.getElementById('showOutdoor').onchange=draw;document.getElementById('magnifyIndoor').onchange=draw;window.addEventListener('resize',draw);
function value(number,digits=1){return Number(number).toFixed(digits)}
function pm25(number){return Number.isFinite(number)?`${value(number)} µg/m³`:'Missing'}
function escapeHtml(text){return String(text).replace(/[&<>"']/g,character=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character]))}
function renderLocations(){const counts=new Map;events.forEach((event,index)=>{const key=String(event.indoor_sensor_id),entry=counts.get(key)||{key,name:event.indoor_name||key,indices:[]};entry.indices.push(index);counts.set(key,entry)});const all=document.createElement('option');all.value='';all.textContent=`All locations (${events.length})`;locationSelect.append(all);[...counts.values()].sort((a,b)=>b.indices.length-a.indices.length||a.name.localeCompare(b.name)).forEach(entry=>{const option=document.createElement('option');option.value=entry.key;option.textContent=`${entry.name} (${entry.indices.length})`;option.dataset.indices=entry.indices.join(',');locationSelect.append(option)})}
function filterEvents(){const option=locationSelect.selectedOptions[0];visibleIndices=option.value?option.dataset.indices.split(',').map(Number):events.map((_,index)=>index);select.replaceChildren();visibleIndices.forEach(index=>{const event=events[index],eventOption=document.createElement('option');eventOption.value=index;eventOption.textContent=option.value?event.event_start_utc:`${event.indoor_name||event.indoor_sensor_id} — ${event.event_start_utc}`;select.append(eventOption)});pickedTime=null;renderRows();draw()}
function renderRows(){const body=document.getElementById('rows');body.replaceChildren();visibleIndices.forEach(index=>{const event=events[index],row=document.createElement('tr');row.dataset.index=index;row.innerHTML=`<td>${event.indoor_sensor_id} · ${event.event_start_utc}</td><td>${value(event.outdoor_rise_pm25)}</td><td>${value(event.indoor_rise_pm25)}</td><td class="optional">${value(event.peak_response_ratio,3)}</td><td class="optional">${value(event.area_response_ratio,3)}</td>`;row.onclick=()=>{select.value=String(index);pickedTime=null;draw();scrollTo({top:0,behavior:'smooth'})};body.append(row)})}
function svgNode(name,attributes={}){const node=document.createElementNS('http://www.w3.org/2000/svg',name);Object.entries(attributes).forEach(([key,val])=>node.setAttribute(key,val));return node}
function nearestIndex(series,time){let low=0,high=series.length-1;while(low<high){const middle=Math.floor((low+high)/2);if(series[middle].time<time)low=middle+1;else high=middle}if(low&&Math.abs(series[low-1].time-time)<=Math.abs(series[low].time-time))return low-1;return low}
function draw(){
 if(!events.length)return;
 const event=events[Number(select.value)],metrics=document.getElementById('metrics');
 metrics.innerHTML=[['Indoor sensor',`${event.indoor_sensor_id} · ${event.indoor_name}`],['Outdoor sensor',`${event.outdoor_sensor_id} · ${event.outdoor_name}`],['Outdoor rise',`${value(event.outdoor_rise_pm25)} µg/m³`],['Indoor rise',`${value(event.indoor_rise_pm25)} µg/m³`],['Peak ratio',value(event.peak_response_ratio,3)],['Area ratio',value(event.area_response_ratio,3)],['Peak lag',`${value(event.peak_lag_hours)} hours`],['Paired coverage',`${value(event.response_coverage*100)}%`]].map(([a,b])=>`<div><dt>${escapeHtml(a)}</dt><dd>${escapeHtml(b)}</dd></div>`).join('');
 const svg=document.getElementById('chart'),W=1000,H=500,L=70,R=70,T=25,B=55,series=event.series.map(row=>({...row,time:new Date(row.timestamp_utc).getTime()})).sort((a,b)=>a.time-b.time),times=series.map(row=>row.time),indoorValues=series.map(row=>row.indoor_pm25).filter(Number.isFinite),outdoorValues=series.map(row=>row.outdoor_pm25).filter(Number.isFinite),allValues=[...indoorValues,...outdoorValues],minT=Math.min(...times),maxT=Math.max(...times),magnify=document.getElementById('magnifyIndoor').checked,sharedMax=Math.max(1,...allValues)*1.08,outdoorMax=magnify?Math.max(1,...outdoorValues)*1.08:sharedMax,indoorMax=magnify?Math.max(1,...indoorValues)*1.08:sharedMax,x=time=>L+(time-minT)/(maxT-minT)*(W-L-R),scale=(reading,maximum)=>H-B-reading/maximum*(H-T-B),yOutdoor=reading=>scale(reading,outdoorMax),yIndoor=reading=>scale(reading,indoorMax),eventStart=new Date(event.event_start_utc).getTime(),eventEnd=new Date(event.event_end_utc).getTime();
 svg.replaceChildren();
 const title=svgNode('title');title.textContent=`${event.indoor_name}, event starting ${event.event_start_utc}. Click any hour before, during, or after the event to inspect both readings.`;svg.append(title);
 svg.append(svgNode('rect',{class:'event',x:x(eventStart),y:T,width:Math.max(2,x(eventEnd)-x(eventStart)),height:H-T-B}));
 for(let i=0;i<=5;i++){const gy=T+i*(H-T-B)/5;svg.append(svgNode('line',{class:'axis',x1:L,x2:W-R,y1:gy,y2:gy}));const left=svgNode('text',{x:L-8,y:gy+4,'text-anchor':'end'});left.textContent=value(outdoorMax*(5-i)/5,0);svg.append(left);if(magnify&&document.getElementById('showIndoor').checked){const right=svgNode('text',{class:'indoor-axis',x:W-R+8,y:gy+4,'text-anchor':'start'});right.textContent=value(indoorMax*(5-i)/5,1);svg.append(right)}}
 for(let i=0;i<=4;i++){const tx=L+i*(W-L-R)/4,text=svgNode('text',{x:tx,y:H-20,'text-anchor':i===0?'start':i===4?'end':'middle'});text.textContent=new Date(minT+(maxT-minT)*i/4).toISOString().slice(5,16).replace('T',' ');svg.append(text)}
 [['indoor_pm25','indoor','showIndoor',yIndoor],['outdoor_pm25','outdoor','showOutdoor',yOutdoor]].forEach(([key,klass,toggle,y])=>{if(!document.getElementById(toggle).checked)return;let d='',active=false;series.forEach(row=>{if(!Number.isFinite(row[key])){active=false;return}d+=`${active?'L':'M'}${x(row.time).toFixed(1)},${y(row[key]).toFixed(1)}`;active=true});svg.append(svgNode('path',{class:`series ${klass}`,d}))});
 const xlabel=svgNode('text',{x:(L+W-R)/2,y:H-2,'text-anchor':'middle'});xlabel.textContent='UTC date and hour';svg.append(xlabel);const ylabel=svgNode('text',{transform:`translate(16 ${(T+H-B)/2}) rotate(-90)`,'text-anchor':'middle'});ylabel.textContent=magnify?'Outdoor PM2.5 (µg/m³)':'PM2.5 (µg/m³)';svg.append(ylabel);if(magnify){const rightLabel=svgNode('text',{class:'indoor-axis',transform:`translate(${W-12} ${(T+H-B)/2}) rotate(90)`,'text-anchor':'middle'});rightLabel.textContent='Indoor PM2.5 (µg/m³)';svg.append(rightLabel)}
 const selected=series[nearestIndex(series,pickedTime??eventStart)];pickedTime=selected.time;svg.append(svgNode('line',{class:'guide',x1:x(selected.time),x2:x(selected.time),y1:T,y2:H-B}));[['indoor_pm25','indoor',yIndoor],['outdoor_pm25','outdoor',yOutdoor]].forEach(([key,klass,y])=>{if(Number.isFinite(selected[key])&&document.getElementById(klass==='indoor'?'showIndoor':'showOutdoor').checked)svg.append(svgNode('circle',{class:`marker ${klass}`,cx:x(selected.time),cy:y(selected[key]),r:5}))});
 const phase=selected.time<eventStart?'Before event':selected.time<=eventEnd?'Inside event':'After event';document.getElementById('reading').innerHTML=`<strong>${phase}</strong><span>${escapeHtml(selected.timestamp_utc)}</span><span class="indoor-value">Indoor: ${escapeHtml(pm25(selected.indoor_pm25))}</span><span class="outdoor-value">Outdoor: ${escapeHtml(pm25(selected.outdoor_pm25))}</span>`;
 const zeros=indoorValues.filter(reading=>reading===0).length,distinct=new Set(indoorValues.map(reading=>reading.toFixed(3))).size,missing=series.length-indoorValues.length,minimum=indoorValues.length?Math.min(...indoorValues):NaN,maximum=indoorValues.length?Math.max(...indoorValues):NaN,range=indoorValues.length?`${value(minimum)}–${value(maximum)} µg/m³`:'unavailable',nearZero=indoorValues.length>=12&&zeros/indoorValues.length>=.9&&maximum<=1,state=nearZero?'potentially stuck near zero; verify sensor health':distinct===1&&indoorValues.length>1?'constant across the plotted window; verify sensor health':`${distinct} distinct values`;document.getElementById('indoorDiagnostics').textContent=`Indoor window diagnostic: ${indoorValues.length} readings, ${missing} missing, ${zeros} exactly zero; range ${range}; ${state}.`;
 const hit=svgNode('rect',{class:'hit','data-chart-hit':'',x:L,y:T,width:W-L-R,height:H-T-B});hit.onclick=pointer=>{const bounds=svg.getBoundingClientRect(),chartX=(pointer.clientX-bounds.left)*W/bounds.width,requested=minT+(Math.max(L,Math.min(W-R,chartX))-L)/(W-L-R)*(maxT-minT);pickedTime=series[nearestIndex(series,requested)].time;draw()};svg.append(hit)
}
</script>
</body>
</html>'''


def write_outputs(
    output: Path,
    events: list[dict[str, object]],
    coverage: list[dict[str, object]],
    criteria: Criteria,
    inputs: dict[str, object] | None = None,
    cohort_selection: list[dict[str, object]] | None = None,
    review_outdoor_ids: set[int] | None = None,
    review_indoor_ids: set[int] | None = None,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    sensors = sensor_rows(events, criteria)
    selected_events = [row for row in events if row["selected_for_exclusion"]]
    excluded_sensors = [row for row in sensors if row["selected_for_exclusion"]]
    summary = {
        "source": "paired PurpleAir pm2.5_atm only",
        "total_pairs": len(coverage),
        "complete_pairs": sum(row["status"] == "complete_pair" for row in coverage),
        "missing_indoor_pairs": sum(row["status"] == "missing_indoor" for row in coverage),
        "missing_outdoor_pairs": sum(row["status"] == "missing_outdoor" for row in coverage),
        "missing_both_pairs": sum(row["status"] == "missing_both" for row in coverage),
        "detected_events": len(events),
        "analyzed_events": sum("outdoor_rise_pm25" in row for row in events),
        "selected_events": len(selected_events),
        "excluded_sensors": len(excluded_sensors),
        "criteria": asdict(criteria),
    }
    if inputs:
        summary["inputs"] = inputs
    if cohort_selection is not None:
        write_csv(
            output / "cohort_selection.csv",
            COHORT_SELECTION_FIELDS,
            cohort_selection,
        )
    if review_outdoor_ids is not None:
        review_candidates = [
            row
            for row in excluded_sensors
            if int(row["outdoor_sensor_id"]) in review_outdoor_ids
            or int(row["sensor_id"]) in (review_indoor_ids or set())
        ]
        summary["k12_1km_indoor_exclusion_candidates"] = len(review_candidates)
        write_csv(
            output / "k12_1km_indoor_exclusion_candidates.csv",
            SENSOR_FIELDS,
            review_candidates,
        )
    write_csv(output / "selected_pairs.csv", SELECTED_PAIR_FIELDS, coverage)
    write_csv(output / "pair_coverage.csv", COVERAGE_FIELDS, coverage)
    write_csv(output / "events.csv", EVENT_FIELDS, events)
    write_csv(output / "sensor_summary.csv", SENSOR_FIELDS, sensors)
    write_csv(output / "excluded_sensors.csv", SENSOR_FIELDS, excluded_sensors)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "summary": summary,
        "events": [
            {
                **{key: value for key, value in row.items() if key != "_series"},
                "series": row["_series"],
            }
            for row in selected_events
        ],
    }
    (output / "event_explorer.html").write_text(
        HTML.replace("__REPORT__", json.dumps(report).replace("</", "<\\/")),
        encoding="utf-8",
    )
    write_plot(output / "event_response_summary.png", events, coverage, criteria)
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Audit paired PurpleAir outdoor events with weak indoor responses."
    )
    result.add_argument(
        "--pairs",
        type=Path,
        default=Path("../purple-air-pull/purpleair_continental_us_pairs.csv"),
    )
    result.add_argument(
        "--school-smoke-overlap",
        type=Path,
        default=Path(
            "../purple-air-pull/smoke_plume_intersection/results/"
            "purpleair_indoor_school_sparse_wildfire_ranges.csv"
        ),
        help="school smoke-overlap ranges whose unique indoor IDs select pairs",
    )
    result.add_argument(
        "--fema-school-sensors",
        type=Path,
        default=Path("../purple-air-pull/purpleair_indoor_school_sensors.csv"),
        help="validated FEMA/NSD indoor-school cohort",
    )
    result.add_argument(
        "--sensor-inventory",
        type=Path,
        default=Path("../purple-air-pull/purpleair_continental_us_sensors.csv"),
        help="active PurpleAir inventory used to pair each school independently",
    )
    result.add_argument("--school-pair-distance", type=float, default=1000.0)
    result.add_argument("--indoor-history", type=Path, action="append")
    result.add_argument(
        "--review-indoor-history",
        type=Path,
        action="append",
        help="1 km indoor history source to analyze and show on the review tab",
    )
    result.add_argument("--outdoor-history", type=Path, action="append")
    result.add_argument(
        "--review-outdoor-history",
        type=Path,
        action="append",
        help="isolated 1 km outdoor archive to analyze and show on its own explorer tab",
    )
    result.add_argument(
        "--include-downloaded-pairs",
        action="store_true",
        help="include every pair whose indoor sensor occurs in the supplied histories",
    )
    result.add_argument(
        "--output-dir",
        type=Path,
        default=Path("purpleair_pair_exclusions/results"),
    )
    result.add_argument(
        "--training-output-dir", type=Path, default=DEFAULT_TRAINING_OUTPUT
    )
    for field, value in asdict(Criteria()).items():
        option = "--" + field.replace("_", "-")
        result.add_argument(option, type=type(value), default=value)
    return result


def main() -> None:
    args = parser().parse_args()
    primary_indoor_paths = args.indoor_history or [HISTORY_ROOT / "all_indoor_pm25.csv"]
    primary_outdoor_paths = args.outdoor_history or [HISTORY_ROOT / "all_outdoor_pm25.csv"]
    criteria = Criteria(
        **{field: getattr(args, field) for field in asdict(Criteria())}
    )
    snapshot_pairs = read_pairs(args.pairs)
    overlap_ids = read_overlap_indoor_ids(args.school_smoke_overlap)
    fema_ids = read_fema_school_ids(args.fema_school_sensors)
    permanent_exclusion_rows = read_permanent_exclusions(PERMANENT_EXCLUSIONS_PATH)
    permanent_exclusions = set().union(
        *(read_excluded_sensor_ids(path) for path in INDOOR_EXCLUSION_PATHS)
    )
    names = read_sensor_names(args.sensor_inventory)
    permanent_ids = {int(row["sensor_id"]) for row in permanent_exclusion_rows}
    permanent_exclusion_rows.extend(
        {
            "sensor_id": sensor_id,
            "sensor_name": names.get(sensor_id, f"Sensor {sensor_id}"),
            "reason": "whole-sensor exclusion: PM2.5 readings above 1,000 ug/m3",
        }
        for sensor_id in sorted(permanent_exclusions - permanent_ids)
    )
    indoor_exclusions = read_indoor_exclusions(INDOOR_RANGE_EXCLUSIONS_PATH)
    outdoor_exclusions = read_outdoor_exclusions(OUTDOOR_EXCLUSIONS_PATH)
    reviewed_outdoor_ids = {item.sensor_id for item in outdoor_exclusions}
    excluded_outdoor_ids = {
        item.sensor_id
        for item in outdoor_exclusions
        if item.start is None and item.end is None
    }
    school_ids = overlap_ids | fema_ids
    school_pairs = read_reusable_school_pairs(
        args.sensor_inventory,
        school_ids - permanent_exclusions,
        args.school_pair_distance,
        excluded_outdoor_ids,
    )
    non_school_pairs = [
        row
        for row in snapshot_pairs
        if int(row["indoor_sensor_id"]) not in school_ids
        and int(row["indoor_sensor_id"]) not in permanent_exclusions
    ]
    excluded_snapshot_pairs = [
        row
        for row in non_school_pairs
        if int(row["outdoor_sensor_id"]) in excluded_outdoor_ids
    ]
    replacement_pairs = read_reusable_school_pairs(
        args.sensor_inventory,
        {int(row["indoor_sensor_id"]) for row in excluded_snapshot_pairs},
        args.school_pair_distance,
        excluded_outdoor_ids,
    )
    all_pairs = [
        row
        for row in non_school_pairs
        if int(row["outdoor_sensor_id"]) not in excluded_outdoor_ids
    ] + replacement_pairs + school_pairs
    cohorts = {"smoke_overlap_school": overlap_ids, "fema_school": fema_ids}
    review_indoor_paths = args.review_indoor_history or []
    indoor_paths = [*primary_indoor_paths, *review_indoor_paths]
    review_indoor_ids = (
        set(read_histories(review_indoor_paths)) if review_indoor_paths else set()
    )
    indoor = None
    downloaded_ids: set[int] = set()
    if args.include_downloaded_pairs:
        indoor = read_histories(
            indoor_paths,
            {
                int(row["indoor_sensor_id"])
                for row in all_pairs + excluded_snapshot_pairs
            },
        )
        downloaded_ids = {sensor for sensor, values in indoor.items() if values}
        cohorts["downloaded_history"] = downloaded_ids
    pairs, cohort_selection = select_pairs(
        all_pairs,
        cohorts,
        permanent_exclusions,
    )
    if indoor is None:
        indoor = read_histories(
            indoor_paths, {int(row["indoor_sensor_id"]) for row in pairs}
        )
    review_paths = args.review_outdoor_history or []
    outdoor_paths = [*primary_outdoor_paths, *review_paths]
    review_outdoor_ids = set(read_histories(review_paths)) if review_paths else set()
    outdoor = read_histories(
        outdoor_paths, {int(row["outdoor_sensor_id"]) for row in pairs}
    )
    explorer_outdoor = outdoor | read_histories(
        outdoor_paths, reviewed_outdoor_ids
    )
    explorer_indoor = read_histories(indoor_paths)
    training_candidates = read_ranked_candidates(
        args.sensor_inventory,
        school_ids - permanent_exclusions,
        args.school_pair_distance,
    )
    training_intervals, unresolved_intervals = build_training_intervals(
        training_candidates,
        explorer_indoor,
        {"smoke_overlap_school": overlap_ids, "fema_school": fema_ids},
        permanent_exclusions,
        indoor_exclusions,
        outdoor_exclusions,
    )
    interval_metadata = write_training_contract(
        args.training_output_dir,
        training_intervals,
        unresolved_intervals,
        args.school_pair_distance,
        {
            "sensor_inventory": source_record(args.sensor_inventory),
            "school_cohorts": [
                source_record(args.school_smoke_overlap),
                source_record(args.fema_school_sensors),
            ],
            "indoor_history": [source_record(path) for path in primary_indoor_paths],
        },
        [
            source_record(path)
            for path in (
                *INDOOR_EXCLUSION_PATHS,
                INDOOR_RANGE_EXCLUSIONS_PATH,
                OUTDOOR_EXCLUSIONS_PATH,
            )
        ],
        len(school_ids - permanent_exclusions),
    )
    paired_indoor_ids = {int(row["indoor_sensor_id"]) for row in all_pairs}
    unpaired_sensor_ids = (
        set(explorer_indoor) - paired_indoor_ids - permanent_exclusions
    )
    analysis_indoor, excluded_indoor_hours = exclude_outdoor_readings(
        indoor, indoor_exclusions
    )
    analysis_outdoor, excluded_outdoor_hours = exclude_outdoor_readings(
        outdoor, outdoor_exclusions
    )
    events, coverage = analyze_pairs(
        pairs, analysis_indoor, analysis_outdoor, criteria
    )
    inputs = {
        "purpleair_pair_snapshot": str(args.pairs.resolve()),
        "school_smoke_overlap": str(args.school_smoke_overlap.resolve()),
        "fema_school_sensors": str(args.fema_school_sensors.resolve()),
        "purpleair_pair_snapshot_count": len(snapshot_pairs),
        "sensor_inventory": str(args.sensor_inventory.resolve()),
        "indoor_whole_sensor_exclusions": [
            str(path.resolve()) for path in INDOOR_EXCLUSION_PATHS
        ],
        "permanently_excluded_indoor_sensors": len(permanent_exclusions),
        "indoor_range_exclusions": str(INDOOR_RANGE_EXCLUSIONS_PATH.resolve()),
        "indoor_exclusion_ranges": len(indoor_exclusions),
        "excluded_indoor_hours": excluded_indoor_hours,
        "outdoor_exclusions": str(OUTDOOR_EXCLUSIONS_PATH.resolve()),
        "outdoor_exclusion_ranges": len(outdoor_exclusions),
        "excluded_outdoor_sensors": len(excluded_outdoor_ids),
        "reviewed_outdoor_sensors": len(reviewed_outdoor_ids),
        "excluded_non_school_snapshot_pairs": len(excluded_snapshot_pairs),
        "replacement_non_school_snapshot_pairs": len(replacement_pairs),
        "excluded_outdoor_hours": excluded_outdoor_hours,
        "excluded_candidate_sensors": sum(
            row["selection_status"] == "permanently_excluded_indoor_sensor"
            for row in cohort_selection
        ),
        "school_pair_distance_meters": args.school_pair_distance,
        "reusable_school_pair_count": len(school_pairs),
        "distinct_school_outdoor_sensors": len(
            {row["outdoor_sensor_id"] for row in school_pairs}
        ),
        "school_smoke_overlap_indoor_sensors": len(overlap_ids),
        "fema_school_indoor_sensors": len(fema_ids),
        "combined_school_indoor_sensors": len(overlap_ids | fema_ids),
        "selected_pair_count": len(pairs),
        "selected_smoke_overlap_pairs": sum(
            "smoke_overlap_school" in row["cohort_sources"] for row in pairs
        ),
        "selected_fema_school_pairs": sum(
            "fema_school" in row["cohort_sources"] for row in pairs
        ),
        "selected_in_both_school_cohorts": sum(
            "smoke_overlap_school" in row["cohort_sources"]
            and "fema_school" in row["cohort_sources"]
            for row in pairs
        ),
        "downloaded_indoor_sensors": len(downloaded_ids),
        "selected_downloaded_pairs": sum(
            "downloaded_history" in row["cohort_sources"] for row in pairs
        ),
        "school_sensors_without_outdoor_purpleair_pairs": sum(
            row["selection_status"] == "no_outdoor_purpleair_pair"
            and "school" in row["cohort_sources"]
            for row in cohort_selection
        ),
        "candidate_sensors_without_outdoor_purpleair_pairs": sum(
            row["selection_status"] == "no_outdoor_purpleair_pair"
            for row in cohort_selection
        ),
        "indoor_history": [str(path.resolve()) for path in primary_indoor_paths],
        "review_indoor_history": [
            str(path.resolve()) for path in review_indoor_paths
        ],
        "outdoor_history": [str(path.resolve()) for path in primary_outdoor_paths],
        "review_outdoor_history": [str(path.resolve()) for path in review_paths],
        "training_intervals": interval_metadata["counts"],
    }
    inputs["history_explorer_locations"] = write_location_history_explorer(
        args.output_dir,
        pairs,
        explorer_indoor,
        explorer_outdoor,
        outdoor_exclusions,
        unpaired_sensor_ids,
        permanent_exclusion_rows,
        indoor_exclusions,
        fema_ids,
        review_outdoor_ids=review_outdoor_ids,
        review_indoor_ids=review_indoor_ids,
    )
    inputs["history_explorer_1km_review_indoor_sensors"] = len(review_indoor_ids)
    inputs["history_explorer_1km_review_outdoor_sensors"] = len(review_outdoor_ids)
    inputs["history_explorer_unpaired_sensors"] = len(unpaired_sensor_ids)
    inputs["history_explorer_excluded_sensors"] = sum(
        bool(explorer_indoor.get(int(row["sensor_id"])))
        for row in permanent_exclusion_rows
    ) + sum(
        bool(explorer_indoor.get(sensor_id))
        for sensor_id in {item.sensor_id for item in indoor_exclusions}
    ) + sum(
        bool(explorer_outdoor.get(sensor_id))
        for sensor_id in reviewed_outdoor_ids
    )
    summary = write_outputs(
        args.output_dir,
        events,
        coverage,
        criteria,
        inputs,
        cohort_selection,
        review_outdoor_ids,
        review_indoor_ids,
    )
    print(" ".join(f"{key}={value}" for key, value in summary.items() if key != "criteria"))
    print(f"output={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
