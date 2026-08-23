"""Build the static outdoor assignments consumed by masked training."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

from purpleair_pair_exclusions.outdoor_quality import OutdoorExclusion


HOUR = 3600
EARTH_RADIUS_METERS = 6_371_008.8
SENSOR_COLUMNS = ("sensor_index", "name", "location_type", "latitude", "longitude")
INTERVAL_FIELDS = (
    "assignment_id",
    "indoor_sensor_id",
    "indoor_name",
    "outdoor_sensor_id",
    "outdoor_name",
    "distance_meters",
    "candidate_rank",
    "start_utc",
    "end_utc",
    "cohort_sources",
    "selection_reason",
)
UNRESOLVED_FIELDS = (
    "indoor_sensor_id",
    "indoor_name",
    "start_utc",
    "end_utc",
    "cohort_sources",
    "reason",
)


def read_ranked_candidates(
    path: Path, indoor_ids: set[int], maximum_distance: float
) -> dict[int, list[dict[str, object]]]:
    """Return every outdoor candidate in distance/ID order for each indoor sensor."""
    if maximum_distance <= 0 or not math.isfinite(maximum_distance):
        raise ValueError("maximum matching distance must be finite and positive")
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = set(SENSOR_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing sensor inventory columns: {', '.join(sorted(missing))}")
        indoors, outdoors = {}, []
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
            if sensor["location_type"] == "outside":
                outdoors.append(sensor)
            elif sensor["sensor_index"] in indoor_ids:
                indoors[int(sensor["sensor_index"])] = sensor
    if not outdoors:
        raise ValueError("sensor inventory contains no outdoor sensors")

    candidates = {}
    for indoor_id in sorted(indoor_ids & indoors.keys()):
        indoor = indoors[indoor_id]
        ranked = sorted(
            (
                (_distance_meters(indoor, outdoor), int(outdoor["sensor_index"]), outdoor)
                for outdoor in outdoors
            ),
            key=lambda item: (item[0], item[1]),
        )
        candidates[indoor_id] = [
            {
                "indoor_sensor_id": indoor_id,
                "indoor_name": indoor["name"],
                "outdoor_sensor_id": outdoor_id,
                "outdoor_name": outdoor["name"],
                "distance_meters": round(distance, 2),
                "candidate_rank": rank,
            }
            for rank, (distance, outdoor_id, outdoor) in enumerate(ranked, 1)
            if distance <= maximum_distance
        ]
    return candidates


def build_training_intervals(
    candidates: dict[int, list[dict[str, object]]],
    indoor_history: dict[int, dict[int, float]],
    cohorts: dict[str, set[int]],
    excluded_indoor_ids: set[int],
    indoor_exclusions: tuple[OutdoorExclusion, ...],
    outdoor_exclusions: tuple[OutdoorExclusion, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Resolve the closest non-excluded outdoor sensor over each indoor history span."""
    indoor_ranges = _by_sensor(indoor_exclusions)
    outdoor_ranges = _by_sensor(outdoor_exclusions)
    intervals, unresolved = [], []
    for indoor_id in sorted(set().union(*cohorts.values())):
        readings = indoor_history.get(indoor_id, {})
        if not readings:
            continue
        start, end = min(readings), max(readings) + HOUR
        if start % HOUR or end % HOUR:
            raise ValueError(f"indoor history for sensor {indoor_id} is not hourly")
        sources = ";".join(name for name, ids in cohorts.items() if indoor_id in ids)
        choices = candidates.get(indoor_id, [])
        name = str(choices[0]["indoor_name"]) if choices else f"Sensor {indoor_id}"
        if indoor_id in excluded_indoor_ids:
            _append_unresolved(
                unresolved,
                indoor_id,
                name,
                start,
                end,
                sources,
                "indoor_sensor_excluded",
            )
            continue
        boundaries = {start, end}
        relevant = [*indoor_ranges.get(indoor_id, ())]
        for choice in choices:
            relevant.extend(outdoor_ranges.get(int(choice["outdoor_sensor_id"]), ()))
        for exclusion in relevant:
            left = max(start, exclusion.start if exclusion.start is not None else start)
            right = min(end, exclusion.end if exclusion.end is not None else end)
            if left < right:
                if left % HOUR or right % HOUR:
                    raise ValueError("training exclusion boundaries must be exact UTC hours")
                boundaries.update((left, right))

        ordered_boundaries = sorted(boundaries)
        for left, right in zip(ordered_boundaries, ordered_boundaries[1:]):
            if any(item.contains(left) for item in indoor_ranges.get(indoor_id, ())):
                _append_unresolved(
                    unresolved, indoor_id, name, left, right, sources, "indoor_excluded"
                )
                continue
            selected = next(
                (
                    choice
                    for choice in choices
                    if not any(
                        item.contains(left)
                        for item in outdoor_ranges.get(
                            int(choice["outdoor_sensor_id"]), ()
                        )
                    )
                ),
                None,
            )
            if selected is None:
                _append_unresolved(
                    unresolved,
                    indoor_id,
                    name,
                    left,
                    right,
                    sources,
                    "no_eligible_outdoor_sensor",
                )
                continue
            row = {
                **selected,
                "start": left,
                "end": right,
                "cohort_sources": sources,
                "selection_reason": (
                    "nearest_outdoor_sensor"
                    if int(selected["candidate_rank"]) == 1
                    else "fallback_due_to_exclusion"
                ),
            }
            if (
                intervals
                and intervals[-1]["indoor_sensor_id"] == indoor_id
                and intervals[-1]["outdoor_sensor_id"] == row["outdoor_sensor_id"]
                and intervals[-1]["end"] == left
            ):
                intervals[-1]["end"] = right
            else:
                intervals.append(row)

    return [_finalize(row) for row in intervals], [_finalize_gap(row) for row in unresolved]


def write_training_contract(
    output: Path,
    intervals: list[dict[str, object]],
    unresolved: list[dict[str, object]],
    matching_distance: float,
    sources: dict[str, object],
    exclusions: list[dict[str, str]],
    candidate_sensor_count: int,
) -> dict[str, object]:
    if not intervals:
        raise ValueError("no static training intervals were generated")
    output.mkdir(parents=True, exist_ok=True)
    interval_path = output / "training_intervals.csv"
    unresolved_path = output / "unresolved_training_intervals.csv"
    _write_csv(interval_path, INTERVAL_FIELDS, intervals)
    _write_csv(unresolved_path, UNRESOLVED_FIELDS, unresolved)
    by_indoor: dict[int, list[dict[str, object]]] = {}
    for row in intervals:
        by_indoor.setdefault(int(row["indoor_sensor_id"]), []).append(row)
    handoffs = sum(
        first["end_utc"] == second["start_utc"]
        and first["outdoor_sensor_id"] != second["outdoor_sensor_id"]
        for rows in by_indoor.values()
        for first, second in zip(rows, rows[1:])
    )
    metadata = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "interval_semantics": "start_utc inclusive; end_utc exclusive; exact UTC hours",
        "matching_distance_meters": matching_distance,
        "manifest": {
            "path": str(interval_path.resolve()),
            "sha256": file_sha256(interval_path),
        },
        "sources": sources,
        "exclusions": exclusions,
        "counts": {
            "candidate_indoor_sensors": candidate_sensor_count,
            "assigned_indoor_sensors": len(by_indoor),
            "training_intervals": len(intervals),
            "distinct_outdoor_sensors": len(
                {int(row["outdoor_sensor_id"]) for row in intervals}
            ),
            "fallback_intervals": sum(
                row["selection_reason"] == "fallback_due_to_exclusion"
                for row in intervals
            ),
            "outdoor_handoffs": handoffs,
            "unresolved_intervals": len(unresolved),
            "unresolved_hours": sum(
                (_timestamp(row["end_utc"]) - _timestamp(row["start_utc"])) // HOUR
                for row in unresolved
            ),
            "indoor_exclusion_intervals": sum(
                str(row["reason"]).startswith("indoor_") for row in unresolved
            ),
            "indoor_exclusion_hours": sum(
                (_timestamp(row["end_utc"]) - _timestamp(row["start_utc"])) // HOUR
                for row in unresolved
                if str(row["reason"]).startswith("indoor_")
            ),
            "unresolved_outdoor_intervals": sum(
                row["reason"] == "no_eligible_outdoor_sensor" for row in unresolved
            ),
            "unresolved_outdoor_hours": sum(
                (_timestamp(row["end_utc"]) - _timestamp(row["start_utc"])) // HOUR
                for row in unresolved
                if row["reason"] == "no_eligible_outdoor_sensor"
            ),
        },
    }
    metadata_path = output / "training_intervals.meta.json"
    _write_json(metadata_path, metadata)
    return metadata


def source_record(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path) if path.is_file() else _directory_sha256(path),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_sha256(path: Path) -> str:
    if not path.is_dir():
        raise FileNotFoundError(f"source path not found: {path}")
    digest = hashlib.sha256()
    for file in sorted(path.rglob("*.csv")):
        digest.update(str(file.relative_to(path)).encode())
        digest.update(file_sha256(file).encode())
    return digest.hexdigest()


def _distance_meters(first: dict[str, object], second: dict[str, object]) -> float:
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


def _by_sensor(
    exclusions: tuple[OutdoorExclusion, ...],
) -> dict[int, list[OutdoorExclusion]]:
    result: dict[int, list[OutdoorExclusion]] = {}
    for exclusion in exclusions:
        result.setdefault(exclusion.sensor_id, []).append(exclusion)
    return result


def _append_unresolved(
    rows: list[dict[str, object]],
    indoor_id: int,
    name: str,
    start: int,
    end: int,
    sources: str,
    reason: str,
) -> None:
    if (
        rows
        and rows[-1]["indoor_sensor_id"] == indoor_id
        and rows[-1]["end"] == start
        and rows[-1]["reason"] == reason
    ):
        rows[-1]["end"] = end
    else:
        rows.append(
            {
                "indoor_sensor_id": indoor_id,
                "indoor_name": name,
                "start": start,
                "end": end,
                "cohort_sources": sources,
                "reason": reason,
            }
        )


def _finalize(row: dict[str, object]) -> dict[str, object]:
    start, end = int(row.pop("start")), int(row.pop("end"))
    indoor, outdoor = int(row["indoor_sensor_id"]), int(row["outdoor_sensor_id"])
    return {
        "assignment_id": f"{indoor}-{outdoor}-{start}-{end}",
        **row,
        "start_utc": _iso_utc(start),
        "end_utc": _iso_utc(end),
    }


def _finalize_gap(row: dict[str, object]) -> dict[str, object]:
    start, end = int(row.pop("start")), int(row.pop("end"))
    return {**row, "start_utc": _iso_utc(start), "end_utc": _iso_utc(end)}


def _iso_utc(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def _timestamp(value: object) -> int:
    return int(datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp())


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
