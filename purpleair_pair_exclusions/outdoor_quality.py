"""Known-bad PurpleAir outdoor sensor time ranges."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class OutdoorExclusion:
    sensor_id: int
    start: int | None
    end: int | None
    reason: str

    def contains(self, timestamp: int) -> bool:
        return (self.start is None or timestamp >= self.start) and (
            self.end is None or timestamp < self.end
        )


def _timestamp(text: str) -> int | None:
    if not text.strip():
        return None
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.utcoffset() is None:
        raise ValueError("outdoor exclusion timestamps must include a UTC offset")
    return int(parsed.timestamp())


def _read_exclusions(
    path: Path, sensor_column: str
) -> tuple[OutdoorExclusion, ...]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {sensor_column, "start_utc", "end_utc", "reason"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"missing exclusion columns: {', '.join(sorted(missing))}"
            )
        ranges = []
        for number, row in enumerate(reader, 2):
            try:
                sensor_id = int(row[sensor_column])
                start, end = _timestamp(row["start_utc"]), _timestamp(row["end_utc"])
                if sensor_id < 1 or start is not None and end is not None and start >= end:
                    raise ValueError
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid outdoor exclusion row {number}") from error
            ranges.append(OutdoorExclusion(sensor_id, start, end, row["reason"]))
    if not ranges:
        raise ValueError("outdoor exclusion CSV contains no ranges")
    return tuple(ranges)


def read_outdoor_exclusions(path: Path) -> tuple[OutdoorExclusion, ...]:
    return _read_exclusions(path, "outdoor_sensor_id")


def read_indoor_exclusions(path: Path) -> tuple[OutdoorExclusion, ...]:
    return _read_exclusions(path, "indoor_sensor_id")


def exclude_outdoor_readings(
    values: dict[int, dict[int, float]],
    exclusions: tuple[OutdoorExclusion, ...],
) -> tuple[dict[int, dict[int, float]], int]:
    by_sensor: dict[int, list[OutdoorExclusion]] = {}
    for exclusion in exclusions:
        by_sensor.setdefault(exclusion.sensor_id, []).append(exclusion)
    filtered, removed = {}, 0
    for sensor_id, readings in values.items():
        ranges = by_sensor.get(sensor_id, ())
        filtered[sensor_id] = {
            timestamp: value
            for timestamp, value in readings.items()
            if not any(item.contains(timestamp) for item in ranges)
        }
        removed += len(readings) - len(filtered[sensor_id])
    return filtered, removed
