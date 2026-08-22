"""Detect recurring PurpleAir hardware-error concentration levels."""

from __future__ import annotations

from dataclasses import dataclass


HOUR = 3600


@dataclass(frozen=True)
class ErrorLevelCriteria:
    bands: tuple[tuple[float, float], ...] = (
        (1500.0, 1800.0),
        (2250.0, 2700.0),
        (3000.0, 3600.0),
    )
    minimum_readings: int = 3
    maximum_gap_hours: int = 48

    def validate(self) -> None:
        if self.minimum_readings < 1 or self.maximum_gap_hours < 1:
            raise ValueError("error-level counts and gaps must be positive")
        if any(low < 0 or high <= low for low, high in self.bands):
            raise ValueError("error-level bands must have valid increasing bounds")


@dataclass(frozen=True)
class ErrorLevelPeriod:
    start: int
    end: int
    readings: int
    minimum_pm25: float
    maximum_pm25: float


def is_error_level(value: float, criteria: ErrorLevelCriteria) -> bool:
    return any(low <= value <= high for low, high in criteria.bands)


def detect_error_level_periods(
    readings: dict[int, float],
    criteria: ErrorLevelCriteria = ErrorLevelCriteria(),
) -> tuple[ErrorLevelPeriod, ...]:
    criteria.validate()
    groups: list[list[tuple[int, float]]] = []
    maximum_gap = criteria.maximum_gap_hours * HOUR
    for reading in sorted(
        (item for item in readings.items() if is_error_level(item[1], criteria))
    ):
        if not groups or reading[0] - groups[-1][-1][0] > maximum_gap:
            groups.append([])
        groups[-1].append(reading)
    return tuple(
        ErrorLevelPeriod(
            group[0][0],
            group[-1][0] + HOUR,
            len(group),
            min(value for _, value in group),
            max(value for _, value in group),
        )
        for group in groups
        if len(group) >= criteria.minimum_readings
    )
