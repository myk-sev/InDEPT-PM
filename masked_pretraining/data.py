from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


HOUR = 3600
TRAINING_COLUMNS = (
    "indoor_sensor_id",
    "outdoor_sensor_id",
    "start_utc",
    "end_utc",
    "responsiveness_tier",
    "indoor_pm25",
    "outdoor_pm25",
)
RESPONSIVENESS_TIERS = {"high", "moderate", "low", "unclassified"}


@dataclass(frozen=True)
class TrainingInterval:
    assignment_id: str
    indoor_id: int
    outdoor_id: int
    start_time: int
    end_time: int


@dataclass(frozen=True)
class IntervalSelection:
    intervals: tuple[TrainingInterval, ...]
    indoor_sensor_count: int
    responsiveness_filtered_interval_count: int = 0

    @property
    def indoor_sensor_ids(self) -> tuple[int, ...]:
        return tuple(sorted({item.indoor_id for item in self.intervals}))


@dataclass(frozen=True)
class HistoryData:
    values: dict[int, dict[int, float]]
    row_count: int


@dataclass(frozen=True)
class IndoorSeries:
    indoor_id: int
    start_time: int
    values: np.ndarray
    allowed: np.ndarray
    outdoor_sensor_ids: np.ndarray
    assignment_indices: np.ndarray
    assignment_ids: tuple[str, ...]
    handoff_offsets: tuple[int, ...]
    window_starts: tuple[int, ...]


@dataclass(frozen=True)
class PairDatabase:
    selection: IntervalSelection
    series: tuple[IndoorSeries, ...]
    sensor_summaries: tuple[dict[str, object], ...]
    history: HistoryData
    history_hours: int
    minimum_observed_hours: int
    stride_hours: int

    @property
    def window_count(self) -> int:
        return sum(len(item.window_starts) for item in self.series)

    @property
    def outdoor_handoff_count(self) -> int:
        return sum(len(item.handoff_offsets) for item in self.series)

    @property
    def windows_crossing_handoffs(self) -> int:
        return sum(
            any(start < handoff < start + self.history_hours for handoff in item.handoff_offsets)
            for item in self.series
            for start in item.window_starts
        )


@dataclass(frozen=True)
class Normalizer:
    mean: tuple[float, float]
    standard_deviation: tuple[float, float]

    @classmethod
    def fit(cls, series: tuple[IndoorSeries, ...], indices: list[int]) -> Normalizer:
        means, deviations = [], []
        for channel in range(2):
            count = total = squared = 0.0
            for index in indices:
                values = series[index].values[series[index].allowed, channel]
                values = values[np.isfinite(values)].astype(np.float64)
                count += values.size
                total += values.sum()
                squared += np.square(values).sum()
            if not count:
                raise ValueError(f"training sensors have no channel {channel} observations")
            mean = total / count
            variance = max(squared / count - mean * mean, 0.0)
            means.append(mean)
            deviations.append(max(math.sqrt(variance), 1e-6))
        return cls(tuple(means), tuple(deviations))


class PairWindowDataset(Dataset):
    """Static indoor histories with an interval-selected outdoor channel."""

    def __init__(
        self,
        database: PairDatabase,
        series_indices: list[int],
        normalizer: Normalizer,
    ) -> None:
        self.database = database
        self.normalizer = normalizer
        self.windows = [
            (series_index, start)
            for series_index in series_indices
            for start in database.series[series_index].window_starts
        ]
        if not self.windows:
            raise ValueError("dataset split contains no eligible windows")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        series_index, start = self.windows[index]
        series = self.database.series[series_index]
        end = start + self.database.history_hours
        raw = series.values[start:end]
        observed = np.isfinite(raw)
        values = np.zeros_like(raw, dtype=np.float32)
        for channel in range(2):
            values[observed[:, channel], channel] = (
                raw[observed[:, channel], channel] - self.normalizer.mean[channel]
            ) / self.normalizer.standard_deviation[channel]
        timestamps = series.start_time + np.arange(start, end, dtype=np.int64) * HOUR
        return {
            "values": torch.from_numpy(values),
            "observed": torch.from_numpy(observed),
            "time_features": torch.from_numpy(_time_features(timestamps)),
            "pair_index": torch.tensor(series_index, dtype=torch.int64),
            "indoor_sensor_id": torch.tensor(series.indoor_id, dtype=torch.int64),
            "outdoor_sensor_ids": torch.from_numpy(series.outdoor_sensor_ids[start:end]),
            "assignment_indices": torch.from_numpy(series.assignment_indices[start:end]),
            "start_time_utc": torch.tensor(int(timestamps[0]), dtype=torch.int64),
        }


def load_training_data(
    path: Path,
    responsiveness_tiers: set[str] | None = None,
) -> tuple[IntervalSelection, HistoryData]:
    """Load the complete interval and PM2.5 contract from one CSV."""
    _require_file(path, "masked-training CSV")
    csv.field_size_limit(sys.maxsize)
    records, assignment_ids = [], set()
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        _require_columns(reader.fieldnames, TRAINING_COLUMNS, path)
        for number, row in enumerate(reader, 2):
            try:
                indoor, outdoor = int(row["indoor_sensor_id"]), int(row["outdoor_sensor_id"])
                start, end = _timestamp(row["start_utc"]), _timestamp(row["end_utc"])
                tier = row["responsiveness_tier"].strip().lower()
                assignment_id = f"{indoor}-{outdoor}-{start}-{end}"
                if (
                    assignment_id in assignment_ids
                    or min(indoor, outdoor) < 1
                    or indoor == outdoor
                    or start >= end
                    or tier not in RESPONSIVENESS_TIERS
                ):
                    raise ValueError
                indoor_values = _read_sparse_values(row["indoor_pm25"], start, end)
                outdoor_values = _read_sparse_values(row["outdoor_pm25"], start, end)
            except (AttributeError, TypeError, ValueError) as error:
                raise ValueError(f"invalid masked-training row {number} in {path}") from error
            assignment_ids.add(assignment_id)
            records.append(
                (
                    TrainingInterval(assignment_id, indoor, outdoor, start, end),
                    tier,
                    indoor_values,
                    outdoor_values,
                )
            )
    if not records:
        raise ValueError("masked-training CSV contains no rows")
    records.sort(key=lambda item: (item[0].indoor_id, item[0].start_time, item[0].end_time))
    intervals = [record[0] for record in records]
    for previous, current in zip(intervals, intervals[1:]):
        if (
            previous.indoor_id == current.indoor_id
            and previous.end_time > current.start_time
        ):
            raise ValueError(
                f"overlapping training intervals for indoor sensor {current.indoor_id}"
            )
    indoor_count = len({item.indoor_id for item in intervals})
    if responsiveness_tiers:
        selected_records = [record for record in records if record[1] in responsiveness_tiers]
    else:
        selected_records = records
    if not selected_records:
        raise ValueError("no training intervals passed the responsiveness filter")
    values: dict[int, dict[int, float]] = {}
    for interval, _, indoor_values, outdoor_values in selected_records:
        _merge_values(values, interval.indoor_id, indoor_values)
        _merge_values(values, interval.outdoor_id, outdoor_values)
    selection = IntervalSelection(
        tuple(record[0] for record in selected_records),
        indoor_count,
        len(records) - len(selected_records),
    )
    return selection, HistoryData(values, sum(map(len, values.values())))


def build_database(
    selection: IntervalSelection,
    history: HistoryData,
    history_hours: int = 168,
    minimum_observed_hours: int = 144,
    stride_hours: int = 24,
) -> PairDatabase:
    if history_hours < 1 or stride_hours < 1:
        raise ValueError("history_hours and stride_hours must be positive")
    if not 1 <= minimum_observed_hours <= history_hours:
        raise ValueError("minimum_observed_hours must be within the history window")
    grouped: dict[int, list[tuple[int, TrainingInterval]]] = {}
    for index, interval in enumerate(selection.intervals):
        grouped.setdefault(interval.indoor_id, []).append((index, interval))
    series, summaries = [], []
    for indoor_id, indexed in sorted(grouped.items()):
        intervals = [item for _, item in indexed]
        start, end = intervals[0].start_time, intervals[-1].end_time
        length = (end - start) // HOUR
        values = np.full((length, 2), np.nan, dtype=np.float32)
        allowed = np.zeros(length, dtype=bool)
        outdoor_ids = np.zeros(length, dtype=np.int64)
        assignment_indices = np.full(length, -1, dtype=np.int64)
        indoor = history.values[indoor_id]
        for timestamp, value in indoor.items():
            if start <= timestamp < end:
                values[(timestamp - start) // HOUR, 1] = value
        for assignment_index, interval in indexed:
            left = (interval.start_time - start) // HOUR
            right = (interval.end_time - start) // HOUR
            allowed[left:right] = True
            outdoor_ids[left:right] = interval.outdoor_id
            assignment_indices[left:right] = assignment_index
            for timestamp, value in history.values[interval.outdoor_id].items():
                if interval.start_time <= timestamp < interval.end_time:
                    values[(timestamp - start) // HOUR, 0] = value
        handoffs = tuple(
            (second.start_time - start) // HOUR
            for first, second in zip(intervals, intervals[1:])
            if first.end_time == second.start_time and first.outdoor_id != second.outdoor_id
        )
        starts = _eligible_starts(
            np.isfinite(values), allowed, history_hours, minimum_observed_hours, stride_hours
        )
        summaries.append(
            {
                "indoor_sensor_id": indoor_id,
                "training_intervals": len(intervals),
                "assignment_ids": [item.assignment_id for item in intervals],
                "outdoor_sensor_ids": sorted({item.outdoor_id for item in intervals}),
                "indoor_observations": int(np.isfinite(values[:, 1])[allowed].sum()),
                "outdoor_observations": int(np.isfinite(values[:, 0])[allowed].sum()),
                "hard_gap_hours": int((~allowed).sum()),
                "outdoor_handoffs": len(handoffs),
                "eligible_windows": len(starts),
                "windows_crossing_handoffs": sum(
                    any(window < handoff < window + history_hours for handoff in handoffs)
                    for window in starts
                ),
            }
        )
        if starts:
            series.append(
                IndoorSeries(
                    indoor_id,
                    start,
                    values,
                    allowed,
                    outdoor_ids,
                    assignment_indices,
                    tuple(item.assignment_id for item in intervals),
                    handoffs,
                    tuple(starts),
                )
            )
    return PairDatabase(
        selection,
        tuple(series),
        tuple(summaries),
        history,
        history_hours,
        minimum_observed_hours,
        stride_hours,
    )


def split_series(
    database: PairDatabase, validation_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    indices = list(range(len(database.series)))
    if len(indices) < 2:
        raise ValueError("at least two indoor sensors with eligible windows are required")
    random.Random(seed).shuffle(indices)
    validation_count = min(len(indices) - 1, max(1, round(len(indices) * validation_fraction)))
    return indices[validation_count:], indices[:validation_count]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_sparse_values(text: str, start: int, end: int) -> dict[int, float]:
    values: dict[int, float] = {}
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError
    interval_hours = (end - start) // HOUR
    for item in parsed:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError
        offset, value = item
        if (
            not isinstance(offset, int)
            or isinstance(offset, bool)
            or not 0 <= offset < interval_hours
            or offset in values
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value < 0
            or not math.isfinite(value)
        ):
            raise ValueError
        values[offset] = float(value)
    return {start + offset * HOUR: value for offset, value in values.items()}


def _merge_values(
    histories: dict[int, dict[int, float]],
    sensor_id: int,
    readings: dict[int, float],
) -> None:
    selected = histories.setdefault(sensor_id, {})
    for timestamp, value in readings.items():
        previous = selected.get(timestamp)
        if previous is not None and not math.isclose(previous, value, abs_tol=1e-6):
            raise ValueError(
                f"conflicting PM2.5 value for sensor {sensor_id} at {timestamp}"
            )
        selected[timestamp] = value


def _eligible_starts(
    observed: np.ndarray,
    allowed: np.ndarray,
    history_hours: int,
    minimum_observed_hours: int,
    stride_hours: int,
) -> list[int]:
    if len(observed) < history_hours:
        return []
    cumulative = np.vstack((np.zeros((1, 2), dtype=np.int64), observed.cumsum(axis=0)))
    starts = range(0, len(observed) - history_hours + 1, stride_hours)
    return [
        start
        for start in starts
        if allowed[start : start + history_hours].all()
        and np.all(cumulative[start + history_hours] - cumulative[start] >= minimum_observed_hours)
    ]


def _timestamp(text: str) -> int:
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("training interval timestamps must use UTC")
    timestamp = int(parsed.timestamp())
    if timestamp % HOUR:
        raise ValueError("training interval timestamps must be exact UTC hours")
    return timestamp


def _time_features(timestamps: np.ndarray) -> np.ndarray:
    hours = timestamps.astype(np.float64) / HOUR
    phases = (hours / 24.0, hours / (24.0 * 7.0), hours / (24.0 * 365.2425))
    return np.stack(
        tuple(
            value
            for phase in phases
            for value in (np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase))
        ),
        axis=1,
    ).astype(np.float32)


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def _require_columns(actual: list[str] | None, required: tuple[str, ...], path: Path) -> None:
    missing = set(required) - set(actual or ())
    if missing:
        raise ValueError(f"missing columns {sorted(missing)} in {path}")
