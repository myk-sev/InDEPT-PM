from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


HOUR = 3600
INTERVAL_COLUMNS = (
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
HISTORY_COLUMNS = ("time_stamp", "sensor_index", "pm2.5_atm")
RESPONSIVENESS_COLUMNS = (
    "indoor_sensor_id",
    "outdoor_sensor_id",
    "responsiveness_tier",
)
SCHOOL_COHORT_SOURCES = {"fema_school", "smoke_overlap_school"}


@dataclass(frozen=True)
class TrainingInterval:
    assignment_id: str
    indoor_id: int
    indoor_name: str
    outdoor_id: int
    outdoor_name: str
    distance_meters: float
    candidate_rank: int
    start_time: int
    end_time: int
    cohort_sources: str
    selection_reason: str


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
    files: tuple[Path, ...]
    row_count: int


@dataclass(frozen=True)
class IndoorSeries:
    indoor_id: int
    indoor_name: str
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


def load_training_intervals(
    path: Path,
    responsiveness_path: Path | None = None,
    responsiveness_tiers: set[str] | None = None,
) -> IntervalSelection:
    _require_file(path, "training interval CSV")
    pair_tiers = _read_responsiveness(responsiveness_path) if responsiveness_tiers else {}
    intervals, assignment_ids = [], set()
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        _require_columns(reader.fieldnames, INTERVAL_COLUMNS, path)
        for number, row in enumerate(reader, 2):
            try:
                assignment_id = row["assignment_id"].strip()
                indoor, outdoor = int(row["indoor_sensor_id"]), int(row["outdoor_sensor_id"])
                distance, rank = float(row["distance_meters"]), int(row["candidate_rank"])
                start, end = _timestamp(row["start_utc"]), _timestamp(row["end_utc"])
                sources = row["cohort_sources"].strip()
                expected_assignment = f"{indoor}-{outdoor}-{start}-{end}"
                expected_reason = (
                    "nearest_outdoor_sensor"
                    if rank == 1
                    else "fallback_due_to_exclusion"
                )
                if (
                    not assignment_id
                    or assignment_id != expected_assignment
                    or assignment_id in assignment_ids
                    or min(indoor, outdoor, rank) < 1
                    or indoor == outdoor
                    or distance < 0
                    or not math.isfinite(distance)
                    or start >= end
                    or not set(sources.split(";")) & SCHOOL_COHORT_SOURCES
                    or row["selection_reason"].strip() != expected_reason
                ):
                    raise ValueError
            except (AttributeError, TypeError, ValueError) as error:
                raise ValueError(f"invalid training interval row {number} in {path}") from error
            assignment_ids.add(assignment_id)
            intervals.append(
                TrainingInterval(
                    assignment_id,
                    indoor,
                    row["indoor_name"].strip(),
                    outdoor,
                    row["outdoor_name"].strip(),
                    distance,
                    rank,
                    start,
                    end,
                    sources,
                    row["selection_reason"].strip(),
                )
            )
    if not intervals:
        raise ValueError("training interval CSV contains no rows")
    intervals.sort(key=lambda item: (item.indoor_id, item.start_time, item.end_time))
    for previous, current in zip(intervals, intervals[1:]):
        if (
            previous.indoor_id == current.indoor_id
            and previous.end_time > current.start_time
        ):
            raise ValueError(f"overlapping training intervals for indoor sensor {current.indoor_id}")
    indoor_count = len({item.indoor_id for item in intervals})
    if responsiveness_tiers:
        selected = tuple(
            item
            for item in intervals
            if pair_tiers.get((item.indoor_id, item.outdoor_id)) in responsiveness_tiers
        )
        filtered = len(intervals) - len(selected)
    else:
        selected, filtered = tuple(intervals), 0
    if not selected:
        raise ValueError("no training intervals passed the responsiveness filter")
    return IntervalSelection(selected, indoor_count, filtered)


def load_interval_metadata(
    metadata_path: Path,
    interval_path: Path,
    exclusion_paths: tuple[Path, ...],
) -> dict[str, object]:
    _require_file(metadata_path, "training interval metadata")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    if metadata.get("schema_version") != 1:
        raise ValueError("unsupported training interval metadata schema")
    if metadata.get("manifest", {}).get("sha256") != file_sha256(interval_path):
        raise ValueError("training interval manifest hash does not match its metadata")
    recorded = {
        str(Path(row["path"]).resolve()): row["sha256"]
        for row in metadata.get("exclusions", ())
    }
    current = {str(path.resolve()): file_sha256(path) for path in exclusion_paths}
    if recorded != current:
        raise ValueError("training intervals are stale: reviewed exclusion hashes changed")
    return metadata


def read_purpleair_history(roots: list[Path], sensor_ids: set[int]) -> HistoryData:
    if not roots:
        raise ValueError("at least one PurpleAir history path is required")
    files: set[Path] = set()
    for root in roots:
        if root.is_file():
            files.add(root)
        elif root.is_dir():
            files.update(
                path
                for path in root.rglob("*.csv")
                if path.stem.split("_", 1)[0].isdigit()
                and int(path.stem.split("_", 1)[0]) in sensor_ids
            )
        else:
            raise FileNotFoundError(f"PurpleAir history path not found: {root}")
    values = {sensor_id: {} for sensor_id in sensor_ids}
    rows = 0
    for path in sorted(files):
        prefix = path.stem.split("_", 1)[0]
        expected = int(prefix) if prefix.isdigit() else None
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            _require_columns(reader.fieldnames, HISTORY_COLUMNS, path)
            for number, row in enumerate(reader, 2):
                try:
                    sensor_id = int(row["sensor_index"])
                    if expected is not None and sensor_id != expected:
                        raise ValueError
                    if sensor_id not in sensor_ids:
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
                existing = values[sensor_id].get(timestamp)
                if existing is not None and not math.isclose(existing, value, abs_tol=1e-6):
                    raise ValueError(
                        f"conflicting PurpleAir value for sensor {sensor_id} at "
                        f"{timestamp} in {path}"
                    )
                values[sensor_id][timestamp] = value
                rows += existing is None
    return HistoryData(values, tuple(sorted(files)), rows)


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
                "indoor_name": intervals[0].indoor_name,
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
                    intervals[0].indoor_name,
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


def history_inventory_sha256(files: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in files:
        stat = path.stat()
        digest.update(f"{path.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def _read_responsiveness(path: Path | None) -> dict[tuple[int, int], str]:
    if path is None:
        raise ValueError("a responsiveness CSV is required when tiers are selected")
    _require_file(path, "pair responsiveness CSV")
    tiers, valid = {}, {"high", "moderate", "low", "unclassified"}
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        _require_columns(reader.fieldnames, RESPONSIVENESS_COLUMNS, path)
        for number, row in enumerate(reader, 2):
            try:
                key = int(row["indoor_sensor_id"]), int(row["outdoor_sensor_id"])
                tier = row["responsiveness_tier"].strip().lower()
                if min(key) < 1 or key in tiers or tier not in valid:
                    raise ValueError
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid or duplicate responsiveness row {number}") from error
            tiers[key] = tier
    if not tiers:
        raise ValueError("pair responsiveness CSV contains no rows")
    return tiers


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
