from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset, Subset


OBSERVATION_COLUMNS = (
    "location_id",
    "time_utc",
    "outdoor_pm25_ug_m3",
    "indoor_pm25_ug_m3",
)
FORECAST_COLUMNS = (
    "location_id",
    "cycle_time_utc",
    "forecast_hour",
    "valid_time_utc",
    "pm25_corrected_ug_m3",
)


@dataclass(frozen=True)
class DualEncoderLoaders:
    train: DataLoader
    validation: DataLoader
    temporal_test: DataLoader
    location_test: DataLoader


class DualEncoderDataset(Dataset):
    """Hourly history, forecast, and future-indoor-PM2.5 windows.

    History features are outdoor PM2.5, indoor PM2.5, normalized UTC hour,
    weekday, month, and day. Forecasts and targets begin one hour after the
    anchor. Only rows with complete history, forecast, and target windows are
    indexed.
    """

    def __init__(
        self,
        observations_path: str | Path,
        forecast_root: str | Path,
        history_hours: int = 168,
        forecast_hours: int = 55,
    ) -> None:
        if history_hours < 1 or forecast_hours < 1:
            raise ValueError("history_hours and forecast_hours must be positive")

        self.history_hours = history_hours
        self.forecast_hours = forecast_hours
        location_ids, timestamps, observations = _read_observations(Path(observations_path))
        cycles, forecasts = _read_forecasts(Path(forecast_root), location_ids)

        if forecast_hours > forecasts.shape[2]:
            raise ValueError(
                f"forecast_hours={forecast_hours} exceeds the available "
                f"{forecasts.shape[2]} forecast hours"
            )

        self.location_ids = tuple(location_ids)
        self.timestamps = torch.from_numpy(timestamps)
        self.observations = torch.from_numpy(observations)
        self.forecasts = torch.from_numpy(forecasts)
        self.time_features = torch.from_numpy(_calendar_features(timestamps))
        self._steps = len(timestamps)
        self._cycles = cycles
        self._anchor_cycles, self._anchor_leads = _align_forecast_cycles(
            timestamps, cycles, forecasts.shape[2], forecast_hours
        )
        self._sample_codes = _valid_sample_codes(
            observations,
            forecasts,
            self._anchor_cycles,
            self._anchor_leads,
            history_hours,
            forecast_hours,
        )
        if not len(self._sample_codes):
            raise ValueError("no complete history, forecast, and target windows were found")

    def __len__(self) -> int:
        return len(self._sample_codes)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        code = int(self._sample_codes[index])
        location = code // self._steps
        anchor = code % self._steps
        cycle = int(self._anchor_cycles[anchor])
        lead = int(self._anchor_leads[anchor])
        history_start = anchor - self.history_hours + 1
        target_end = anchor + self.forecast_hours + 1

        history = torch.cat(
            (
                self.observations[location, history_start : anchor + 1],
                self.time_features[history_start : anchor + 1],
            ),
            dim=1,
        )
        forecast = self.forecasts[
            cycle, location, lead : lead + self.forecast_hours
        ].unsqueeze(1)
        target = self.observations[location, anchor + 1 : target_end, 1]

        return {
            "history": history,
            "forecast": forecast,
            "target": target,
            "location_index": torch.tensor(location, dtype=torch.int64),
            "anchor_time_utc": self.timestamps[anchor],
        }


def create_data_loaders(
    dataset: DualEncoderDataset,
    batch_size: int = 64,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    location_holdout_fraction: float = 0.20,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DualEncoderLoaders:
    """Split by location and time without overlapping future labels."""
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers cannot be negative")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("train_fraction and validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be less than 1")
    if not 0 < location_holdout_fraction < 1:
        raise ValueError("location_holdout_fraction must be between 0 and 1")

    codes = dataset._sample_codes
    locations = codes // dataset._steps
    anchors = codes % dataset._steps
    unique_anchors = np.unique(anchors)
    if len(unique_anchors) < 3:
        raise ValueError("at least three eligible anchor times are required")

    validation_position = int(len(unique_anchors) * train_fraction)
    test_position = int(
        len(unique_anchors) * (train_fraction + validation_fraction)
    )
    if validation_position == 0 or test_position <= validation_position:
        raise ValueError("the temporal fractions leave an empty split")
    validation_start = unique_anchors[validation_position]
    test_start = unique_anchors[min(test_position, len(unique_anchors) - 1)]

    generator = np.random.default_rng(seed)
    location_order = generator.permutation(len(dataset.location_ids))
    held_out_count = round(len(location_order) * location_holdout_fraction)
    held_out_count = min(max(held_out_count, 1), len(location_order) - 1)
    held_out = np.zeros(len(location_order), dtype=bool)
    held_out[location_order[:held_out_count]] = True
    seen = ~held_out[locations]
    unseen = held_out[locations]
    target_ends = anchors + dataset.forecast_hours

    split_indices = (
        np.flatnonzero(seen & (target_ends <= validation_start)),
        np.flatnonzero(
            seen
            & (anchors >= validation_start)
            & (target_ends <= test_start)
        ),
        np.flatnonzero(seen & (anchors >= test_start)),
        np.flatnonzero(unseen & (anchors >= test_start)),
    )
    if any(not len(indices) for indices in split_indices):
        counts = ", ".join(str(len(indices)) for indices in split_indices)
        raise ValueError(
            "one or more data splits are empty; add data or adjust the split "
            f"fractions (train, validation, temporal test, location test: {counts})"
        )

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    torch_generator = torch.Generator().manual_seed(seed)
    return DualEncoderLoaders(
        train=DataLoader(
            Subset(dataset, split_indices[0]),
            shuffle=True,
            generator=torch_generator,
            **common,
        ),
        validation=DataLoader(Subset(dataset, split_indices[1]), **common),
        temporal_test=DataLoader(Subset(dataset, split_indices[2]), **common),
        location_test=DataLoader(Subset(dataset, split_indices[3]), **common),
    )


def _read_observations(
    path: Path,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"observation Parquet not found: {path}")
    schema = pq.read_schema(path)
    _require_columns(schema, OBSERVATION_COLUMNS, path)
    if not pa.types.is_string(schema.field("location_id").type):
        raise ValueError("location_id must be a string column")
    _require_utc_timestamp(schema.field("time_utc").type, "time_utc")
    for column in OBSERVATION_COLUMNS[2:]:
        if not pa.types.is_floating(schema.field(column).type):
            raise ValueError(f"{column} must be a floating-point column")

    table = pq.read_table(path, columns=list(OBSERVATION_COLUMNS)).combine_chunks()
    location_values = table["location_id"].to_pylist()
    if any(not value for value in location_values):
        raise ValueError("location_id cannot contain null or empty values")
    times = _timestamp_seconds(table["time_utc"], schema.field("time_utc").type)
    outdoor = table["outdoor_pm25_ug_m3"].to_numpy(zero_copy_only=False)
    indoor = table["indoor_pm25_ug_m3"].to_numpy(zero_copy_only=False)
    location_ids = sorted(set(location_values))
    location_lookup = {value: index for index, value in enumerate(location_ids)}
    locations = np.fromiter(
        (location_lookup[value] for value in location_values),
        dtype=np.int32,
        count=len(location_values),
    )

    order = np.lexsort((times, locations))
    if np.any(
        (locations[order][1:] == locations[order][:-1])
        & (times[order][1:] == times[order][:-1])
    ):
        raise ValueError("duplicate observation rows for the same location and hour")

    timestamps = np.unique(times)
    if len(timestamps) < 2 or np.any(np.diff(timestamps) != 3600):
        raise ValueError("time_utc must form a continuous hourly UTC grid")
    expected_rows = len(location_ids) * len(timestamps)
    if len(table) != expected_rows:
        raise ValueError(
            "every location must contain one row for every hour in the shared UTC grid"
        )
    time_indices = (times - timestamps[0]) // 3600
    if np.any(timestamps[time_indices] != times):
        raise ValueError("every observation timestamp must lie on the shared hourly grid")

    observations = np.full(
        (len(location_ids), len(timestamps), 2), np.nan, dtype=np.float32
    )
    observations[locations, time_indices, 0] = outdoor
    observations[locations, time_indices, 1] = indoor
    return location_ids, timestamps.astype(np.int64), observations


def _read_forecasts(
    root: Path, location_ids: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    files = [root] if root.is_file() else sorted(root.rglob("*.parquet"))
    files = [path for path in files if path.name != "locations.parquet"]
    if not files:
        raise FileNotFoundError(f"no forecast Parquet files found under: {root}")

    lookup = {value: index for index, value in enumerate(location_ids)}
    cycles: dict[int, np.ndarray] = {}
    max_hour = 0
    found_locations = np.zeros(len(location_ids), dtype=bool)

    for path in files:
        schema = pq.read_schema(path)
        _require_columns(schema, FORECAST_COLUMNS, path)
        if not pa.types.is_string(schema.field("location_id").type):
            raise ValueError("location_id must be a string column")
        if not pa.types.is_integer(schema.field("forecast_hour").type):
            raise ValueError("forecast_hour must be an integer column")
        if not pa.types.is_floating(schema.field("pm25_corrected_ug_m3").type):
            raise ValueError("pm25_corrected_ug_m3 must be a floating-point column")
        _require_utc_timestamp(schema.field("cycle_time_utc").type, "cycle_time_utc")
        _require_utc_timestamp(schema.field("valid_time_utc").type, "valid_time_utc")
        table = pq.read_table(path, columns=list(FORECAST_COLUMNS)).combine_chunks()
        ids = table["location_id"].to_pylist()
        cycle_times = _timestamp_seconds(
            table["cycle_time_utc"], schema.field("cycle_time_utc").type
        )
        valid_times = _timestamp_seconds(
            table["valid_time_utc"], schema.field("valid_time_utc").type
        )
        hours = table["forecast_hour"].to_numpy(zero_copy_only=False).astype(np.int64)
        values = table["pm25_corrected_ug_m3"].to_numpy(
            zero_copy_only=False
        ).astype(np.float32)
        if np.any((hours < 1) | (hours > 72)):
            raise ValueError("forecast_hour must be between 1 and 72")
        if np.any(valid_times != cycle_times + hours * 3600):
            raise ValueError(
                f"valid_time_utc does not match cycle_time_utc + forecast_hour in {path}"
            )
        max_hour = max(max_hour, int(hours.max(initial=0)))
        row_locations = np.fromiter(
            (lookup.get(value, -1) for value in ids),
            dtype=np.int32,
            count=len(ids),
        )

        for cycle in np.unique(cycle_times):
            rows = (cycle_times == cycle) & (row_locations >= 0)
            if not np.any(rows):
                continue
            cycle = int(cycle)
            required_hours = int(hours[rows].max())
            current = cycles.get(cycle)
            if current is None:
                current = np.full(
                    (len(location_ids), required_hours), np.nan, dtype=np.float32
                )
                cycles[cycle] = current
            elif current.shape[1] < required_hours:
                current = np.pad(
                    current,
                    ((0, 0), (0, required_hours - current.shape[1])),
                    constant_values=np.nan,
                )
                cycles[cycle] = current

            locations = row_locations[rows]
            leads = hours[rows] - 1
            flat = locations * current.shape[1] + leads
            if len(np.unique(flat)) != len(flat):
                raise ValueError(f"duplicate forecast rows in {path}")
            if np.any(np.isfinite(current[locations, leads])):
                raise ValueError(f"duplicate forecast rows across files at cycle {cycle}")
            current[locations, leads] = values[rows]
            found_locations[locations] = True

    missing = [
        location_ids[index]
        for index in np.flatnonzero(~found_locations)
    ]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"observation locations missing from forecasts: {preview}")
    if not cycles or max_hour < 1:
        raise ValueError("no usable forecast rows matched the observation locations")

    cycle_times = np.array(sorted(cycles), dtype=np.int64)
    forecasts = np.full(
        (len(cycle_times), len(location_ids), max_hour),
        np.nan,
        dtype=np.float32,
    )
    for index, cycle in enumerate(cycle_times):
        values = cycles[int(cycle)]
        forecasts[index, :, : values.shape[1]] = values
    return cycle_times, forecasts


def _align_forecast_cycles(
    timestamps: np.ndarray,
    cycles: np.ndarray,
    available_hours: int,
    forecast_hours: int,
) -> tuple[np.ndarray, np.ndarray]:
    anchor_cycles = np.full(len(timestamps), -1, dtype=np.int32)
    anchor_leads = np.full(len(timestamps), -1, dtype=np.int16)
    cycle_list = cycles.tolist()
    for anchor, timestamp in enumerate(timestamps):
        cycle = bisect_right(cycle_list, int(timestamp)) - 1
        if cycle < 0:
            continue
        elapsed = int((timestamp - cycles[cycle]) // 3600)
        if cycles[cycle] + elapsed * 3600 != timestamp:
            continue
        lead = elapsed
        if lead + forecast_hours <= available_hours:
            anchor_cycles[anchor] = cycle
            anchor_leads[anchor] = lead
    return anchor_cycles, anchor_leads


def _valid_sample_codes(
    observations: np.ndarray,
    forecasts: np.ndarray,
    anchor_cycles: np.ndarray,
    anchor_leads: np.ndarray,
    history_hours: int,
    forecast_hours: int,
) -> np.ndarray:
    locations, steps, _ = observations.shape
    complete_history = np.isfinite(observations).all(axis=2)
    complete_target = np.isfinite(observations[:, :, 1])
    history_missing = np.pad(
        (~complete_history).cumsum(axis=1, dtype=np.int32),
        ((0, 0), (1, 0)),
    )
    target_missing = np.pad(
        (~complete_target).cumsum(axis=1, dtype=np.int32),
        ((0, 0), (1, 0)),
    )

    blocks = []
    for anchor in range(history_hours - 1, steps - forecast_hours):
        cycle = anchor_cycles[anchor]
        if cycle < 0:
            continue
        lead = anchor_leads[anchor]
        history_ok = (
            history_missing[:, anchor + 1]
            - history_missing[:, anchor - history_hours + 1]
            == 0
        )
        target_ok = (
            target_missing[:, anchor + forecast_hours + 1]
            - target_missing[:, anchor + 1]
            == 0
        )
        forecast_ok = np.isfinite(
            forecasts[cycle, :, lead : lead + forecast_hours]
        ).all(axis=1)
        valid_locations = np.flatnonzero(history_ok & target_ok & forecast_ok)
        if len(valid_locations):
            blocks.append(valid_locations.astype(np.int64) * steps + anchor)
    return np.concatenate(blocks) if blocks else np.empty(0, dtype=np.int64)


def _calendar_features(timestamps: np.ndarray) -> np.ndarray:
    datetimes = timestamps.astype("datetime64[s]")
    days = datetimes.astype("datetime64[D]")
    months = datetimes.astype("datetime64[M]")
    hour = (datetimes - days).astype("timedelta64[h]").astype(np.float32) / 23
    weekday = ((days.astype(np.int64) + 3) % 7).astype(np.float32) / 6
    month = (months.astype(np.int64) % 12).astype(np.float32) / 11
    day = (days - months.astype("datetime64[D]")).astype(np.float32) / 30
    return np.column_stack((hour, weekday, month, day)).astype(np.float32)


def _require_columns(schema: pa.Schema, columns: tuple[str, ...], path: Path) -> None:
    missing = [column for column in columns if column not in schema.names]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def _require_utc_timestamp(data_type: pa.DataType, name: str) -> None:
    if not pa.types.is_timestamp(data_type) or data_type.tz != "UTC":
        raise ValueError(f"{name} must be a timezone-aware UTC timestamp")


def _timestamp_seconds(column: pa.ChunkedArray, data_type: pa.TimestampType) -> np.ndarray:
    raw = column.cast(pa.int64()).to_numpy(zero_copy_only=False)
    divisor = {"s": 1, "ms": 1_000, "us": 1_000_000, "ns": 1_000_000_000}[
        data_type.unit
    ]
    if np.any(raw % divisor):
        raise ValueError("timestamps must resolve to whole seconds")
    return (raw // divisor).astype(np.int64)
