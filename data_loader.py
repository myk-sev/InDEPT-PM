from __future__ import annotations

import csv
import math
from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset, Subset


PAIR_COLUMNS = (
    "indoor_sensor_index",
    "outdoor_latitude",
    "outdoor_longitude",
)
BALANCER_PAIR_COLUMNS = ("sensor_id", "latitude", "longitude")
INDOOR_COLUMNS = ("time_stamp", "sensor_index", "pm2.5_atm")
OUTDOOR_COLUMNS = ("sensor_id", "timestamp_utc", "tempo_pm25_ug_m3")
BALANCE_COLUMNS = ("sensor_id", "timestamp_utc", "balance_cell")
FORECAST_COLUMNS = (
    "location_id",
    "cycle_time_utc",
    "forecast_hour",
    "valid_time_utc",
    "pm25_corrected_ug_m3",
)
REQUIRED_RECENT_OUTDOOR_HOURS = 3
SINGULAR_SPLITS = ("train", "validation", "temporal_test", "location_test")
LINEAR_TIME_FEATURES = ("hour", "weekday", "month", "day")
CYCLICAL_TIME_FEATURES = (
    "daily_sin",
    "daily_cos",
    "weekly_sin",
    "weekly_cos",
    "annual_sin",
    "annual_cos",
)
PERMANENT_EXCLUSIONS_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "exclusions"
    / "permanently_excluded_indoor_sensors.csv"
)


@dataclass(frozen=True)
class DualEncoderLoaders:
    train: DataLoader
    validation: DataLoader
    temporal_test: DataLoader
    location_test: DataLoader
    balance_report: dict[str, object] | None = None
    initial_training_exclusion_report: dict[str, int] | None = None


class DualEncoderDataset(Dataset):
    """Hourly history, forecast, and future-indoor-PM2.5 windows.

    History features are sparse TEMPO outdoor PM2.5, complete PurpleAir indoor
    PM2.5, and UTC time features. Forecasts and targets begin one hour after the
    anchor. Cyclical mode supplies daily, weekly, and annual sine/cosine pairs
    to both the history and forecast inputs.
    """

    def __init__(
        self,
        pairs_path: str | Path,
        indoor_history: str | Path | Iterable[str | Path],
        outdoor_history_path: str | Path,
        forecast_root: str | Path,
        history_hours: int = 168,
        forecast_hours: int = 36,
        minimum_outdoor_history_hours: int = 24,
        excluded_sensors_path: str | Path | None = None,
        cyclical_time: bool = False,
    ) -> None:
        if history_hours < 1 or forecast_hours < 1:
            raise ValueError("history_hours and forecast_hours must be positive")
        if not 1 <= minimum_outdoor_history_hours <= history_hours:
            raise ValueError(
                "minimum_outdoor_history_hours must be between 1 and history_hours"
            )
        if history_hours < REQUIRED_RECENT_OUTDOOR_HOURS:
            raise ValueError(
                f"history_hours must be at least {REQUIRED_RECENT_OUTDOOR_HOURS}"
            )

        self.history_hours = history_hours
        self.forecast_hours = forecast_hours
        self.minimum_outdoor_history_hours = minimum_outdoor_history_hours
        all_pairs = _read_pairs(Path(pairs_path))
        excluded = _read_excluded_sensors(PERMANENT_EXCLUSIONS_PATH)
        if excluded_sensors_path is not None:
            excluded |= _read_excluded_sensors(Path(excluded_sensors_path))
        pairs = [pair for pair in all_pairs if pair[1] not in excluded]
        if not pairs:
            raise ValueError("excluded sensor list removes every pair")
        location_ids = [pair[0] for pair in pairs]
        sensor_ids = [pair[1] for pair in pairs]
        indoor = _read_indoor_history(_paths(indoor_history), sensor_ids)
        outdoor = _read_outdoor_history(Path(outdoor_history_path), sensor_ids)
        timestamps, observations = _observations(sensor_ids, indoor, outdoor)
        cycles, forecasts = _read_forecasts(Path(forecast_root), location_ids)
        _validate_forecast_locations(Path(forecast_root), all_pairs)

        if forecast_hours > forecasts.shape[2]:
            raise ValueError(
                f"forecast_hours={forecast_hours} exceeds the available "
                f"{forecasts.shape[2]} forecast hours"
            )

        self.location_ids = tuple(location_ids)
        self.sensor_ids = tuple(sensor_ids)
        paired_sensors = {sensor for _, sensor, _, _ in all_pairs}
        self.excluded_sensor_ids = tuple(sorted(excluded & paired_sensors))
        self.timestamps = torch.from_numpy(timestamps)
        self.observations = torch.from_numpy(observations)
        self.forecasts = torch.from_numpy(forecasts)
        time_features = (
            _cyclical_time_features(timestamps)
            if cyclical_time
            else _calendar_features(timestamps)
        )
        self.time_features = torch.from_numpy(time_features)
        self.cyclical_time = cyclical_time
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
            minimum_outdoor_history_hours,
        )
        if not len(self._sample_codes):
            raise ValueError(
                "no windows met the indoor, TEMPO coverage, forecast, and target requirements"
            )

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
        if self.cyclical_time:
            forecast = torch.cat(
                (forecast, self.time_features[anchor + 1 : target_end]),
                dim=1,
            )
        target = self.observations[location, anchor + 1 : target_end, 1]

        return {
            "history": history,
            "forecast": forecast,
            "target": target,
            "location_index": torch.tensor(location, dtype=torch.int64),
            "anchor_time_utc": self.timestamps[anchor],
        }


class SingularTrainingDataset(Dataset):
    """Materialized windows and authoritative splits from one training CSV."""

    def __init__(
        self,
        path: str | Path,
        history_hours: int = 168,
        forecast_hours: int = 36,
        cyclical_time: bool = False,
    ) -> None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"singular training CSV not found: {path}")
        if history_hours < 1 or forecast_hours < 1:
            raise ValueError("history_hours and forecast_hours must be positive")

        time_features = (
            CYCLICAL_TIME_FEATURES if cyclical_time else LINEAR_TIME_FEATURES
        )
        history_features = (
            "tempo_pm25_ug_m3",
            "indoor_pm25_ug_m3",
            *time_features,
        )
        forecast_features = (
            ("naqfc_pm25_ug_m3", *time_features)
            if cyclical_time
            else ("naqfc_pm25_ug_m3",)
        )
        history_columns = tuple(
            f"history_{hour:03d}_{feature}"
            for hour in range(history_hours)
            for feature in history_features
        )
        forecast_columns = tuple(
            f"forecast_{hour:03d}_{feature}"
            for hour in range(1, forecast_hours + 1)
            for feature in forecast_features
        )
        target_columns = tuple(
            f"target_{hour:03d}_indoor_pm25_ug_m3"
            for hour in range(1, forecast_hours + 1)
        )
        metadata_columns = (
            "sample_index",
            "split",
            "location_id",
            "sensor_id",
            "model_name",
            "history_hours",
            "prediction_hours",
            "anchor_time_utc",
        )
        required = metadata_columns + history_columns + forecast_columns + target_columns
        with path.open(encoding="utf-8-sig", newline="") as source:
            fieldnames = next(csv.reader(source), None)
        _require_csv_columns(fieldnames, required, path)

        column_types = {column: pa.float32() for column in required}
        column_types.update(
            {
                "sample_index": pa.int64(),
                "split": pa.string(),
                "location_id": pa.string(),
                "sensor_id": pa.int64(),
                "model_name": pa.string(),
                "history_hours": pa.int32(),
                "prediction_hours": pa.int32(),
                "anchor_time_utc": pa.string(),
            }
        )
        try:
            table = pacsv.read_csv(
                path,
                convert_options=pacsv.ConvertOptions(
                    include_columns=list(required), column_types=column_types
                ),
            ).combine_chunks()
        except (pa.ArrowInvalid, pa.ArrowKeyError) as error:
            raise ValueError(f"invalid singular training CSV {path}: {error}") from error
        if not len(table):
            raise ValueError(f"singular training CSV contains no rows: {path}")

        stored_history_hours = table["history_hours"].to_numpy()
        stored_forecast_hours = np.unique(table["prediction_hours"].to_numpy())
        if np.any(stored_history_hours != history_hours):
            raise ValueError(
                f"singular training CSV history_hours does not match {history_hours}"
            )
        if len(stored_forecast_hours) != 1 or stored_forecast_hours[0] < forecast_hours:
            raise ValueError(
                "singular training CSV prediction_hours is shorter than or "
                f"inconsistent with {forecast_hours}"
            )

        histories = np.column_stack(
            [table[column].to_numpy(zero_copy_only=False) for column in history_columns]
        ).reshape(len(table), history_hours, len(history_features))
        forecasts = np.column_stack(
            [table[column].to_numpy(zero_copy_only=False) for column in forecast_columns]
        ).reshape(len(table), forecast_hours, len(forecast_features))
        targets = np.column_stack(
            [table[column].to_numpy(zero_copy_only=False) for column in target_columns]
        )
        if (
            np.isinf(histories[..., 0]).any()
            or np.any(histories[..., 0][np.isfinite(histories[..., 0])] < 0)
            or not np.isfinite(histories[..., 1:]).all()
            or not np.isfinite(forecasts).all()
            or not np.isfinite(targets).all()
            or np.any(histories[..., 1] < 0)
            or np.any(targets < 0)
        ):
            raise ValueError(f"singular training CSV contains invalid model values: {path}")

        sample_ids = table["sample_index"].to_numpy()
        if len(np.unique(sample_ids)) != len(sample_ids):
            raise ValueError(f"singular training CSV has duplicate sample_index values: {path}")
        splits = table["split"].to_pylist()
        if any(split not in SINGULAR_SPLITS for split in splits):
            raise ValueError(f"singular training CSV contains an invalid split: {path}")
        self.split_indices = {
            split: np.flatnonzero(np.asarray(splits) == split)
            for split in SINGULAR_SPLITS
        }
        if any(not len(indices) for indices in self.split_indices.values()):
            raise ValueError(f"singular training CSV contains an empty split: {path}")

        locations = table["location_id"].to_pylist()
        sensors = table["sensor_id"].to_numpy()
        location_lookup: dict[str, int] = {}
        sensor_by_location: dict[str, int] = {}
        sensor_locations: dict[int, str] = {}
        location_indices = []
        for location, sensor in zip(locations, sensors):
            sensor = int(sensor)
            if not location or sensor < 1:
                raise ValueError(f"singular training CSV has invalid sensor identity: {path}")
            if location in sensor_by_location and sensor_by_location[location] != sensor:
                raise ValueError(
                    "singular training CSV maps one location to multiple sensors: "
                    f"{path}"
                )
            if sensor in sensor_locations and sensor_locations[sensor] != location:
                raise ValueError(
                    "singular training CSV maps one sensor to multiple locations: "
                    f"{path}"
                )
            sensor_by_location[location] = sensor
            sensor_locations[sensor] = location
            location_indices.append(location_lookup.setdefault(location, len(location_lookup)))

        anchor_times = np.array(
            [_utc_hour(value, path) for value in table["anchor_time_utc"].to_pylist()],
            dtype=np.int64,
        )
        if len(set(zip(locations, anchor_times.tolist()))) != len(anchor_times):
            raise ValueError(f"singular training CSV has duplicate sensor-hour windows: {path}")
        model_names = set(table["model_name"].to_pylist())
        if len(model_names) != 1 or not next(iter(model_names)):
            raise ValueError(f"singular training CSV has inconsistent model_name values: {path}")

        self.path = path
        self.sample_ids = sample_ids
        self.history_hours = history_hours
        self.forecast_hours = forecast_hours
        self.source_forecast_hours = int(stored_forecast_hours[0])
        self.cyclical_time = cyclical_time
        self.source_model_name = next(iter(model_names))
        self.location_ids = tuple(location_lookup)
        self.sensor_ids = tuple(sensor_by_location[location] for location in self.location_ids)
        self.excluded_sensor_ids: tuple[int, ...] = ()
        self.histories = torch.from_numpy(histories)
        self.forecasts = torch.from_numpy(forecasts)
        self.targets = torch.from_numpy(targets)
        self.location_indices = torch.tensor(location_indices, dtype=torch.int64)
        self.anchor_times = torch.from_numpy(anchor_times)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "history": self.histories[index],
            "forecast": self.forecasts[index],
            "target": self.targets[index],
            "location_index": self.location_indices[index],
            "anchor_time_utc": self.anchor_times[index],
        }


def create_singular_data_loaders(
    dataset: SingularTrainingDataset,
    batch_size: int = 64,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    initial_training_interval_audit: str | Path | None = None,
    training_data_sha256: str | None = None,
) -> DualEncoderLoaders:
    """Create loaders from the split labels stored in the singular CSV."""
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers cannot be negative")
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    generator = torch.Generator().manual_seed(seed)
    train_indices = dataset.split_indices["train"]
    exclusion_report = None
    if initial_training_interval_audit is not None:
        excluded = _read_initial_training_interval_audit(
            Path(initial_training_interval_audit), dataset, training_data_sha256
        )
        train_indices = train_indices[~np.isin(train_indices, excluded)]
        if not len(train_indices):
            raise ValueError("initial training interval audit excludes every train sample")
        exclusion_report = {
            "eligible_training_intervals": len(dataset.split_indices["train"]),
            "excluded_training_intervals": len(excluded),
            "retained_training_intervals": len(train_indices),
        }
    loader = lambda split, **options: DataLoader(
        Subset(dataset, train_indices if split == "train" else dataset.split_indices[split]),
        **common,
        **options,
    )
    return DualEncoderLoaders(
        train=loader("train", shuffle=True, generator=generator),
        validation=loader("validation"),
        temporal_test=loader("temporal_test"),
        location_test=loader("location_test"),
        initial_training_exclusion_report=exclusion_report,
    )


def _read_initial_training_interval_audit(
    path: Path,
    dataset: SingularTrainingDataset,
    training_data_sha256: str | None,
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"initial training interval audit not found: {path}")
    required = {
        "sample_index",
        "split",
        "training_data_sha256",
        "exclude_from_initial_training",
    }
    decisions: dict[int, bool] = {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                "initial training interval audit is missing columns: "
                + ", ".join(sorted(missing))
            )
        for number, row in enumerate(reader, 2):
            try:
                sample_id = int(row["sample_index"])
                decision = row["exclude_from_initial_training"].strip().lower()
                if row["split"] != "train" or decision not in {"true", "false"}:
                    raise ValueError
                if training_data_sha256 and row["training_data_sha256"] != training_data_sha256:
                    raise ValueError("training data SHA-256 mismatch")
                if sample_id in decisions:
                    raise ValueError("duplicate sample_index")
            except (AttributeError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid initial training interval audit row {number} in {path}: {error}"
                ) from error
            decisions[sample_id] = decision == "true"

    sample_to_row = {int(sample_id): row for row, sample_id in enumerate(dataset.sample_ids)}
    expected = {
        int(dataset.sample_ids[row]) for row in dataset.split_indices["train"]
    }
    if decisions.keys() != expected:
        raise ValueError(
            "initial training interval audit sample_index values do not exactly match "
            "the singular CSV train split"
        )
    return np.array(
        [sample_to_row[sample_id] for sample_id, exclude in decisions.items() if exclude],
        dtype=np.int64,
    )


def create_data_loaders(
    dataset: DualEncoderDataset,
    batch_size: int = 64,
    train_fraction: float = 0.75,
    validation_fraction: float = 0.15,
    location_holdout_fraction: float = 0.20,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    balanced_training_index: str | Path | None = None,
) -> DualEncoderLoaders:
    """Split without label overlap and optionally balance only training."""
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
    eligible_locations = np.unique(locations)
    location_order = generator.permutation(eligible_locations)
    held_out_count = round(len(location_order) * location_holdout_fraction)
    held_out_count = min(max(held_out_count, 1), len(location_order) - 1)
    held_out = np.zeros(len(dataset.location_ids), dtype=bool)
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

    balance_report = None
    if balanced_training_index is not None:
        train_indices, balance_report = _balanced_training_indices(
            dataset, split_indices[0], Path(balanced_training_index)
        )
        split_indices = (train_indices, *split_indices[1:])

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
        balance_report=balance_report,
    )


def _balanced_training_indices(
    dataset: DualEncoderDataset, train_indices: np.ndarray, path: Path
) -> tuple[np.ndarray, dict[str, object]]:
    records = _read_balance_index(path)
    first_timestamp = int(dataset.timestamps[0])
    locations = {sensor: index for index, sensor in enumerate(dataset.sensor_ids)}
    records_by_code = {}
    for (sensor, timestamp), record in records.items():
        location = locations.get(sensor)
        anchor, remainder = divmod(timestamp - first_timestamp, 3600)
        if location is not None and not remainder and 0 <= anchor < dataset._steps:
            records_by_code[location * dataset._steps + anchor] = record

    record_codes = np.fromiter(records_by_code, dtype=np.int64)
    valid_count = int(np.isin(record_codes, dataset._sample_codes).sum())
    groups = {cell: [] for cell in sorted({cell for cell, _ in records.values()})}
    for index in train_indices:
        code = int(dataset._sample_codes[index])
        record = records_by_code.get(code)
        if record is not None:
            cell, rank = record
            groups[cell].append((rank, code, int(index)))

    missing = [cell for cell, rows in groups.items() if not rows]
    if missing:
        raise ValueError(
            "balanced training cells have no eligible training windows: "
            + ", ".join(missing)
        )

    quota = min(map(len, groups.values()))
    selected = [
        index
        for rows in groups.values()
        for _, _, index in sorted(rows)[:quota]
    ]
    return np.array(sorted(selected), dtype=np.int64), {
        "requested_anchors": len(records),
        "valid_anchors": valid_count,
        "training_eligible_anchors": sum(map(len, groups.values())),
        "selected_training_anchors": len(selected),
        "quota_per_cell": quota,
        "eligible_cell_counts": {
            cell: len(rows) for cell, rows in groups.items()
        },
    }


def _read_balance_index(path: Path) -> dict[tuple[int, int], tuple[str, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"balanced training index not found: {path}")
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        _require_csv_columns(reader.fieldnames, BALANCE_COLUMNS, path)
        records = {}
        for number, row in enumerate(reader, 2):
            selected = (row.get("selected") or "true").strip().lower()
            if selected in {"false", "0", "no"}:
                continue
            if selected not in {"true", "1", "yes"}:
                raise ValueError(f"invalid selected value on row {number} in {path}")
            try:
                sensor = int(row["sensor_id"])
                value = datetime.fromisoformat(
                    row["timestamp_utc"].replace("Z", "+00:00")
                )
                if (
                    value.tzinfo is None
                    or value.utcoffset() is None
                    or value.utcoffset().total_seconds() != 0
                ):
                    raise ValueError
                timestamp = int(value.timestamp())
                cell = row["balance_cell"].strip()
                rank = int(row.get("selection_rank") or 0)
                if timestamp % 3600 or not cell or rank < 0:
                    raise ValueError
            except (AttributeError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid balanced training row {number} in {path}"
                ) from error
            key = (sensor, timestamp)
            if key in records:
                raise ValueError(f"duplicate balanced training anchor in {path}")
            records[key] = (cell, rank)
    if not records:
        raise ValueError(f"balanced training index contains no selected rows: {path}")
    return records


def _paths(value: str | Path | Iterable[str | Path]) -> list[Path]:
    values = [value] if isinstance(value, (str, Path)) else list(value)
    paths = [Path(item) for item in values]
    if not paths:
        raise ValueError("at least one indoor history path is required")
    return paths


def _utc_hour(value: str, path: Path) -> int:
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if (
            timestamp.tzinfo is None
            or timestamp.utcoffset() is None
            or timestamp.utcoffset().total_seconds() != 0
        ):
            raise ValueError
        seconds = int(timestamp.timestamp())
        if seconds % 3600:
            raise ValueError
        return seconds
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"invalid anchor_time_utc in {path}") from error


def _read_pairs(path: Path) -> list[tuple[str, int, float, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"pair CSV not found: {path}")
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = set(reader.fieldnames or ())
        if set(PAIR_COLUMNS) <= fields:
            sensor_column, latitude_column, longitude_column = PAIR_COLUMNS
        elif set(BALANCER_PAIR_COLUMNS) <= fields:
            sensor_column, latitude_column, longitude_column = BALANCER_PAIR_COLUMNS
        else:
            _require_csv_columns(reader.fieldnames, PAIR_COLUMNS, path)
        pairs = []
        seen = set()
        for number, row in enumerate(reader, 1):
            try:
                sensor = int(row[sensor_column])
                latitude = float(row[latitude_column])
                longitude = float(row[longitude_column])
                if sensor in seen or not (-90 <= latitude <= 90) or not (
                    -180 <= longitude <= 180
                ):
                    raise ValueError
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid or duplicate pair row {number} in {path}"
                ) from error
            seen.add(sensor)
            pairs.append(
                (f"location_{number:06d}", sensor, latitude, longitude)
            )
    if not pairs:
        raise ValueError(f"pair CSV contains no rows: {path}")
    return pairs


def _read_excluded_sensors(path: Path) -> set[int]:
    if not path.is_file():
        raise FileNotFoundError(f"excluded sensor CSV not found: {path}")
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        _require_csv_columns(reader.fieldnames, ("sensor_id",), path)
        sensors = set()
        for number, row in enumerate(reader, 2):
            try:
                sensor = int(row["sensor_id"])
                if sensor < 1 or sensor in sensors:
                    raise ValueError
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid or duplicate excluded sensor row {number} in {path}"
                ) from error
            sensors.add(sensor)
    return sensors


def _history_files(paths: list[Path]) -> list[Path]:
    files = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(
                file
                for file in path.rglob("*.csv")
                if file.stem.split("_", 1)[0].isdigit()
            )
        else:
            raise FileNotFoundError(f"indoor history path not found: {path}")
    if not files:
        raise FileNotFoundError("no PurpleAir hourly history CSV files were found")
    return sorted(set(files))


def _read_indoor_history(
    paths: list[Path], sensor_ids: list[int]
) -> dict[int, dict[int, float]]:
    requested = set(sensor_ids)
    values = {sensor: {} for sensor in sensor_ids}
    for path in _history_files(paths):
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            _require_csv_columns(reader.fieldnames, INDOOR_COLUMNS, path)
            for row_number, row in enumerate(reader, 2):
                try:
                    sensor = int(row["sensor_index"])
                    if sensor not in requested:
                        continue
                    timestamp = int(row["time_stamp"])
                    if timestamp % 3600:
                        raise ValueError
                    text = (row["pm2.5_atm"] or "").strip().lower()
                    if text in {"", "null", "nan"}:
                        continue
                    value = float(text)
                    if value < 0 or not math.isfinite(value):
                        raise ValueError
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"invalid PurpleAir row {row_number} in {path}"
                    ) from error
                _store(values[sensor], timestamp, value, path)
    return values


def _read_outdoor_history(
    path: Path, sensor_ids: list[int]
) -> dict[int, dict[int, float]]:
    files = (
        [path]
        if path.is_file()
        else sorted(path.rglob("tempo_pm25_*.csv")) if path.is_dir() else []
    )
    if not files:
        raise FileNotFoundError(f"TEMPO history CSV or directory not found: {path}")
    requested = np.array(sorted(sensor_ids), dtype=np.int64)
    values = {sensor: {} for sensor in sensor_ids}
    for file in files:
        try:
            batches = pacsv.open_csv(
                file,
                read_options=pacsv.ReadOptions(block_size=16 * 1024 * 1024),
                convert_options=pacsv.ConvertOptions(
                    include_columns=list(OUTDOOR_COLUMNS),
                    column_types={
                        "sensor_id": pa.int64(),
                        "timestamp_utc": pa.timestamp("s", tz="UTC"),
                        "tempo_pm25_ug_m3": pa.float32(),
                    },
                ),
            )
            for batch in batches:
                sensors = batch["sensor_id"].to_numpy(zero_copy_only=False)
                selected = np.isin(sensors, requested)
                if not np.any(selected):
                    continue
                timestamps = (
                    batch["timestamp_utc"]
                    .cast(pa.int64())
                    .to_numpy(zero_copy_only=False)[selected]
                )
                concentrations = batch["tempo_pm25_ug_m3"].to_numpy(
                    zero_copy_only=False
                )[selected]
                for sensor, timestamp, concentration in zip(
                    sensors[selected], timestamps, concentrations
                ):
                    value = float(concentration)
                    if timestamp % 3600 or value < 0 or not math.isfinite(value):
                        raise ValueError("invalid TEMPO value or timestamp")
                    _store(values[int(sensor)], int(timestamp), value, file)
        except (pa.ArrowInvalid, pa.ArrowKeyError) as error:
            raise ValueError(f"invalid TEMPO history CSV {file}: {error}") from error
    return values


def _store(
    values: dict[int, float], timestamp: int, value: float, path: Path
) -> None:
    previous = values.get(timestamp)
    if previous is not None and not math.isclose(previous, value, abs_tol=1e-6):
        raise ValueError(f"conflicting duplicate sensor-hour in {path}")
    values[timestamp] = value


def _observations(
    sensor_ids: list[int],
    indoor: dict[int, dict[int, float]],
    outdoor: dict[int, dict[int, float]],
) -> tuple[np.ndarray, np.ndarray]:
    outdoor_times = [timestamp for rows in outdoor.values() for timestamp in rows]
    indoor_times = [timestamp for rows in indoor.values() for timestamp in rows]
    if not outdoor_times:
        raise ValueError("no TEMPO rows matched the paired indoor sensors")
    if not indoor_times:
        raise ValueError("no PurpleAir rows matched the paired indoor sensors")
    start = min(outdoor_times)
    end = max(indoor_times)
    if end < start:
        raise ValueError("PurpleAir history ends before TEMPO history begins")
    timestamps = np.arange(start, end + 3600, 3600, dtype=np.int64)
    observations = np.full(
        (len(sensor_ids), len(timestamps), 2), np.nan, dtype=np.float32
    )
    for location, sensor in enumerate(sensor_ids):
        for feature, source in enumerate((outdoor[sensor], indoor[sensor])):
            for timestamp, value in source.items():
                index, remainder = divmod(timestamp - start, 3600)
                if 0 <= index < len(timestamps) and not remainder:
                    observations[location, index, feature] = value
    return timestamps, observations


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


def _validate_forecast_locations(
    root: Path, pairs: list[tuple[str, int, float, float]]
) -> None:
    path = root / "locations.parquet"
    if not root.is_dir() or not path.is_file():
        return
    table = pq.read_table(
        path, columns=["location_id", "latitude", "longitude"]
    ).combine_chunks()
    expected = {
        location: (latitude, longitude)
        for location, _, latitude, longitude in pairs
    }
    actual = {
        location: (float(latitude), float(longitude))
        for location, latitude, longitude in zip(
            table["location_id"].to_pylist(),
            table["latitude"].to_pylist(),
            table["longitude"].to_pylist(),
        )
    }
    if actual.keys() != expected.keys() or any(
        not (
            math.isclose(actual[key][0], value[0], abs_tol=1e-7)
            and math.isclose(actual[key][1], value[1], abs_tol=1e-7)
        )
        for key, value in expected.items()
    ):
        raise ValueError("pair rows do not match the forecast locations.parquet")


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
    minimum_outdoor_history_hours: int,
) -> np.ndarray:
    locations, steps, _ = observations.shape
    outdoor_valid = np.isfinite(observations[:, :, 0])
    indoor_valid = np.isfinite(observations[:, :, 1])
    indoor_missing = np.pad(
        (~indoor_valid).cumsum(axis=1, dtype=np.int32),
        ((0, 0), (1, 0)),
    )
    outdoor_count = np.pad(
        outdoor_valid.cumsum(axis=1, dtype=np.int32),
        ((0, 0), (1, 0)),
    )
    target_missing = np.pad(
        (~indoor_valid).cumsum(axis=1, dtype=np.int32),
        ((0, 0), (1, 0)),
    )

    blocks = []
    for anchor in range(history_hours - 1, steps - forecast_hours):
        cycle = anchor_cycles[anchor]
        if cycle < 0:
            continue
        lead = anchor_leads[anchor]
        indoor_history_ok = (
            indoor_missing[:, anchor + 1]
            - indoor_missing[:, anchor - history_hours + 1]
            == 0
        )
        outdoor_history_ok = (
            outdoor_count[:, anchor + 1]
            - outdoor_count[:, anchor - history_hours + 1]
            >= minimum_outdoor_history_hours
        )
        recent_start = anchor - REQUIRED_RECENT_OUTDOOR_HOURS + 1
        outdoor_recent_ok = (
            outdoor_count[:, anchor + 1] - outdoor_count[:, recent_start]
            == REQUIRED_RECENT_OUTDOOR_HOURS
        )
        target_ok = (
            target_missing[:, anchor + forecast_hours + 1]
            - target_missing[:, anchor + 1]
            == 0
        )
        forecast_ok = np.isfinite(
            forecasts[cycle, :, lead : lead + forecast_hours]
        ).all(axis=1)
        valid_locations = np.flatnonzero(
            indoor_history_ok
            & outdoor_history_ok
            & outdoor_recent_ok
            & target_ok
            & forecast_ok
        )
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


def _cyclical_time_features(timestamps: np.ndarray) -> np.ndarray:
    datetimes = timestamps.astype("datetime64[s]")
    days = datetimes.astype("datetime64[D]")
    years = datetimes.astype("datetime64[Y]")
    hours = (datetimes - days).astype("timedelta64[s]").astype(np.float64) / 3600
    weekdays = ((days.astype(np.int64) + 3) % 7) + hours / 24
    next_years = (years.astype(np.int64) + 1).astype("datetime64[Y]")
    year_seconds = (
        next_years.astype("datetime64[s]") - years.astype("datetime64[s]")
    ).astype("timedelta64[s]").astype(np.float64)
    elapsed_seconds = (
        datetimes - years.astype("datetime64[s]")
    ).astype("timedelta64[s]").astype(np.float64)
    phases = 2 * np.pi * np.column_stack(
        (hours / 24, weekdays / 7, elapsed_seconds / year_seconds)
    )
    return np.stack((np.sin(phases), np.cos(phases)), axis=2).reshape(
        len(timestamps), -1
    ).astype(np.float32)


def _require_columns(schema: pa.Schema, columns: tuple[str, ...], path: Path) -> None:
    missing = [column for column in columns if column not in schema.names]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def _require_csv_columns(
    fieldnames: list[str] | None, columns: tuple[str, ...], path: Path
) -> None:
    missing = [column for column in columns if column not in (fieldnames or [])]
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
