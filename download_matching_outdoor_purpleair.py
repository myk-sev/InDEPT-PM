"""Download paired outdoor PurpleAir PM2.5 for existing indoor sensor-hours.

During execution, each outdoor sensor is checkpointed in
``OUTPUT_DIR/outdoor_sensor_history/<sensor_index>.csv``. After every request
succeeds, those files are retained as backups and atomically consolidated into
the requested final CSV. Status CSVs are retained beside the sensor histories.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_URL = "https://api.purpleair.com/v1/sensors/{sensor}/history"
HOUR = 3600
MAX_REQUEST_HOURS = 180 * 24
DEFAULT_FINAL_FILENAME = "outdoor_pm25.csv"
CHECKPOINT_DIRECTORY = "outdoor_sensor_history"
HISTORY_COLUMNS = ("time_stamp", "sensor_index", "pm2.5_atm")
STATUS_COLUMNS = (
    "start_timestamp",
    "end_timestamp",
    "target_hours",
    "returned_target_rows",
    "status",
)


@dataclass(frozen=True)
class Plan:
    sensor: int
    targets: frozenset[int]
    ranges: tuple[tuple[int, int], ...]
    downloaded: int
    attempted_without_data: int


class RateLimiter:
    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.next_request = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            time.sleep(max(0.0, self.next_request - time.monotonic()))
            self.next_request = time.monotonic() + self.interval


class Progress:
    def __init__(
        self,
        total_hours: int,
        completed_hours: int,
        total_requests: int,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not 0 <= completed_hours <= total_hours or total_requests < 0:
            raise ValueError("invalid progress totals")
        self.total_hours = total_hours
        self.completed_hours = completed_hours
        self.total_requests = total_requests
        self.completed_requests = 0
        self.run_completed_hours = 0
        self.clock = clock or time.monotonic
        self.started = self.clock()
        self.lock = threading.Lock()

    def report(self) -> None:
        with self.lock:
            self._report()

    def advance(self, hours: int) -> None:
        with self.lock:
            if hours < 1 or self.completed_hours + hours > self.total_hours:
                raise ValueError("invalid progress increment")
            self.completed_hours += hours
            self.run_completed_hours += hours
            self.completed_requests += 1
            self._report()

    def _report(self) -> None:
        elapsed = self.clock() - self.started
        remaining = self.total_hours - self.completed_hours
        if not remaining:
            eta = 0.0
        elif elapsed > 0 and self.run_completed_hours:
            eta = elapsed * remaining / self.run_completed_hours
        else:
            eta = None
        percentage = (
            100.0
            if not self.total_hours
            else 100 * self.completed_hours / self.total_hours
        )
        print(
            f"Progress: {percentage:6.2f}% | "
            f"target hours {self.completed_hours:,}/{self.total_hours:,} | "
            f"requests this run {self.completed_requests:,}/{self.total_requests:,} | "
            f"elapsed {format_duration(elapsed)} | "
            f"ETA {format_duration(eta) if eta is not None else 'calculating'}",
            flush=True,
        )


def format_duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs", type=Path, required=True, help="Indoor/outdoor pair CSV"
    )
    parser.add_argument(
        "--indoor-history",
        type=Path,
        action="append",
        required=True,
        help="Indoor history CSV or directory; may be repeated",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the final CSV and outdoor_sensor_history backups",
    )
    parser.add_argument(
        "--final-csv-name",
        type=final_csv_name,
        default=DEFAULT_FINAL_FILENAME,
        help=f"Final CSV filename within --output-dir (default: {DEFAULT_FINAL_FILENAME})",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Make paid API requests using PURPLEAIR_API_KEY from .env",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--request-delay", type=float, default=1.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    if (
        args.threads < 1
        or args.request_delay < 0
        or args.retries < 0
        or args.timeout <= 0
    ):
        parser.error(
            "threads and timeout must be positive; delay and retries cannot be negative"
        )
    return args


def final_csv_name(value: str) -> str:
    path = Path(value)
    if (
        path.name != value
        or path.suffix.lower() != ".csv"
        or path.stem.isdigit()
        or value.lower().endswith(".status.csv")
    ):
        raise argparse.ArgumentTypeError(
            "final CSV name must be a non-numeric .csv filename without a directory"
        )
    return value


def read_api_key(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"PurpleAir API key file not found: {path}")
    with path.open(encoding="utf-8-sig") as source:
        for line in source:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            if text.startswith("export "):
                text = text[7:].lstrip()
            name, separator, value = text.partition("=")
            if separator and name.strip() == "PURPLEAIR_API_KEY":
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                if value:
                    return value
                break
    raise SystemExit(f"PURPLEAIR_API_KEY is missing or empty in {path}")


def read_pairs(path: Path) -> dict[int, int]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = set(reader.fieldnames or ())
        columns = next(
            (
                columns
                for columns in (
                    ("indoor_sensor_index", "outdoor_sensor_index"),
                    ("indoor_sensor_id", "outdoor_sensor_id"),
                )
                if set(columns) <= fields
            ),
            None,
        )
        if columns is None:
            raise ValueError(f"missing indoor/outdoor sensor columns in {path}")
        indoor_column, outdoor_column = columns
        pairs: dict[int, int] = {}
        for line, row in enumerate(reader, 2):
            try:
                indoor = int(row[indoor_column])
                outdoor = int(row[outdoor_column])
                if indoor < 1 or outdoor < 1 or indoor in pairs:
                    raise ValueError
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid or duplicate pair at {path}:{line}") from error
            pairs[indoor] = outdoor
    if not pairs:
        raise ValueError(f"pair CSV contains no rows: {path}")
    return pairs


def history_files(paths: list[Path]) -> list[tuple[Path, bool]]:
    files: list[tuple[Path, bool]] = []
    for path in paths:
        if path.is_file():
            files.append((path, True))
        elif path.is_dir():
            files.extend(
                (file, False)
                for file in path.rglob("*.csv")
                if not file.name.endswith(".part")
            )
        else:
            raise FileNotFoundError(f"indoor history path not found: {path}")
    if not files:
        raise FileNotFoundError("no indoor history CSV files were found")
    return sorted(set(files))


def read_indoor_targets(
    paths: list[Path], sensors: set[int] | None = None
) -> dict[int, set[int]]:
    targets = {} if sensors is None else {sensor: set() for sensor in sensors}
    compatible_files = 0
    for path, explicit in history_files(paths):
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if not set(HISTORY_COLUMNS) <= set(reader.fieldnames or ()):
                if explicit:
                    require_columns(reader.fieldnames, HISTORY_COLUMNS, path)
                continue
            compatible_files += 1
            for line, row in enumerate(reader, 2):
                try:
                    sensor = int(row["sensor_index"])
                    text = (row["pm2.5_atm"] or "").strip().lower()
                    if text in {"", "null", "nan"}:
                        continue
                    timestamp, value = int(row["time_stamp"]), float(text)
                    if (
                        sensor < 1
                        or timestamp % HOUR
                        or value < 0
                        or not math.isfinite(value)
                    ):
                        raise ValueError
                except (TypeError, ValueError) as error:
                    raise ValueError(f"invalid indoor history row at {path}:{line}") from error
                if sensors is None or sensor in sensors:
                    targets.setdefault(sensor, set()).add(timestamp)
    if not compatible_files:
        raise ValueError("no CSV has the required indoor history columns")
    if not any(targets.values()):
        raise ValueError("indoor history contains no PM2.5 rows for paired sensors")
    return targets


def contiguous_ranges(timestamps: set[int]) -> tuple[tuple[int, int], ...]:
    if not timestamps:
        return ()
    ranges: list[tuple[int, int]] = []
    start = previous = min(timestamps)
    for timestamp in sorted(timestamps - {start}):
        if timestamp != previous + HOUR or timestamp - start >= MAX_REQUEST_HOURS * HOUR:
            ranges.append((start, previous + HOUR))
            start = timestamp
        previous = timestamp
    ranges.append((start, previous + HOUR))
    return tuple(ranges)


def read_download(path: Path, sensor: int) -> dict[int, float]:
    if not path.exists():
        return {}
    values: dict[int, float] = {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        require_columns(reader.fieldnames, HISTORY_COLUMNS, path)
        for line, row in enumerate(reader, 2):
            try:
                timestamp = int(row["time_stamp"])
                row_sensor = int(row["sensor_index"])
                value = float(row["pm2.5_atm"])
                if (
                    row_sensor != sensor
                    or timestamp % HOUR
                    or value < 0
                    or not math.isfinite(value)
                ):
                    raise ValueError
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid downloaded row at {path}:{line}") from error
            previous = values.setdefault(timestamp, value)
            if previous != value:
                raise ValueError(f"conflicting downloaded row at {path}:{line}")
    return values


def read_final_download(path: Path) -> dict[int, dict[int, float]]:
    if not path.exists():
        return {}
    values: dict[int, dict[int, float]] = {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        require_columns(reader.fieldnames, HISTORY_COLUMNS, path)
        for line, row in enumerate(reader, 2):
            try:
                timestamp = int(row["time_stamp"])
                sensor = int(row["sensor_index"])
                value = float(row["pm2.5_atm"])
                if (
                    sensor < 1
                    or timestamp % HOUR
                    or value < 0
                    or not math.isfinite(value)
                ):
                    raise ValueError
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid final output row at {path}:{line}") from error
            sensor_values = values.setdefault(sensor, {})
            previous = sensor_values.setdefault(timestamp, value)
            if previous != value:
                raise ValueError(f"conflicting final output row at {path}:{line}")
    return values


def read_attempted(path: Path, targets: frozenset[int]) -> set[int]:
    if not path.exists():
        return set()
    attempted: set[int] = set()
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        require_columns(reader.fieldnames, STATUS_COLUMNS, path)
        for line, row in enumerate(reader, 2):
            try:
                start, end = int(row["start_timestamp"]), int(row["end_timestamp"])
                target_hours = int(row["target_hours"])
                returned_rows = int(row["returned_target_rows"])
                if (
                    row["status"] not in {"complete", "no_data"}
                    or start % HOUR
                    or end % HOUR
                    or end <= start
                    or target_hours != (end - start) // HOUR
                    or not 0 <= returned_rows <= target_hours
                ):
                    raise ValueError
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid status row at {path}:{line}") from error
            attempted.update(timestamp for timestamp in targets if start <= timestamp < end)
    return attempted


def checkpoint_directory(output_dir: Path) -> Path:
    return output_dir / CHECKPOINT_DIRECTORY


def migrate_legacy_checkpoints(output_dir: Path) -> None:
    paths = [
        path
        for path in output_dir.glob("*.csv")
        if path.stem.isdigit()
        or path.name.removesuffix(".status.csv").isdigit()
    ]
    if not paths:
        return
    directory = checkpoint_directory(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for source in paths:
        destination = directory / source.name
        if destination.exists():
            if source.read_bytes() != destination.read_bytes():
                raise ValueError(f"conflicting checkpoint files: {source}, {destination}")
            source.unlink()
        else:
            source.replace(destination)


def make_plan(
    sensor: int,
    targets: set[int],
    output_dir: Path,
    final_values: dict[int, float] | None = None,
) -> Plan:
    frozen = frozenset(targets)
    downloaded_values = dict(final_values or {})
    directory = checkpoint_directory(output_dir)
    for timestamp, value in read_download(directory / f"{sensor}.csv", sensor).items():
        previous = downloaded_values.setdefault(timestamp, value)
        if previous != value:
            raise ValueError(f"conflicting saved values for outdoor sensor {sensor}")
    downloaded = set(downloaded_values) & frozen
    attempted = read_attempted(directory / f"{sensor}.status.csv", frozen) - downloaded
    pending = set(frozen) - downloaded - attempted
    return Plan(sensor, frozen, contiguous_ranges(pending), len(downloaded), len(attempted))


def fetch_history(
    sensor: int, start: int, end: int, api_key: str, timeout: float
) -> dict[int, float]:
    query = urlencode(
        {
            "start_timestamp": start,
            "end_timestamp": end,
            "average": 60,
            "fields": "pm2.5_atm",
        }
    )
    request = Request(
        f"{API_URL.format(sensor=sensor)}?{query}",
        headers={"X-API-Key": api_key, "User-Agent": "AirGuard outdoor PM2.5 matcher"},
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    fields = payload.get("fields")
    data = payload.get("data")
    if not isinstance(fields, list) or not isinstance(data, list):
        raise ValueError(f"unexpected PurpleAir response for sensor {sensor}")
    try:
        timestamp_index = fields.index("time_stamp")
        value_index = fields.index("pm2.5_atm")
    except ValueError as error:
        raise ValueError(
            f"PurpleAir response omitted requested fields for sensor {sensor}"
        ) from error
    values: dict[int, float] = {}
    for row in data:
        if row[value_index] is None:
            continue
        timestamp, value = int(row[timestamp_index]), float(row[value_index])
        if timestamp % HOUR or value < 0 or not math.isfinite(value):
            raise ValueError(f"invalid PurpleAir value for sensor {sensor}")
        previous = values.setdefault(timestamp, value)
        if previous != value:
            raise ValueError(
                f"conflicting PurpleAir values for sensor {sensor} at {timestamp}"
            )
    return values


def atomic_write(
    path: Path, columns: tuple[str, ...], rows: list[tuple[object, ...]]
) -> None:
    part = path.with_suffix(path.suffix + ".part")
    with part.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)
    part.replace(path)


def consolidate_downloads(
    output_dir: Path, final_filename: str = DEFAULT_FINAL_FILENAME
) -> tuple[Path, int, int, int]:
    final_path = output_dir / final_csv_name(final_filename)
    values = read_final_download(final_path)
    directory = checkpoint_directory(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    sensor_paths = sorted(
        path for path in directory.glob("*.csv") if path.stem.isdigit()
    )
    for path in sensor_paths:
        sensor = int(path.stem)
        sensor_values = values.setdefault(sensor, {})
        for timestamp, value in read_download(path, sensor).items():
            previous = sensor_values.setdefault(timestamp, value)
            if previous != value:
                raise ValueError(f"conflicting saved values for outdoor sensor {sensor}")
    values = {sensor: sensor_values for sensor, sensor_values in values.items() if sensor_values}
    rows = [
        (timestamp, sensor, sensor_values[timestamp])
        for sensor, sensor_values in sorted(values.items())
        for timestamp in sorted(sensor_values)
    ]
    atomic_write(final_path, HISTORY_COLUMNS, rows)
    if read_final_download(final_path) != values:
        raise ValueError(f"final output verification failed: {final_path}")
    for sensor, sensor_values in values.items():
        path = directory / f"{sensor}.csv"
        atomic_write(
            path,
            HISTORY_COLUMNS,
            [(timestamp, sensor, sensor_values[timestamp]) for timestamp in sorted(sensor_values)],
        )
        if read_download(path, sensor) != sensor_values:
            raise ValueError(f"sensor backup verification failed: {path}")
    return final_path, len(rows), len(values), len(values)


def download_plan(
    plan: Plan,
    output_dir: Path,
    api_key: str,
    limiter: RateLimiter,
    retries: int,
    timeout: float,
    progress: Progress | None = None,
    existing_values: dict[int, float] | None = None,
) -> tuple[int, int, int]:
    directory = checkpoint_directory(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    data_path = directory / f"{plan.sensor}.csv"
    status_path = directory / f"{plan.sensor}.status.csv"
    values = dict(existing_values or {})
    for timestamp, value in read_download(data_path, plan.sensor).items():
        previous = values.setdefault(timestamp, value)
        if previous != value:
            raise ValueError(f"conflicting saved values for outdoor sensor {plan.sensor}")
    statuses: list[tuple[object, ...]] = []
    if status_path.exists():
        with status_path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            require_columns(reader.fieldnames, STATUS_COLUMNS, status_path)
            statuses.extend(tuple(row[column] for column in STATUS_COLUMNS) for row in reader)
    returned = 0
    for start, end in plan.ranges:
        for attempt in range(retries + 1):
            try:
                limiter.wait()
                response = fetch_history(plan.sensor, start, end, api_key, timeout)
                break
            except Exception:
                if attempt == retries:
                    raise
                time.sleep(2**attempt)
        matches = {
            timestamp: value
            for timestamp, value in response.items()
            if timestamp in plan.targets
        }
        values.update(matches)
        atomic_write(
            data_path,
            HISTORY_COLUMNS,
            [(timestamp, plan.sensor, values[timestamp]) for timestamp in sorted(values)],
        )
        target_hours = (end - start) // HOUR
        statuses.append(
            (
                start,
                end,
                target_hours,
                len(matches),
                "complete" if matches else "no_data",
            )
        )
        atomic_write(status_path, STATUS_COLUMNS, statuses)
        returned += len(matches)
        if progress is not None:
            progress.advance(target_hours)
    return plan.sensor, len(plan.ranges), returned


def summarize(
    plans: list[Plan],
    all_indoor_targets: dict[int, set[int]],
    matched_indoor_targets: dict[int, set[int]],
    pairs: dict[int, int],
    mode: str,
) -> dict[str, object]:
    request_hours = sum(
        (end - start) // HOUR for plan in plans for start, end in plan.ranges
    )
    requests = sum(len(plan.ranges) for plan in plans)
    unpaired = sorted(set(all_indoor_targets) - set(pairs))
    return {
        "mode": mode,
        "input_indoor_sensors": len(all_indoor_targets),
        "input_indoor_sensor_hours": sum(map(len, all_indoor_targets.values())),
        "matched_indoor_sensors": len(matched_indoor_targets),
        "matched_indoor_sensor_hours": sum(map(len, matched_indoor_targets.values())),
        "unpaired_indoor_sensors": len(unpaired),
        "unpaired_indoor_sensor_hours": sum(
            len(all_indoor_targets[sensor]) for sensor in unpaired
        ),
        "unpaired_indoor_sensor_ids": unpaired,
        "paired_sensors_without_input_readings": len(set(pairs) - set(all_indoor_targets)),
        "outdoor_sensors": len(plans),
        "outdoor_sensor_hours_required": sum(len(plan.targets) for plan in plans),
        "already_downloaded": sum(plan.downloaded for plan in plans),
        "previously_attempted_without_data": sum(plan.attempted_without_data for plan in plans),
        "pending_target_hours": sum(
            len(plan.targets) - plan.downloaded - plan.attempted_without_data
            for plan in plans
        ),
        "requests": requests,
        "requested_span_hours": request_hours,
        "estimated_api_points_upper_bound": 2 * request_hours + 2 * requests,
    }


def require_columns(
    actual: list[str] | None, required: tuple[str, ...], path: Path
) -> None:
    missing = set(required) - set(actual or ())
    if missing:
        raise ValueError(f"missing columns {sorted(missing)} in {path}")


def main() -> None:
    args = arguments()
    if args.execute:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        migrate_legacy_checkpoints(args.output_dir)
    pairs = read_pairs(args.pairs)
    all_indoor_targets = read_indoor_targets(args.indoor_history)
    indoor_targets = {
        sensor: timestamps
        for sensor, timestamps in all_indoor_targets.items()
        if sensor in pairs
    }
    if not indoor_targets:
        raise ValueError("indoor history contains no sensors from the pair manifest")
    outdoor_targets: dict[int, set[int]] = {}
    for indoor, timestamps in indoor_targets.items():
        if timestamps:
            outdoor_targets.setdefault(pairs[indoor], set()).update(timestamps)
    final_values = read_final_download(args.output_dir / args.final_csv_name)
    plans = [
        make_plan(sensor, targets, args.output_dir, final_values.get(sensor))
        for sensor, targets in sorted(outdoor_targets.items())
    ]
    mode = "execute" if args.execute else "dry-run"
    print(
        json.dumps(
            summarize(plans, all_indoor_targets, indoor_targets, pairs, mode),
            indent=2,
        )
    )
    if not args.execute:
        return
    api_key = read_api_key(Path.cwd() / ".env")
    pending = [plan for plan in plans if plan.ranges]
    limiter = RateLimiter(args.request_delay)
    progress = Progress(
        total_hours=sum(len(plan.targets) for plan in plans),
        completed_hours=sum(
            plan.downloaded + plan.attempted_without_data for plan in plans
        ),
        total_requests=sum(len(plan.ranges) for plan in pending),
    )
    progress.report()
    failures = []
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(
                download_plan,
                plan,
                args.output_dir,
                api_key,
                limiter,
                args.retries,
                args.timeout,
                progress=progress,
                existing_values=final_values.get(plan.sensor),
            ): plan.sensor
            for plan in pending
        }
        for completed, future in enumerate(as_completed(futures), 1):
            sensor = futures[future]
            try:
                _, requests, rows = future.result()
                print(
                    f"[{completed}/{len(futures)}] sensor {sensor}: "
                    f"{requests} requests, {rows} rows"
                )
            except Exception as error:
                failures.append(sensor)
                print(f"[{completed}/{len(futures)}] sensor {sensor}: ERROR: {error}")
    if failures:
        raise SystemExit(f"downloads failed for outdoor sensors: {failures}")
    final_path, rows, sensors, backups = consolidate_downloads(
        args.output_dir, args.final_csv_name
    )
    print(
        f"Final CSV: {final_path} ({rows:,} rows, {sensors:,} sensors); "
        f"retained {backups:,} verified per-sensor backups in "
        f"{checkpoint_directory(args.output_dir)}"
    )


if __name__ == "__main__":
    main()
