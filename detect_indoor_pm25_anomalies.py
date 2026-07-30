from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_COLUMNS = ("time_stamp", "sensor_index", "pm2.5_atm")
OUTPUT_COLUMNS = (
    "sensor_id",
    "anomalous_hours",
    "maximum_pm25_ug_m3",
    "first_anomaly_utc",
    "last_anomaly_utc",
    "criterion",
)


def history_files(paths: list[Path]) -> list[Path]:
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


def detect_anomalous_sensors(
    paths: list[Path], maximum_pm25: float
) -> list[dict[str, object]]:
    if maximum_pm25 <= 0 or not math.isfinite(maximum_pm25):
        raise ValueError("maximum_pm25 must be positive and finite")
    anomalies: dict[int, dict[int, float]] = {}
    for path in history_files(paths):
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"missing columns in {path}: {', '.join(sorted(missing))}")
            for row_number, row in enumerate(reader, 2):
                text = (row["pm2.5_atm"] or "").strip().lower()
                if text in {"", "null", "nan"}:
                    continue
                try:
                    sensor = int(row["sensor_index"])
                    timestamp = int(row["time_stamp"])
                    value = float(text)
                    if (
                        sensor < 1
                        or timestamp % 3600
                        or value < 0
                        or not math.isfinite(value)
                    ):
                        raise ValueError
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"invalid PurpleAir row {row_number} in {path}"
                    ) from error
                if value > maximum_pm25:
                    previous = anomalies.setdefault(sensor, {}).get(timestamp)
                    if previous is not None and not math.isclose(
                        previous, value, abs_tol=1e-6
                    ):
                        raise ValueError(
                            f"conflicting duplicate sensor-hour in {path}"
                        )
                    anomalies[sensor][timestamp] = value

    criterion = f"pm2.5_atm>{maximum_pm25:g}"
    return [
        {
            "sensor_id": sensor,
            "anomalous_hours": len(values),
            "maximum_pm25_ug_m3": max(values.values()),
            "first_anomaly_utc": iso_utc(min(values)),
            "last_anomaly_utc": iso_utc(max(values)),
            "criterion": criterion,
        }
        for sensor, values in sorted(anomalies.items())
    ]


def iso_utc(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List indoor PurpleAir sensors with implausibly high hourly PM2.5."
    )
    parser.add_argument(
        "--indoor-history", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-pm25", type=float, default=1000.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    files = history_files(args.indoor_history)
    rows = detect_anomalous_sensors(args.indoor_history, args.maximum_pm25)
    write_report(args.output, rows)
    print(
        f"files={len(files)} excluded_sensors={len(rows)} "
        f"anomalous_hours={sum(int(row['anomalous_hours']) for row in rows)} "
        f"maximum_pm25={args.maximum_pm25:g} output={args.output}"
    )


if __name__ == "__main__":
    main()
