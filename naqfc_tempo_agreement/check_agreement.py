"""Compare NAQFC forecasts with a balanced sample of TEMPO-derived PM2.5."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent.parent
UTC = timezone.utc
CYCLE_PATTERN = re.compile(r"naqfc_(\d{8}T\d{2})\.parquet$")
OUTPUT_COLUMNS = (
    "location_id",
    "sensor_id",
    "sensor_name",
    "timestamp_utc",
    "cycle_time_utc",
    "forecast_hour",
    "model_version",
    "tempo_pm25_ug_m3",
    "naqfc_pm25_ug_m3",
    "error_ug_m3",
    "absolute_error_ug_m3",
    "data_context",
    "outdoor_range",
    "pair_distance_km",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tempo-sample",
        type=Path,
        default=ROOT / "tempo_indoor_balanced_intersections.csv",
        help="Balanced TEMPO intersection CSV.",
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=(
            ROOT
            / "data"
            / "legacy"
            / "purpleair_continental_us_pairs_thinned_20km.csv"
        ),
        help="Indoor/outdoor pair CSV used to assign NAQFC location IDs.",
    )
    parser.add_argument(
        "--forecast-root",
        type=Path,
        default=ROOT / "naqfc_output",
        help="Root of the partitioned NAQFC Parquet archive.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory for matches.csv and summary.json.",
    )
    parser.add_argument(
        "--max-observations",
        type=int,
        default=200,
        help="Maximum TEMPO observations, sampled round-robin across balance cells.",
    )
    return parser.parse_args()


def read_pairs(path: Path) -> dict[int, dict[str, object]]:
    pairs = {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        for number, row in enumerate(csv.DictReader(source), 1):
            indoor_lat = float(row["indoor_latitude"])
            indoor_lon = float(row["indoor_longitude"])
            pairs[int(row["indoor_sensor_index"])] = {
                "location_id": f"location_{number:06d}",
                "outdoor_latitude": float(row["outdoor_latitude"]),
                "outdoor_longitude": float(row["outdoor_longitude"]),
                "pair_distance_km": float(row["distance_meters"]) / 1000,
                "indoor_latitude": indoor_lat,
                "indoor_longitude": indoor_lon,
            }
    return pairs


def read_sample(
    path: Path, pairs: dict[int, dict[str, object]], limit: int
) -> tuple[list[dict[str, object]], dict[str, int]]:
    if limit < 1:
        raise ValueError("--max-observations must be positive")
    cells: dict[str, list[dict[str, object]]] = defaultdict(list)
    counts = {"source_rows": 0, "unmapped_rows": 0}
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            if row.get("selected", "true").lower() not in {"true", "1", "yes"}:
                continue
            counts["source_rows"] += 1
            sensor_id = int(row["sensor_id"])
            pair = pairs.get(sensor_id)
            if pair is None:
                counts["unmapped_rows"] += 1
                continue
            cells[row["balance_cell"]].append(
                {
                    "record_id": row["record_id"],
                    "episode_id": row["episode_id"],
                    "sensor_id": sensor_id,
                    "sensor_name": row["sensor_name"],
                    "timestamp": parse_time(row["timestamp_utc"]),
                    "tempo_pm25_ug_m3": float(row["tempo_outdoor_pm25_ug_m3"]),
                    "data_context": row["data_context"],
                    "outdoor_range": row["outdoor_range"],
                    **pair,
                }
            )

    for rows in cells.values():
        rows.sort(key=lambda row: (row["timestamp"], row["sensor_id"]))
    sample = []
    while len(sample) < limit:
        added = False
        for cell in sorted(cells):
            if cells[cell]:
                sample.append(cells[cell].pop(0))
                added = True
                if len(sample) == limit:
                    break
        if not added:
            break
    counts["eligible_rows"] = sum(len(rows) for rows in cells.values()) + len(sample)
    counts["sampled_rows"] = len(sample)
    return sample, counts


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def forecast_cycles(root: Path) -> dict[datetime, Path]:
    cycles = {}
    for path in root.glob("model_version=*/year=*/month=*/*.parquet"):
        match = CYCLE_PATTERN.fullmatch(path.name)
        if match:
            cycle = datetime.strptime(match.group(1), "%Y%m%dT%H").replace(tzinfo=UTC)
            if cycle in cycles:
                raise ValueError(f"duplicate NAQFC cycle: {cycle.isoformat()}")
            cycles[cycle] = path
    if not cycles:
        raise FileNotFoundError(f"no partitioned NAQFC forecasts under {root}")
    return cycles


def choose_cycles(
    sample: list[dict[str, object]], cycles: dict[datetime, Path]
) -> dict[datetime, list[dict[str, object]]]:
    available = sorted(cycles)
    grouped: dict[datetime, list[dict[str, object]]] = defaultdict(list)
    for row in sample:
        timestamp = row["timestamp"]
        candidates = [
            cycle
            for cycle in available
            if cycle < timestamp and (timestamp - cycle).total_seconds() <= 72 * 3600
        ]
        if candidates:
            grouped[candidates[-1]].append(row)
    return grouped


def validate_locations(
    root: Path, sample: list[dict[str, object]]
) -> None:
    table = pq.ParquetFile(root / "locations.parquet").read()
    locations = {
        location_id: (latitude, longitude)
        for location_id, latitude, longitude in zip(
            table["location_id"].to_pylist(),
            table["latitude"].to_pylist(),
            table["longitude"].to_pylist(),
        )
    }
    for row in sample:
        actual = locations.get(row["location_id"])
        expected = (row["outdoor_latitude"], row["outdoor_longitude"])
        if actual is None or not np.allclose(actual, expected, atol=1e-6):
            raise ValueError(f"pair does not match locations.parquet: {row['location_id']}")


def match_forecasts(
    grouped: dict[datetime, list[dict[str, object]]],
    cycles: dict[datetime, Path],
) -> list[dict[str, object]]:
    matches = []
    for cycle, observations in sorted(grouped.items()):
        table = pq.ParquetFile(cycles[cycle]).read(
            columns=(
                "location_id",
                "valid_time_utc",
                "forecast_hour",
                "model_version",
                "pm25_corrected_ug_m3",
            )
        )
        location_ids = pa.array(sorted({row["location_id"] for row in observations}))
        timestamps = pa.array(
            sorted({row["timestamp"] for row in observations}),
            type=table["valid_time_utc"].type,
        )
        selected = table.filter(
            pc.and_(
                pc.is_in(table["location_id"], value_set=location_ids),
                pc.is_in(table["valid_time_utc"], value_set=timestamps),
            )
        )
        lookup = {
            (location_id, valid_time): (lead, version, value)
            for location_id, valid_time, lead, version, value in zip(
                selected["location_id"].to_pylist(),
                selected["valid_time_utc"].to_pylist(),
                selected["forecast_hour"].to_pylist(),
                selected["model_version"].to_pylist(),
                selected["pm25_corrected_ug_m3"].to_pylist(),
            )
        }
        for observation in observations:
            forecast = lookup.get(
                (observation["location_id"], observation["timestamp"])
            )
            if forecast is None:
                continue
            lead, version, value = forecast
            tempo = observation["tempo_pm25_ug_m3"]
            error = float(value) - float(tempo)
            matches.append(
                {
                    "location_id": observation["location_id"],
                    "sensor_id": observation["sensor_id"],
                    "sensor_name": observation["sensor_name"],
                    "timestamp_utc": observation["timestamp"].isoformat().replace(
                        "+00:00", "Z"
                    ),
                    "cycle_time_utc": cycle.isoformat().replace("+00:00", "Z"),
                    "forecast_hour": lead,
                    "model_version": version,
                    "tempo_pm25_ug_m3": tempo,
                    "naqfc_pm25_ug_m3": float(value),
                    "error_ug_m3": error,
                    "absolute_error_ug_m3": abs(error),
                    "data_context": observation["data_context"],
                    "outdoor_range": observation["outdoor_range"],
                    "pair_distance_km": observation["pair_distance_km"],
                }
            )
    return matches


def metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    observed = np.array([row["tempo_pm25_ug_m3"] for row in rows], dtype=float)
    forecast = np.array([row["naqfc_pm25_ug_m3"] for row in rows], dtype=float)
    errors = forecast - observed
    correlation = (
        float(np.corrcoef(observed, forecast)[0, 1])
        if len(rows) > 1 and observed.std() and forecast.std()
        else None
    )
    positive = (observed > 0) & (forecast > 0)
    denominator = observed + forecast
    fractional = denominator > 0
    return {
        "comparisons": len(rows),
        "tempo_mean_ug_m3": float(observed.mean()),
        "tempo_median_ug_m3": float(np.median(observed)),
        "naqfc_mean_ug_m3": float(forecast.mean()),
        "naqfc_median_ug_m3": float(np.median(forecast)),
        "mean_bias_ug_m3": float(errors.mean()),
        "mae_ug_m3": float(np.abs(errors).mean()),
        "median_absolute_error_ug_m3": float(np.median(np.abs(errors))),
        "rmse_ug_m3": float(np.sqrt(np.mean(errors**2))),
        "pearson_r": correlation,
        "fac2_percent": (
            float(
                np.mean(
                    (forecast[positive] / observed[positive] >= 0.5)
                    & (forecast[positive] / observed[positive] <= 2)
                )
                * 100
            )
            if np.any(positive)
            else None
        ),
        "mean_fractional_bias_percent": (
            float(np.mean(2 * errors[fractional] / denominator[fractional]) * 100)
            if np.any(fractional)
            else None
        ),
        "mean_fractional_error_percent": (
            float(
                np.mean(2 * np.abs(errors[fractional]) / denominator[fractional])
                * 100
            )
            if np.any(fractional)
            else None
        ),
        "within_5_ug_m3_percent": float(np.mean(np.abs(errors) <= 5) * 100),
        "within_10_ug_m3_percent": float(np.mean(np.abs(errors) <= 10) * 100),
    }


def summarize(
    rows: list[dict[str, object]], counts: dict[str, int]
) -> dict[str, object]:
    if not rows:
        raise ValueError("no NAQFC forecasts matched the sampled TEMPO observations")
    by_context = {
        context: metrics([row for row in rows if row["data_context"] == context])
        for context in sorted({row["data_context"] for row in rows})
    }
    by_range = {
        value: metrics([row for row in rows if row["outdoor_range"] == value])
        for value in sorted({row["outdoor_range"] for row in rows})
    }
    by_lead = {
        label: metrics(
            [
                row
                for row in rows
                if lower <= int(row["forecast_hour"]) <= upper
            ]
        )
        for label, lower, upper in (
            ("1-6", 1, 6),
            ("7-12", 7, 12),
            ("13-24", 13, 24),
            ("25-48", 25, 48),
            ("49-72", 49, 72),
        )
        if any(lower <= int(row["forecast_hour"]) <= upper for row in rows)
    }
    return {
        "method": (
            "Balanced TEMPO sample; latest archived NAQFC cycle strictly before "
            "each TEMPO valid hour, with a maximum 72-hour lead."
        ),
        "sample": {
            **counts,
            "matched_rows": len(rows),
            "locations": len({row["location_id"] for row in rows}),
            "balance_cells": len(
                {(row["data_context"], row["outdoor_range"]) for row in rows}
            ),
            "start_utc": min(row["timestamp_utc"] for row in rows),
            "end_utc": max(row["timestamp_utc"] for row in rows),
            "maximum_pair_distance_km": max(row["pair_distance_km"] for row in rows),
        },
        "overall": metrics(rows),
        "by_context": by_context,
        "by_tempo_range": by_range,
        "by_forecast_hour": by_lead,
        "interpretation_limits": [
            "TEMPO-ABI PM2.5 is a derived outdoor estimate, not an independent ground measurement.",
            "NAQFC is evaluated at the paired outdoor coordinate; TEMPO is evaluated at the indoor sensor coordinate.",
            "This concentration- and context-balanced sample is not prevalence-weighted.",
        ],
    }


def write_results(
    output: Path, rows: list[dict[str, object]], summary: dict[str, object]
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "matches.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["timestamp_utc"]))
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = arguments()
    pairs = read_pairs(args.pairs)
    sample, counts = read_sample(args.tempo_sample, pairs, args.max_observations)
    validate_locations(args.forecast_root, sample)
    cycles = forecast_cycles(args.forecast_root)
    matches = match_forecasts(choose_cycles(sample, cycles), cycles)
    summary = summarize(matches, counts)
    write_results(args.output, matches, summary)
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {args.output / 'matches.csv'}")
    print(f"Wrote {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
