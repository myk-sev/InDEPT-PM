"""Detailed NAQFC/TEMPO agreement, lead-time, and event-shape analysis."""

from __future__ import annotations

import argparse
import csv
import json
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.compute as pc
import pyarrow.parquet as pq

from check_agreement import (
    OUTPUT_COLUMNS,
    ROOT,
    UTC,
    forecast_cycles,
    match_forecasts,
    metrics,
    parse_time,
    read_pairs,
    read_sample,
    validate_locations,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


TEMPO_HISTORY = (
    ROOT.parent
    / "purple-air-pull"
    / "tempo_pm25_sensor_match"
    / "tempo_pm25_indoor_sensors.csv"
)
RANGES = (
    "<5",
    "5-<9.1",
    "9.1-<12",
    "12-<20",
    "20-<35.5",
    "35.5-<55.5",
    "55.5-<125.5",
    ">=125.5",
)
HIGH_RANGES = RANGES[-3:]
LEAD_BINS = (
    ("1-6", 1, 6),
    ("7-12", 7, 12),
    ("13-24", 13, 24),
    ("25-48", 25, 48),
    ("49-72", 49, 72),
)
SHAPE_COLUMNS = (
    "event_id",
    "sensor_id",
    "sensor_name",
    "location_id",
    "event_time_utc",
    "cycle_time_utc",
    "valid_time_utc",
    "forecast_hour",
    "tempo_pm25_ug_m3",
    "naqfc_pm25_ug_m3",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tempo-sample",
        type=Path,
        default=ROOT / "tempo_indoor_balanced_intersections.csv",
    )
    parser.add_argument(
        "--tempo-history",
        type=Path,
        default=TEMPO_HISTORY,
        help="Consolidated TEMPO history used only for selected event windows.",
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
    )
    parser.add_argument("--forecast-root", type=Path, default=ROOT / "naqfc_output")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument("--max-observations", type=int, default=200)
    parser.add_argument(
        "--shape-events",
        type=int,
        default=18,
        help="High-PM wildfire event windows, balanced across the three high ranges.",
    )
    return parser.parse_args()


def group_all_cycles(
    sample: list[dict[str, object]], cycles: dict[datetime, Path]
) -> dict[datetime, list[dict[str, object]]]:
    available = sorted(cycles)
    grouped: dict[datetime, list[dict[str, object]]] = defaultdict(list)
    for row in sample:
        timestamp = row["timestamp"]
        start = bisect_left(available, timestamp - timedelta(hours=72))
        end = bisect_left(available, timestamp)
        for cycle in available[start:end]:
            grouped[cycle].append(row)
    return grouped


def latest_forecasts(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    latest = {}
    for row in rows:
        key = (row["location_id"], row["timestamp_utc"])
        if key not in latest or row["cycle_time_utc"] > latest[key]["cycle_time_utc"]:
            latest[key] = row
    return list(latest.values())


def threshold_statistics(
    rows: list[dict[str, object]], threshold: float
) -> dict[str, object]:
    observed = np.array([row["tempo_pm25_ug_m3"] for row in rows]) >= threshold
    forecast = np.array([row["naqfc_pm25_ug_m3"] for row in rows]) >= threshold
    hits = int(np.sum(observed & forecast))
    misses = int(np.sum(observed & ~forecast))
    false_alarms = int(np.sum(~observed & forecast))
    correct_negatives = int(np.sum(~observed & ~forecast))

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    precision = ratio(hits, hits + false_alarms)
    recall = ratio(hits, hits + misses)
    return {
        "threshold_ug_m3": threshold,
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_negatives": correct_negatives,
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else None
        ),
        "critical_success_index": ratio(
            hits, hits + misses + false_alarms
        ),
    }


def grouped_metrics(
    rows: list[dict[str, object]], field: str, order: tuple[str, ...]
) -> dict[str, dict[str, object]]:
    return {
        value: metrics([row for row in rows if row[field] == value])
        for value in order
        if any(row[field] == value for row in rows)
    }


def lead_metrics(
    rows: list[dict[str, object]],
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    bins = {
        label: metrics(
            [
                row
                for row in rows
                if lower <= int(row["forecast_hour"]) <= upper
            ]
        )
        for label, lower, upper in LEAD_BINS
    }
    exact = {
        str(hour): metrics(
            [row for row in rows if int(row["forecast_hour"]) == hour]
        )
        for hour in range(1, 73)
        if any(int(row["forecast_hour"]) == hour for row in rows)
    }
    return bins, exact


def select_shape_events(
    sample: list[dict[str, object]], limit: int
) -> list[dict[str, object]]:
    groups = {
        value: sorted(
            [
                row
                for row in sample
                if row["data_context"] == "wildfire"
                and row["outdoor_range"] == value
            ],
            key=lambda row: row["tempo_pm25_ug_m3"],
            reverse=True,
        )
        for value in HIGH_RANGES
    }
    selected = []
    used_sensors = set()
    while len(selected) < limit:
        added = False
        for value in HIGH_RANGES:
            while groups[value] and groups[value][0]["sensor_id"] in used_sensors:
                groups[value].pop(0)
            if groups[value]:
                row = groups[value].pop(0)
                selected.append(row)
                used_sensors.add(row["sensor_id"])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
    return selected


def assign_shape_cycles(
    events: list[dict[str, object]], cycles: dict[datetime, Path]
) -> list[dict[str, object]]:
    available = sorted(cycles)
    assigned = []
    for event in events:
        event = dict(event)
        cutoff = event["timestamp"] - timedelta(hours=24)
        index = bisect_left(available, cutoff)
        candidates = available[: index + (index < len(available) and available[index] <= cutoff)]
        if not candidates:
            continue
        cycle = candidates[-1]
        if event["timestamp"] - cycle <= timedelta(hours=72):
            event["shape_cycle"] = cycle
            assigned.append(event)
    return assigned


def read_tempo_windows(
    path: Path, events: list[dict[str, object]]
) -> dict[str, dict[int, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"TEMPO history not found: {path}")
    windows: dict[int, list[dict[str, object]]] = defaultdict(list)
    for event in events:
        event["window_start"] = event["shape_cycle"] + timedelta(hours=1)
        event["window_end"] = event["shape_cycle"] + timedelta(hours=72)
        windows[event["sensor_id"]].append(event)

    values: dict[str, dict[int, float]] = {
        event["record_id"]: {} for event in events
    }
    reader = pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=64 * 1024 * 1024),
        convert_options=pacsv.ConvertOptions(
            include_columns=("sensor_id", "timestamp_utc", "tempo_pm25_ug_m3"),
            column_types={
                "sensor_id": pa.int64(),
                "timestamp_utc": pa.timestamp("s", tz="UTC"),
                "tempo_pm25_ug_m3": pa.float32(),
            },
        ),
    )
    requested = np.array(sorted(windows), dtype=np.int64)
    for batch in reader:
        sensors = batch["sensor_id"].to_numpy(zero_copy_only=False)
        selected = np.isin(sensors, requested)
        if not np.any(selected):
            continue
        times = (
            batch["timestamp_utc"]
            .cast(pa.int64())
            .to_numpy(zero_copy_only=False)[selected]
        )
        concentrations = batch["tempo_pm25_ug_m3"].to_numpy(
            zero_copy_only=False
        )[selected]
        for sensor, timestamp, concentration in zip(
            sensors[selected], times, concentrations
        ):
            observed_time = datetime.fromtimestamp(int(timestamp), UTC)
            for event in windows[int(sensor)]:
                if event["window_start"] <= observed_time <= event["window_end"]:
                    stored = values[event["record_id"]].get(int(timestamp))
                    value = float(concentration)
                    if stored is not None and not np.isclose(stored, value):
                        raise ValueError("conflicting duplicate TEMPO event value")
                    values[event["record_id"]][int(timestamp)] = value
    return values


def match_event_shapes(
    events: list[dict[str, object]],
    cycles: dict[datetime, Path],
    tempo: dict[str, dict[int, float]],
) -> list[dict[str, object]]:
    matches = []
    for event in events:
        cycle = event["shape_cycle"]
        table = pq.ParquetFile(cycles[cycle]).read(
            columns=(
                "location_id",
                "valid_time_utc",
                "forecast_hour",
                "pm25_corrected_ug_m3",
            )
        )
        selected = table.filter(pc.equal(table["location_id"], event["location_id"]))
        forecasts = {
            int(valid.timestamp()): (lead, float(value))
            for valid, lead, value in zip(
                selected["valid_time_utc"].to_pylist(),
                selected["forecast_hour"].to_pylist(),
                selected["pm25_corrected_ug_m3"].to_pylist(),
            )
        }
        for timestamp, observed in sorted(tempo[event["record_id"]].items()):
            forecast = forecasts.get(timestamp)
            if forecast is None:
                continue
            lead, value = forecast
            matches.append(
                {
                    "event_id": event["record_id"],
                    "sensor_id": event["sensor_id"],
                    "sensor_name": event["sensor_name"],
                    "location_id": event["location_id"],
                    "event_time_utc": event["timestamp"].isoformat().replace(
                        "+00:00", "Z"
                    ),
                    "cycle_time_utc": cycle.isoformat().replace("+00:00", "Z"),
                    "valid_time_utc": datetime.fromtimestamp(timestamp, UTC)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "forecast_hour": lead,
                    "tempo_pm25_ug_m3": observed,
                    "naqfc_pm25_ug_m3": value,
                }
            )
    return matches


def correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    return (
        float(np.corrcoef(first, second)[0, 1])
        if len(first) > 1 and first.std() and second.std()
        else None
    )


def event_shape_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    rows = sorted(rows, key=lambda row: row["forecast_hour"])
    hours = np.array([row["forecast_hour"] for row in rows], dtype=float)
    observed = np.array([row["tempo_pm25_ug_m3"] for row in rows], dtype=float)
    forecast = np.array([row["naqfc_pm25_ug_m3"] for row in rows], dtype=float)
    observed_slopes = np.diff(observed) / np.diff(hours)
    forecast_slopes = np.diff(forecast) / np.diff(hours)
    meaningful = np.abs(np.diff(observed)) >= 1
    observed_range = float(np.ptp(observed))
    return {
        "event_id": rows[0]["event_id"],
        "sensor_id": rows[0]["sensor_id"],
        "sensor_name": rows[0]["sensor_name"],
        "event_time_utc": rows[0]["event_time_utc"],
        "cycle_time_utc": rows[0]["cycle_time_utc"],
        "matched_points": len(rows),
        "tempo_peak_ug_m3": float(observed.max()),
        "naqfc_peak_at_tempo_times_ug_m3": float(forecast.max()),
        "level_pearson_r": correlation(observed, forecast),
        "change_pearson_r": correlation(observed_slopes, forecast_slopes),
        "direction_agreement_percent": (
            float(
                np.mean(
                    np.sign(observed_slopes[meaningful])
                    == np.sign(forecast_slopes[meaningful])
                )
                * 100
            )
            if np.any(meaningful)
            else None
        ),
        "peak_timing_error_hours": float(
            hours[int(np.argmax(forecast))] - hours[int(np.argmax(observed))]
        ),
        "amplitude_ratio": (
            float(np.ptp(forecast) / observed_range) if observed_range else None
        ),
        "mean_bias_ug_m3": float(np.mean(forecast - observed)),
        "mae_ug_m3": float(np.mean(np.abs(forecast - observed))),
    }


def aggregate_shapes(events: list[dict[str, object]]) -> dict[str, object]:
    usable = [event for event in events if event["matched_points"] >= 6]

    def median(field: str) -> float | None:
        values = [event[field] for event in usable if event[field] is not None]
        return float(np.median(values)) if values else None

    return {
        "events_with_at_least_6_tempo_points": len(usable),
        "matched_tempo_points": sum(event["matched_points"] for event in usable),
        "median_level_pearson_r": median("level_pearson_r"),
        "events_with_positive_level_correlation_percent": (
            float(
                np.mean(
                    [
                        event["level_pearson_r"] > 0
                        for event in usable
                        if event["level_pearson_r"] is not None
                    ]
                )
                * 100
            )
            if usable
            else None
        ),
        "median_change_pearson_r": median("change_pearson_r"),
        "median_direction_agreement_percent": median(
            "direction_agreement_percent"
        ),
        "median_absolute_peak_timing_error_hours": (
            float(
                np.median(
                    [abs(event["peak_timing_error_hours"]) for event in usable]
                )
            )
            if usable
            else None
        ),
        "median_amplitude_ratio": median("amplitude_ratio"),
        "median_event_bias_ug_m3": median("mean_bias_ug_m3"),
        "median_event_mae_ug_m3": median("mae_ug_m3"),
    }


def write_csv(path: Path, rows: list[dict[str, object]], columns) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot_regimes(
    path: Path, regimes: dict[str, dict[str, object]]
) -> None:
    labels = list(regimes)
    x = np.arange(len(labels))
    width = 0.36
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    axes[0].bar(
        x - width / 2,
        [regimes[label]["tempo_median_ug_m3"] for label in labels],
        width,
        label="TEMPO-derived",
    )
    axes[0].bar(
        x + width / 2,
        [regimes[label]["naqfc_median_ug_m3"] for label in labels],
        width,
        label="NAQFC",
    )
    axes[0].set_ylabel("Median PM2.5 (µg/m³)")
    axes[0].legend()
    axes[0].set_title("Latest-forecast agreement by TEMPO concentration regime")
    axes[1].bar(
        x - width / 2,
        [regimes[label]["mean_bias_ug_m3"] for label in labels],
        width,
        label="Mean bias",
    )
    axes[1].bar(
        x + width / 2,
        [regimes[label]["mae_ug_m3"] for label in labels],
        width,
        label="MAE",
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Error (µg/m³)")
    axes[1].legend()
    for axis in axes:
        axis.set_xticks(x, labels, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_horizon(
    path: Path, bins: dict[str, dict[str, object]]
) -> None:
    labels = list(bins)
    x = np.arange(len(labels))
    width = 0.36
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    axes[0].bar(
        x - width / 2,
        [bins[label]["mean_bias_ug_m3"] for label in labels],
        width,
        label="Mean bias",
    )
    axes[0].bar(
        x + width / 2,
        [bins[label]["mae_ug_m3"] for label in labels],
        width,
        label="MAE",
    )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Error (µg/m³)")
    axes[0].legend()
    axes[1].bar(
        x - width / 2,
        [bins[label]["pearson_r"] for label in labels],
        width,
        label="Pearson r",
    )
    axes[1].bar(
        x + width / 2,
        [bins[label]["fac2_percent"] / 100 for label in labels],
        width,
        label="FAC2 fraction",
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Agreement statistic")
    axes[1].legend()
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.set_xlabel("Forecast-hour bin")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Agreement across the 72-hour forecast")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_shapes(
    path: Path,
    rows: list[dict[str, object]],
    event_metrics: list[dict[str, object]],
) -> None:
    selected = sorted(
        event_metrics, key=lambda event: event["tempo_peak_ug_m3"], reverse=True
    )[:6]
    figure, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    for axis, event in zip(axes.flat, selected):
        event_rows = sorted(
            [row for row in rows if row["event_id"] == event["event_id"]],
            key=lambda row: row["forecast_hour"],
        )
        hours = [row["forecast_hour"] for row in event_rows]
        axis.plot(
            hours,
            [row["tempo_pm25_ug_m3"] for row in event_rows],
            marker="o",
            label="TEMPO-derived",
        )
        axis.plot(
            hours,
            [row["naqfc_pm25_ug_m3"] for row in event_rows],
            marker="o",
            label="NAQFC",
        )
        event_hour = (
            parse_time(event["event_time_utc"])
            - parse_time(event["cycle_time_utc"])
        ).total_seconds() / 3600
        axis.axvline(event_hour, color="black", linestyle=":", alpha=0.6)
        correlation_label = (
            f"{event['level_pearson_r']:.2f}"
            if event["level_pearson_r"] is not None
            else "n/a"
        )
        axis.set_title(
            f"{event['sensor_name']} | peak {event['tempo_peak_ug_m3']:.0f} | "
            f"r={correlation_label}"
        )
        axis.set_xlabel("Forecast hour")
        axis.set_ylabel("PM2.5 (µg/m³)")
        axis.grid(alpha=0.25)
    for axis in axes.flat[len(selected) :]:
        axis.set_visible(False)
    if selected:
        axes.flat[0].legend()
    figure.suptitle(
        "High-PM wildfire event shapes (points available from TEMPO)"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    pairs = read_pairs(args.pairs)
    sample, counts = read_sample(args.tempo_sample, pairs, args.max_observations)
    validate_locations(args.forecast_root, sample)
    cycles = forecast_cycles(args.forecast_root)

    print("Matching every available NAQFC lead to the balanced TEMPO sample...")
    all_matches = match_forecasts(group_all_cycles(sample, cycles), cycles)
    latest = latest_forecasts(all_matches)
    bins, exact = lead_metrics(all_matches)
    regimes = grouped_metrics(latest, "outdoor_range", RANGES)
    contexts = grouped_metrics(latest, "data_context", ("routine", "wildfire"))

    events = assign_shape_cycles(
        select_shape_events(sample, args.shape_events), cycles
    )
    print(
        f"Scanning TEMPO history for {len(events)} selected 72-hour event windows..."
    )
    tempo = read_tempo_windows(args.tempo_history, events)
    shape_rows = match_event_shapes(events, cycles, tempo)
    shape_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in shape_rows:
        shape_groups[row["event_id"]].append(row)
    event_metrics = [
        event_shape_metrics(rows) for rows in shape_groups.values() if len(rows) >= 2
    ]

    thresholds = (12.0, 35.5, 55.5, 125.5)
    summary = {
        "method": {
            "regimes": "Latest available NAQFC forecast for each sampled TEMPO hour.",
            "horizon": "Every archived forecast valid at each sampled TEMPO hour, grouped through 72 hours.",
            "shape": (
                f"{args.shape_events} wildfire events balanced across TEMPO "
                "ranges >=35.5; "
                "72-hour forecast issued at least 24 hours before the event, "
                "compared only at times with TEMPO retrievals."
            ),
        },
        "sample": {
            **counts,
            "latest_forecast_matches": len(latest),
            "all_lead_matches": len(all_matches),
            "locations": len({row["location_id"] for row in latest}),
            "start_utc": min(row["timestamp_utc"] for row in latest),
            "end_utc": max(row["timestamp_utc"] for row in latest),
        },
        "overall_latest_forecast": metrics(latest),
        "agreement_by_tempo_range": regimes,
        "agreement_by_context": contexts,
        "agreement_by_model_version": grouped_metrics(
            latest, "model_version", ("AQMv6", "AQMv7")
        ),
        "threshold_detection_latest_forecast": {
            str(threshold): threshold_statistics(latest, threshold)
            for threshold in thresholds
        },
        "overall_all_forecasts": metrics(all_matches),
        "agreement_by_forecast_hour_bin": bins,
        "agreement_by_forecast_hour": exact,
        "high_pm_detection_by_forecast_hour_bin": {
            label: {
                str(threshold): threshold_statistics(
                    [
                        row
                        for row in all_matches
                        if lower <= int(row["forecast_hour"]) <= upper
                    ],
                    threshold,
                )
                for threshold in (35.5, 55.5)
            }
            for label, lower, upper in LEAD_BINS
        },
        "wildfire_shape": {
            "events_requested": args.shape_events,
            "events_assigned": len(events),
            "aggregate": aggregate_shapes(event_metrics),
            "events": sorted(
                event_metrics,
                key=lambda event: event["tempo_peak_ug_m3"],
                reverse=True,
            ),
        },
        "interpretation_limits": [
            "TEMPO-ABI PM2.5 is a derived estimate, not independent ground truth.",
            "TEMPO retrievals are intermittent and daytime-weighted; shape metrics use matched times only.",
            "Repeated lead-time comparisons for one TEMPO hour are not statistically independent.",
            "The concentration- and context-balanced sample is not prevalence-weighted.",
        ],
    }

    write_csv(args.output / "all_lead_matches.csv", all_matches, OUTPUT_COLUMNS)
    write_csv(
        args.output / "horizon_metrics.csv",
        [{"forecast_hour": hour, **values} for hour, values in exact.items()],
        ("forecast_hour", *next(iter(exact.values())).keys()),
    )
    write_csv(args.output / "wildfire_shape_matches.csv", shape_rows, SHAPE_COLUMNS)
    if event_metrics:
        write_csv(
            args.output / "wildfire_shape_metrics.csv",
            event_metrics,
            event_metrics[0].keys(),
        )
    (args.output / "detailed_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    plot_regimes(args.output / "agreement_by_regime.png", regimes)
    plot_horizon(args.output / "agreement_by_horizon.png", bins)
    plot_shapes(
        args.output / "wildfire_event_shapes.png", shape_rows, event_metrics
    )
    print(json.dumps(summary["wildfire_shape"]["aggregate"], indent=2))
    print(f"Wrote detailed results to {args.output}")


if __name__ == "__main__":
    main()
