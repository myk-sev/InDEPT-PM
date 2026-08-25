"""Balance model-eligible training anchors by outdoor PM2.5 and smoke context."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loader import (
    REQUIRED_RECENT_OUTDOOR_HOURS,
    DualEncoderDataset,
    create_data_loaders,
)


HOUR = 3600
STRATA = (
    ("<5", -math.inf, 5.0),
    ("5-<9.1", 5.0, 9.1),
    ("9.1-<12", 9.1, 12.0),
    ("12-<20", 12.0, 20.0),
    ("20-<35.5", 20.0, 35.5),
    ("35.5-<55.5", 35.5, 55.5),
    ("55.5-<125.5", 55.5, 125.5),
    (">=125.5", 125.5, math.inf),
)
CONTEXTS = ("routine", "wildfire")
RECORD_FIELDS = (
    "record_id",
    "location_id",
    "sensor_id",
    "timestamp_utc",
    "tempo_outdoor_pm25_ug_m3",
    "indoor_pm25_ug_m3",
    "data_context",
    "wildfire_event_ids",
    "episode_id",
    "outdoor_range",
    "balance_cell",
    "selected",
    "selection_rank",
)
CELL_FIELDS = (
    "data_context",
    "outdoor_range",
    "eligible_training_candidates",
    "independent_capacity",
    "selected_hours",
    "common_quota",
    "status",
)
OUTPUT_NAMES = (
    "eligible_training_candidates.csv",
    "balanced_training_index.csv",
    "balance_cells.csv",
    "report.md",
)


def parse_time(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())


def format_time(value: int) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def concentration_stratum(value: float) -> str:
    return next(name for name, low, high in STRATA if low <= value < high)


def stable_order(seed: int, *values: object) -> bytes:
    text = ":".join((str(seed), *(str(value) for value in values)))
    return hashlib.sha256(text.encode()).digest()


def read_wildfire_ranges(
    path: Path, sensor_ids: set[int]
) -> dict[int, list[dict[str, object]]]:
    ranges: dict[int, list[dict[str, object]]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = set(reader.fieldnames or ())
        start_field = (
            "smoke_start_time_utc"
            if "smoke_start_time_utc" in fields
            else "start_time_utc"
        )
        end_field = (
            "smoke_end_time_utc"
            if "smoke_end_time_utc" in fields
            else "end_time_utc"
        )
        required = {"sensor_index", start_field, end_field}
        if not required <= fields:
            raise ValueError(f"{path} must contain {', '.join(sorted(required))}")
        for row in reader:
            try:
                sensor = int(row["sensor_index"])
                start = parse_time(row[start_field])
                end = parse_time(row[end_field])
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid wildfire range in {path}") from error
            if sensor not in sensor_ids:
                continue
            if row.get("range_type") not in (None, "", "smoke"):
                continue
            if end < start:
                raise ValueError(f"wildfire range ends before it starts in {path}")
            ranges[sensor].append(
                {
                    "start": start,
                    "end": end,
                    "event_ids": tuple(
                        sorted(filter(None, (row.get("event_ids") or "").split(";")))
                    ),
                }
            )
    for rows in ranges.values():
        rows.sort(key=lambda row: (row["start"], row["end"]))
    return ranges


def hour_context(
    ranges: dict[int, list[dict[str, object]]], sensor: int, timestamp: int
) -> tuple[str, str, str]:
    overlaps = [
        row
        for row in ranges.get(sensor, ())
        if row["start"] < timestamp + HOUR and row["end"] >= timestamp
    ]
    if not overlaps:
        year, week, _ = datetime.fromtimestamp(timestamp, timezone.utc).isocalendar()
        return "routine", "", f"sensor_{sensor}_routine_{year}W{week:02d}"
    events = sorted(
        {event for row in overlaps for event in row["event_ids"]}
    )
    identity = ";".join(events) or ";".join(
        f"{row['start']}-{row['end']}" for row in overlaps
    )
    episode = hashlib.sha256(identity.encode()).hexdigest()[:16]
    return "wildfire", ";".join(events), f"wildfire_{episode}"


def eligible_candidates(
    dataset: DualEncoderDataset,
    train_indices: np.ndarray,
    wildfire_ranges: dict[int, list[dict[str, object]]],
) -> list[dict[str, object]]:
    rows = []
    for index in train_indices:
        code = int(dataset._sample_codes[index])
        location, anchor = divmod(code, dataset._steps)
        outdoor = float(dataset.observations[location, anchor, 0])
        if not math.isfinite(outdoor):
            continue
        indoor = float(dataset.observations[location, anchor, 1])
        timestamp = int(dataset.timestamps[anchor])
        sensor = dataset.sensor_ids[location]
        context, events, episode = hour_context(
            wildfire_ranges, sensor, timestamp
        )
        stratum = concentration_stratum(outdoor)
        rows.append(
            {
                "record_id": f"sensor_{sensor}_{timestamp}",
                "location_id": dataset.location_ids[location],
                "sensor_id": sensor,
                "timestamp_utc": format_time(timestamp),
                "tempo_outdoor_pm25_ug_m3": f"{outdoor:.7g}",
                "indoor_pm25_ug_m3": f"{indoor:.7g}",
                "data_context": context,
                "wildfire_event_ids": events,
                "episode_id": episode,
                "outdoor_range": stratum,
                "balance_cell": f"{context}|{stratum}",
                "selected": "false",
                "selection_rank": "",
            }
        )
    return rows


def select_balanced(
    rows: list[dict[str, object]],
    seed: int,
    max_per_episode: int,
    target_per_cell: int | None,
) -> tuple[int, list[dict[str, object]]]:
    cells: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        cells[str(row["balance_cell"])][str(row["episode_id"])].append(row)
    capacities = {
        cell: sum(min(len(records), max_per_episode) for records in episodes.values())
        for cell, episodes in cells.items()
    }
    quota = min(capacities.values(), default=0)
    if target_per_cell is not None:
        quota = min(quota, target_per_cell)

    for cell, episodes in sorted(cells.items()):
        ordered_episodes = sorted(
            episodes, key=lambda episode: stable_order(seed, cell, episode)
        )
        for episode, records in episodes.items():
            records.sort(
                key=lambda row: stable_order(
                    seed, cell, episode, row["record_id"]
                )
            )
        selected = []
        for round_number in range(max_per_episode):
            for episode in ordered_episodes:
                records = episodes[episode]
                if round_number < len(records):
                    selected.append(records[round_number])
                    if len(selected) == quota:
                        break
            if len(selected) == quota:
                break
        for rank, row in enumerate(selected, 1):
            row["selected"] = "true"
            row["selection_rank"] = rank

    availability = []
    for context in CONTEXTS:
        for stratum, _, _ in STRATA:
            cell = f"{context}|{stratum}"
            candidates = sum(map(len, cells.get(cell, {}).values()))
            capacity = capacities.get(cell, 0)
            selected = sum(
                row["selected"] == "true"
                for records in cells.get(cell, {}).values()
                for row in records
            )
            availability.append(
                {
                    "data_context": context,
                    "outdoor_range": stratum,
                    "eligible_training_candidates": candidates,
                    "independent_capacity": capacity,
                    "selected_hours": selected,
                    "common_quota": quota,
                    "status": (
                        "empty"
                        if not candidates
                        else "limiting"
                        if capacity == quota
                        else "balanced"
                    ),
                }
            )
    return quota, availability


def atomic_csv(
    path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]
) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_report(
    path: Path,
    args: argparse.Namespace,
    dataset: DualEncoderDataset,
    loaders,
    candidates: list[dict[str, object]],
    availability: list[dict[str, object]],
    quota: int,
) -> None:
    selected = sum(row["selected"] == "true" for row in candidates)
    populated = [row for row in availability if row["eligible_training_candidates"]]
    empty = [
        f"{row['data_context']}|{row['outdoor_range']}"
        for row in availability
        if not row["eligible_training_candidates"]
    ]
    lines = [
        "# Model-aware PM2.5 training balance",
        "",
        "The model loader established window eligibility and data splits before",
        "outdoor-PM2.5 balancing. No missing value or training window was synthesized.",
        "",
        "## Counts",
        "",
        f"- Configured sensors after exclusions: {len(dataset.sensor_ids):,}",
        f"- Eligible model windows across all splits: {len(dataset):,}",
        f"- Natural training windows: {len(loaders.train.dataset):,}",
        f"- Training windows with TEMPO at the anchor: {len(candidates):,}",
        f"- Populated balance cells: {len(populated):,} of {len(availability):,}",
        f"- Common independent quota per populated cell: {quota:,}",
        f"- Selected balanced training anchors: {selected:,}",
        f"- Empty cells omitted from the index: {', '.join(empty) or 'none'}",
        "",
        "## Eligibility and split configuration",
        "",
        f"- History hours: {args.history_hours}",
        f"- Prediction/target hours: {args.prediction_hours}",
        f"- Minimum TEMPO history observations: {args.minimum_outdoor_history_hours}",
        f"- Required consecutive recent TEMPO hours: {REQUIRED_RECENT_OUTDOOR_HOURS}",
        f"- Train fraction: {args.train_fraction}",
        f"- Validation fraction: {args.validation_fraction}",
        f"- Location holdout fraction: {args.location_holdout_fraction}",
        f"- Seed: {args.seed}",
        f"- Maximum selected hours per episode/cell: {args.max_hours_per_episode}",
        "",
        "Only populated eligible cells appear in `balanced_training_index.csv`.",
        "With the same inputs and split configuration, the training loader cannot",
        "recreate the old zero-eligible-cell failure.",
    ]
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def build(args: argparse.Namespace) -> dict[str, object]:
    if args.max_hours_per_episode < 1:
        raise ValueError("max-hours-per-episode must be positive")
    if args.target_per_cell is not None and args.target_per_cell < 1:
        raise ValueError("target-per-cell must be positive")

    output = args.output_dir
    existing = [output / name for name in OUTPUT_NAMES if (output / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"output files already exist under {output}; pass --overwrite to replace them"
        )
    output.mkdir(parents=True, exist_ok=True)

    print("Loading model inputs and calculating eligible windows...")
    dataset = DualEncoderDataset(
        args.pairs,
        args.indoor_history,
        args.outdoor_history,
        args.forecast_root,
        history_hours=args.history_hours,
        forecast_hours=args.prediction_hours,
        minimum_outdoor_history_hours=args.minimum_outdoor_history_hours,
        excluded_sensors_path=args.excluded_sensors,
    )
    loaders = create_data_loaders(
        dataset,
        batch_size=1,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        location_holdout_fraction=args.location_holdout_fraction,
        seed=args.seed,
    )
    train_indices = np.asarray(loaders.train.dataset.indices, dtype=np.int64)
    ranges = read_wildfire_ranges(
        args.wildfire_ranges, set(dataset.sensor_ids)
    )
    candidates = eligible_candidates(dataset, train_indices, ranges)
    if not candidates:
        raise ValueError(
            "no eligible training windows have a finite TEMPO value at the anchor"
        )
    quota, availability = select_balanced(
        candidates,
        args.seed,
        args.max_hours_per_episode,
        args.target_per_cell,
    )
    selected = [row for row in candidates if row["selected"] == "true"]
    if not selected:
        raise ValueError("no balanced training anchors were selected")

    candidates.sort(key=lambda row: (str(row["balance_cell"]), str(row["record_id"])))
    selected.sort(
        key=lambda row: (
            str(row["balance_cell"]),
            int(row["selection_rank"]),
            str(row["record_id"]),
        )
    )
    atomic_csv(
        output / "eligible_training_candidates.csv",
        RECORD_FIELDS,
        candidates,
    )
    atomic_csv(output / "balanced_training_index.csv", RECORD_FIELDS, selected)
    atomic_csv(output / "balance_cells.csv", CELL_FIELDS, availability)
    write_report(
        output / "report.md",
        args,
        dataset,
        loaders,
        candidates,
        availability,
        quota,
    )

    validation = create_data_loaders(
        dataset,
        batch_size=1,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        location_holdout_fraction=args.location_holdout_fraction,
        seed=args.seed,
        balanced_training_index=output / "balanced_training_index.csv",
    ).balance_report
    if validation is None or validation["selected_training_anchors"] != len(selected):
        raise RuntimeError("the generated index failed loader validation")
    return {
        "candidates": len(candidates),
        "selected": len(selected),
        "quota": quota,
        "populated_cells": sum(
            bool(row["eligible_training_candidates"]) for row in availability
        ),
        "output": output / "balanced_training_index.csv",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument(
        "--indoor-history", type=Path, action="append", required=True
    )
    parser.add_argument("--outdoor-history", type=Path, required=True)
    parser.add_argument("--forecast-root", type=Path, required=True)
    parser.add_argument("--wildfire-ranges", type=Path, required=True)
    parser.add_argument("--excluded-sensors", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "inputs" / "model_aware_training_balance",
    )
    parser.add_argument("--history-hours", type=int, default=168)
    parser.add_argument("--prediction-hours", type=int, default=36)
    parser.add_argument("--minimum-outdoor-history-hours", type=int, default=24)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--location-holdout-fraction", type=float, default=0.20)
    parser.add_argument("--max-hours-per-episode", type=int, default=1)
    parser.add_argument("--target-per-cell", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    result = build(parse_args(argv))
    print(
        "Balanced training index written: "
        f"{result['output']} | candidates={result['candidates']:,} "
        f"cells={result['populated_cells']}/16 "
        f"quota={result['quota']:,} selected={result['selected']:,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
