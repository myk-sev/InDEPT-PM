"""Restore the balance index recorded by a completed singular export."""

import csv
from pathlib import Path

from model_aware_training_balance.build_training_index import (
    RECORD_FIELDS,
    concentration_stratum,
    format_time,
    hour_context,
    parse_time,
    read_wildfire_ranges,
)


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "inputs/forecasting/balanced_old_training_data.csv"
TARGET = ROOT / "inputs/model_aware_training_balance/balanced_training_index.csv"
WILDFIRE = ROOT.parent / "purple-air-pull/smoke_plume_intersection/results/all_indoor_sensor_light_smoke_ranges.csv"


with SOURCE.open(encoding="utf-8-sig", newline="") as source:
    training = [row for row in csv.DictReader(source) if row["split"] == "train"]
ranges = read_wildfire_ranges(WILDFIRE, {int(row["sensor_id"]) for row in training})
records = []
for row in training:
    sensor = int(row["sensor_id"])
    timestamp = parse_time(row["anchor_time_utc"])
    outdoor = float(row["history_167_tempo_pm25_ug_m3"])
    context, events, episode = hour_context(ranges, sensor, timestamp)
    stratum = concentration_stratum(outdoor)
    records.append(
        {
            "record_id": f"sensor_{sensor}_{timestamp}",
            "location_id": row["location_id"],
            "sensor_id": sensor,
            "timestamp_utc": format_time(timestamp),
            "tempo_outdoor_pm25_ug_m3": f"{outdoor:.7g}",
            "indoor_pm25_ug_m3": f"{float(row['history_167_indoor_pm25_ug_m3']):.7g}",
            "data_context": context,
            "wildfire_event_ids": events,
            "episode_id": episode,
            "outdoor_range": stratum,
            "balance_cell": f"{context}|{stratum}",
            "selected": "true",
            "selection_rank": 1,
        }
    )
if len(records) != 16 or len({row["balance_cell"] for row in records}) != 16:
    raise ValueError("balanced export must contain one training anchor per balance cell")
partial = TARGET.with_suffix(".csv.part")
with partial.open("w", encoding="utf-8", newline="") as target:
    writer = csv.DictWriter(target, fieldnames=RECORD_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(sorted(records, key=lambda row: row["balance_cell"]))
partial.replace(TARGET)
print("Restored 16 balanced anchors across 16 cells")
