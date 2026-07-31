# Indoor school PurpleAir PM2.5

This folder contains a reproducible copy and consolidation of hourly PurpleAir
`pm2.5_atm` for the validated 119-sensor indoor-school cohort.

The cohort is restricted to rows where `location_type=inside`,
`k12_status=school`, and `is_k12=true`. The build combines, in precedence order:

1. the copied 119-sensor TEMPO-overlap history downloads;
2. school-sensor rows from the current fixed-48-hour trend downloads;
3. school-sensor rows from the current adaptive TEMPO-overlap downloads.

The output has one row per `sensor_index` and `time_stamp`. A later collection
replaces an earlier value only when the same sensor-hour was downloaded again.
Counts of duplicate and changed sensor-hours are recorded in
`data/build_summary.json`.

The cohort count describes requested sensor downloads. Some request files may
contain only a header because PurpleAir returned no observations. The summary
separately reports sensors with rows and lists cohort sensor IDs without rows.

Run from the repository root with the project virtual environment:

```powershell
.\.venv\Scripts\python.exe .\school_indoor_pm25\build_dataset.py
```

Outputs:

- `data/purpleair_school_indoor_pm25.csv`: the single combined dataset;
- `data/purpleair_indoor_school_sensors.csv`: the copied cohort definition;
- `data/source_history/`: the unchanged history-download copy, including its
  download ledger;
- `data/build_summary.json`: source counts, coverage, and output SHA-256.
