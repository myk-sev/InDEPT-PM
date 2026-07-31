# Model-aware PM2.5 training balance

This tool replaces the old anchor-only balance workflow for model training.
It constructs the current `DualEncoderDataset`, applies the normal temporal and
location splits, and then balances only eligible training windows that have a
finite TEMPO PM2.5 value at the anchor.

Consequently, every selected record already satisfies:

- complete indoor history;
- configured TEMPO history coverage and recency;
- complete future indoor target;
- complete causal NAQFC forecast;
- the sensor exclusion list; and
- the training portion of the temporal and location splits.

Balance cells remain `routine/wildfire × 8 outdoor-PM2.5 ranges`. Selection is
deterministic, episode-capped, and based only on TEMPO outdoor PM2.5. Empty
eligible cells are reported and omitted rather than written as unusable cells.
No record is duplicated, interpolated, synthesized, or assigned a weight.
Training must use the same inputs, eligibility parameters, split fractions,
exclusions, and seed recorded in the generated report.

## Run

Use this repository's virtual environment:

```powershell
.\.venv\Scripts\python.exe -m model_aware_training_balance.build_training_index `
  --pairs .\purpleair_continental_us_pairs_thinned_20km.csv `
  --indoor-history ..\purple-air-pull\purpleair_hourly_pm25_atm `
  --outdoor-history ..\purple-air-pull\tempo_pm25_sensor_match\tempo_pm25_indoor_sensors.csv `
  --forecast-root .\naqfc_output `
  --wildfire-ranges ..\purple-air-pull\smoke_plume_intersection\results\all_indoor_sensor_light_smoke_ranges.csv `
  --excluded-sensors .\excluded_indoor_sensors_pm25_gt1000.csv `
  --output-dir .\model_aware_training_balance\results\current
```

The model defaults are 168 history hours, 36 prediction/target hours, at least
24 TEMPO observations in the history, and a maximum TEMPO age of 48 hours.
Pass the corresponding options whenever training uses different values.

Use the generated index during training:

```powershell
.\.venv\Scripts\python.exe pm25_transformer.py train ... `
  --balanced-training-index .\model_aware_training_balance\results\current\balanced_training_index.csv
```

## Outputs

- `balanced_training_index.csv`: selected loader-compatible training anchors.
- `eligible_training_candidates.csv`: every eligible training anchor with
  TEMPO present at the anchor, including unselected records.
- `balance_cells.csv`: availability, independent capacity, and selection for
  all 16 possible cells.
- `report.md`: inputs, constraints, split counts, empty cells, and final quota.

Existing outputs are protected unless `--overwrite` is passed.

## Test

```powershell
.\.venv\Scripts\python.exe -m unittest -v `
  model_aware_training_balance.test_build_training_index
```
