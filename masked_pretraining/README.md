# School-pair masked PM2.5 pretraining

This package pretrains a history encoder by reconstructing deliberately hidden
PurpleAir PM2.5 observations. It does not read TEMPO or NAQFC. TEMPO can be
introduced later when the pretrained history encoder is transferred into the
forecasting model.

## Data contract

The loader filters the original one-to-one PurpleAir pair table by the validated
119-sensor indoor K-12 cohort. The current source table contains 43 genuine
school indoor-outdoor pairs. It rejects the zero-distance school file whose
"outdoor" IDs equal its indoor IDs because that file represents ambient grid
coordinates, not outdoor PurpleAir sensors.

The permanent broken-sensor list, `permanently_excluded_indoor_sensors.csv`,
and both prior high-reading lists, `excluded_indoor_sensors_pm25_gt1000.csv`
and `school_indoor_pm25/data/excluded_indoor_schools_pm25_gt1000.csv`, are
always applied before histories, windows, splits, or normalization. Their
SHA-256 values are stored in every checkpoint manifest. Additional exclusion
files can be supplied with repeated `--excluded-sensors` options.

Known-bad outdoor PurpleAir periods in
`excluded_outdoor_purpleair_ranges.csv` are also mandatory. The loader removes
only readings inside each half-open UTC range before paired histories, windows,
splits, or normalization are built; a blank start and end excludes that outdoor
sensor's full downloaded history. Indoor sensors paired with these devices
remain valid.

Bounded indoor bad-reading periods in `excluded_indoor_purpleair_ranges.csv`
are removed with the same half-open UTC semantics, while valid readings from
those sensors remain available.

Each history CSV must contain `time_stamp,sensor_index,pm2.5_atm`. Directories
are scanned recursively for numeric sensor-named CSVs. Pass indoor and outdoor
history collections with repeated `--history` options:

```powershell
.\.venv\Scripts\python.exe -m masked_pretraining audit `
  --history C:\path\to\school_indoor_purpleair `
  --history C:\path\to\school_outdoor_purpleair
```

The audit reports observations and eligible windows for every pair. Training
requires at least two pairs with 168-hour windows and, by default, at least 144
observed hours in each channel. Natural gaps remain missing and are never
reconstruction targets.

## Training

```powershell
.\.venv\Scripts\python.exe -m masked_pretraining train `
  --history C:\path\to\school_indoor_purpleair `
  --history C:\path\to\school_outdoor_purpleair `
  --model transformer
```

The curriculum advances after validation plateaus:

1. isolated points;
2. two- and three-hour blocks;
3. mixed one-, three-, and six-hour blocks;
4. indoor cross-channel blocks with outdoor values visible;
5. final 3-, 6-, and 12-hour indoor suffixes.

Input features are normalized outdoor/indoor values, visible-value indicators,
artificial-mask indicators, and daily/weekly/annual sine-cosine time features.
Only deliberately hidden known values contribute to loss. Pair-level splitting
holds out entire schools, and normalization is fitted only on training pairs.

The post-exclusion responsiveness classifier can stage pair difficulty without
adding a fixed building label to the model. Its join uses the exact indoor and
outdoor sensor IDs. Start with the highest-response third, then add the middle
and lowest thirds:

```powershell
.\.venv\Scripts\python.exe -m masked_pretraining train `
  --history ".\data\purple air\school_indoor_pm25.csv" `
  --history ".\data\purple air\general_non_school_indoor_pm25.csv" `
  --history ".\data\purple air\outdoor_school\school_outdoor_pm25.csv" `
  --history ".\data\purple air\outdoor_non_school\non_school_outdoor_pm25.csv" `
  --responsiveness-tiers high

# Later stages use: --responsiveness-tiers high moderate
# Final classified stage: --responsiveness-tiers high moderate low
```

Omitting `--responsiveness-tiers` preserves the unfiltered pair set. Filtered
runs exclude `unclassified` pairs and record the classifier path, SHA-256,
included tiers, and removed-pair count in the audit/checkpoint provenance.

`--model transformer` and `--model gru` use the same reconstruction contract.
Add another architecture by decorating a `ModelConfig -> nn.Module` builder with
`register_model()` in `models.py`; the module must return `[batch, time, 2]`.
Stage checkpoints and JSON provenance manifests are written under
`masked_pretraining/runs/` by default.

## Current data gate

The available PurpleAir history archive contains the school indoor sensors but
none of the paired outdoor sensor IDs. The default audit therefore reports zero
trainable paired windows. Outdoor PurpleAir histories must be acquired before a
real training run; the code intentionally does not replace them with TEMPO or
duplicate indoor readings.
