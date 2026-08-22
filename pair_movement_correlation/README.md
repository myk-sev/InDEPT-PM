# Indoor/outdoor PM2.5 movement correlation

This analysis tests whether hourly indoor PM2.5 *movement* tracks the nearest
outdoor PurpleAir sensor as the allowable school-pair distance increases. It
does not correlate raw concentration levels, interpolate missing hours, or
substitute TEMPO, NAQFC, or a different outdoor sensor when history is absent.

For every validated FEMA/NSD indoor school, the tool independently selects the
nearest active-snapshot outdoor PurpleAir sensor. Outdoor reuse is allowed.
The repository's three established indoor exclusion lists are applied before
pairing. `excluded_indoor_purpleair_ranges.csv` masks bounded bad-reading
periods. `excluded_outdoor_purpleair_ranges.csv` removes full-history outdoor
sensors before pairing and masks only the specified hours for bounded ranges.
It then computes simultaneous one-hour changes at exact consecutive UTC hours.
Each pair needs at least 168 movement observations. The primary statistic is
Pearson correlation after separate 1%/99% winsorization of indoor and outdoor
changes; raw Pearson, Spearman, direction agreement, and outdoor-leading lags
from zero through six hours are retained as checks.

The cumulative distance result is the median pair-level correlation, giving
each building equal weight regardless of history length. Its 95% confidence
interval uses 10,000 bootstrap resamples of outdoor-sensor clusters, which
keeps schools sharing an outdoor sensor together.

Run from the repository root with its virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pair_movement_correlation.analyze_movement_correlation `
  --school-sensors ..\purple-air-pull\purpleair_indoor_school_sensors.csv `
  --sensor-inventory ..\purple-air-pull\purpleair_continental_us_sensors.csv `
  --indoor-history ".\data\purple air\school_indoor_pm25.csv" `
  --outdoor-history ".\data\purple air\outdoor_school\school_outdoor_pm25.csv" `
  --outdoor-history ".\data\purple air\outdoor_non_school\non_school_outdoor_pm25.csv" `
  --excluded-indoor-ranges .\excluded_indoor_purpleair_ranges.csv `
  --excluded-outdoor-ranges .\excluded_outdoor_purpleair_ranges.csv `
  --distance-limits 100 250 500 1000 `
  --output-dir .\pair_movement_correlation\results
```

Outputs are `eligible_pairs.csv`, `pair_metrics.csv`, `distance_summary.csv`,
`summary.json`, and the run-specific `report.md`. Spatial eligibility and
history availability are always reported separately.
