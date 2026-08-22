# Outdoor PurpleAir failure-period detector

`detect_outdoor_quality_periods.py` is a review tool for identifying periods
that resemble confirmed outdoor PurpleAir sensor failures. It uses only paired
indoor/outdoor PurpleAir `pm2.5_atm`; it does not use TEMPO or NAQFC.

The tool reads raw outdoor histories before applying the reviewed ranges in
`../excluded_outdoor_purpleair_ranges.csv`. Those ranges are validation labels,
not input filters. The tool never edits the manifest, and its candidate files
are not consumed by training.

## Detection process

For every indoor/outdoor PurpleAir pair, the tool:

1. Requires at least 12 observations in the preceding 24 hours to establish an
   outdoor baseline.
2. Detects an outdoor event when PM2.5 is at least 55.5 µg/m³ and at least
   25 µg/m³ above baseline for three trigger hours. Trigger gaps of up to two
   hours are merged.
3. Examines paired readings through 24 hours after the event and requires at
   least 80% paired coverage.
4. Labels the event as low-response when indoor rise is at most 5 µg/m³ and
   both the peak-rise and area-above-baseline indoor/outdoor ratios are at most
   0.10.
5. Groups events by indoor/outdoor pair and UTC calendar year.
6. Selects a sensor-year for review only after at least four low-response events
   and a low-response rate of at least 55%.

The period thresholds were set just below the weakest reviewed bad sensor-year
in the current data: four low-response events and a 57.9% rate. This allows
isolated false-positive events without proposing a period for review.

## Run after future downloads

From the repository root, using the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m purpleair_pair_exclusions.detect_outdoor_quality_periods `
  --pairs .\purpleair_pair_exclusions\results\selected_pairs.csv `
  --indoor-history ".\data\purple air\school_indoor_pm25.csv" `
  --indoor-history ".\data\purple air\general_non_school_indoor_pm25.csv" `
  --outdoor-history ".\data\purple air\outdoor_school\school_outdoor_pm25.csv" `
  --outdoor-history ".\data\purple air\outdoor_non_school\non_school_outdoor_pm25.csv"
```

Use `--minimum-period-events` and `--minimum-low-response-rate` only when an
intentional threshold change is required. The selected values are recorded in
the run summary.

## Outputs

The default output directory is `purpleair_pair_exclusions/outdoor_quality_results`:

- `event_scores.csv`: every raw elevated-outdoor event and its paired response
  metrics.
- `period_scores.csv`: every analyzable pair-year, including periods below the
  repeated-event threshold.
- `candidate_periods.csv`: threshold-passing known and new periods.
- `new_candidate_periods.csv`: only threshold-passing periods not already in
  the reviewed manifest.
- `summary.json`: inputs, event and period thresholds, known-period recovery,
  and candidate counts.
- `outdoor_quality_period_summary.png`: event count versus low-response rate,
  distinguishing recovered known periods from new candidates.

Candidate output is evidence for review, not an exclusion decision. Examine
the candidate in `location_history_explorer.html`; only confirmed outdoor
failures should be manually added to `excluded_outdoor_purpleair_ranges.csv`.
