# Outdoor PurpleAir failure-period detector

`detect_outdoor_quality_periods.py` is a review tool for identifying periods
that resemble confirmed outdoor PurpleAir sensor failures. It uses paired
indoor/outdoor PurpleAir `pm2.5_atm` for response testing and scans every
downloaded outdoor PurpleAir history for recurring error levels. It does not
use TEMPO or NAQFC.

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
4. Labels the event as low-response when both the peak-rise and
   area-above-baseline indoor/outdoor ratios are at most 0.10. The absolute
   indoor rise remains in the review output but does not veto an extreme
   proportional mismatch.
5. Groups events by indoor/outdoor pair and UTC calendar year.
6. Selects a sensor-year for review only after at least three low-response events
   and a low-response rate of at least 55%.
7. Independently imports the recurring error-level detector and runs it across
   every downloaded outdoor history, including sensors absent from the current
   pair file.
8. Marks one event as an extreme mismatch when outdoor rise is at least 700
   µg/m³, peak response ratio is at most 0.05, area response ratio is at most
   0.10, and paired coverage is at least 80%. This rule has no absolute indoor
   rise veto and does not require repeated events.

The recurring error-level and extreme-mismatch rules generate concrete range
candidates. Repeated low response by itself remains review-only and is never
written as an exclusion-range candidate. All outputs still require review and
do not edit the authoritative manifest. The recall-first thresholds are
intentionally broader than the indoor-sensor exclusion rule, which retains its
5 µg/m³ absolute-rise gate. Candidate rows never modify training exclusions;
they require manual review before addition to the exclusion manifest.

## Recurring error-level detector

`outdoor_sensor_error.py` is a standalone detector operating on one outdoor
sensor history at a time. It recognizes the recurring device-failure levels
seen in the reviewed histories: 1,500–1,800, 2,250–2,700, and 3,000–3,600
µg/m³. These are PM2.5 mass concentrations, not ppm.

A period requires at least three error-level readings, with no more than 48
hours between successive readings. The output is a half-open UTC interval plus
its reading count and observed range. An isolated extreme reading is not
enough. `ErrorLevelCriteria` is importable so a future partial-history error
classifier can reuse the same interval detector instead of discarding a sensor
whose remaining history is valid.

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

Use `--minimum-period-events`, `--minimum-low-response-rate`,
`--maximum-peak-response-ratio`, and `--maximum-area-response-ratio` only when
an intentional threshold change is required. The selected values are recorded
in the run summary.

## Outputs

The default output directory is `purpleair_pair_exclusions/outdoor_quality_results`:

- `event_scores.csv`: every raw elevated-outdoor event and its paired response
  metrics. `selected_for_period_review` is the recall-first proportional flag;
  `selected_for_exclusion` retains the stricter absolute-and-proportional flag;
  `selected_for_error_exclusion` is the magnitude-and-ratio rule.
- `period_scores.csv`: every analyzable pair-year, including periods below the
  repeated-event threshold.
- `candidate_periods.csv`: threshold-passing known and new periods.
- `new_candidate_periods.csv`: only threshold-passing periods not already in
  the reviewed manifest.
- `error_level_periods.csv`: standalone error-level intervals from all
  downloaded outdoor histories, with sensor IDs, UTC bounds, reading counts,
  and observed PM2.5 range.
- `exclusion_range_candidates.csv`: merged recurring error-level and extreme
  mismatch ranges, including ranges already covered by the reviewed manifest.
- `new_exclusion_range_candidates.csv`: only ranges not fully covered by the
  reviewed manifest. Low-response-only candidates are deliberately absent.
- `summary.json`: inputs, event and period thresholds, known-period recovery,
  and candidate counts.
- `outdoor_quality_period_summary.png`: event count versus low-response rate,
  distinguishing recovered known periods from new candidates.

Candidate output is evidence for review, not an exclusion decision. Examine
the candidate in `location_history_explorer.html`; only confirmed outdoor
failures should be manually added to `excluded_outdoor_purpleair_ranges.csv`.
Every added row records the local review date in `decision_date` and the exact
UTC batch time in `added_at_utc`. Rerun this detector after approval to confirm
the applied rows are no longer reported in
`new_exclusion_range_candidates.csv`, then rerun the pair-exclusion workflow to
refresh training intervals and both exclusion tabs in the data explorer.
