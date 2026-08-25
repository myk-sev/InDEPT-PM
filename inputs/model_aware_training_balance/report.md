# Model-aware PM2.5 training balance

The model loader established window eligibility and data splits before
outdoor-PM2.5 balancing. No missing value or training window was synthesized.

## Counts

- Configured sensors after exclusions: 388
- Eligible model windows across all splits: 19,342
- Natural training windows: 7,757
- Training windows with TEMPO at the anchor: 7,757
- Populated balance cells: 16 of 16
- Common independent quota per populated cell: 1
- Selected balanced training anchors: 16
- Empty cells omitted from the index: none

## Eligibility and split configuration

- History hours: 168
- Prediction/target hours: 36
- Minimum TEMPO history observations: 24
- Required consecutive recent TEMPO hours: 3
- Train fraction: 0.75
- Validation fraction: 0.15
- Location holdout fraction: 0.2
- Seed: 42
- Maximum selected hours per episode/cell: 1

Only populated eligible cells appear in `balanced_training_index.csv`.
With the same inputs and split configuration, the training loader cannot
recreate the old zero-eligible-cell failure.
