# Data and training inputs

The repository keeps source data and model-ready inputs separate:

- `data/` contains raw histories, reviewed exclusions, location/pair manifests,
  and staged downloads consumed by input generators.
- `inputs/` contains final model-ready datasets and composed training contracts.
- Generator scripts remain in the source tree; they write to one of those two
  roots according to whether their output is source data or a final input.

## Source data

- `data/purple air/`: consolidated PurpleAir histories, staged 1 km histories,
  and resumable download archives.
- `data/exclusions/`: reviewed whole-sensor and bounded-range exclusions.
- `data/legacy/`: legacy pairing and location manifests.

## Final training inputs

- `inputs/unmasked_type/old_training_data.csv` and
  `inputs/unmasked_type/school_old_training_data.csv`: model-ready exports from
  the legacy forecasting pipeline.
- `inputs/masked_pretraining/exclusion_aware/k12_exclusion_aware_masked_training_data.csv`:
  default K-12 contract.
- `inputs/masked_pretraining/no_exclusions/k12_no_exclusions_masked_training_data.csv`:
  opt-in K-12 exclusion-free contract.
- `inputs/masked_pretraining/exclusion_informed_finetuned/k12_exclusion_informed_finetuned_masked_training_data.csv`:
  K-12 fine-tuning contract.
- `inputs/masked_pretraining/all_sensors/`: descriptively named exclusion-aware,
  exclusion-free, and fine-tuning contracts for every retrieved indoor history
  with a PurpleAir pair.
- `inputs/masked_pretraining/old_non_masked_purpleair/`: descriptively named
  legacy-PurpleAir exclusion-aware and exclusion-free masked-training contracts.
- `inputs/masked_pretraining/responsiveness/`: generation-time curriculum
  classification outputs.
- `inputs/model_aware_training_balance/`: default destination for a generated
  legacy forecasting balance index.

Each masked-pretraining model input has a self-describing filename containing
its cohort and exclusion policy, such as
`k12_exclusion_aware_masked_training_data.csv` or
`all_sensors_no_exclusions_masked_training_data.csv`. Every row contains an
assignment interval, its embedded responsiveness tier, and sparse
indoor/outdoor PM2.5 readings. No source data,
exclusion manifest, metadata sidecar, or additional history file is required at
training time.

## Generators

- `school_indoor_pm25/build_dataset.py`: builds raw school indoor history in
  `data/purple air`.
- `download_matching_outdoor_purpleair.py`: downloads raw matched outdoor history.
- `download_k12_1km_outdoor_review.ps1`: downloads isolated K-12 review history.
- `merge_1km_history_archives.py`: merges reviewed staged history into the raw
  school archives.
- `reorganize_purpleair_csvs.py`: builds the raw all-indoor and all-outdoor CSVs.
- `purpleair_pair_exclusions/detect_pair_exclusions.py`: writes the default
  exclusion-aware masked-training CSV to `inputs`.
- `purpleair_pair_exclusions/generate_k12_training_inputs.py`: writes the
  exclusion-aware and exclusion-free K-12 contracts to `inputs`.
- `purpleair_pair_exclusions/generate_all_sensor_training_inputs.py`: writes
  exclusion-aware and exclusion-free all-sensor contracts to `inputs`.
- `pair_responsiveness/classify.py`: writes the final responsiveness manifest.
- `model_aware_training_balance/build_training_index.py`: writes the final
  forecasting balance index.
- `export_old_training_csv.py`: generates a model-ready forecasting CSV; the
  current finalized exports are organized under `inputs/unmasked_type/`.
- `extract_old_training_purpleair.py`: reconstructs raw PurpleAir history under
  `data/purple air` from the legacy model-ready CSV.
