# Bridge-to-forecast workflow

This is the current-PC workflow. It uses only paths that exist in this checkout.
A training-PC layout should be added as a separate launcher later rather than
adding path fallbacks to these commands.

## Runtime contracts

Base reconstruction reads one paired-PurpleAir CSV and writes one checkpoint
per reconstruction architecture. Bridge training resumes each completed base
checkpoint and adds the three synthetic TEMPO-missingness stages. Neither step
reads TEMPO or NAQFC.

Supervised forecasting reads:

- the completed bridge checkpoint;
- the matching materialized cyclical forecasting CSV under
  `inputs\forecasting`; and
- no raw TEMPO, NAQFC, PurpleAir, exclusion, or pair files at runtime.

The materialized forecasting CSV contains the TEMPO history, NAQFC future
forecast, indoor target, source provenance, and authoritative train,
validation, temporal-test, and location-test labels.

## 1. Inspect the base reconstruction matrix

```bat
scripts\run_base_reconstruction_model_family.bat --dry-run auto 20 64 0
```

Run it after inspection:

```bat
scripts\run_base_reconstruction_model_family.bat auto 20 64 0
```

It writes eleven architectures for each cohort. Every artifact uses the same
`<dataset>_<model>` stem: checkpoints go to `inference\checkpoints`, metrics to
`inference\metrics`, cumulative loss graphs to `inference\graphs`, and the
multiple stage/epoch reconstruction images to a matching folder under
`inference\reconstructions`.

## 2. Inspect and run the bridge matrix

```bat
scripts\run_bridge_training_model_family.bat --dry-run auto 20 64 0
scripts\run_bridge_training_model_family.bat auto 20 64 0
```

After all runs, the launcher calls the family verifier. It requires exactly 22
checkpoints with:

- the matching model name and training-data SHA-256;
- all seven reconstruction and all three bridge stages completed;
- `tempo_bridge_86` as the final stage;
- synthetic-only bridge metadata with no TEMPO data claimed; and
- optimizer state retained.

The verifier can also be rerun directly:

```bat
.venv\Scripts\python.exe -m masked_pretraining.verify_bridge_family
```

It is expected to fail on a preparation PC until the checkpoints exist.

## 3. Inspect and run supervised forecasting

```bat
scripts\run_bridge_forecast_model_family.bat --dry-run auto 64 0
scripts\run_bridge_forecast_model_family.bat auto 64 0
```

The matrix contains 44 runs: two cohorts, eleven architectures, and two history
initializations. `pretrained` strictly transfers the matching bridge encoder.
`random-control` keeps the identical forecasting architecture and bridge
normalization but does not load encoder weights.

Each run uses prefix-loss stages of 3, 6, 12, 24, and 36 hours over 50 epochs.
The pretrained history encoder is frozen for the first three epochs and then
fine-tuned with the complete model. The random control trains its history
encoder from the first epoch.

Every successful training run must also complete:

1. production cached inference with a unique output directory; and
2. validation, temporal-test, and location-test evaluation against persistence
   and fitted per-lead linear baselines.

Failure of training, inference, or evaluation stops the matrix.

## One-model command

The trainer selects the matching `bridge-forecast-*` model automatically when
`--model` is omitted:

```bat
.venv\Scripts\python.exe pm25_transformer.py train ^
  --pretrained-checkpoint inference\checkpoints\k12_excl_fine_t_dual-encoder-cross-fusion.pt ^
  --training-data inputs\forecasting\k12_exclusion_informed_finetuned_tempo_naqfc_forecast_training_cyclical.csv ^
  --history-initialization pretrained ^
  --epochs 50 ^
  --freeze-history-epochs 3 ^
  --forecast-horizons 3 6 12 24 36 ^
  --horizon-stage-epochs 5 5 10 10 20 ^
  --checkpoint inference\checkpoints\k12_excl_fine_t_bridge-forecast-dual-encoder-cross-fusion-pretrained.pt ^
  --metrics-output inference\metrics\k12_excl_fine_t_bridge-forecast-dual-encoder-cross-fusion-pretrained.csv ^
  --loss-plot inference\graphs\k12_excl_fine_t_bridge-forecast-dual-encoder-cross-fusion-pretrained.png
```

The resulting forecast checkpoint contains all transferred history weights,
new forecast/decoder weights, downstream and bridge history normalization,
training configuration, and source-checkpoint path and SHA-256. Forecast
inference therefore does not reopen the bridge checkpoint.

## Outputs

Every output type has a dedicated folder. Single-file artifacts share the
unambiguous `<dataset>_<model>` stem; only multi-image outputs add one unnested
folder per training artifact. Base reconstruction and bridge stages for the same
dataset/model continuously update the same checkpoint, metrics CSV, loss graph,
and reconstruction folder. For example:

```text
inference\checkpoints\k12_excl_fine_t_dual-encoder-cross-fusion.pt
inference\metrics\k12_excl_fine_t_dual-encoder-cross-fusion.csv
inference\graphs\k12_excl_fine_t_dual-encoder-cross-fusion.png
inference\reconstructions\k12_excl_fine_t_dual-encoder-cross-fusion\*.png
inference\forecasts\k12_excl_fine_t_bridge-forecast-dual-encoder-cross-fusion-pretrained\*.png
inference\evaluations\k12_excl_fine_t_bridge-forecast-dual-encoder-cross-fusion-pretrained.json
```

Shared inference caches live under `inference\caches`; they are generation-time
inputs reused by multiple training runs.
