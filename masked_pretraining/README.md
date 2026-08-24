# PurpleAir masked PM2.5 pretraining

This package pretrains a history encoder by reconstructing deliberately hidden
PurpleAir PM2.5 observations. It does not read TEMPO or NAQFC. TEMPO can be
introduced later when the pretrained history encoder is transferred into the
forecasting model.

## Static data contract

Masked training reads exactly one model input:
`inputs/masked_pretraining/exclusion_aware/training_data.csv`. Each row contains
one half-open UTC indoor/outdoor assignment interval, its responsiveness tier,
and both sensors' PM2.5 readings for that interval. The readings are sparse JSON
lists of `[hour_offset, value]`, where the offset is measured from `start_utc`.
An absent offset is a naturally missing observation.

Distance ranking, reviewed exclusions, fallback selection, and restoration of
the closer sensor are resolved while this file is generated. Training does not
read the source histories, exclusion files, a metadata sidecar, or a separate
responsiveness file. Regenerate the CSV after any generation input changes.

The loader builds one continuous series for each indoor sensor. Adjacent
outdoor assignments are stitched, so a 168-hour window may cross an outdoor
sensor handoff. A manifest gap is a hard boundary, and a window can never
contain more than one indoor sensor ID. Outdoor sensor and assignment IDs are
retained per hour for audits and checkpoint compatibility but are not model
inputs.

Select a different generated dataset with one argument:

```powershell
.\.venv\Scripts\python.exe -m masked_pretraining audit `
  --training-data C:\path\to\training_data.csv
```

Natural missing observations are not reconstruction targets and are governed
by `--minimum-observed-hours`, which defaults to 144 observations per channel
in each 168-hour window.

The default dataset is the exclusion-aware K-12 cohort. All-retrieved-sensor
variants
are stored under `inputs/masked_pretraining/all_sensors/exclusion_aware` and
`inputs/masked_pretraining/all_sensors/no_exclusions`; select one by passing its
`training_data.csv` to `--training-data`.

## Training

```powershell
.\.venv\Scripts\python.exe -m masked_pretraining train `
  --model transformer
```

The curriculum advances after validation plateaus:

1. isolated points;
2. two- and three-hour blocks;
3. mixed one-, three-, and six-hour blocks;
4. indoor cross-channel blocks with outdoor values visible;
5. final 3-, 6-, and 12-hour indoor suffixes.

`--epochs-per-stage` accepts either one value, applied uniformly to every
selected stage, or one value per stage in `--stages` order. Without an explicit
`--stages` list, the seven values follow the curriculum order above:

```powershell
.\.venv\Scripts\python.exe -m masked_pretraining train `
  --epochs-per-stage 30 20 20 50 20 50 50
```

Each hour has the final eight-feature contract: normalized outdoor and indoor
PM2.5 followed by daily, weekly, and annual sine-cosine time features. Natural
gaps and artificially hidden values use `-9` in the affected PM2.5 value slot.
The artificial target mask remains outside the model, so only deliberately
hidden known values contribute to loss. Splitting is by `indoor_sensor_id`, and
normalization is fitted only on training sensors.

The optional responsiveness curriculum classifies indoor/outdoor assignments.
A window is admitted only if every assignment it touches has an included tier;
missing classifications remain `unclassified` and are not admitted by a
classified-only run:

```powershell
.\.venv\Scripts\python.exe -m masked_pretraining train `
  --responsiveness-tiers high

# Later stages use: --responsiveness-tiers high moderate
# Final classified stage: --responsiveness-tiers high moderate low
```

Omitting `--responsiveness-tiers` preserves every static interval. The audit
and checkpoints record any intervals removed by responsiveness filtering.

`--model transformer` and `--model gru` use the same reconstruction contract.
Add an architecture by decorating a `ModelConfig -> nn.Module` builder with
`register_model()` in `models.py`; it must return `[batch, time, 2]`. Future
run artifacts are separated by type under `masked_pretraining/runs/`:

- `checkpoints/`: stage/final model checkpoints and JSON provenance manifests;
- `graphs/`: metrics CSV files and loss-curve graphs;
- `inference/`: validation reconstruction inference graphs.

Each run writes:

- `<run>.metrics.csv`: one row per epoch with training and validation
  loss, validation RMSE by channel, target counts, and checkpoint improvement;
- `<run>.loss_curve.png`: training and validation loss across all
  completed curriculum stages;
- `<run>.reconstruction_examples.png`: four fixed validation windows
  showing the full label history, model-visible history, artificially masked
  labels, predictions, and natural missingness as gaps.

The metrics CSV and loss graph are refreshed after every epoch. The unnumbered
reconstruction plot is refreshed from the best validation checkpoint after
each completed masking stage. Their absolute paths are stored in the checkpoint
JSON.

To refresh the reconstruction plot during a stage, set an epoch interval:

```powershell
.\.venv\Scripts\python.exe -m masked_pretraining train `
  --stages points `
  --reconstruction-every-epochs 5
```

This uses the same fixed validation windows and mask pattern every time, making
changes between epochs directly comparable. Each periodic image is retained as
`<run>.<stage>.epoch_NNN.reconstruction_examples.png`. The interval resets at
the start of each masking stage. The default `0` disables periodic snapshots;
the unnumbered image remains the best-checkpoint plot from the latest completed
stage.

Resume a saved model for more epochs with `--resume`. Unless `--stages` is
provided, training continues the masking stage recorded in the checkpoint.
The continuation runs all requested epochs rather than stopping early.

```powershell
.\.venv\Scripts\python.exe -m masked_pretraining train `
  --resume .\masked_pretraining\runs\checkpoints\masked_pretraining.pt `
  --epochs-per-stage 10 `
  --checkpoint .\masked_pretraining\runs\checkpoints\masked_pretraining_resumed.pt
```

Use the same model and data arguments as the original run; incompatible model
configuration, sources, normalization, or sensor splits are rejected. New
checkpoints include optimizer state. Older weight-only checkpoints can also be
resumed, but begin the continuation with a fresh optimizer. Only resume
checkpoints you trust because PyTorch checkpoint loading can execute serialized
code.

## Readiness check

Run `.\.venv\Scripts\python.exe -m masked_pretraining audit` immediately before
training. It reports selected indoor sensors, intervals, outdoor handoffs,
windows crossing handoffs, hard gaps, and eligible windows using only the
selected `training_data.csv`.
