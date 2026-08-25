# PurpleAir masked PM2.5 pretraining

This package pretrains a history encoder by reconstructing deliberately hidden
PurpleAir PM2.5 observations. It does not read TEMPO or NAQFC. TEMPO can be
introduced later when the pretrained history encoder is transferred into the
forecasting model.

## Static data contract

Masked training reads exactly one model input:
`inputs/masked_pretraining/exclusion_aware/k12_exclusion_aware_masked_training_data.csv`.
Each row contains
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
  --training-data C:\path\to\k12_no_exclusions_masked_training_data.csv
```

Natural missing observations are not reconstruction targets and are governed
by `--minimum-observed-hours`, which defaults to 144 observations per channel
in each 168-hour window.

The default dataset is the exclusion-aware K-12 cohort. All-retrieved-sensor
variants
are stored under `inputs/masked_pretraining/all_sensors/exclusion_aware` and
`inputs/masked_pretraining/all_sensors/no_exclusions`; select one by passing its
descriptively named `*_masked_training_data.csv` to `--training-data`.

## Training

```powershell
.\.venv\Scripts\python.exe -m masked_pretraining train `
  --model single-self-attention-encoder
```

The curriculum advances after validation plateaus:

1. isolated points;
2. two- and three-hour blocks;
3. mixed one-, three-, and six-hour blocks;
4. indoor cross-channel blocks with outdoor values visible;
5. final 3-, 6-, and 12-hour indoor suffixes.

Add the optional PurpleAir-to-TEMPO missingness bridge after that curriculum
with `--tempo-missingness-bridge`:

```powershell
.\.venv\Scripts\python.exe -m masked_pretraining train `
  --model dual-encoder-self-fusion-outdoor-availability `
  --tempo-missingness-bridge
```

The bridge progressively hides 50%, 70%, and finally six-sevenths (85.7%) of
the originally observed outdoor PurpleAir values in each window. It uses
synthetic 6-, 12-, 24-, and 48-hour blocks and masks 15% of observed indoor
values for reconstruction throughout. Naturally absent observations remain
excluded from the loss. The bridge reads no TEMPO data; it only brings the
outdoor input availability toward the downstream 24-of-168-hour floor while
preserving known PurpleAir values as reconstruction labels.

The flag works with every reconstruction model. Base models receive the same
`-9` missing-value sentinel, availability models additionally derive the binary
availability input, and availability-recency models additionally derive hours
since the last model-visible outdoor value. This keeps the three model families
directly comparable under identical synthetic gaps.

`--epochs-per-stage` accepts either one value, applied uniformly to every
selected stage, or one value per stage in `--stages` order. Without an explicit
`--stages` list, the seven values follow the base curriculum order above. With
`--tempo-missingness-bridge`, supply one uniform value or ten values in base
then bridge order:

```powershell
.\.venv\Scripts\python.exe -m masked_pretraining train `
  --epochs-per-stage 30 20 20 50 20 50 50
```

To add only the bridge to a checkpoint that completed all seven base stages:

```powershell
.\.venv\Scripts\python.exe -m masked_pretraining train `
  --resume .\inference\checkpoints\k12_excl_fine_t_dual-encoder-cross-fusion.pt `
  --tempo-missingness-bridge `
  --epochs-per-stage 10 `
  --checkpoint .\inference\checkpoints\k12_excl_fine_t_dual-encoder-cross-fusion.pt
```

An explicit bridge stage such as `--stages tempo_bridge_86` also requires the
flag. Bridge checkpoint metadata records the enabled flag, exact synthetic
fractions, and that no TEMPO data was used.

Each hour has the final eight-feature contract: normalized outdoor and indoor
PM2.5 followed by daily, weekly, and annual sine-cosine time features. Natural
gaps and artificially hidden values use `-9` in the affected PM2.5 value slot.
The artificial target mask remains outside the model, so only deliberately
hidden known values contribute to loss. Splitting is by `indoor_sensor_id`; a
seeded candidate search balances eligible-window volume, PM2.5 mass and high
events, and temporal quartiles. Normalization is fitted only on training sensors.

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

All eleven model selectors use the same reconstruction contract:

- `single-self-attention-encoder` (default) encodes the complete eight-feature
  history with one self-attention encoder;
- `dual-encoder-self-fusion` independently encodes indoor PM2.5 plus time and
  outdoor PM2.5 plus time, then jointly self-attends over both encoded streams;
- `dual-encoder-cross-fusion` uses the same two encoders, then lets each encoded
  PM2.5 stream cross-attend to the other without another self-attention step;
- `separate-stream-self-fusion` independently encodes time, indoor PM2.5, and
  outdoor PM2.5, then combines all three with self-attention only;
- `gru` retains the recurrent baseline.

Each of the three fusion architectures also has an
`-outdoor-availability` variant:

- `dual-encoder-self-fusion-outdoor-availability`;
- `dual-encoder-cross-fusion-outdoor-availability`;
- `separate-stream-self-fusion-outdoor-availability`.

These variants derive an outdoor availability value after masking: `1` when
the model-visible outdoor PM2.5 value is not the `-9` sentinel and `0`
otherwise. Outdoor PM2.5 and availability are projected and encoded together.
The dual-encoder variants then append the time features to that pair; the
separate-stream variant keeps its time encoder independent.

Each availability variant also has a cumulative
`-outdoor-availability-recency` counterpart. Recency is normalized hours since
the last model-visible outdoor observation. It is `0` when outdoor PM2.5 is
currently available, increases by `1 / history_hours` across a gap, and is
clamped to `1` before the first available value or after a full-history gap:

- `dual-encoder-self-fusion-outdoor-availability-recency`;
- `dual-encoder-cross-fusion-outdoor-availability-recency`;
- `separate-stream-self-fusion-outdoor-availability-recency`.

The transformer variants keep projections, positions, independent encoders,
and fusion modules under the checkpoint's retained transfer prefixes. Their
temporary indoor/outdoor reconstruction heads remain under the discarded
prefix.

The implemented supervised handoff, current-layout launchers, strict bridge
family verification, horizon curriculum, random controls, and baseline
evaluation are documented in [`../BRIDGE_FORECAST_WORKFLOW.md`](../BRIDGE_FORECAST_WORKFLOW.md).
Those launchers keep the same dataset/model stem across the dedicated artifact
folders under `inference`.

Add an architecture by decorating a `ModelConfig -> nn.Module` builder with
`register_model()` in `models.py`; it must return `[batch, time, 2]`. Direct CLI
runs derive `<dataset>_<model>` from the training CSV and model name, then write:

- `inference/checkpoints/<dataset_model>.pt`;
- `inference/metrics/<dataset_model>.csv`;
- `inference/graphs/<dataset_model>.png`;
- `inference/reports/<dataset_model>.csv`: final statistics by reconstruction
  and bridge stage;
- `inference/reconstructions/<dataset_model>/`: stage/epoch validation images.

Each run writes:

- `<dataset_model>.csv`: one row per epoch with training and validation
  loss, validation RMSE by channel, target counts, and checkpoint improvement;
- `<dataset_model>.png`: training and validation loss across all
  completed curriculum stages;
- `<run>.<stage>.epoch_NNN.reconstruction_examples.png`: four fixed validation windows
  showing the full label history, model-visible history, artificially masked
  labels, predictions, and natural missingness as gaps. The title identifies
  the dataset, model, masking stage, and selected epoch.

The metrics CSV and loss graph are refreshed after every epoch. After each
masking-stage block, the reconstruction plot uses that block's lowest-validation
epoch (20 epochs by default). Its absolute path is stored in checkpoint metadata.
After the final stage, the trainer writes one report row per completed stage
with indoor/outdoor RMSE in ug/m3, normalized validation loss, validation target
count, selected epoch, stage type, and checkpoint/data hashes.

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
the lowest-validation image is still written after each completed stage block.

Resume a saved model for more epochs with `--resume`. Unless `--stages` is
provided, training continues the masking stage recorded in the checkpoint.
The continuation runs all requested epochs rather than stopping early.

```powershell
.\.venv\Scripts\python.exe -m masked_pretraining train `
  --resume .\inference\checkpoints\k12_excl_fine_t_dual-encoder-cross-fusion.pt `
  --epochs-per-stage 10 `
  --checkpoint .\inference\checkpoints\k12_excl_fine_t_dual-encoder-cross-fusion.pt
```

Use the same model and data arguments as the original run; incompatible model
configuration, sources, normalization, or sensor splits are rejected. New
checkpoints include optimizer state. Older weight-only checkpoints can also be
resumed, but begin the continuation with a fresh optimizer. Only resume
checkpoints you trust because PyTorch checkpoint loading can execute serialized
code.

Checkpoints that record the former `transformer` model name remain resumable;
new checkpoints use `single-self-attention-encoder`.

## Readiness check

Run `.\.venv\Scripts\python.exe -m masked_pretraining audit` immediately before
training. It reports selected indoor sensors, intervals, outdoor handoffs,
windows crossing handoffs, hard gaps, and eligible windows using only the
selected `*_masked_training_data.csv`.
