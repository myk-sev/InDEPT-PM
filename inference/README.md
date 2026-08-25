# Cached inference graphs

These utilities generate retrospective PM2.5 forecast graphs without rebuilding
the complete evaluation dataset every time.

- [`build_inference_cache.py`](build_inference_cache.py) performs the expensive
  data-loading step once and saves selected samples.
- [`run_cached_inference.py`](run_cached_inference.py) loads those prepared
  samples, runs the trained model, and creates forecast graphs.

Project training launchers use dedicated type folders under `inference` and the
same `<dataset>_<model>` stem for every single-file artifact. Cached forecast
images go in one unnested folder under `inference/forecasts/<dataset_model>/`;
generated cache files go under `inference/caches/`.

## Run the utilities as modules

The `inference` directory is a Python package, identified by
[`__init__.py`](__init__.py). Run both utilities as modules from the project
root so Python can import `pm25_transformer.py`:

```powershell
.\.venv\Scripts\python.exe -m inference.build_inference_cache --help
.\.venv\Scripts\python.exe -m inference.run_cached_inference --help
```

Do not invoke the files directly:

```powershell
# Do not use
.\.venv\Scripts\python.exe .\inference\build_inference_cache.py
```

Direct file execution places the `inference` directory, rather than the project
root, at the beginning of Python's import path. The `-m inference...` form
preserves access to the shared components in `pm25_transformer.py`.

All commands below assume they are run from the project root with the project
virtual environment.

## Checkpoints and caches

Both files normally end in `.pt` because both are serialized with PyTorch's
`torch.save()`. The suffix identifies the storage format, not the file's
purpose.

| File | Contents |
|---|---|
| Model checkpoint | Trained model weights, model configuration, normalization values, and training configuration |
| Inference cache | Selected histories, forecasts, measured targets, sample names, and location metadata |

A clearer naming convention is:

```text
pm25_transformer_smoke_test.checkpoint.pt
inference_samples.cache.pt
```

The scripts do not require those additional suffixes. Existing checkpoint names
such as `pm25_transformer_smoke_test.pt` continue to work.

The cache does not contain model predictions or checkpoint-specific
normalization. Predictions are recalculated using the supplied checkpoint each
time `run_cached_inference.py` runs. A cache can therefore be reused across
models whose history length, prediction length, and time-feature layout are
compatible. The inference script validates that data contract before running.

## 1. Build a named cache

Use `--samples NAME=INDEX` to associate each temporal-test index with a
filename-safe name:

```powershell
.\.venv\Scripts\python.exe -m inference.build_inference_cache `
  --checkpoint .\inference\checkpoints\pm25_transformer_smoke_test.pt `
  --split temporal-test `
  --samples `
    normal=6281 `
    elevated=4399 `
    wildfire_incoming=2445 `
    wildfire_ongoing=3118 `
    wildfire_leaving=4227 `
  --output .\inference\caches\inference_samples.cache.pt
```

Names may contain letters, numbers, underscores, and hyphens. Names and indices
must each be unique.

The available splits are:

```text
train
validation
temporal-test
location-test
```

The default is `temporal-test`.

### Build an unnamed cache

`--indices` remains available when names are not needed:

```powershell
.\.venv\Scripts\python.exe -m inference.build_inference_cache `
  --checkpoint .\inference\checkpoints\pm25_transformer_smoke_test.pt `
  --split temporal-test `
  --indices 6281 4399 2445 3118 4227 `
  --output .\inference\caches\inference_samples.cache.pt
```

`--samples` and `--indices` are alternatives and cannot be supplied together.

## 2. Generate every cached graph

Omit `--indices` to generate a graph for every sample in the cache:

```powershell
.\.venv\Scripts\python.exe -m inference.run_cached_inference `
  --cache .\inference\caches\inference_samples.cache.pt `
  --checkpoint .\inference\checkpoints\pm25_transformer_smoke_test.pt
```

For the named example, this creates:

```text
inference/forecasts/pm25_transformer_smoke_test/MODEL_normal.png
inference/forecasts/pm25_transformer_smoke_test/MODEL_elevated.png
inference/forecasts/pm25_transformer_smoke_test/MODEL_wildfire_incoming.png
inference/forecasts/pm25_transformer_smoke_test/MODEL_wildfire_ongoing.png
inference/forecasts/pm25_transformer_smoke_test/MODEL_wildfire_leaving.png
inference/forecasts/pm25_transformer_smoke_test/stacked_inference_graphs.png
```

The stacked image places every selected graph in one vertical image and labels
each panel with its cache name, data split, and sample index. The training loss
graph remains separately at `inference/graphs/<dataset_model>.png`.

For an unnamed cache, filenames follow this pattern:

```text
temporal_test_sample_6281.png
```

## Generate a subset of cached graphs

Graph generation selects cached records by their original numeric indices. The
stored names are still used for the output filenames:

```powershell
.\.venv\Scripts\python.exe -m inference.run_cached_inference `
  --cache .\inference\caches\inference_samples.cache.pt `
  --checkpoint .\pm25_transformer_smoke_test.pt `
  --indices 2445 3118 4227 `
  --output-dir .\inference\wildfire_graphs
```

This creates:

```text
wildfire_incoming.png
wildfire_ongoing.png
wildfire_leaving.png
```

`--samples` is only used while building the cache. The inference utility reads
the names already stored in the cache.

## Device selection

Inference uses `--device auto` by default. It selects CUDA, then Intel XPU, and
then CPU according to availability. A device can be selected explicitly:

```powershell
.\.venv\Scripts\python.exe -m inference.run_cached_inference `
  --cache .\inference\caches\inference_samples.cache.pt `
  --checkpoint .\pm25_transformer_smoke_test.pt `
  --output-dir .\inference\graphs `
  --device cpu
```

## When to rebuild the cache

Rebuild it when:

- different sample indices or names are needed;
- a different data split is needed;
- a model needs different history, prediction, or time-feature inputs; or
- updated source data should be reflected in the samples.

Retraining a model or switching to another compatible model does not require a
cache rebuild. Version 1 caches remain supported and receive the same tensor
shape validation.

The first cache build still scans the complete source dataset and can take
several minutes. Subsequent graph generation loads only the compact cache and
normally takes seconds.

These are retrospective evaluation graphs. Each cached sample includes the
future measured indoor target so the graph can compare it with the model
prediction.

## Command help

```powershell
.\.venv\Scripts\python.exe -m inference.build_inference_cache --help
.\.venv\Scripts\python.exe -m inference.run_cached_inference --help
```
