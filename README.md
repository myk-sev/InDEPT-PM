# NAQFC historical PM2.5 pull

This tool retrieves the corrected hourly PM2.5 NAQFC forecast at each supplied
latitude/longitude. It starts on 2021-07-20, the first 72-hour AQMv6 run, and
uses both the 06Z and 12Z cycles. It deliberately does not request the older
48-hour archive.

## Setup

In Command Prompt, from this folder:

```bat
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe pull_naqfc.py --setup-wgrib2
```

`locations.csv` needs only these columns:

```csv
latitude,longitude
40.4406,-79.9959
```

## Run

```bat
.venv\Scripts\python.exe pull_naqfc.py --locations locations.csv --output naqfc_output
```

Eight forecasts download in parallel by default. Change the limit with
`--download-workers`, for example `--download-workers 4`.

Use `--start` and `--end` for a smaller run:

```bat
.venv\Scripts\python.exe pull_naqfc.py --locations locations.csv --output naqfc_output --start 2024-01-01 --end 2024-01-31
```

The pull keeps up to eight GRIB downloads in progress while extraction and
Parquet writing remain serial. After each cycle is written, its GRIB file is
removed and another download enters the bounded queue. A restart skips valid
completed Parquet files, reuses complete GRIB files already in the scratch
folder, and resumes matching `.part` downloads before retrying failed work.

Output includes `locations.parquet`, `run_manifest.csv`, and cycle Parquet
files partitioned by model version/year/month. Each row contains the original
coordinate, nearest-grid coordinate and distance, cycle and valid times,
forecast hour, and `pm25_corrected_ug_m3`.

## Retain the source forecasts for future coordinate sets

The point pull normally deletes each GRIB after extracting it. To keep the full
archive for later coordinate choices, download it once:

```bat
.venv\Scripts\python.exe download_naqfc_gribs.py --output naqfc_gribs
```

This requires roughly 206 GB of free disk space. It keeps every source GRIB,
resumes `.part` files, and skips already-complete GRIBs. Generate a new
coordinate-specific Parquet output from that local archive with a new output
directory:

```bat
.venv\Scripts\python.exe pull_naqfc.py --locations new_locations.csv --output naqfc_output_v2 --scratch naqfc_gribs --keep-grib
```

## Read the Parquet data

Read one forecast file with PyArrow:

```python
import pyarrow.parquet as pq

table = pq.read_table("naqfc_output/model_version=AQMv7/year=2026/month=07/naqfc_20260722T06.parquet")
```

For all forecast files, select only the partitioned files. Do not scan the
output root directly because it also contains the separate `locations.parquet`:

```python
from pathlib import Path
import pyarrow.dataset as ds

files = [str(path) for path in Path("naqfc_output").glob("model_version=*/year=*/month=*/*.parquet")]
dataset = ds.dataset(files, format="parquet")
table = dataset.to_table(columns=["location_id", "cycle_time_utc", "forecast_hour", "pm25_corrected_ug_m3"])
```

DuckDB can query all forecast files without loading the whole dataset into
memory:

```sql
SELECT *
FROM read_parquet('naqfc_output/model_version=*/year=*/month=*/*.parquet');
```

## Cyclical-time model variants

Two additional model names add UTC daily, weekly, and annual sine/cosine
features to both the historical observations and future NAQFC inputs:

- `transformer-cyclical` uses patches.
- `transformer-no-patches-cyclical` uses one token per hour.

Select either with `pm25_transformer.py train --model MODEL_NAME`. The original
`transformer` and `transformer-no-patches` inputs and checkpoints are unchanged.

## Delta model variants

These models learn changes from the last indoor PM2.5 observation while still
returning absolute PM2.5 predictions:

- `transformer-cyclical-delta`
- `transformer-no-patches-cyclical-delta`
- `patchtst-delta`

PatchTST also has cyclical-input versions:

- `patchtst-cyclical`
- `patchtst-cyclical-delta`

## Resume interrupted training

Training writes a recovery checkpoint after every completed epoch alongside the
best-model checkpoint. For `--checkpoint model.pt`, the recovery file is
`checkpoints/model.last.pt`.

Re-run the original training command with `--resume`. The `--epochs` value is
the total target, not the number of additional epochs:

```bat
.venv\Scripts\python.exe pm25_transformer.py train ... --checkpoint model.pt --epochs 50 --resume
```

If the recovery file does not exist for an older run, `--resume` falls back to
the best-model checkpoint and starts a fresh optimizer. Subsequent epochs create
the full recovery checkpoint.
