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
