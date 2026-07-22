"""Download corrected 72-hour NAQFC PM2.5 forecasts at supplied coordinates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pyarrow as pa
import pyarrow.parquet as pq


ARCHIVE = "https://noaa-nws-naqfc-pds.s3.amazonaws.com"
WGRIB2 = "https://ftp.cpc.ncep.noaa.gov/wd51we/wgrib2/Windows10/v3.1.3"
WGRIB2_FILES = (
    "wgrib2.exe", "cyggcc_s-seh-1.dll", "cyggfortran-5.dll", "cyggomp-1.dll",
    "cygquadmath-0.dll", "cygwin1.dll",
)
START_DATE = date(2021, 7, 20)
V7_START = date(2024, 5, 14)
CYCLES = (6, 12)
POINT = re.compile(r"lon=([^,:]+),lat=([^,:]+)(?::i=[^:]+:ix=[^:]+:iy=[^:]+)?[, :]val=([^:\s]+)")
FIELD = re.compile(r"D=(\d{14}).*?vt=(\d{14}).*?:(\d+)-(\d+) hour ave fcst")


def utc_now_date() -> date:
    return datetime.now(timezone.utc).date()


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def version_for(day: date) -> str:
    return "AQMv7" if day >= V7_START else "AQMv6"


def source_url(day: date, cycle: int) -> str:
    stamp = day.strftime("%Y%m%d")
    name = f"aqm.t{cycle:02d}z.ave_1hr_pm25_bc.{stamp}.227.grib2"
    return f"{ARCHIVE}/{version_for(day)}/CS/{stamp}/{cycle:02d}/{name}"


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_locations(path: Path) -> tuple[list[dict[str, object]], str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"latitude", "longitude"} <= set(reader.fieldnames):
            raise ValueError("locations CSV must have latitude,longitude headers")
        rows = []
        for number, row in enumerate(reader, 1):
            try:
                latitude = float(row["latitude"])
                longitude = float(row["longitude"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"locations row {number} has invalid coordinates") from error
            if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
                raise ValueError(f"locations row {number} is outside latitude/longitude bounds")
            rows.append({"location_id": f"location_{number:06d}", "latitude": latitude, "longitude": longitude})
    if not rows:
        raise ValueError("locations CSV has no data rows")
    if len(rows) > 1000:
        raise ValueError("a maximum of 1,000 locations is supported")
    return rows, digest


def write_locations(output: Path, locations: list[dict[str, object]]) -> None:
    table = pa.table({name: [row[name] for row in locations] for name in ("location_id", "latitude", "longitude")}, schema=pa.schema([
        pa.field("location_id", pa.string()), pa.field("latitude", pa.float64()), pa.field("longitude", pa.float64()),
    ]))
    temporary = output / "locations.parquet.part"
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(output / "locations.parquet")


def setup_wgrib2(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name in WGRIB2_FILES:
        target = folder / name
        if target.exists():
            continue
        print(f"Downloading {name}")
        download_file(f"{WGRIB2}/{name}", target, retries=3)


def remote_headers(url: str) -> tuple[int, str]:
    request = Request(url, method="HEAD")
    with urlopen(request, timeout=60) as response:
        return int(response.headers.get("Content-Length", "0")), response.headers.get("ETag", "")


def download_file(url: str, target: Path, retries: int = 5) -> tuple[int, str]:
    partial = target.with_suffix(target.suffix + ".part")
    sidecar = partial.with_suffix(partial.suffix + ".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            size, etag = remote_headers(url)
            saved = json.loads(sidecar.read_text()) if sidecar.exists() else {}
            offset = partial.stat().st_size if partial.exists() and saved == {"url": url, "size": size, "etag": etag} else 0
            if offset == 0:
                partial.unlink(missing_ok=True)
                atomic_json(sidecar, {"url": url, "size": size, "etag": etag})
            request = Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
            with urlopen(request, timeout=120) as response, partial.open("ab") as file:
                shutil.copyfileobj(response, file, length=1024 * 1024)
            if partial.stat().st_size != size:
                raise OSError(f"download is {partial.stat().st_size} bytes; expected {size}")
            partial.replace(target)
            sidecar.unlink(missing_ok=True)
            return size, etag
        except HTTPError as error:
            if error.code == 404:
                raise
            last_error: Exception = error
        except (URLError, OSError, TimeoutError, ValueError) as error:
            last_error = error
        if attempt + 1 < retries:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"could not download {url}: {last_error}")


def to_utc(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def normal_longitude(value: float) -> float:
    return value - 360 if value > 180 else value


def format_bytes(value: int | str) -> str:
    value = int(value or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return "0 B"


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "estimating"
    seconds = max(0, round(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radians = math.pi / 180
    lat1, lon1, lat2, lon2 = (value * radians for value in (lat1, lon1, lat2, lon2))
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(a))


def extract(wgrib2: Path, grib: Path, locations: list[dict[str, object]], day: date, cycle: int) -> list[dict[str, object]]:
    command = [str(wgrib2), str(grib), "-T", "-VT", "-ftime"]
    for location in locations:
        command.extend(("-lon", str(float(location["longitude"]) % 360), str(location["latitude"])))
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    rows: list[dict[str, object]] = []
    expected_leads = set(range(1, 73))
    found_leads: set[int] = set()
    for line in completed.stdout.splitlines():
        field = FIELD.search(line)
        points = POINT.findall(line)
        if not field or not points:
            continue
        _, valid_text, start, end = field.groups()
        lead = int(end)
        if int(start) != lead - 1 or len(points) != len(locations):
            raise RuntimeError(f"unexpected point output for {day} {cycle:02d}Z lead {lead}")
        found_leads.add(lead)
        valid_time = to_utc(valid_text)
        for location, (grid_lon, grid_lat, value) in zip(locations, points):
            try:
                pm25 = float(value)
            except ValueError:
                pm25 = None
            grid_latitude = float(grid_lat)
            grid_longitude = normal_longitude(float(grid_lon))
            rows.append({
                **location, "grid_latitude": grid_latitude, "grid_longitude": grid_longitude,
                "grid_distance_km": distance_km(float(location["latitude"]), float(location["longitude"]), grid_latitude, grid_longitude),
                "model_version": version_for(day), "cycle_time_utc": datetime(day.year, day.month, day.day, cycle, tzinfo=timezone.utc),
                "forecast_hour": lead, "valid_time_utc": valid_time, "pm25_corrected_ug_m3": pm25,
            })
    if found_leads != expected_leads or len(rows) != 72 * len(locations):
        raise RuntimeError(f"expected 72 forecast hours for {day} {cycle:02d}Z; received {len(found_leads)}")
    return rows


SCHEMA = pa.schema([
    pa.field("location_id", pa.string()), pa.field("latitude", pa.float64()), pa.field("longitude", pa.float64()),
    pa.field("grid_latitude", pa.float64()), pa.field("grid_longitude", pa.float64()), pa.field("grid_distance_km", pa.float32()),
    pa.field("model_version", pa.string()), pa.field("cycle_time_utc", pa.timestamp("s", tz="UTC")),
    pa.field("forecast_hour", pa.uint8()), pa.field("valid_time_utc", pa.timestamp("s", tz="UTC")),
    pa.field("pm25_corrected_ug_m3", pa.float32()),
])


def write_cycle(path: Path, rows: list[dict[str, object]], source: str, etag: str) -> None:
    columns = {field.name: [row[field.name] for row in rows] for field in SCHEMA}
    table = pa.table(columns, schema=SCHEMA).replace_schema_metadata({b"source_url": source.encode(), b"source_etag": etag.encode()})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.part")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(path)


def cycle_path(output: Path, day: date, cycle: int) -> Path:
    return output / f"model_version={version_for(day)}" / f"year={day:%Y}" / f"month={day:%m}" / f"naqfc_{day:%Y%m%d}T{cycle:02d}.parquet"


def valid_cycle(path: Path, expected_rows: int) -> bool:
    try:
        return path.exists() and pq.read_metadata(path).num_rows == expected_rows
    except (OSError, pa.ArrowException):
        return False


def write_manifest(output: Path, manifest: dict[str, dict[str, object]]) -> None:
    fields = ("date", "cycle_utc", "model_version", "status", "source_url", "bytes", "etag", "rows", "error")
    target = output / "run_manifest.csv"
    temporary = target.with_suffix(".csv.part")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest[key] for key in sorted(manifest))
    temporary.replace(target)


def configure(output: Path, locations_hash: str, start: date, end: date) -> None:
    target = output / "run_config.json"
    current = {"locations_sha256": locations_hash, "start": start.isoformat(), "product": "corrected hourly PM2.5, 72-hour NAQFC forecasts"}
    if target.exists():
        prior = json.loads(target.read_text(encoding="utf-8"))
        if prior["locations_sha256"] != locations_hash:
            raise ValueError("locations CSV changed; use a new --output directory")
        return
    atomic_json(target, current)


def jobs_between(start: date, end: date) -> list[tuple[date, int]]:
    jobs = []
    while start <= end:
        jobs.extend((start, cycle) for cycle in CYCLES)
        start += timedelta(days=1)
    return jobs


def run(args: argparse.Namespace) -> int:
    if args.setup_wgrib2:
        setup_wgrib2(args.wgrib2.parent)
        return 0
    if args.locations is None or args.output is None:
        raise ValueError("--locations and --output are required unless using --setup-wgrib2")
    locations, locations_hash = read_locations(args.locations)
    if args.start < START_DATE:
        raise ValueError("--start cannot be earlier than 2021-07-20; earlier forecasts are 48-hour")
    if args.end < args.start:
        raise ValueError("--end must not precede --start")
    args.output.mkdir(parents=True, exist_ok=True)
    configure(args.output, locations_hash, args.start, args.end)
    write_locations(args.output, locations)
    if not args.wgrib2.exists():
        raise FileNotFoundError(f"wgrib2 is missing: run with --setup-wgrib2 first ({args.wgrib2})")
    manifest: dict[str, dict[str, object]] = {}
    if (args.output / "run_manifest.csv").exists():
        with (args.output / "run_manifest.csv").open(newline="", encoding="utf-8") as file:
            manifest = {f"{row['date']}T{row['cycle_utc']}": row for row in csv.DictReader(file)}
    jobs = jobs_between(args.start, args.end)
    complete_paths = {cycle_path(args.output, day, cycle) for day, cycle in jobs if valid_cycle(cycle_path(args.output, day, cycle), 72 * len(locations))}
    pending_total = len(jobs) - len(complete_paths)
    attempted = 0
    durations: list[float] = []
    started = time.monotonic()
    failures = 0
    for number, (day, cycle) in enumerate(jobs, 1):
        key = f"{day.isoformat()}T{cycle:02d}"
        destination = cycle_path(args.output, day, cycle)
        source = source_url(day, cycle)
        record: dict[str, object] = {"date": day.isoformat(), "cycle_utc": f"{cycle:02d}", "model_version": version_for(day), "source_url": source, "bytes": "", "etag": "", "rows": "", "error": ""}
        if destination in complete_paths:
            prior = manifest.get(key, {})
            manifest[key] = {**record, **prior, "status": "complete", "rows": 72 * len(locations)}
            eta = (sum(durations) / len(durations)) * (pending_total - attempted) if durations else None
            print(f"[{number}/{len(jobs)}] Already complete {key} | elapsed {format_duration(time.monotonic() - started)} | ETA {format_duration(eta)}")
            continue
        cycle_started = time.monotonic()
        message = ""
        try:
            size, etag = download_file(source, args.scratch / f"naqfc_{day:%Y%m%d}T{cycle:02d}.grib2")
            grib = args.scratch / f"naqfc_{day:%Y%m%d}T{cycle:02d}.grib2"
            rows = extract(args.wgrib2, grib, locations, day, cycle)
            write_cycle(destination, rows, source, etag)
            grib.unlink(missing_ok=True)
            manifest[key] = {**record, "status": "complete", "bytes": size, "etag": etag, "rows": len(rows)}
            message = f"Complete {key} ({len(rows)} rows, GRIB {format_bytes(size)})"
        except HTTPError as error:
            manifest[key] = {**record, "status": "source_missing", "error": f"HTTP {error.code}"}
            message = f"Missing {key}"
        except Exception as error:  # Keep working; reruns retry only failed cycles.
            failures += 1
            manifest[key] = {**record, "status": "failed", "error": str(error)}
            message = f"Failed {key}: {error}"
        finally:
            durations.append(time.monotonic() - cycle_started)
            attempted += 1
            remaining = pending_total - attempted
            eta = (sum(durations) / len(durations)) * remaining
            print(f"[{number}/{len(jobs)}] {message} | elapsed {format_duration(time.monotonic() - started)} | ETA {format_duration(eta)}", file=sys.stderr if message.startswith("Failed") else sys.stdout)
            write_manifest(args.output, manifest)
    return 1 if failures else 0


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locations", type=Path, help="CSV with latitude,longitude columns")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--start", type=parse_date, default=START_DATE)
    parser.add_argument("--end", type=parse_date, default=utc_now_date())
    parser.add_argument("--scratch", type=Path, default=Path.cwd() / ".naqfc_scratch")
    parser.add_argument("--wgrib2", type=Path, default=Path.cwd() / ".tools" / "wgrib2" / "wgrib2.exe")
    parser.add_argument("--setup-wgrib2", action="store_true", help="download the official Windows wgrib2 files")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(arguments()))
    except (ValueError, FileNotFoundError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)
