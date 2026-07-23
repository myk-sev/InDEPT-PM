"""Download and retain all historical 72-hour corrected NAQFC PM2.5 GRIBs."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from urllib.error import HTTPError

from pull_naqfc import CYCLES, START_DATE, download_file, format_bytes, format_duration, jobs_between, parse_date, source_url, utc_now_date, version_for


def grib_path(output: Path, day: date, cycle: int) -> Path:
    return output / f"naqfc_{day:%Y%m%d}T{cycle:02d}.grib2"


def write_manifest(output: Path, rows: dict[str, dict[str, object]]) -> bool:
    fields = ("date", "cycle_utc", "model_version", "status", "source_url", "bytes", "etag", "error")
    target = output / "grib_manifest.csv"
    temporary = target.with_suffix(".csv.part")
    for _ in range(20):
        try:
            with temporary.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows[key] for key in sorted(rows))
            temporary.replace(target)
            return True
        except PermissionError:
            time.sleep(0.5)
    print(f"Warning: OneDrive is locking {target.name}; progress will be written on a later cycle.", file=sys.stderr)
    return False


def submit(executor: ThreadPoolExecutor, output: Path, day: date, cycle: int) -> Future[tuple[int, str]]:
    return executor.submit(download_file, source_url(day, cycle), grib_path(output, day, cycle))


def run(args: argparse.Namespace) -> int:
    if args.start < START_DATE:
        raise ValueError("--start cannot be earlier than 2021-07-20; earlier forecasts are 48-hour")
    if args.end < args.start:
        raise ValueError("--end must not precede --start")
    if not 1 <= args.download_workers <= 32:
        raise ValueError("--download-workers must be between 1 and 32")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, object]] = {}
    target = args.output / "grib_manifest.csv"
    if target.exists():
        with target.open(newline="", encoding="utf-8") as file:
            manifest = {f"{row['date']}T{row['cycle_utc']}": row for row in csv.DictReader(file)}
    jobs = jobs_between(args.start, args.end)
    pending = iter(jobs)
    futures: dict[Future[tuple[int, str]], tuple[date, int]] = {}
    started = time.monotonic()
    resolved = failures = 0
    with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
        for _ in range(min(args.download_workers, len(jobs))):
            day, cycle = next(pending)
            futures[submit(executor, args.output, day, cycle)] = (day, cycle)
        while futures:
            future = next(as_completed(futures))
            day, cycle = futures.pop(future)
            key = f"{day.isoformat()}T{cycle:02d}"
            record: dict[str, object] = {"date": day.isoformat(), "cycle_utc": f"{cycle:02d}", "model_version": version_for(day), "source_url": source_url(day, cycle), "bytes": "", "etag": "", "error": ""}
            try:
                size, etag = future.result()
                manifest[key] = {**record, "status": "complete", "bytes": size, "etag": etag}
                message = f"Complete {key} (GRIB {format_bytes(size)})"
            except HTTPError as error:
                manifest[key] = {**record, "status": "source_missing", "error": f"HTTP {error.code}"}
                message = f"Missing {key}"
            except Exception as error:
                failures += 1
                manifest[key] = {**record, "status": "failed", "error": str(error)}
                message = f"Failed {key}: {error}"
            resolved += 1
            next_job = next(pending, None)
            if next_job is not None:
                futures[submit(executor, args.output, *next_job)] = next_job
            elapsed = time.monotonic() - started
            eta = elapsed / resolved * (len(jobs) - resolved)
            print(f"[{resolved}/{len(jobs)}] {message} | elapsed {format_duration(elapsed)} | ETA {format_duration(eta)}", file=sys.stderr if message.startswith("Failed") else sys.stdout)
            write_manifest(args.output, manifest)
    return 1 if failures else 0


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="directory that retains all downloaded GRIBs")
    parser.add_argument("--start", type=parse_date, default=START_DATE)
    parser.add_argument("--end", type=parse_date, default=utc_now_date())
    parser.add_argument("--download-workers", type=int, default=8, help="parallel downloads (default: 8)")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(arguments()))
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)
