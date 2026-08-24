"""Combine school and non-school PurpleAir histories into the model inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
AIR = ROOT / "data" / "purple air"
COLUMNS = ["time_stamp", "sensor_index", "pm2.5_atm"]
KEY = ["sensor_index", "time_stamp"]


def read_csv(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    if list(data.columns) != COLUMNS:
        raise ValueError(f"unexpected columns in {path}: {list(data.columns)}")
    if data.isna().any().any():
        raise ValueError(f"missing values in {path}")
    return data


def combine(first: pd.DataFrame, second: pd.DataFrame) -> pd.DataFrame:
    data = pd.concat([first, second], ignore_index=True)
    repeated = data[data.duplicated(KEY, keep=False)]
    if not repeated.empty and repeated.groupby(KEY)["pm2.5_atm"].nunique().gt(1).any():
        raise ValueError("conflicting PM2.5 values for the same sensor-hour")
    return data.drop_duplicates(KEY).sort_values(KEY).reset_index(drop=True)


def write_csv(data: pd.DataFrame, path: Path) -> None:
    part = path.with_suffix(path.suffix + ".part")
    if part.exists():
        raise FileExistsError(f"refusing to overwrite partial file: {part}")
    try:
        data.to_csv(part, index=False, lineterminator="\n")
        if not read_csv(part).equals(data):
            raise ValueError(f"verification failed for {path}")
        part.replace(path)
    except Exception:
        part.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--school-indoor", type=Path, default=AIR / "school_indoor_pm25.csv"
    )
    parser.add_argument("--non-school-indoor", type=Path, required=True)
    parser.add_argument(
        "--school-outdoor", type=Path, default=AIR / "school_outdoor_pm25.csv"
    )
    parser.add_argument("--non-school-outdoor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=AIR)
    parser.add_argument("--execute", action="store_true", help="write the combined CSVs")
    args = parser.parse_args()

    school_indoor = read_csv(args.school_indoor)
    non_school_indoor = read_csv(args.non_school_indoor)
    school_outdoor = read_csv(args.school_outdoor)
    non_school_outdoor = read_csv(args.non_school_outdoor)

    overlap = set(school_indoor.sensor_index) & set(non_school_indoor.sensor_index)
    if overlap:
        raise ValueError(f"school sensors found in non-school indoor input: {sorted(overlap)}")

    outputs = {
        args.output_dir / "all_indoor_pm25.csv": combine(school_indoor, non_school_indoor),
        args.output_dir / "all_outdoor_pm25.csv": combine(school_outdoor, non_school_outdoor),
    }
    for path, data in outputs.items():
        print(f"{path.name}: {len(data):,} rows, {data.sensor_index.nunique():,} sensors")
        if args.execute:
            write_csv(data, path)

    print("The separate k12_1km_outdoor_review archive was not read.")
    if not args.execute:
        print("Dry run only. Add --execute to write the combined CSVs.")


if __name__ == "__main__":
    main()
