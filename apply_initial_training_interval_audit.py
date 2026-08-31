"""Materialize a singular training CSV with audited train rows removed."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path


REQUIRED_AUDIT_COLUMNS = {
    "sample_index",
    "split",
    "training_data_sha256",
    "exclude_from_initial_training",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--interval-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_audit(path: Path, training_hash: str) -> dict[int, bool]:
    decisions: dict[int, bool] = {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = REQUIRED_AUDIT_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError("interval audit is missing columns: " + ", ".join(sorted(missing)))
        for number, row in enumerate(reader, 2):
            try:
                sample = int(row["sample_index"])
                decision = row["exclude_from_initial_training"].strip().lower()
                if row["split"] != "train" or decision not in {"true", "false"}:
                    raise ValueError("expected a train row and true/false decision")
                if row["training_data_sha256"] != training_hash:
                    raise ValueError("training data SHA-256 mismatch")
                if sample in decisions:
                    raise ValueError("duplicate sample_index")
            except (AttributeError, TypeError, ValueError) as error:
                raise ValueError(f"invalid interval audit row {number} in {path}: {error}") from error
            decisions[sample] = decision == "true"
    if not decisions:
        raise ValueError(f"interval audit contains no rows: {path}")
    return decisions


def run(args: argparse.Namespace) -> Path:
    training_data = args.training_data.resolve()
    interval_audit = args.interval_audit.resolve()
    output = args.output.resolve()
    if output == training_data:
        raise ValueError("--output must not overwrite --training-data")

    training_hash = sha256(training_data)
    decisions = read_audit(interval_audit, training_hash)
    excluded = {sample for sample, decision in decisions.items() if decision}
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    train_samples: set[int] = set()

    try:
        with training_data.open(encoding="utf-8-sig", newline="") as source, partial.open(
            "w", encoding="utf-8", newline=""
        ) as target:
            reader = csv.DictReader(source)
            missing = {"sample_index", "split"} - set(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    "training data is missing columns: " + ", ".join(sorted(missing))
                )
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
            writer.writeheader()
            for number, row in enumerate(reader, 2):
                try:
                    sample = int(row["sample_index"])
                    split = row["split"]
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"invalid training data row {number} in {training_data}"
                    ) from error
                if split == "train":
                    if sample in train_samples:
                        raise ValueError(f"duplicate train sample_index: {sample}")
                    train_samples.add(sample)
                if sample in excluded:
                    if split != "train":
                        raise ValueError(f"audit excludes non-train sample_index: {sample}")
                    continue
                writer.writerow(row)
        if train_samples != decisions.keys():
            raise ValueError("interval audit does not exactly cover the training-data train split")
        if not excluded:
            raise ValueError("interval audit does not exclude any train rows")
        os.replace(partial, output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    return output


if __name__ == "__main__":
    output = run(arguments())
    print(f"training_data={output}")
