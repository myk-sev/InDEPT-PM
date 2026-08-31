from __future__ import annotations

import argparse
import csv
import tempfile
import unittest
from pathlib import Path

from apply_initial_training_interval_audit import run, sha256


class ApplyInitialTrainingIntervalAuditTests(unittest.TestCase):
    def test_removes_only_excluded_train_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "training.csv"
            audit = root / "audit.csv"
            output = root / "filtered.csv"
            self.write_csv(
                training,
                ("sample_index", "split", "value"),
                ((1, "train", "a"), (2, "train", "b"), (3, "validation", "c")),
            )
            digest = sha256(training)
            self.write_csv(
                audit,
                (
                    "sample_index",
                    "split",
                    "training_data_sha256",
                    "exclude_from_initial_training",
                ),
                ((1, "train", digest, "false"), (2, "train", digest, "true")),
            )

            result = run(argparse.Namespace(training_data=training, interval_audit=audit, output=output))

            with output.open(encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(result, output.resolve())
            self.assertEqual([row["sample_index"] for row in rows], ["1", "3"])
            self.assertFalse(output.with_suffix(".summary.json").exists())

    def test_requires_complete_train_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "training.csv"
            audit = root / "audit.csv"
            output = root / "filtered.csv"
            self.write_csv(training, ("sample_index", "split"), ((1, "train"), (2, "train")))
            self.write_csv(
                audit,
                (
                    "sample_index",
                    "split",
                    "training_data_sha256",
                    "exclude_from_initial_training",
                ),
                ((1, "train", sha256(training), "true"),),
            )

            with self.assertRaisesRegex(ValueError, "exactly cover"):
                run(argparse.Namespace(training_data=training, interval_audit=audit, output=output))
            self.assertFalse(output.exists())

    @staticmethod
    def write_csv(
        path: Path, fieldnames: tuple[str, ...], rows: tuple[tuple[object, ...], ...]
    ) -> None:
        with path.open("w", encoding="utf-8", newline="") as target:
            writer = csv.writer(target)
            writer.writerow(fieldnames)
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
