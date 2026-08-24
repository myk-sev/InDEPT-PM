import csv
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from masked_pretraining.data import (
    Normalizer,
    PairWindowDataset,
    build_database,
    file_sha256,
    load_interval_metadata,
    load_training_intervals,
    read_purpleair_history,
    split_series,
)


START = 1_704_067_200


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.intervals = self.root / "training_intervals.csv"

    def tearDown(self):
        self.temporary.cleanup()

    def test_stitches_adjacent_outdoor_assignments_inside_window(self):
        _write_intervals(
            self.intervals,
            (
                _interval(1, 2, 0, 84, 1),
                _interval(1, 4, 84, 192, 2),
            ),
        )
        history = self.root / "history"
        history.mkdir()
        for sensor in (1, 2, 4):
            _write_history(history / f"{sensor}_history.csv", sensor, 192)

        selection = load_training_intervals(self.intervals)
        loaded = read_purpleair_history([history], {1, 2, 4})
        database = build_database(selection, loaded, 168, 144, 24)

        self.assertEqual(database.window_count, 2)
        self.assertEqual(database.outdoor_handoff_count, 1)
        self.assertEqual(database.windows_crossing_handoffs, 2)
        normalizer = Normalizer.fit(database.series, [0])
        sample = PairWindowDataset(database, [0], normalizer)[0]
        self.assertEqual(sample["outdoor_sensor_ids"].tolist(), [2] * 84 + [4] * 84)
        self.assertEqual(tuple(sample["values"].shape), (168, 2))

    def test_manifest_gap_is_a_hard_window_boundary(self):
        _write_intervals(
            self.intervals,
            (
                _interval(1, 2, 0, 6, 1),
                _interval(1, 4, 7, 24, 2),
            ),
        )
        history = self.root / "history"
        history.mkdir()
        for sensor in (1, 2, 4):
            _write_history(history / f"{sensor}_history.csv", sensor, 24)

        selection = load_training_intervals(self.intervals)
        database = build_database(
            selection, read_purpleair_history([history], {1, 2, 4}), 12, 10, 6
        )

        self.assertEqual(database.series[0].window_starts, (12,))
        self.assertEqual(database.sensor_summaries[0]["hard_gap_hours"], 1)
        self.assertAlmostEqual(
            Normalizer.fit(database.series, [0]).mean[1],
            sum(1 + hour / 10 for hour in range(24) if hour != 6) / 23,
        )

    def test_rejects_overlapping_intervals(self):
        _write_intervals(
            self.intervals,
            (
                _interval(1, 2, 0, 12, 1),
                _interval(1, 4, 6, 18, 2),
            ),
        )

        with self.assertRaisesRegex(ValueError, "overlapping training intervals"):
            load_training_intervals(self.intervals)

    def test_accepts_downloaded_non_school_interval(self):
        _write_intervals(
            self.intervals,
            (_interval(1, 2, 0, 12, 1, "downloaded_history"),),
        )

        selection = load_training_intervals(self.intervals)

        self.assertEqual(selection.intervals[0].cohort_sources, "downloaded_history")

    def test_responsiveness_filter_turns_disallowed_interval_into_gap(self):
        _write_intervals(
            self.intervals,
            (
                _interval(1, 2, 0, 12, 1),
                _interval(1, 4, 12, 24, 2),
            ),
        )
        responsiveness = self.root / "responsiveness.csv"
        _write_csv(
            responsiveness,
            ("indoor_sensor_id", "outdoor_sensor_id", "responsiveness_tier"),
            ((1, 2, "high"), (1, 4, "low")),
        )

        selection = load_training_intervals(self.intervals, responsiveness, {"high"})

        history = self.root / "history"
        history.mkdir()
        for sensor in (1, 2, 4):
            _write_history(history / f"{sensor}_history.csv", sensor, 24)
        database = build_database(
            selection, read_purpleair_history([history], {1, 2, 4}), 12, 10, 6
        )

        self.assertEqual(
            [item.assignment_id for item in selection.intervals],
            [f"1-2-{START}-{START + 12 * 3600}"],
        )
        self.assertEqual(selection.responsiveness_filtered_interval_count, 1)
        self.assertEqual(database.window_count, 1)

    def test_split_keeps_indoor_sensors_disjoint_when_outdoor_is_reused(self):
        _write_intervals(
            self.intervals,
            (_interval(1, 2, 0, 24, 1), _interval(3, 2, 0, 24, 1)),
        )
        history = self.root / "history"
        history.mkdir()
        for sensor in range(1, 4):
            _write_history(history / f"{sensor}_history.csv", sensor, 24)
        selection = load_training_intervals(self.intervals)
        database = build_database(
            selection, read_purpleair_history([history], set(range(1, 4))), 12, 10, 6
        )

        train, validation = split_series(database, 0.5, 42)

        self.assertNotEqual(
            database.series[train[0]].indoor_id,
            database.series[validation[0]].indoor_id,
        )

    def test_natural_missing_reading_uses_observed_hour_threshold(self):
        _write_intervals(self.intervals, (_interval(1, 2, 0, 12, 1),))
        history = self.root / "history"
        history.mkdir()
        _write_history(history / "1_history.csv", 1, 12)
        _write_csv(
            history / "2_history.csv",
            ("time_stamp", "sensor_index", "pm2.5_atm"),
            (
                (START + hour * 3600, 2, 2 + hour / 10)
                for hour in range(12)
                if hour != 5
            ),
        )
        database = build_database(
            load_training_intervals(self.intervals),
            read_purpleair_history([history], {1, 2}),
            12,
            11,
            1,
        )
        sample = PairWindowDataset(
            database, [0], Normalizer.fit(database.series, [0])
        )[0]

        self.assertEqual(database.window_count, 1)
        self.assertFalse(sample["observed"][5, 0])

    def test_metadata_rejects_changed_reviewed_exclusion(self):
        _write_intervals(self.intervals, (_interval(1, 2, 0, 12, 1),))
        exclusion = self.root / "reviewed.csv"
        exclusion.write_text("sensor_id\n1\n", encoding="utf-8")
        metadata = self.intervals.with_suffix(".meta.json")
        metadata.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "manifest": {"sha256": file_sha256(self.intervals)},
                    "exclusions": [
                        {
                            "path": str(exclusion.resolve()),
                            "sha256": file_sha256(exclusion),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        exclusion.write_text("sensor_id\n2\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "reviewed exclusion hashes changed"):
            load_interval_metadata(metadata, self.intervals, (exclusion,))


def _interval(indoor, outdoor, start, end, rank, cohort="fema_school"):
    return (
        f"{indoor}-{outdoor}-{START + start * 3600}-{START + end * 3600}",
        indoor,
        f"Indoor {indoor}",
        outdoor,
        f"Outdoor {outdoor}",
        rank * 10,
        rank,
        _iso(start),
        _iso(end),
        cohort,
        "nearest_outdoor_sensor" if rank == 1 else "fallback_due_to_exclusion",
    )


def _iso(hour):
    return datetime.fromtimestamp(START + hour * 3600, timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def _write_intervals(path, rows):
    _write_csv(
        path,
        (
            "assignment_id",
            "indoor_sensor_id",
            "indoor_name",
            "outdoor_sensor_id",
            "outdoor_name",
            "distance_meters",
            "candidate_rank",
            "start_utc",
            "end_utc",
            "cohort_sources",
            "selection_reason",
        ),
        rows,
    )


def _write_history(path, sensor, hours):
    _write_csv(
        path,
        ("time_stamp", "sensor_index", "pm2.5_atm"),
        ((START + hour * 3600, sensor, sensor + hour / 10) for hour in range(hours)),
    )


def _write_csv(path: Path, header, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(header)
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
