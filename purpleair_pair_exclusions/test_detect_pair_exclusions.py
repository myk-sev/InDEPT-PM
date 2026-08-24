import csv
import tempfile
import unittest
from pathlib import Path

from purpleair_pair_exclusions.location_history_explorer import (
    write_location_history_explorer,
)
from purpleair_pair_exclusions.detect_outdoor_quality_periods import (
    ExtremeMismatchCriteria,
    PeriodCriteria,
    exclusion_range_rows,
    is_extreme_mismatch,
    period_rows,
)
from purpleair_pair_exclusions.outdoor_quality import (
    OutdoorExclusion,
    exclude_outdoor_readings,
    read_indoor_exclusions,
    read_outdoor_exclusions,
)
from purpleair_pair_exclusions.outdoor_sensor_error import (
    ErrorLevelPeriod,
    detect_error_level_periods,
)
from purpleair_pair_exclusions.training_intervals import build_training_intervals

from purpleair_pair_exclusions.detect_pair_exclusions import (
    HOUR,
    Criteria,
    analyze_pairs,
    read_fema_school_ids,
    read_excluded_sensor_ids,
    read_histories,
    read_overlap_indoor_ids,
    read_reusable_school_pairs,
    select_pairs,
    sensor_rows,
    write_outputs,
)


PAIR = {
    "indoor_sensor_id": 1,
    "indoor_name": "Inside",
    "outdoor_sensor_id": 2,
    "outdoor_name": "Outside",
    "distance_meters": 10.0,
}


def histories(responds=False, two_events=False):
    hours = 140 if two_events else 70
    indoor = {hour * HOUR: 5.0 for hour in range(hours)}
    outdoor = {hour * HOUR: 10.0 for hour in range(hours)}
    starts = (30, 90) if two_events else (30,)
    for start in starts:
        for hour in range(start, start + 3):
            outdoor[hour * HOUR] = 100.0
        if responds:
            indoor[(start + 4) * HOUR] = 30.0
    return {1: indoor}, {2: outdoor}


class DetectorTests(unittest.TestCase):
    def test_training_intervals_restore_primary_after_partial_exclusion(self):
        candidates = {
            1: [
                PAIR | {"candidate_rank": 1},
                PAIR
                | {
                    "outdoor_sensor_id": 3,
                    "outdoor_name": "Fallback",
                    "distance_meters": 20.0,
                    "candidate_rank": 2,
                },
            ]
        }
        history = {1: {hour * HOUR: 5.0 for hour in range(10)}}

        intervals, unresolved = build_training_intervals(
            candidates,
            history,
            {"fema_school": {1}},
            set(),
            (),
            (OutdoorExclusion(2, 3 * HOUR, 5 * HOUR, "failed"),),
        )

        self.assertEqual(unresolved, [])
        self.assertEqual(
            [
                (row["outdoor_sensor_id"], row["start_utc"], row["end_utc"])
                for row in intervals
            ],
            [
                (2, "1970-01-01T00:00:00Z", "1970-01-01T03:00:00Z"),
                (3, "1970-01-01T03:00:00Z", "1970-01-01T05:00:00Z"),
                (2, "1970-01-01T05:00:00Z", "1970-01-01T10:00:00Z"),
            ],
        )

    def test_training_intervals_cascade_and_leave_indoor_gap(self):
        candidates = {
            1: [
                PAIR | {"candidate_rank": 1},
                PAIR
                | {
                    "outdoor_sensor_id": 3,
                    "outdoor_name": "Second",
                    "candidate_rank": 2,
                },
                PAIR
                | {
                    "outdoor_sensor_id": 4,
                    "outdoor_name": "Third",
                    "candidate_rank": 3,
                },
            ]
        }
        history = {1: {hour * HOUR: 5.0 for hour in range(10)}}

        intervals, unresolved = build_training_intervals(
            candidates,
            history,
            {"fema_school": {1}},
            set(),
            (OutdoorExclusion(1, 8 * HOUR, 9 * HOUR, "indoor failed"),),
            (
                OutdoorExclusion(2, 2 * HOUR, 7 * HOUR, "primary failed"),
                OutdoorExclusion(3, 4 * HOUR, 6 * HOUR, "secondary failed"),
            ),
        )

        self.assertEqual([row["outdoor_sensor_id"] for row in intervals], [2, 3, 4, 3, 2, 2])
        self.assertEqual(len(unresolved), 1)
        self.assertEqual(unresolved[0]["reason"], "indoor_excluded")
        self.assertEqual(unresolved[0]["start_utc"], "1970-01-01T08:00:00Z")

    def test_whole_indoor_exclusion_is_a_hard_gap(self):
        intervals, unresolved = build_training_intervals(
            {1: [PAIR | {"candidate_rank": 1}]},
            {1: {hour * HOUR: 5.0 for hour in range(10)}},
            {"fema_school": {1}},
            {1},
            (),
            (),
        )

        self.assertEqual(intervals, [])
        self.assertEqual(len(unresolved), 1)
        self.assertEqual(unresolved[0]["reason"], "indoor_sensor_excluded")

    def test_repeated_nonresponses_select_the_indoor_sensor(self):
        indoor, outdoor = histories(two_events=True)
        criteria = Criteria()
        events, coverage = analyze_pairs([PAIR], indoor, outdoor, criteria)
        sensors = sensor_rows(events, criteria)

        self.assertEqual(len(events), 2)
        self.assertTrue(all(row["selected_for_exclusion"] for row in events))
        self.assertTrue(sensors[0]["selected_for_exclusion"])
        self.assertEqual(coverage[0]["status"], "complete_pair")

    def test_delayed_indoor_response_is_retained(self):
        indoor, outdoor = histories(responds=True)
        events, _ = analyze_pairs([PAIR], indoor, outdoor, Criteria())

        self.assertEqual(len(events), 1)
        self.assertFalse(events[0]["selected_for_exclusion"])
        self.assertEqual(events[0]["selection_reason"], "measurable_indoor_response")

    def test_outputs_are_reviewable_and_loader_compatible(self):
        indoor, outdoor = histories(two_events=True)
        events, coverage = analyze_pairs([PAIR], indoor, outdoor, Criteria())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = write_outputs(
                output, events, coverage, Criteria(), review_outdoor_ids={2}
            )

            self.assertEqual(summary["source"], "paired PurpleAir pm2.5_atm only")
            self.assertTrue((output / "event_response_summary.png").is_file())
            explorer = (output / "event_explorer.html").read_text()
            self.assertIn("const report=", explorer)
            self.assertNotIn("Selected school pair coverage", explorer)
            self.assertIn('"outdoor_sensor_id": 2', explorer)
            self.assertIn('id="reading"', explorer)
            self.assertIn('id="location"', explorer)
            self.assertIn("b.indices.length-a.indices.length", explorer)
            self.assertIn("data-chart-hit", explorer)
            self.assertIn("Indoor window diagnostic", explorer)
            with (output / "selected_pairs.csv").open(newline="") as source:
                pair_rows = list(csv.DictReader(source))
            self.assertEqual(pair_rows[0]["outdoor_sensor_id"], "2")
            with (output / "excluded_sensors.csv").open(newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(rows[0]["sensor_id"], "1")
            self.assertEqual(summary["k12_1km_indoor_exclusion_candidates"], 1)
            self.assertTrue(
                (output / "k12_1km_indoor_exclusion_candidates.csv").is_file()
            )

    def test_full_history_explorer_is_searchable_and_defaults_to_all_time(self):
        indoor, outdoor = histories()
        indoor[3] = {0: 7.0, HOUR: 8.0}
        indoor[4] = {0: 9.0, HOUR: 10.0}
        outdoor[5] = {0: 20.0, HOUR: 21.0}
        exclusions = (
            OutdoorExclusion(2, 30 * HOUR, 40 * HOUR, "broken"),
            OutdoorExclusion(
                5,
                0,
                HOUR,
                "bad range",
                "Outside Five",
                "2026-08-24T00:00:37Z",
                "2026-08-23",
            ),
        )
        permanent = [
            {
                "sensor_id": 4,
                "sensor_name": "Inside Four",
                "reason": "permanently broken",
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            count = write_location_history_explorer(
                output,
                [PAIR],
                indoor,
                outdoor,
                exclusions,
                {3},
                permanent,
                k12_sensor_ids={1, 3, 4},
                review_outdoor_ids={2},
                review_indoor_ids={1},
            )
            explorer = (output / "location_history_explorer.html").read_text()
            data = (output / "location_history_data" / "1.js").read_text()
            unpaired = (
                output / "location_history_data" / "unpaired_indoor_3.js"
            ).read_text()
            excluded_indoor = (
                output / "location_history_data" / "excluded_indoor_4.js"
            ).read_text()
            excluded_outdoor = (
                output / "location_history_data" / "excluded_outdoor_5.js"
            ).read_text()

        self.assertEqual(count, 1)
        self.assertIn('type="search" list="locations"', explorer)
        self.assertEqual(explorer.count('type="datetime-local"'), 2)
        self.assertIn("Blank dates show the entire available history", explorer)
        self.assertIn('id="resetZoom"', explorer)
        self.assertIn("addEventListener('pointerdown'", explorer)
        self.assertIn("setPointerCapture", explorer)
        self.assertIn("Release to zoom", explorer)
        self.assertIn('href="#unpaired"', explorer)
        self.assertIn('href="#review"', explorer)
        self.assertIn('href="#recent"', explorer)
        self.assertIn('href="#excluded"', explorer)
        self.assertIn('id="previousLocation" type="button"', explorer)
        self.assertIn('id="nextLocation" type="button"', explorer)
        self.assertIn('id="k12Only" type="checkbox"', explorer)
        self.assertIn('id="undoExclusion" type="button"', explorer)
        self.assertIn('id="redoExclusion" type="button"', explorer)
        self.assertIn('id="deleteExclusion" type="button"', explorer)
        self.assertIn('id="saveExclusions" type="button"', explorer)
        self.assertIn('id="exportIndoor" type="button"', explorer)
        self.assertIn('id="exportOutdoor" type="button"', explorer)
        self.assertIn("window.showDirectoryPicker", explorer)
        self.assertIn("mode:'create'", explorer)
        self.assertIn("excluded_${type}_purpleair_ranges.csv", explorer)
        self.assertIn('originalExclusionManifest={"indoor":[],"outdoor":[', explorer)
        self.assertIn('"decision_date":"2026-08-23"', explorer)
        self.assertIn("k12Only.onchange=()=>showPage(true)", explorer)
        self.assertIn("allLocations.filter(location=>location.k12)", explorer)
        self.assertIn('"file":"1.js","k12":true', explorer)
        self.assertIn('reviewLocations=[{"indoor_sensor_id":1', explorer)
        self.assertIn('recentExcludedLocations=[{"sensor_id":5', explorer)
        self.assertIn(
            "Recent-data exclusions (${recentExcludedLocations.length})", explorer
        )
        self.assertIn("Inside (10.00 m)", explorer)
        self.assertIn('"file":"unpaired_indoor_3.js","k12":true', explorer)
        self.assertIn('"file":"excluded_indoor_4.js","k12":true', explorer)
        self.assertIn('"file":"excluded_outdoor_5.js","k12":false', explorer)
        self.assertIn("stepLocation(-1)", explorer)
        self.assertIn("stepLocation(1)", explorer)
        self.assertIn("previousLocation.disabled=index<=0", explorer)
        self.assertIn("nextLocation.disabled=index<0||index>=locations.length-1", explorer)
        self.assertIn('"file":"unpaired_indoor_3.js"', explorer)
        self.assertIn('"file":"excluded_indoor_4.js"', explorer)
        self.assertIn('"file":"excluded_outdoor_5.js"', explorer)
        self.assertIn("window.__loadPurpleAirLocation", data)
        self.assertIn('"reason":"broken"', data)
        self.assertIn('"paired":false', unpaired)
        self.assertIn('"series":[[0,7.0,null],[3600,8.0,null]]', unpaired)
        self.assertIn('"start":null,"end":null', excluded_indoor)
        self.assertIn('"reason":"permanently broken"', excluded_indoor)
        self.assertIn('"sensor_type":"outdoor"', excluded_outdoor)
        self.assertIn('"start":0,"end":3600', excluded_outdoor)

    def test_bad_outdoor_ranges_remove_only_matching_sensor_hours(self):
        values = {2: {10: 1.0, 20: 2.0, 30: 3.0}, 4: {20: 4.0}}
        filtered, removed = exclude_outdoor_readings(
            values, (OutdoorExclusion(2, 20, 30, "broken"),)
        )

        self.assertEqual(filtered, {2: {10: 1.0, 30: 3.0}, 4: {20: 4.0}})
        self.assertEqual(removed, 1)

    def test_isolated_high_readings_use_indoor_ranges(self):
        root = Path(__file__).resolve().parent.parent
        exclusions = root / "data" / "exclusions"
        ranges = read_indoor_exclusions(exclusions / "excluded_indoor_purpleair_ranges.csv")
        badger = [item for item in ranges if item.sensor_id == 106678]
        foster = [item for item in ranges if item.sensor_id == 163677]
        whole_sensor_ids = set().union(
            *(
                read_excluded_sensor_ids(path)
                for path in (
                    exclusions / "permanently_excluded_indoor_sensors.csv",
                    exclusions / "excluded_indoor_sensors_pm25_gt1000.csv",
                    exclusions / "excluded_indoor_schools_pm25_gt1000.csv",
                )
            )
        )

        self.assertEqual(len(badger), 1)
        self.assertEqual((badger[0].start, badger[0].end), (1630382400, 1630386000))
        self.assertEqual(len(foster), 1)
        self.assertEqual((foster[0].start, foster[0].end), (1725951600, 1725962400))
        self.assertNotIn(106678, whole_sensor_ids)
        self.assertNotIn(163677, whole_sensor_ids)

    def test_reviewed_outdoor_manifest_has_partial_sensor_ranges(self):
        manifest = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "exclusions"
            / "excluded_outdoor_purpleair_ranges.csv"
        )
        ranges = read_outdoor_exclusions(manifest)
        by_sensor = {}
        for item in ranges:
            by_sensor.setdefault(item.sensor_id, []).append(item)

        self.assertEqual(
            set(by_sensor),
            {
                3590,
                40209,
                46639,
                58699,
                64375,
                66451,
                70175,
                70249,
                78205,
                92169,
                96513,
                104680,
                112476,
                112880,
                114473,
                118897,
                120423,
                127669,
                130163,
                132807,
                134770,
                134792,
                138092,
                138110,
                147502,
                162157,
                164283,
                164965,
                175887,
                175913,
                179901,
                181779,
                185731,
                186881,
                187247,
                196427,
                218835,
                23195,
                227437,
                233187,
                234471,
                236857,
                252471,
                255423,
                268443,
                175889,
                198223,
            },
        )
        self.assertEqual(len(ranges), 88)
        self.assertEqual(
            sum(item.added_at_utc == "2026-08-24T00:00:37Z" for item in ranges),
            14,
        )
        self.assertTrue(all(item.decision_date for item in ranges))
        self.assertEqual(len(by_sensor[175913]), 3)
        self.assertEqual(len(by_sensor[198223]), 3)
        self.assertEqual(len(by_sensor[268443]), 4)
        self.assertEqual(len(by_sensor[23195]), 3)
        self.assertEqual(
            (by_sensor[23195][0].start, by_sensor[23195][0].end),
            (1609459200, 1672531200),
        )
        self.assertEqual(len(by_sensor[46639]), 2)
        self.assertEqual(len(by_sensor[58699]), 5)
        self.assertEqual(len(by_sensor[78205]), 2)
        self.assertEqual(len(by_sensor[114473]), 2)
        self.assertEqual(len(by_sensor[118897]), 4)
        self.assertEqual(len(by_sensor[70175]), 2)
        self.assertEqual(len(by_sensor[236857]), 3)
        self.assertEqual(len(by_sensor[252471]), 3)
        self.assertEqual(len(by_sensor[120423]), 4)
        self.assertEqual(len(by_sensor[138092]), 9)
        self.assertEqual(len(by_sensor[218835]), 3)
        self.assertEqual(by_sensor[46639][0].start, 1725400800)
        self.assertEqual(by_sensor[127669][0].start, 1721214000)
        self.assertIsNone(by_sensor[127669][0].end)
        self.assertIsNone(by_sensor[132807][0].start)
        self.assertIsNone(by_sensor[132807][0].end)
        self.assertIsNone(by_sensor[164283][0].start)
        self.assertIsNone(by_sensor[164283][0].end)
        self.assertEqual(len(by_sensor[138110]), 2)
        self.assertEqual(by_sensor[255423][0].start, 1755954000)
        self.assertEqual(by_sensor[255423][0].end, 1756684800)
        self.assertEqual(by_sensor[40209][0].start, 1609459200)
        self.assertEqual(by_sensor[40209][0].end, 1633046400)
        self.assertIsNone(by_sensor[196427][0].start)
        self.assertIsNone(by_sensor[196427][0].end)
        self.assertIsNone(by_sensor[233187][0].start)
        self.assertIsNone(by_sensor[233187][0].end)
        self.assertEqual(by_sensor[255423][0].sensor_name, "Colmesneil TX, outside")

    def test_recurring_error_level_detector_groups_observed_bands(self):
        readings = {0: 1664.0, HOUR: 2499.8, 2 * HOUR: 3332.0, 3 * HOUR: 20.0}

        periods = detect_error_level_periods(readings)

        self.assertEqual(len(periods), 1)
        self.assertEqual(periods[0].readings, 3)
        self.assertEqual(periods[0].end, 3 * HOUR)

    def test_error_level_detector_requires_repetition(self):
        readings = {0: 1664.0, HOUR: 3332.0}

        self.assertEqual(detect_error_level_periods(readings), ())

    def test_error_level_period_is_imported_as_review_candidate(self):
        start = 1735689600
        errors = {(2, 2025): (ErrorLevelPeriod(start, start + HOUR, 3, 1664, 1668),)}

        row = period_rows([], (), PeriodCriteria(), [PAIR], errors)[0]

        self.assertTrue(row["selected_for_review"])
        self.assertEqual(row["candidate_signals"], "recurring_error_level")
        self.assertEqual(row["error_level_readings"], 3)

    def test_outdoor_quality_periods_require_repeated_low_responses(self):
        def event(sensor, year, selected):
            ratio = 0.03 if selected else 0.20
            return {
                "indoor_sensor_id": sensor + 1,
                "indoor_name": "Inside",
                "outdoor_sensor_id": sensor,
                "outdoor_name": "Outside",
                "event_start_utc": f"{year}-07-01T00:00:00Z",
                "outdoor_rise_pm25": 80.0,
                "indoor_rise_pm25": 2.0,
                "peak_response_ratio": ratio,
                "area_response_ratio": ratio,
                "selected_for_exclusion": selected,
            }

        events = [event(10, 2024, True) for _ in range(4)]
        events += [event(10, 2024, False)]
        events += [event(20, 2025, True) for _ in range(2)]
        events += [event(20, 2025, False) for _ in range(2)]
        events += [event(30, 2025, True) for _ in range(4)]
        events += [event(30, 2025, False) for _ in range(3)]
        exclusions = (OutdoorExclusion(10, 1704067200, 1735689600, "broken"),)

        rows = period_rows(events, exclusions, PeriodCriteria())

        self.assertEqual(len(rows), 3)
        self.assertTrue(rows[0]["known_exclusion_overlap"])
        self.assertTrue(rows[0]["selected_for_review"])
        self.assertEqual(rows[0]["selection_reason"], "known_period_recovered")
        self.assertFalse(rows[1]["selected_for_review"])
        self.assertEqual(rows[1]["selection_reason"], "below_repeated_event_threshold")
        self.assertTrue(rows[2]["selected_for_review"])
        self.assertEqual(rows[2]["selection_reason"], "new_period_candidate")

    def test_outdoor_period_screen_uses_proportional_response(self):
        events = [
            {
                "indoor_sensor_id": 1,
                "indoor_name": "Inside",
                "outdoor_sensor_id": 2,
                "outdoor_name": "Outside",
                "event_start_utc": "2025-07-01T00:00:00Z",
                "outdoor_rise_pm25": 1656.3,
                "indoor_rise_pm25": 6.75,
                "peak_response_ratio": 0.0041,
                "area_response_ratio": 0.00044,
                "selected_for_exclusion": False,
            }
            for _ in range(3)
        ]

        row = period_rows(events, (), PeriodCriteria())[0]

        self.assertEqual(row["low_response_events"], 3)
        self.assertTrue(row["selected_for_review"])

    def test_extreme_mismatch_promotes_one_large_proportional_event(self):
        event = {
            "indoor_sensor_id": 1,
            "indoor_name": "Inside",
            "outdoor_sensor_id": 2,
            "outdoor_name": "Outside",
            "event_start_utc": "2025-11-01T02:00:00Z",
            "event_end_utc": "2025-11-01T07:00:00Z",
            "outdoor_rise_pm25": 740.4,
            "indoor_rise_pm25": 5.6,
            "peak_response_ratio": 0.0076,
            "area_response_ratio": 0.0106,
            "response_coverage": 1.0,
            "selected_for_exclusion": False,
        }

        self.assertTrue(is_extreme_mismatch(event, ExtremeMismatchCriteria()))
        row = period_rows([event], (), PeriodCriteria())[0]
        self.assertTrue(row["selected_for_review"])
        self.assertEqual(row["extreme_mismatch_events"], 1)
        self.assertEqual(row["candidate_signals"], "extreme_mismatch")

    def test_extreme_mismatch_rejects_confirmed_low_response_non_error(self):
        event = {
            "outdoor_rise_pm25": 550.1,
            "peak_response_ratio": 0.01,
            "area_response_ratio": 0.01,
            "response_coverage": 1.0,
        }

        self.assertFalse(is_extreme_mismatch(event, ExtremeMismatchCriteria()))
        self.assertEqual(exclusion_range_rows([event], [], ()), [])

    def test_extreme_mismatch_catches_reviewed_high_magnitude_events(self):
        reviewed = (
            (2499.7, 0.00254, 0.00555),
            (2019.0, 0.00909, 0.08467),
            (1664.4, 0.00487, 0.00520),
            (1629.45, 0.00644, 0.01133),
            (1663.3, 0.00637, 0.00242),
            (1661.25, 0.00512, 0.02987),
            (890.6, 0.00606, 0.02068),
            (1650.6, 0.00370, 0.00053),
            (1347.5, 0.01344, 0.00181),
            (740.4, 0.00757, 0.01058),
        )

        self.assertTrue(
            all(
                is_extreme_mismatch(
                    {
                        "outdoor_rise_pm25": rise,
                        "peak_response_ratio": peak,
                        "area_response_ratio": area,
                        "response_coverage": 1.0,
                    },
                    ExtremeMismatchCriteria(),
                )
                for rise, peak, area in reviewed
            )
        )

    def test_exclusion_range_includes_extreme_ramp_before_error_band(self):
        event = {
            "outdoor_sensor_id": 2,
            "outdoor_name": "Outside",
            "event_start_utc": "2024-09-03T22:00:00Z",
            "event_end_utc": "2024-09-04T00:00:00Z",
            "outdoor_rise_pm25": 1290.0,
            "peak_response_ratio": 0.01,
            "area_response_ratio": 0.01,
            "response_coverage": 1.0,
        }
        errors = [
            {
                "outdoor_sensor_id": 2,
                "outdoor_name": "Outside",
                "period_start_utc": "2024-09-03T23:00:00Z",
                "period_end_utc": "2024-09-05T00:00:00Z",
            }
        ]

        row = exclusion_range_rows([event], errors, ())[0]

        self.assertEqual(row["start_utc"], "2024-09-03T22:00:00Z")
        self.assertEqual(row["end_utc"], "2024-09-05T00:00:00Z")
        self.assertEqual(
            row["candidate_signals"], "extreme_mismatch;recurring_error_level"
        )

    def test_missing_both_histories_are_reported_separately(self):
        events, coverage = analyze_pairs([PAIR], {1: {}}, {2: {}}, Criteria())

        self.assertEqual(events, [])
        self.assertEqual(coverage[0]["status"], "missing_both")

    def test_empty_history_directory_returns_requested_empty_sensors(self):
        with tempfile.TemporaryDirectory() as directory:
            histories = read_histories([Path(directory)], {1, 2})

        self.assertEqual(histories, {1: {}, 2: {}})

    def test_histories_can_load_every_sensor_for_the_unpaired_page(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "histories.csv"
            path.write_text(
                "time_stamp,sensor_index,pm2.5_atm\n0,1,5\n3600,3,7\n",
                encoding="utf-8",
            )
            loaded = read_histories([path])

        self.assertEqual(loaded, {1: {0: 5.0}, 3: {HOUR: 7.0}})

    def test_school_smoke_overlap_selects_only_matching_snapshot_pairs(self):
        other = PAIR | {"indoor_sensor_id": 3, "outdoor_sensor_id": 4}
        selected, audit = select_pairs([PAIR, other], {"smoke_overlap_school": {1}})

        self.assertEqual(selected, [PAIR | {"cohort_sources": "smoke_overlap_school"}])
        self.assertEqual(
            audit[0]["selection_status"], "selected_outdoor_purpleair_pair"
        )

    def test_permanent_exclusion_removes_pair_before_analysis(self):
        retained = PAIR | {"indoor_sensor_id": 3, "outdoor_sensor_id": 4}
        selected, audit = select_pairs(
            [PAIR, retained], {"fema_school": {1, 3}}, excluded_ids={1}
        )

        self.assertEqual([row["indoor_sensor_id"] for row in selected], [3])
        self.assertEqual(
            audit[0]["selection_status"], "permanently_excluded_indoor_sensor"
        )

    def test_permanent_exclusion_manifest_contains_reviewed_sensors(self):
        manifest = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "exclusions"
            / "permanently_excluded_indoor_sensors.csv"
        )

        excluded = read_excluded_sensor_ids(manifest)
        self.assertIn(112546, excluded)
        self.assertIn(171059, excluded)

    def test_school_smoke_overlap_ids_are_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overlap.csv"
            path.write_text("sensor_index,event\n1,a\n1,b\n3,c\n", encoding="utf-8")

            self.assertEqual(read_overlap_indoor_ids(path), {1, 3})

    def test_unpaired_school_sensor_is_reported(self):
        selected, audit = select_pairs(
            [PAIR], {"smoke_overlap_school": {1}, "fema_school": {1, 3}}
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(audit[1]["selection_status"], "no_outdoor_purpleair_pair")

    def test_downloaded_history_extends_school_cohort(self):
        other = PAIR | {"indoor_sensor_id": 3, "outdoor_sensor_id": 4}
        selected, _ = select_pairs(
            [PAIR, other],
            {"fema_school": {1}, "downloaded_history": {1, 3}},
        )

        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[1]["cohort_sources"], "downloaded_history")

    def test_school_pairing_allows_outdoor_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sensors.csv"
            path.write_text(
                "sensor_index,name,location_type,latitude,longitude\n"
                "1,School A,inside,40,-75\n"
                "2,School B,inside,40.0001,-75\n"
                "3,Outdoor,outside,40.00005,-75\n",
                encoding="utf-8",
            )
            pairs = read_reusable_school_pairs(path, {1, 2}, 100)

        self.assertEqual(len(pairs), 2)
        self.assertEqual({row["outdoor_sensor_id"] for row in pairs}, {3})

    def test_school_pairing_replaces_excluded_outdoor_sensor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sensors.csv"
            path.write_text(
                "sensor_index,name,location_type,latitude,longitude\n"
                "1,School,inside,40,-75\n"
                "2,Broken outside,outside,40.00005,-75\n"
                "3,Working outside,outside,40.0001,-75\n",
                encoding="utf-8",
            )
            pairs = read_reusable_school_pairs(path, {1}, 100, {2})

        self.assertEqual(pairs[0]["outdoor_sensor_id"], 3)

    def test_fema_school_filter_uses_only_validated_inside_schools(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schools.csv"
            path.write_text(
                "sensor_index,location_type,k12_status,is_k12\n"
                "1,inside,school,true\n2,outside,school,true\n"
                "3,inside,not_school,false\n",
                encoding="utf-8",
            )

            self.assertEqual(read_fema_school_ids(path), {1})


if __name__ == "__main__":
    unittest.main()
