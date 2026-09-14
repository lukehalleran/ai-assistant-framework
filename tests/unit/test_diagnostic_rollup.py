import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import diagnostic_rollup as rollup_module


class DiagnosticRollupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input_path = self.root / "turns.jsonl"

    def write_rows(self, rows):
        self.input_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_timezone_selection_naive_and_out_of_window(self):
        self.write_rows([
            {"ts": "2026-09-13T05:00:00Z", "wall_elapsed_s": 4},
            {"ts": "2026-09-13T04:30:00Z", "wall_elapsed_s": 4},
            {"ts": "2026-09-13T00:30:00", "wall_elapsed_s": 8},
            {"ts": "2026-09-14T05:00:00Z", "wall_elapsed_s": 16},
        ])
        result = rollup_module.rollup(str(self.input_path), date(2026, 9, 13), "America/North_Dakota/Center")
        self.assertEqual(result["counts"]["selected"], 2)
        self.assertEqual(result["counts"]["naive_timestamps"], 1)
        self.assertEqual(result["counts"]["out_of_window"], 2)
        self.assertEqual(result["wall_elapsed_s"]["median"], 6)

    def test_malformed_test_exclusion_bad_numeric_and_shape_are_distinct(self):
        self.input_path.write_text("{oops}\n" + "\n".join(map(json.dumps, [
            {"ts": "2026-09-13T12:00:00-05:00", "model": "test-model", "wall_elapsed_s": 99},
            {"ts": "bad", "wall_elapsed_s": 12},
            {"ts": "2026-09-13T12:00:00-05:00", "test_env": True},
            {"ts": "2026-09-13T12:00:00-05:00", "wall_elapsed_s": True,
             "phase_timings": ["free text"], "mode": "private-label"},
        ])) + "\n", encoding="utf-8")
        result = rollup_module.rollup(str(self.input_path), date(2026, 9, 13), "America/North_Dakota/Center")
        self.assertEqual(result["counts"]["malformed"], 1)
        self.assertEqual(result["counts"]["test_excluded"], 2)
        self.assertEqual(result["counts"]["invalid_time"], 1)
        self.assertEqual(result["wall_elapsed_s"]["count"], 0)
        self.assertIsNone(result["wall_elapsed_s"]["median"])
        self.assertEqual(len(result["groups"]), 1)
        self.assertEqual(result["groups"][0]["rows"], 1)
        self.assertEqual(result["groups"][0]["wall_elapsed_s"]["count"], 0)

    def test_percentiles_nested_timings_and_missing_are_not_zero(self):
        self.write_rows([
            {"ts": "2026-09-13T12:00:00-05:00", "wall_elapsed_s": 1,
             "phase_timings": {"prompt_build": 80, "private phase": 5},
             "task_timings": {"memories": 2}},
            {"ts": "2026-09-13T12:01:00-05:00", "wall_elapsed_s": 2,
             "phase_timings": {"prompt_build": -3, "context_pipeline": 4},
             "task_timings": {"memories": True}},
            {"ts": "2026-09-13T12:02:00-05:00", "wall_elapsed_s": 1000,
             "phase_timings": {"prompt_build": float("inf")}},
        ])
        result = rollup_module.rollup(str(self.input_path), date(2026, 9, 13), "America/North_Dakota/Center")
        self.assertEqual(result["wall_elapsed_s"]["count"], 3)
        self.assertEqual(result["wall_elapsed_s"]["median"], 2)
        self.assertAlmostEqual(result["wall_elapsed_s"]["p90"], 800.4)
        self.assertEqual(result["wall_elapsed_s"]["max"], 1000)
        self.assertEqual(result["phases_elapsed_s"]["prompt_build"]["count"], 1)
        self.assertNotIn("other", result["phases_elapsed_s"])
        self.assertEqual(result["unknown_timing_entries_dropped"], 1)
        self.assertEqual(result["tasks_elapsed_s"]["memories"]["count"], 1)
        self.assertEqual(result["coverage"]["correlation_id"]["present"], 0)
        self.assertEqual(result["coverage"]["wall_elapsed_s"]["present"], 3)
        self.assertEqual(result["coverage"]["phase_timings"]["valid_values"], 2)
        self.assertFalse(result["privacy"]["nested_timings_summed"])

    def test_sample_cap_and_unknown_labels_do_not_leak(self):
        self.write_rows([{
            "ts": "2026-09-13T12:00:00-05:00", "wall_elapsed_s": 20,
            "mode": "secret-user-label", "model": "private model",
            "query": "CANARY_QUERY", "response": "CANARY_RESPONSE",
            "phase_timings": {"CANARY_PHASE": 4}, "task_timings": {"CANARY_TASK": 2},
        }])
        result = rollup_module.rollup(str(self.input_path), date(2026, 9, 13), "America/North_Dakota/Center")
        encoded = json.dumps(result)
        for canary in ("CANARY_QUERY", "CANARY_RESPONSE", "CANARY_PHASE", "CANARY_TASK", "secret-user-label", "private model"):
            self.assertNotIn(canary, encoded)
        self.assertEqual(result["top_slow_rows"][0]["mode"], "other")
        self.assertEqual(result["top_slow_rows"][0]["model"], "other")

    def test_record_and_line_bounds_are_reported(self):
        self.input_path.write_bytes(b"x" * 256 + b"\n" + json.dumps({"ts": "2026-09-13T12:00:00-05:00"}).encode() + b"\n")
        result = rollup_module.rollup(str(self.input_path), date(2026, 9, 13), "America/North_Dakota/Center",
                                      max_records=1, max_line_bytes=128)
        self.assertEqual(result["counts"]["oversized"], 1)
        self.assertTrue(result["truncation"]["record_cap_reached"])
        self.assertTrue(result["truncation"]["notice"])

    def test_huge_numbers_deep_json_and_unknown_timing_keys_stay_bounded(self):
        huge_integer = "9" * 5000
        deep_json = "[" * 1200 + "0" + "]" * 1200
        unknown_timings = {"private-key-%d" % index: 1 for index in range(500)}
        self.input_path.write_text("\n".join([
            '{"ts":"2026-09-13T12:00:00-05:00","wall_elapsed_s":' + huge_integer + '}',
            deep_json,
            json.dumps({"ts": "2026-09-13T12:00:00-05:00", "wall_elapsed_s": 3,
                        "task_timings": unknown_timings}),
        ]) + "\n", encoding="utf-8")
        result = rollup_module.rollup(str(self.input_path), date(2026, 9, 13), "America/North_Dakota/Center")
        self.assertEqual(result["counts"]["malformed"], 2)
        self.assertEqual(result["wall_elapsed_s"]["count"], 1)
        self.assertEqual(result["unknown_timing_entries_dropped"], 500)
        self.assertNotIn("private-key", json.dumps(result))
        self.assertLessEqual(len(result["top_slow_rows"]), rollup_module.MAX_SAMPLE_ROWS)

    def test_groups_count_selected_rows_without_wall_measurements(self):
        self.write_rows([
            {"ts": "2026-09-13T12:00:00-05:00", "mode": "enhanced", "model": "kimi-3"},
            {"ts": "2026-09-13T12:01:00-05:00", "mode": "enhanced", "model": "kimi-3", "wall_elapsed_s": 2},
        ])
        result = rollup_module.rollup(str(self.input_path), date(2026, 9, 13), "America/North_Dakota/Center")
        self.assertEqual(result["groups"][0]["rows"], 2)
        self.assertEqual(result["groups"][0]["wall_elapsed_s"]["count"], 1)

    def test_token_and_retry_coverage_requires_nonnegative_integers(self):
        self.write_rows([
            {"ts": "2026-09-13T12:00:00-05:00", "input_tokens": 4, "output_tokens": 0,
             "cache_tokens": True, "provider_retries": 1.5},
        ])
        result = rollup_module.rollup(str(self.input_path), date(2026, 9, 13), "America/North_Dakota/Center")
        self.assertEqual(result["coverage"]["input_tokens"]["present"], 1)
        self.assertEqual(result["coverage"]["output_tokens"]["present"], 1)
        self.assertEqual(result["coverage"]["cache_tokens"]["present"], 0)
        self.assertEqual(result["coverage"]["provider_retries"]["present"], 0)

    def test_data_components_and_symlink_targets_are_rejected_before_open(self):
        data_dir = self.root / "data"
        data_dir.mkdir()
        target = data_dir / "turns.jsonl"
        target.write_text("should never open", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "data component"):
            rollup_module.rollup(str(target), date(2026, 9, 13), "UTC")
        link = self.root / "alias.jsonl"
        try:
            link.symlink_to(target)
        except OSError:
            self.skipTest("symlinks unavailable")
        with self.assertRaisesRegex(ValueError, "data component"):
            rollup_module.rollup(str(link), date(2026, 9, 13), "UTC")

    def test_directory_and_invalid_timezone_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "regular file"):
            rollup_module.rollup(str(self.root), date(2026, 9, 13), "UTC")
        self.input_path.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "IANA timezone"):
            rollup_module.rollup(str(self.input_path), date(2026, 9, 13), "Nowhere/NoZone")


if __name__ == "__main__":
    unittest.main()
