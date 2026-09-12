"""Tests for core/performance.py: event records and the per-stage summary."""

import json
import os
import tempfile
import unittest

from protassem.core.performance import Metrics, configure_cpu_threads


class MetricsTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

    def _jsonl_rows(self):
        path = os.path.join(self.tmp, "performance.jsonl")
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def test_record_appends_jsonl(self):
        metrics = Metrics(output_dir=self.tmp, run_id="run-1")
        metrics.record("stage_a", 1.5, mode="chain")
        metrics.record("stage_a", 0.5)
        rows = self._jsonl_rows()
        self.assertEqual([row["stage"] for row in rows], ["stage_a", "stage_a"])
        self.assertEqual(rows[0]["mode"], "chain")
        self.assertEqual(rows[0]["run_id"], "run-1")
        self.assertEqual(metrics.records[1]["elapsed_s"], 0.5)

    def test_stage_records_even_when_block_raises(self):
        metrics = Metrics(output_dir=self.tmp, run_id="run-2")
        with self.assertRaises(ValueError):
            with metrics.stage("boom"):
                raise ValueError("intentional")
        self.assertEqual(len(metrics.records), 1)
        self.assertEqual(metrics.records[0]["stage"], "boom")

    def test_summary_totals_by_stage(self):
        metrics = Metrics(output_dir=self.tmp, run_id="run-3")
        metrics.record("a", 1.0)
        metrics.record("a", 2.0)
        metrics.record("b", 0.25)
        metrics.write_summary()
        path = os.path.join(self.tmp, "performance_summary.json")
        with open(path, encoding="utf-8") as handle:
            summary = json.load(handle)
        self.assertEqual(summary["run_id"], "run-3")
        self.assertEqual(summary["stages"]["a"], {"count": 2, "seconds": 3.0})
        self.assertEqual(summary["stages"]["b"]["count"], 1)

    def test_without_output_dir_nothing_is_written(self):
        metrics = Metrics(output_dir=None, run_id="run-4")
        metrics.record("a", 1.0)
        metrics.write_summary()
        self.assertEqual(len(metrics.records), 1)
        self.assertEqual(os.listdir(self.tmp), [])

    def test_configure_cpu_threads_never_returns_zero(self):
        self.assertEqual(configure_cpu_threads(0), 1)
        self.assertEqual(configure_cpu_threads(None), 1)
        self.assertEqual(configure_cpu_threads(4), 4)


if __name__ == "__main__":
    unittest.main()
