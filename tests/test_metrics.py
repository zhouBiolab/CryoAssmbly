"""Tests for runtime/metrics.py: event records, per-stage summary and totals."""

import json
import os
import tempfile
import unittest

from protassem.runtime.metrics import Metrics, worker_count


class MetricsTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

    def _jsonl_rows(self):
        path = os.path.join(self.tmp, "performance.jsonl")
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def test_record_appends_jsonl_with_optional_fields(self):
        metrics = Metrics(output_dir=self.tmp, run_id="run-1")
        metrics.record("cc_batch", 1.5, task_id="A_d1", mask_version=3,
                       candidate_count=4)
        metrics.record("cc_batch", 0.5)
        rows = self._jsonl_rows()
        self.assertEqual([row["stage"] for row in rows], ["cc_batch", "cc_batch"])
        self.assertEqual(rows[0]["task_id"], "A_d1")
        self.assertEqual(rows[0]["mask_version"], 3)
        self.assertEqual(rows[0]["run_id"], "run-1")
        self.assertEqual(metrics.records[1]["elapsed_s"], 0.5)

    def test_stage_records_even_when_block_raises(self):
        metrics = Metrics(output_dir=self.tmp, run_id="run-2")
        with self.assertRaises(ValueError):
            with metrics.stage("boom"):
                raise ValueError("intentional")
        self.assertEqual(len(metrics.records), 1)
        self.assertEqual(metrics.records[0]["stage"], "boom")

    def test_summary_totals_by_stage_and_wall(self):
        metrics = Metrics(output_dir=self.tmp, run_id="run-3")
        with metrics.stage("pipeline_total"):
            metrics.record("sampling", 1.0)
            metrics.record("sampling", 2.0)
        summary = metrics.summary()
        self.assertEqual(summary["run_id"], "run-3")
        self.assertEqual(summary["stages"]["sampling"], {"count": 2, "seconds": 3.0})
        self.assertEqual(summary["stages"]["pipeline_total"]["count"], 1)
        self.assertGreaterEqual(summary["total_wall_s"], 0.0)

    def test_write_summary_returns_path(self):
        metrics = Metrics(output_dir=self.tmp, run_id="run-5")
        metrics.record("mask", 0.25)
        path = metrics.write_summary()
        self.assertEqual(os.path.basename(path), "performance_summary.json")
        with open(path, encoding="utf-8") as handle:
            written = json.load(handle)
        self.assertEqual(written["stages"]["mask"]["count"], 1)

    def test_without_output_dir_nothing_is_written(self):
        metrics = Metrics(output_dir=None, run_id="run-4")
        metrics.record("a", 1.0)
        self.assertIsNone(metrics.write_summary())
        self.assertEqual(len(metrics.records), 1)
        self.assertEqual(os.listdir(self.tmp), [])

    def test_worker_count_never_returns_zero(self):
        self.assertEqual(worker_count(0), 1)
        self.assertEqual(worker_count(None), 1)
        self.assertEqual(worker_count(4), 4)


if __name__ == "__main__":
    unittest.main()
