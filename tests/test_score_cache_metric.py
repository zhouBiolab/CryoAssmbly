"""评分缓存统计写入指标的测试（审计#4）。

旧实现读取 `snapshot["density"]["bytes"]` / `["peak_bytes"]`，而共享预算改造后这两个字段
已经移到顶层 —— 真实运行日志里是 `Score cache snapshot failed: 'bytes'`，指标事件直接丢失。
这里测的是"真实 snapshot → 指标写入"，不是缓存内部。
"""

import os
import tempfile
import unittest

from protassem.core import scoring
from protassem.core.scoring import calculate_cc_mask, configure_score_cache
from protassem.pipeline import _record_score_cache, _record_tm_cache
from protassem.runtime.metrics import Metrics
from tests import fixtures


class ScoreCacheMetricTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(configure_score_cache, 128)
        self.dir = self._tmp.name

    def _exercise_cache(self):
        """真正产生一次命中，让 snapshot 里有非零计数与占用。"""
        import mrcfile
        import numpy as np
        path = os.path.join(self.dir, "map.mrc")
        data = np.zeros((16, 16, 16), dtype=np.float32)
        data[2:-2, 2:-2, 2:-2] = 5.0
        with mrcfile.new(path, overwrite=True) as mrc:
            mrc.set_data(data)
            mrc.voxel_size = (2.0, 2.0, 2.0)
            mrc.header.origin.x = 1.0
            mrc.header.origin.y = 2.0
            mrc.header.origin.z = 3.0
            mrc.update_header_from_data()
        structure = fixtures.make_chain_structure(
            os.path.join(self.dir, "chain.pdb"), [("A", (8.0, 8.0, 8.0))], residues=6)
        configure_score_cache(128)
        calculate_cc_mask(path, structure, 6.0, 0.04)
        calculate_cc_mask(path, structure, 6.0, 0.04)      # 命中
        return path

    def test_score_cache_event_is_written_with_shared_budget_fields(self):
        self._exercise_cache()
        metrics = Metrics(output_dir=self.dir, run_id="score-cache-metric")
        _record_score_cache(metrics)
        rows = [row for row in metrics.records if row["stage"] == "score_cache"]
        self.assertEqual(len(rows), 1, "score_cache 事件必须写出来（旧实现被 KeyError 吞掉）")
        row = rows[0]
        for key in ("score_cache_mb", "entries", "bytes", "peak_bytes", "evictions",
                    "density_hits", "density_misses",
                    "structure_hits", "structure_misses"):
            self.assertIn(key, row)
        self.assertGreaterEqual(row["density_hits"], 1)
        self.assertGreaterEqual(row["density_misses"], 1)
        self.assertGreater(row["bytes"], 0)
        self.assertGreaterEqual(row["peak_bytes"], row["bytes"])
        self.assertEqual(row["score_cache_mb"], 128)

    def test_snapshot_shape_matches_what_the_recorder_reads(self):
        """守护契约：recorder 读取的键必须都在真实 snapshot 里（防止再次搬字段）。"""
        self._exercise_cache()
        snapshot = scoring.score_cache_snapshot()
        for key in ("score_cache_mb", "entries", "bytes", "peak_bytes", "evictions"):
            self.assertIn(key, snapshot)
        for kind in ("density", "structure"):
            self.assertIn(kind, snapshot)
            for key in ("hits", "misses"):
                self.assertIn(key, snapshot[kind])

    def test_tm_cache_event_is_written(self):
        metrics = Metrics(output_dir=self.dir, run_id="tm-cache-metric")
        _record_tm_cache(metrics)
        rows = [row for row in metrics.records if row["stage"] == "tm_cache"]
        self.assertEqual(len(rows), 1)
        self.assertIn("mode", rows[0])

    def test_disabled_cache_still_records(self):
        configure_score_cache(0)
        metrics = Metrics(output_dir=self.dir, run_id="score-cache-disabled")
        _record_score_cache(metrics)
        rows = [row for row in metrics.records if row["stage"] == "score_cache"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["score_cache_mb"], 0)


if __name__ == "__main__":
    unittest.main()
