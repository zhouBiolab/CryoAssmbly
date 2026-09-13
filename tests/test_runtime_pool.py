"""进程池辅助测试：池启动/关闭事件被记录，异常路径也释放。"""

import os
import tempfile
import unittest

from protassem.runtime.metrics import Metrics
from protassem.runtime.pool import timed_pool


class TimedPoolTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

    def test_records_start_and_close_and_runs_tasks(self):
        metrics = Metrics(output_dir=self.tmp, run_id="pool-1")
        with timed_pool(metrics, 2, "unit_pool") as pool:
            results = pool.map(abs, [-1, -2, -3])
        self.assertEqual(results, [1, 2, 3])
        stages = [row["stage"] for row in metrics.records]
        self.assertEqual(stages, ["pool_start", "pool_close"])
        self.assertEqual(metrics.records[0]["pool"], "unit_pool")
        self.assertEqual(metrics.records[0]["workers"], 2)
        self.assertEqual(metrics.records[0]["start_method"], "default")

    def test_pool_is_released_when_block_raises(self):
        metrics = Metrics(output_dir=None, run_id="pool-2")
        with self.assertRaises(ValueError):
            with timed_pool(metrics, 1, "boom_pool") as pool:
                pool.map(abs, [-1])
                raise ValueError("intentional")
        self.assertEqual([row["stage"] for row in metrics.records],
                         ["pool_start", "pool_close"])

    def test_metrics_is_optional(self):
        with timed_pool(None, 1, "no_metrics") as pool:
            self.assertEqual(pool.map(abs, [-5]), [5])


if __name__ == "__main__":
    unittest.main()
