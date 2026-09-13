"""P3 复用池测试：顺序归并、复用（只建一次池）、串行回退、异常与幂等释放。"""

import os
import tempfile
import unittest

from protassem.runtime.execution import ExecutionContext
from protassem.runtime.metrics import Metrics
from protassem.runtime.pool import open_pool, close_pool


def _double(value):
    """模块级 worker（可被 spawn 序列化）。"""
    return value * 2


class ExecutionContextTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

    def test_map_preserves_input_order_with_pool(self):
        metrics = Metrics(output_dir=self.tmp, run_id="ctx-1")
        with ExecutionContext(metrics=metrics, pool_workers=4) as context:
            results = context.map(_double, list(range(20)))
        self.assertEqual(results, [value * 2 for value in range(20)])

    def test_pool_is_created_once_and_reused(self):
        metrics = Metrics(output_dir=self.tmp, run_id="ctx-2")
        with ExecutionContext(metrics=metrics, pool_workers=3) as context:
            for _ in range(5):
                self.assertEqual(context.map(_double, [1, 2, 3]), [2, 4, 6])
        starts = [row for row in metrics.records if row["stage"] == "pool_start"]
        closes = [row for row in metrics.records if row["stage"] == "pool_close"]
        self.assertEqual(len(starts), 1)
        self.assertEqual(len(closes), 1)
        self.assertEqual(starts[0]["pool"], "shared")
        self.assertEqual(starts[0]["workers"], 3)

    def test_single_worker_runs_serially_without_pool(self):
        metrics = Metrics(output_dir=self.tmp, run_id="ctx-3")
        with ExecutionContext(metrics=metrics, pool_workers=1) as context:
            self.assertEqual(context.map(_double, [4, 5]), [8, 10])
        self.assertEqual([row["stage"] for row in metrics.records], [])

    def test_empty_items_do_not_create_pool(self):
        metrics = Metrics(output_dir=self.tmp, run_id="ctx-4")
        with ExecutionContext(metrics=metrics, pool_workers=4) as context:
            self.assertEqual(context.map(_double, []), [])
        self.assertEqual([row["stage"] for row in metrics.records], [])

    def test_close_is_idempotent(self):
        metrics = Metrics(output_dir=self.tmp, run_id="ctx-5")
        context = ExecutionContext(metrics=metrics, pool_workers=2)
        context.map(_double, [1])
        context.close()
        context.close()
        closes = [row for row in metrics.records if row["stage"] == "pool_close"]
        self.assertEqual(len(closes), 1)

    def test_context_manager_closes_on_exception(self):
        metrics = Metrics(output_dir=self.tmp, run_id="ctx-6")
        with self.assertRaises(ValueError):
            with ExecutionContext(metrics=metrics, pool_workers=2) as context:
                context.map(_double, [1])
                raise ValueError("intentional")
        self.assertEqual(len([row for row in metrics.records
                              if row["stage"] == "pool_close"]), 1)

    def test_spawn_start_method_is_recorded(self):
        metrics = Metrics(output_dir=self.tmp, run_id="ctx-7")
        with ExecutionContext(metrics=metrics, pool_workers=2,
                              start_method="spawn") as context:
            self.assertEqual(context.map(_double, [3]), [6])
        starts = [row for row in metrics.records if row["stage"] == "pool_start"]
        self.assertEqual(starts[0]["start_method"], "spawn")


class PoolPrimitivesTest(unittest.TestCase):

    def test_open_and_close_pool(self):
        metrics = Metrics(output_dir=None, run_id="pool-prim")
        pool = open_pool(metrics, 2, "unit")
        self.assertEqual(pool.map(_double, [2, 3]), [4, 6])
        close_pool(metrics, pool, "unit")
        self.assertEqual([row["stage"] for row in metrics.records],
                         ["pool_start", "pool_close"])


if __name__ == "__main__":
    unittest.main()
