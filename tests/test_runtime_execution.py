"""P3 复用池测试：顺序归并、复用（只建一次池）、串行回退、异常与幂等释放、异常回收。"""

import multiprocessing
import unittest

from protassem.runtime.execution import ExecutionContext


def _double(value):
    """模块级 worker（可被 spawn 序列化）。"""
    return value * 2


def _boom(value):
    """模块级 worker：故意失败，用于异常回收路径。"""
    raise RuntimeError("worker failure: %s" % value)


def _active_child_pids():
    return {child.pid for child in multiprocessing.active_children()}


class ExecutionContextTest(unittest.TestCase):

    def test_map_preserves_input_order_with_pool(self):
        with ExecutionContext(pool_workers=4) as context:
            results = context.map(_double, list(range(20)))
        self.assertEqual(results, [value * 2 for value in range(20)])

    def test_pool_is_created_once_and_reused(self):
        context = ExecutionContext(pool_workers=3)
        for _ in range(5):
            self.assertEqual(context.map(_double, [1, 2, 3]), [2, 4, 6])
        created = context._pool is not None
        context.close()
        self.assertTrue(created, "复用池应在首次 map 时建立")

    def test_single_worker_runs_serially_without_pool(self):
        with ExecutionContext(pool_workers=1) as context:
            self.assertEqual(context.map(_double, [4, 5]), [8, 10])
            self.assertIsNone(context._pool, "workers<=1 不应建池")

    def test_empty_items_do_not_create_pool(self):
        with ExecutionContext(pool_workers=4) as context:
            self.assertEqual(context.map(_double, []), [])
            self.assertIsNone(context._pool)

    def test_close_is_idempotent(self):
        context = ExecutionContext(pool_workers=2)
        context.map(_double, [1])
        context.close()
        context.close()
        self.assertIsNone(context._pool)

    def test_context_manager_closes_on_exception(self):
        context = ExecutionContext(pool_workers=2)
        with self.assertRaises(ValueError):
            with context:
                context.map(_double, [1])
                raise ValueError("intentional")
        self.assertIsNone(context._pool)

    def test_spawn_start_method_runs_tasks(self):
        with ExecutionContext(pool_workers=2, start_method="spawn") as context:
            self.assertEqual(context.map(_double, [3]), [6])


class PoolRecoveryTest(unittest.TestCase):
    """异常路径的资源回收：worker 抛异常后池必须能被回收干净。"""

    def test_context_manager_reaps_children_after_worker_exception(self):
        before = _active_child_pids()
        context = ExecutionContext(pool_workers=2)
        with self.assertRaises(RuntimeError):
            context.map(_boom, [1, 2])
        context.close()
        self.assertEqual(_active_child_pids() - before, set())


if __name__ == "__main__":
    unittest.main()
