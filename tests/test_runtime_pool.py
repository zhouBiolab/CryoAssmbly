"""进程池原语测试：创建/回收、worker 数下限保护、异常路径也释放。"""

import multiprocessing
import unittest

from protassem.runtime.pool import close_pool, open_pool, worker_count


def _double(value):
    """模块级 worker（可被 spawn 序列化）。"""
    return value * 2


def _boom(value):
    """模块级 worker：故意失败，用于异常回收路径。"""
    raise RuntimeError("worker failure: %s" % value)


def _active_child_pids():
    return {child.pid for child in multiprocessing.active_children()}


class WorkerCountTest(unittest.TestCase):
    """worker 数下限保护：0 / None 都钳到 1。"""

    def test_never_returns_zero(self):
        self.assertEqual(worker_count(0), 1)
        self.assertEqual(worker_count(None), 1)
        self.assertEqual(worker_count(4), 4)


class PoolPrimitivesTest(unittest.TestCase):

    def test_open_and_close_pool(self):
        pool = open_pool(2)
        self.assertEqual(pool.map(_double, [2, 3]), [4, 6])
        close_pool(pool)

    def test_single_worker_pool_runs_tasks(self):
        pool = open_pool(1)
        self.assertEqual(pool.map(abs, [-5]), [5])
        close_pool(pool)

    def test_spawn_context_is_honoured(self):
        pool = open_pool(2, "spawn")
        self.assertEqual(pool.map(_double, [3]), [6])
        close_pool(pool)


class PoolRecoveryTest(unittest.TestCase):
    """异常路径的资源回收：worker 抛异常后池必须能被回收干净。"""

    def test_worker_exception_propagates_and_children_are_reaped(self):
        before = _active_child_pids()
        pool = open_pool(2)
        with self.assertRaises(RuntimeError):
            pool.map(_boom, [1, 2], chunksize=1)
        close_pool(pool)
        self.assertEqual(_active_child_pids() - before, set())


if __name__ == "__main__":
    unittest.main()
