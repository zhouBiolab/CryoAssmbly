"""T09 有界尾部流水线契约测试。

覆盖任务卡验收："开关结果/顺序一致，取消/错误/正常退出均释放资源"：
  - 顺序：单 worker FIFO，提交顺序 = 执行顺序；
  - 深度 1：最多积压一个任务（worker 忙时 submit 等待）；
  - 开关等价：enabled=False 时就地执行，结果与开启时一致；
  - 资源：close() 幂等、worker 线程退出、异常任务在 wait()/close() 抛出且不吞掉；
  - 上下文管理器：异常退出也会关闭（不残留线程），且不掩盖原始异常。
"""

import threading
import time
import unittest

from protassem.runtime.tail_pipeline import TailPipeline


def thread_names():
    return {item.name for item in threading.enumerate()}


class TailPipelineTest(unittest.TestCase):
    def test_disabled_runs_inline_and_preserves_order(self):
        pipeline = TailPipeline(enabled=False, name="test-tail-off")
        order = []
        for index in range(5):
            pipeline.submit(lambda index=index: order.append(index))
        self.assertEqual([0, 1, 2, 3, 4], order)          # 就地、立即执行
        self.assertEqual(5, pipeline.completed)
        self.assertEqual(5, pipeline.submitted)
        self.assertIsNone(pipeline._worker)
        pipeline.wait()
        pipeline.close()

    def test_enabled_preserves_submission_order(self):
        order = []
        with TailPipeline(enabled=True, name="test-tail-order") as pipeline:
            for index in range(20):
                pipeline.submit(lambda index=index: order.append(index))
            pipeline.wait()
        self.assertEqual(list(range(20)), order)
        self.assertEqual(20, pipeline.completed)

    def test_depth_is_bounded_to_one_pending_task(self):
        state = {"running": 0, "peak": 0}
        lock = threading.Lock()

        def task():
            with lock:
                state["running"] += 1
                state["peak"] = max(state["peak"], state["running"])
            time.sleep(0.02)
            with lock:
                state["running"] -= 1

        with TailPipeline(enabled=True, name="test-tail-depth") as pipeline:
            for _ in range(5):
                pipeline.submit(task)
            pipeline.wait()
        self.assertEqual(1, state["peak"], "最多只能有一个任务在跑（深度 1）")

    def test_close_is_idempotent_and_joins_worker(self):
        pipeline = TailPipeline(enabled=True, name="test-tail-close")
        pipeline.submit(lambda: None)
        pipeline.close()
        pipeline.close()                                  # 幂等
        self.assertFalse(pipeline._worker.is_alive())
        self.assertNotIn("test-tail-close", thread_names())

    def test_task_error_propagates_and_worker_is_joined(self):
        pipeline = TailPipeline(enabled=True, name="test-tail-error")

        def boom():
            raise RuntimeError("尾部任务失败")

        pipeline.submit(boom)
        with self.assertRaises(RuntimeError):
            pipeline.wait()
        with self.assertRaises(RuntimeError):
            pipeline.close()
        self.assertNotIn("test-tail-error", thread_names())

    def test_error_after_success_is_not_swallowed(self):
        pipeline = TailPipeline(enabled=True, name="test-tail-late-error")
        results = []
        pipeline.submit(lambda: results.append(1))
        pipeline.submit(lambda: (_ for _ in ()).throw(ValueError("late")))
        with self.assertRaises(ValueError):
            pipeline.close()
        self.assertEqual([1], results)
        self.assertNotIn("test-tail-late-error", thread_names())

    def test_context_manager_closes_on_exception(self):
        with self.assertRaises(KeyError):
            with TailPipeline(enabled=True, name="test-tail-ctx") as pipeline:
                pipeline.submit(lambda: None)
                raise KeyError("原始异常")
        self.assertNotIn("test-tail-ctx", thread_names())

    def test_snapshot_reports_state(self):
        with TailPipeline(enabled=True, name="test-tail-snapshot") as pipeline:
            pipeline.submit(lambda: None)
            pipeline.wait()
            snapshot = pipeline.snapshot()
        self.assertTrue(snapshot["enabled"])
        self.assertEqual(1, snapshot["submitted"])
        self.assertEqual(1, snapshot["completed"])
        self.assertFalse(snapshot["failed"])


if __name__ == "__main__":
    unittest.main()
