"""有界（深度 1）的 CPU 尾部流水线（任务卡 T09）。

用途：把"模型之后的 CPU 尾部"（后处理计算 + 写盘）从一个预取 worker 上执行，
使它与**下一次配准的 GPU 工作**重叠；主线程只负责准备与模型调用。

边界与约定
----------
- **深度 1**：`submit()` 在下一次任务开始时最多积压一个任务；worker 忙时 `submit()` 会等待
  （内存最多增加一个预取任务，符合任务卡）。
- **顺序**：worker 单线程 FIFO，提交顺序 = 执行顺序 = 消费顺序。
- **资源**：`close()` 幂等，正常/异常/早停路径都必须调用；worker 是 daemon 线程并显式 join，
  超时报错而不是静默留下线程。
- **失败**：任务里的异常被 worker 捕获并**在 `wait()`/`close()` 抛出**（不吞掉、不伪装成功）。
- **开关等价**：`enabled=False` 时 `submit()` 就地同步执行，与开启时走**同一份调用代码**。
- **随机过程**：只搬运确定性的后处理/写盘；不并行任何消耗随机数的步骤（任务卡要求）。

主线程在**读取任何已提交任务的结果之前**必须调用 `wait()`——这是使用契约，不是可选优化。
"""

import logging
import queue
import threading

log = logging.getLogger(__name__)

_SENTINEL = object()


class TailPipeline:
    """单 worker、深度 1 的尾部流水线；`enabled=False` 时退化为同步执行。"""

    def __init__(self, enabled=True, name="dsh-tail"):
        self.enabled = bool(enabled)
        self.name = name
        self.submitted = 0
        self.completed = 0
        self._error = None
        self._closed = False
        if not self.enabled:
            self._queue = None
            self._worker = None
            return
        self._queue = queue.Queue(maxsize=1)
        self._worker = threading.Thread(target=self._run, name=name, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------
    def submit(self, function):
        """提交一个任务；`enabled=False` 时就地执行。"""
        if self._error is not None:
            raise self._error
        self.submitted += 1
        if not self.enabled:
            self.completed += 1
            return function()
        self._queue.put(function)      # 深度 1：worker 忙时在此等待
        return None

    def wait(self):
        """等待已提交任务全部完成；有失败则抛出（首个异常）。"""
        if self.enabled and self._worker is not None:
            self._queue.join()
        if self._error is not None:
            raise self._error

    def close(self):
        """停止 worker 并 join；幂等。异常任务会在关闭时抛出。"""
        if self._closed:
            return
        self._closed = True
        if not self.enabled or self._worker is None:
            if self._error is not None:
                raise self._error
            return
        self._queue.join()             # 先把已提交任务做完
        self._queue.put(_SENTINEL)
        self._worker.join(timeout=30.0)
        if self._worker.is_alive():
            raise RuntimeError("尾部流水线 worker 未在超时内退出（可能有任务卡住）")
        if self._error is not None:
            raise self._error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.close()
        except Exception as close_error:       # 关闭失败不应掩盖原始异常
            if exc_type is None:
                raise
            log.error("尾部流水线关闭失败：%s", close_error)
        return False

    def snapshot(self):
        return {"enabled": self.enabled, "submitted": self.submitted,
                "completed": self.completed, "failed": self._error is not None}

    # ------------------------------------------------------------------
    def _run(self):
        while True:
            function = self._queue.get()
            try:
                if function is _SENTINEL:
                    return
                function()
                self.completed += 1
            except Exception as error:                 # 记录首个异常，等待主线程抛出
                if self._error is None:
                    self._error = error
                log.error("尾部任务失败：%s", error)
            finally:
                self._queue.task_done()
