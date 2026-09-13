"""运行级执行上下文（阶段二 P3）：一次运行共用一个惰性 CPU 池。

约束与语义：
  - 池惰性创建、整个运行内复用；`close()` 幂等，正常与异常路径都必须调用；
  - `map()` 结果按输入顺序归并（multiprocessing.Pool.map 的语义，不按完成顺序）；
  - worker 数 <= 1 时不建池，直接在当前进程串行执行（等价结果，避免无谓启停）；
  - worker 函数必须是模块级可序列化对象；**禁止在 worker 内再建池**。
"""

from protassem.runtime.pool import close_pool, open_pool


class ExecutionContext:
    """持有 metrics 与一个惰性共享池；作为运行级资源句柄在调用链上显式传递。"""

    def __init__(self, metrics=None, pool_workers=1, start_method=None):
        self.metrics = metrics
        self.pool_workers = max(1, int(pool_workers or 1))
        self.start_method = start_method
        self._pool = None

    def map(self, func, items):
        """按输入顺序执行 func(item) 并归并结果；worker 数 <= 1 时串行。"""
        items = list(items)
        if not items:
            return []
        if self.pool_workers <= 1:
            return [func(item) for item in items]
        return self._shared_pool().map(func, items)

    def close(self):
        """释放共享池（幂等）；应在运行的 finally 中调用。"""
        if self._pool is None:
            return
        close_pool(self.metrics, self._pool, "shared")
        self._pool = None

    def _shared_pool(self):
        if self._pool is None:
            self._pool = open_pool(self.metrics, self.pool_workers, "shared",
                                   self.start_method)
        return self._pool

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
