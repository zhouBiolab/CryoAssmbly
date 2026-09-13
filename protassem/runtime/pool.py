"""进程池辅助：记录创建/关闭耗时（P2 细化），并为 P3 的复用池提供统一入口。

约定：池的创建与关闭都在父进程计时；worker 内不得再建池（P3 约束）。
`start_method=None` 表示沿用系统默认（Linux 上为 fork），显式传入可测 spawn 成本。
"""

import multiprocessing
import time
from contextlib import contextmanager


def pool_context(start_method=None):
    """返回 multiprocessing 上下文；None 表示系统默认（Linux 为 fork）。"""
    if start_method:
        return multiprocessing.get_context(start_method)
    return multiprocessing


@contextmanager
def timed_pool(metrics, processes, label, start_method=None):
    """创建一个进程池并记录 pool_start / pool_close，退出时确保释放。

    Args:
        metrics: 运行级 Metrics（记录事件）；None 时不记录，只保证释放。
        processes: worker 数量
        label: 调用点标识（如 "batch_cc"、"local_optimize_copies"）
        start_method: "fork" / "spawn" / None（默认）
    """
    context = pool_context(start_method)
    started = time.perf_counter()
    pool = context.Pool(processes=processes)
    created = time.perf_counter()
    if metrics is not None:
        metrics.record("pool_start", created - started, pool=label,
                       workers=processes,
                       start_method=start_method or "default")
    try:
        yield pool
    finally:
        # 只计关闭+回收的耗时；池的"存活时长"与上层阶段重叠，不单独记录，避免重复计数。
        closing_started = time.perf_counter()
        pool.close()
        pool.join()
        if metrics is not None:
            metrics.record("pool_close", time.perf_counter() - closing_started,
                           pool=label, workers=processes)
