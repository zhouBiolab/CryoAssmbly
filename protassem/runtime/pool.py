"""进程池辅助：创建/关闭计时（P2 细化）与 P3 的复用池原语。

约定：池的创建与关闭都在父进程计时；worker 内不得再建池。
`start_method=None` 表示系统默认（Linux 为 fork）；显式传 "spawn" 可单独测启动成本。
"""

import multiprocessing
import time
from contextlib import contextmanager


def pool_context(start_method=None):
    """返回 multiprocessing 上下文；None 表示系统默认（Linux 为 fork）。"""
    if start_method:
        return multiprocessing.get_context(start_method)
    return multiprocessing


def open_pool(metrics, processes, label, start_method=None):
    """创建进程池并记录 pool_start；调用方负责用 close_pool() 释放。"""
    started = time.perf_counter()
    pool = pool_context(start_method).Pool(processes=processes)
    if metrics is not None:
        metrics.record("pool_start", time.perf_counter() - started, pool=label,
                       workers=processes, start_method=start_method or "default")
    return pool


def close_pool(metrics, pool, label):
    """关闭并回收池，只计 close()+join()；池的存活时长与上层阶段重叠，不单独记录。"""
    started = time.perf_counter()
    pool.close()
    pool.join()
    if metrics is not None:
        metrics.record("pool_close", time.perf_counter() - started, pool=label)


@contextmanager
def timed_pool(metrics, processes, label, start_method=None):
    """创建临时池并保证释放（P2 计量用；P3 之后主路径改用 ExecutionContext）。"""
    pool = open_pool(metrics, processes, label, start_method)
    try:
        yield pool
    finally:
        close_pool(metrics, pool, label)
