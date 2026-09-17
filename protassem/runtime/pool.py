"""进程池辅助：P3 的复用池原语。

约定：池惰性创建、整个运行内复用；**worker 内不得再建池**。
`start_method=None` 表示系统默认（Linux 为 fork）；显式传 "spawn" 可单独测启动成本。
"""

import multiprocessing


def worker_count(workers):
    """worker 数量的下限保护（>=1）；线程 env 由 RuntimeConfig 在入口负责。"""
    return max(1, int(workers or 1))


def pool_context(start_method=None):
    """返回 multiprocessing 上下文；None 表示系统默认（Linux 为 fork）。"""
    if start_method:
        return multiprocessing.get_context(start_method)
    return multiprocessing


def open_pool(processes, start_method=None):
    """创建进程池；调用方负责用 `close_pool()` 释放。"""
    return pool_context(start_method).Pool(processes=processes)


def close_pool(pool):
    """关闭并回收池。"""
    pool.close()
    pool.join()
