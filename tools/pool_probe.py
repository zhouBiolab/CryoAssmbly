"""P3 池生命周期探针（老卡收口第 2 步）。

测什么：
  1. 构造耗时        —— `Pool(processes=N)` 返回所需时间；
  2. 全部 worker 就绪 —— N 个任务（chunksize=1，一 worker 一个）各写一个 pid 标记，
                        再忙等到出现 N 个不同 pid；这段时间 = 建池 + 全部 worker 首次执行；
  3. 首批任务耗时    —— 就绪屏障这一批的总耗时（含进程启动与首次导入）；
  4. 暖任务耗时      —— 就绪之后单次小任务的往返耗时（池已预热）；
  5. 回收耗时        —— `close()` + `join()`；
  6. 异常回收        —— worker 抛异常后 `close()+join()` 是否回收干净（无残留子进程）；
  7. fork 适用性     —— 建池瞬间父进程的线程数与 CUDA 是否已初始化（worker 只做 CPU 计算）。

约定：
  - worker 函数一律模块级（spawn 也能序列化）；
  - 只用标准库 + 可选 torch（CUDA 状态查询失败时记 None，不影响结论）；
  - 子进程数用 `multiprocessing.active_children()` 与 `/proc` 双重核对。

用法：
    python tools/pool_probe.py --workers 10 --repeat 3 [--start-method spawn] [--json out.json]
"""

import argparse
import json
import multiprocessing
import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protassem.runtime.pool import close_pool, open_pool  # noqa: E402


def _square(value):
    """最小 CPU 任务（模块级，可被 spawn 序列化）。"""
    return value * value


def _boom(value):
    """故意失败的 worker，用于异常回收路径。"""
    raise RuntimeError("probe worker failure: %s" % value)


def _ready_marker(args):
    """写自己的 pid 标记，然后等到出现 `total` 个不同 pid（文件栅栏）。"""
    index, marker_dir, total, timeout = args
    pid = os.getpid()
    with open(os.path.join(marker_dir, "w%04d_%d" % (index, pid)), "w") as handle:
        handle.write(str(pid))
    deadline = time.time() + timeout
    while time.time() < deadline:
        pids = set()
        for name in os.listdir(marker_dir):
            if name.startswith("w") and "_" in name:
                pids.add(name.split("_", 1)[1])
        if len(pids) >= total:
            return pid
        time.sleep(0.005)
    raise RuntimeError("ready barrier timeout: %d/%d workers" % (len(pids), total))


def _children_pids():
    return sorted(child.pid for child in multiprocessing.active_children())


def _alive(pid):
    return os.path.exists("/proc/%d" % pid)


def _cuda_state():
    """父进程 CUDA 是否已初始化（查询失败记 None）。"""
    try:
        import torch
    except Exception:
        return None
    try:
        return bool(torch.cuda.is_initialized())
    except Exception:
        return None


def _measure_once(workers, start_method, timeout):
    """跑一轮完整生命周期，返回各阶段耗时与 pids。"""
    metrics = None
    with tempfile.TemporaryDirectory() as marker_dir:
        children_before = _children_pids()
        row = {
            "workers": workers,
            "start_method": start_method or "default",
            "parent_threads_at_open": threading.active_count(),
            "parent_cuda_initialized_at_open": _cuda_state(),
            "children_before": len(children_before),
        }

        started = time.perf_counter()
        pool = open_pool(metrics, workers, "probe", start_method)
        row["construct_s"] = time.perf_counter() - started

        started = time.perf_counter()
        args = [(index, marker_dir, workers, timeout) for index in range(workers)]
        pids = pool.map(_ready_marker, args, chunksize=1)
        row["ready_batch_s"] = time.perf_counter() - started
        row["distinct_worker_pids"] = len(set(pids))

        started = time.perf_counter()
        pool.map(_square, [1])
        row["first_task_after_ready_s"] = time.perf_counter() - started

        warm = []
        for _ in range(5):
            started = time.perf_counter()
            pool.map(_square, list(range(10)))
            warm.append(time.perf_counter() - started)
        row["warm_task_s"] = sorted(warm)[len(warm) // 2]

        pool_children = _children_pids()
        started = time.perf_counter()
        close_pool(metrics, pool, "probe")
        row["close_s"] = time.perf_counter() - started
        row["children_after_close"] = len(_children_pids())
        row["pool_children_alive_after_close"] = sum(
            1 for pid in pool_children if _alive(pid))

        # 异常回收：worker 抛异常后仍要能干净回收
        pool = open_pool(metrics, workers, "probe-error", start_method)
        raised = False
        try:
            pool.map(_boom, [1, 2], chunksize=1)
        except RuntimeError:
            raised = True
        error_children = _children_pids()
        close_pool(metrics, pool, "probe-error")
        row["error_raised"] = raised
        row["children_after_error_close"] = len(_children_pids())
        row["error_children_alive_after_close"] = sum(
            1 for pid in error_children if _alive(pid))
    return row


def main():
    parser = argparse.ArgumentParser(description="P3 池生命周期探针")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--start-method", default=None,
                        help="fork / spawn；缺省 = 系统默认")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="就绪栅栏超时（秒）")
    parser.add_argument("--json", default=None, help="结果写入的 JSON 路径")
    args = parser.parse_args()

    rows = []
    for index in range(args.repeat):
        row = _measure_once(args.workers, args.start_method, args.timeout)
        row["repeat"] = index + 1
        rows.append(row)
        print("[%d/%d] workers=%d method=%s 构造=%.3fs 就绪批=%.3fs "
              "就绪后首任务=%.3fs 暖任务=%.3fs 回收=%.3fs "
              "子进程(前后)=%d→%d 异常回收后存活=%d 线程=%d cuda=%s"
              % (index + 1, args.repeat, row["workers"], row["start_method"],
                 row["construct_s"], row["ready_batch_s"],
                 row["first_task_after_ready_s"], row["warm_task_s"],
                 row["close_s"], row["children_before"],
                 row["children_after_close"],
                 row["error_children_alive_after_close"],
                 row["parent_threads_at_open"],
                 row["parent_cuda_initialized_at_open"]))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"rows": rows}, handle, ensure_ascii=False, indent=2)
        print("JSON: %s" % args.json)


if __name__ == "__main__":
    main()
