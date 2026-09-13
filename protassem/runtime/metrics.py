"""运行级计时（阶段二 P2）。

一个运行一个 `Metrics`：主进程拥有并写出 `performance.jsonl`（逐事件）与
`performance_summary.json`（按 stage 汇总）。事件字段：`run_id` / `stage` /
`elapsed_s` / `timestamp`，可选 `task_id`、`mask_version`、`status`、
`candidate_count`、`accepted`。worker 只返回耗时，不并发写文件。

线程环境变量由入口的 `RuntimeConfig.apply_thread_env()`（P1）在导入 NumPy/Torch 之前设定；
本模块不再负责线程控制。
"""

import json
import os
import platform
import threading
import time
from contextlib import contextmanager


def worker_count(workers):
    """worker 数量的下限保护（>=1）；线程 env 由 RuntimeConfig 在入口负责。"""
    return max(1, int(workers or 1))


class Metrics:
    """收集每个 stage 的事件；边跑边写 JSONL，结束时写汇总。"""

    def __init__(self, output_dir=None, run_id=None):
        self.run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
        self.output_dir = output_dir
        self.records = []
        self._lock = threading.Lock()

    @contextmanager
    def stage(self, name, **fields):
        """给代码块计时；块内抛异常时同样记录事件。"""
        started = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, time.perf_counter() - started, **fields)

    def record(self, stage, elapsed_s, **fields):
        """记录一条事件到内存与 performance.jsonl。"""
        row = {"run_id": self.run_id,
               "stage": stage,
               "elapsed_s": round(float(elapsed_s), 6),
               "timestamp": time.time()}
        row.update(fields)
        with self._lock:
            self.records.append(row)
        if not self.output_dir:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, "performance.jsonl")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def summary(self):
        """按 stage 汇总：次数、累计秒数；并给出总墙钟（pipeline_total 事件）。"""
        by_stage = {}
        for record in self.records:
            entry = by_stage.setdefault(record["stage"],
                                        {"count": 0, "seconds": 0.0})
            entry["count"] += 1
            entry["seconds"] += record["elapsed_s"]
        total_wall = sum(record["elapsed_s"] for record in self.records
                         if record["stage"] == "pipeline_total")
        return {"run_id": self.run_id,
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "total_wall_s": total_wall,
                "stages": by_stage}

    def write_summary(self):
        """写 performance_summary.json（并行阶段的累计秒数不能相加当作墙钟）。"""
        if not self.output_dir:
            return None
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, "performance_summary.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.summary(), handle, indent=2)
        return path
