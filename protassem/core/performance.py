"""Lightweight runtime metrics for a pipeline run.

One Metrics instance per fitting request: events are appended to a JSONL file as
they happen, and a per-stage summary is written at the end. Only the process that
owns the instance writes to it; workers must not share one.
"""

import json
import os
import platform
import threading
import time
from contextlib import contextmanager


def configure_cpu_threads(workers):
    """Default the BLAS thread limits to 1 and return the usable worker count.

    Note: ``os.environ.setdefault`` only affects libraries that have not been
    imported yet, and this call happens after NumPy is imported. It therefore
    does not guarantee that the already-loaded BLAS uses a single thread; real
    thread control is phase-2 work (plan step P1).
    """
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    return max(1, int(workers or 1))


class Metrics:
    """Collect one event per stage; write JSONL while running, summary at the end."""

    def __init__(self, output_dir=None, run_id=None):
        self.run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
        self.output_dir = output_dir
        self.records = []
        self._lock = threading.Lock()

    @contextmanager
    def stage(self, name, **fields):
        """Time a block; the event is recorded even when the block raises."""
        started = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, time.perf_counter() - started, **fields)

    def record(self, stage, elapsed_s, **fields):
        """Append one event to memory and to performance.jsonl."""
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

    def write_summary(self):
        """Write performance_summary.json: per-stage count and total seconds."""
        if not self.output_dir:
            return
        by_stage = {}
        for record in self.records:
            entry = by_stage.setdefault(record["stage"],
                                        {"count": 0, "seconds": 0.0})
            entry["count"] += 1
            entry["seconds"] += record["elapsed_s"]
        summary = {"run_id": self.run_id,
                   "platform": platform.platform(),
                   "cpu_count": os.cpu_count(),
                   "stages": by_stage}
        path = os.path.join(self.output_dir, "performance_summary.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
