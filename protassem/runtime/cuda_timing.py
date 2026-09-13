"""CUDA 事件分阶段计时（任务卡 T02）。

为什么不用 perf_counter：CUDA 调用是异步的，`time.perf_counter()` 只能测到 kernel 入队耗时。
本模块用 `torch.cuda.Event` 记录阶段边界（记录时不强制同步），在 `flush()` 时**同步一次**再把
各阶段耗时交给 sink——避免"在异步流上读时间"造成的错误计时。
"""

import time
from contextlib import contextmanager

import torch


class CudaStageRecorder:
    """记录若干 CUDA 阶段耗时；`flush()` 同步一次后回放给 sink(name, seconds)。"""

    def __init__(self, sink):
        self.sink = sink
        self._events = []

    @contextmanager
    def stage(self, name):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._events.append((name, start, end))

    def flush(self):
        """同步一次并回放所有阶段耗时（秒）；空记录时不做同步。"""
        if not self._events:
            return
        torch.cuda.synchronize()
        for name, start, end in self._events:
            self.sink(name, start.elapsed_time(end) / 1000.0)
        self._events = []


def cuda_stage(recorder, name):
    """recorder 为 None 时返回空上下文，便于保持原调用路径不变。"""
    if recorder is None:
        class _Null:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc_value, traceback):
                return False
        return _Null()
    return recorder.stage(name)


def host_stage(sink, name):
    """CPU 侧阶段：用 perf_counter 计时并在退出时回放（同一 sink 口径）。"""
    class _Host:
        def __enter__(self):
            self.started = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            if sink is not None:
                sink(name, time.perf_counter() - self.started)
            return False
    return _Host()
