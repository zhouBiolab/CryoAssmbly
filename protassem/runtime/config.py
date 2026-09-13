"""运行级配置与线程控制（阶段二 P1）。

要点：BLAS/Torch 的线程数必须在导入 NumPy/Torch **之前**用环境变量设定。
既有的 `core/performance.configure_cpu_threads` 在 NumPy 已导入后才调用，
只能影响此后再加载的库，因此实测（threadpoolctl）显示 OpenBLAS 仍为 128 线程。
入口 `main.py` 现在采用两段式导入：先解析参数并调用 `apply_thread_env()`，再导入重依赖。
"""

import json
import os
from dataclasses import dataclass, fields

# 需要在导入 NumPy/Torch 前设定的线程环境变量
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


@dataclass
class RuntimeConfig:
    """运行配置；当前只有线程与随机种子，P2–P5 在此基础上扩展。"""

    blas_threads: int = 1
    seed: int = 7351

    @classmethod
    def from_json(cls, path):
        """从 JSON 读取配置；未给出的键取默认值，未知键报错。"""
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError("%s: unknown runtime config key(s): %s"
                             % (path, ", ".join(unknown)))
        return cls(**data)

    def apply_thread_env(self):
        """设定线程环境变量（覆盖旧值，保证同一次运行可复现）。

        必须在导入 NumPy/Torch 之前调用，否则对已加载的 BLAS 无效。
        """
        value = str(max(1, int(self.blas_threads)))
        for name in THREAD_ENV_VARS:
            os.environ[name] = value
        return value

    def describe_effective_threads(self):
        """实测生效线程数（threadpoolctl / torch）：用于验证，不靠假设。"""
        import threadpoolctl
        import torch
        return {
            "env": {name: os.environ.get(name) for name in THREAD_ENV_VARS},
            "blas": [{"internal_api": info.get("internal_api"),
                      "num_threads": info.get("num_threads")}
                     for info in threadpoolctl.threadpool_info()],
            "torch_intra": torch.get_num_threads(),
        }
