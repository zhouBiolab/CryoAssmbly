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

# 单侧几何 CPU 缓存默认容量（MiB，0 = 关闭）。常量放在这里是为了让本模块保持"无重依赖"：
# main.py 必须在导入 NumPy/Torch 之前导入它（两段式导入，P1）。
DEFAULT_GEOMETRY_CACHE_MB = 512

# 模型推理是否允许 TF32。T06 实测：TF32 会让卷积/matmul 的数值依赖张量形状，从而让"单侧编码"
# 这种改变形状的重构改变结果（真实输入位姿差 0.127）；关闭后差 3.8e-06。该精度模式在本模型上
# 也没有带来可测加速，因此 **split 模式强制关闭**；joint（旧路径）跟随框架默认（True）以保持
# 既有数值与冻结基线。
DEFAULT_ALLOW_TF32 = None       # None = 跟随 inference_mode

# 推理路径：joint = 联合布局 + forward（旧路径，默认）；split = 单侧编码 + 双侧配准（T06 起可选）
INFERENCE_MODES = ("joint", "split")
DEFAULT_INFERENCE_MODE = "joint"

# 源编码缓存（T07）：GPU 预算（MiB，0 = 关闭）；只在 split 模式下生效（joint 不做单侧编码）
DEFAULT_ENCODING_CACHE_MB = 256

# 位姿假设评分分块（T08）：0 = 原整批路径（默认）；>0 时按该大小分块（任务卡建议先测 64）
DEFAULT_HYPOTHESIS_CHUNK = 0

# CPU 尾部流水线（T09）：默认**关闭**（原路径）。真实运行 A/B（test/1，同配置）实测
# 1082.95 → 985.19 s（−9.0%）且三个 CIF md5 与冻结基线完全一致；但同一份代码在单客户端
# 微基准里反而慢 9.5%（尾部线程与主线程争 CPU/GIL）——收益依赖"多 worker 把 GPU 压满"，
# 尚无重复测量，故不提升默认；用 tail_pipeline=true 开启（T10 复测后决定）。
DEFAULT_TAIL_PIPELINE = False


def effective_allow_tf32(inference_mode, allow_tf32=None):
    """解析 TF32 策略：显式值优先；未给定时跟随推理模式。

    split 模式在 TF32 下**不等价**（实测位姿差 0.127），因此显式要求 allow_tf32=True 时直接报错，
    而不是给出"看起来正常"的错误结果。
    """
    if inference_mode not in INFERENCE_MODES:
        raise ValueError("未知的 inference_mode: %r（支持 %s）"
                         % (inference_mode, "/".join(INFERENCE_MODES)))
    if allow_tf32 is None:
        return inference_mode == "joint"
    if inference_mode == "split" and allow_tf32:
        raise ValueError("inference_mode=split 要求 allow_tf32=false：TF32 下单侧编码不等价"
                         "（真实输入位姿差 0.127）")
    return bool(allow_tf32)


def apply_tf32_policy(allow_tf32):
    """设定 TF32 精度策略（在**模型推理进程**里调用，早于任何前向）。

    allow_tf32=False：关闭 cuDNN 卷积与 matmul 的 TF32，数值与张量形状无关（split 模式要求）。
    allow_tf32=True ：保留框架默认（PyTorch 1.10 两者默认 True），即 joint 旧路径的既有数值。
    """
    import torch
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    return {"cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32)}


@dataclass
class RuntimeConfig:
    """运行配置；当前只有线程、随机种子与 T05 的几何缓存容量，P2–P5 在此基础上扩展。"""

    blas_threads: int = 1
    seed: int = 7351
    # 进程池启动方式：None 表示系统默认（Linux 为 fork）；"spawn" 用于单独测启动成本
    pool_start_method: str = None
    # 单侧几何 CPU 缓存容量（MiB，0 = 关闭）；透传给 PARENet 常驻服务进程（T05）
    geometry_cache_mb: int = DEFAULT_GEOMETRY_CACHE_MB
    # 推理路径（T06）：joint = 联合布局 + forward（默认，保持既有数值）；split = 单侧编码 + 双侧配准
    inference_mode: str = DEFAULT_INFERENCE_MODE
    # TF32：None = 跟随 inference_mode（joint→True、split→False）；split 下不允许 True
    allow_tf32: bool = DEFAULT_ALLOW_TF32
    # 源编码缓存 GPU 预算（MiB，T07；0 = 关闭）；joint 模式不使用
    encoding_cache_mb: int = DEFAULT_ENCODING_CACHE_MB
    # 位姿假设评分分块（T08；0 = 原整批路径，>0 = 按假设分块）
    hypothesis_chunk: int = DEFAULT_HYPOTHESIS_CHUNK
    # CPU 尾部流水线（T09；False = 就地执行后处理与写盘）
    tail_pipeline: bool = DEFAULT_TAIL_PIPELINE

    def __post_init__(self):
        # 早失败：非法模式、不安全的精度组合或负的分块在构造配置时就报错
        effective_allow_tf32(self.inference_mode, self.allow_tf32)
        if int(self.hypothesis_chunk) < 0:
            raise ValueError("hypothesis_chunk 不能为负：%r" % (self.hypothesis_chunk,))

    def tf32(self):
        """实测生效的 TF32 策略（解析 inference_mode 与显式覆盖）。"""
        return effective_allow_tf32(self.inference_mode, self.allow_tf32)

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
