"""protassem 装配流水线命令行入口。

用法:
  python main.py <data_dir>
      在目录内自动查找 .mrc / 结构文件(.pdb/.cif) / resolution.txt / contour_level.txt
  python main.py <mrc> <struct_dir> <res> <cont> [out_dir]
      直接给定密度图、结构目录、分辨率、contour（可选输出目录）

可选参数见下方 assembly_kwargs 各项注释。带值参数登记在 _VALUE_OPTS 中，
解析位置参数时需连同其值一起跳过。
"""

import os
import sys
from protassem.core.io import find_files, read_param_file
from protassem.pipeline import run_pipeline


def _parse_float_opt(argv, name, default):
    """从 argv 取 `--name <float>`，缺省返回 default。"""
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return float(argv[i + 1])
    return default


def _parse_int_opt(argv, name, default):
    """从 argv 取 `--name <int>`，缺省返回 default。"""
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return int(argv[i + 1])
    return default


if __name__ == "__main__":
    argv = sys.argv[1:]
    # 带值选项：解析位置参数时要连同其后的值一起跳过，避免把值误当作位置参数
    _VALUE_OPTS = {"--log-file", "--chain-threshold", "--domain-threshold",
                   "--domain-min-cc", "--similarity-threshold",
                   "--num-processes", "--batch-size", "--complex-min-cc",
                   "--refine-tm", "--no-domain-split", "--complex-threshold"}
    positional = []
    _i = 0
    while _i < len(argv):
        _a = argv[_i]
        if _a.startswith("--"):
            _i += 2 if _a in _VALUE_OPTS else 1
            continue
        positional.append(_a)
        _i += 1

    # 日志：--log 写默认日志文件；--log-file <path> 指定日志路径
    log_file = None
    if "--log" in argv:
        log_file = True
    for i, a in enumerate(argv):
        if a == "--log-file" and i + 1 < len(argv):
            log_file = argv[i + 1]

    assembly_kwargs = {
        # 整链拟合接受阈值：链级 cc_mask >= 此值才作为整链直接接受
        "chain_threshold": _parse_float_opt(argv, "--chain-threshold", 0.45),
        # 结构域第 1 轮接受阈值（随轮次向 min_domain_threshold 衰减）
        "initial_domain_threshold": _parse_float_opt(argv, "--domain-threshold", 0.45),
        # 结构域阈值下限/地板：轮次衰减与同源宽松后不低于此值
        "min_domain_threshold": _parse_float_opt(argv, "--domain-min-cc", 0.35),
        # 同源判定的 TM-score 阈值（装配阶段的同源分组）
        "similarity_threshold": _parse_float_opt(argv, "--similarity-threshold", 0.85),
        # 多链复合物模板整体拟合的接受阈值（无复合物模板时不触发）
        "complex_threshold": _parse_float_opt(argv, "--complex-threshold", 0.35),
        # 复合物模板是否做域级优化（默认关）
        "complex_domain_opt": "--complex-domain-opt" in argv,
        # 已接受链是否再尝试用域改进（默认开，--no-improve-accepted 关）
        "improve_accepted": "--no-improve-accepted" not in argv,
        # 结构域局部优化（默认开，--no-domain-opt 关）
        "domain_opt": "--no-domain-opt" not in argv,
        # 是否保存每次拟合尝试用于调试（默认关）
        "save_all_attempts": "--save-all-attempts" in argv,
        # 结束后清理临时文件（默认关）
        "cleanup": "--cleanup" in argv,
        # Step4 同源域枚举精修 -> refined_complex.cif（默认开，--no-refine 关）
        "do_refine": "--no-refine" not in argv,
        # 最终复合物的"结构域过滤阈值"：域 cc < 此值不写入 assembled_complex.cif；
        # assembled_complex_all.cif 不过滤、保留全部域
        "complex_min_cc": _parse_float_opt(argv, "--complex-min-cc", 0.25),
        # Step4 精修枚举时判定同源域的 TM 阈值
        "refine_tm": _parse_float_opt(argv, "--refine-tm", 0.75),
        # 并行进程数
        "num_processes": _parse_int_opt(argv, "--num-processes", 10),
        # 采样/拟合批大小
        "batch_size": _parse_int_opt(argv, "--batch-size", 10),
        # Step5 同源链精修（借同源链补回缺失域；默认关，需显式 --homo-chain-refine）
        "homo_chain_refine": "--homo-chain-refine" in argv,
        # 装配前并行预筛：用模板原始位姿直接接受高 cc 同源链（默认开，--no-pre-screen 关）
        "pre_screen": "--no-pre-screen" not in argv,
        # 可选：预筛按结构域而非整链接受；未命中任何域的链仍走常规整链流程
        "pre_screen_by_domain": "--pre-screen-by-domain" in argv,
        # 不做结构域拆分的链 id 集合（逗号分隔，如 --no-domain-split A,B）
        "no_domain_split_chains": frozenset(
            next((argv[i+1] for i, a in enumerate(argv)
                  if a == "--no-domain-split" and i+1 < len(argv)), "").split(",")
        ) - {""},
    }

    if len(positional) == 1:
        data_dir = positional[0]
        run_pipeline(
            density_mrc=find_files(data_dir, ".mrc")[0],
            structure_files=find_files(data_dir, ".pdb", ".cif"),
            resolution=read_param_file(os.path.join(data_dir, "resolution.txt")),
            contour=read_param_file(os.path.join(data_dir, "contour_level.txt")),
            log_file=log_file,
            assembly_kwargs=assembly_kwargs,
        )
    elif len(positional) >= 4:
        run_pipeline(
            density_mrc=positional[0],
            structure_files=find_files(positional[1], ".pdb", ".cif"),
            resolution=float(positional[2]),
            contour=float(positional[3]),
            output_dir=positional[4] if len(positional) > 4 else None,
            log_file=log_file,
            assembly_kwargs=assembly_kwargs,
        )
    else:
        print(__doc__)
