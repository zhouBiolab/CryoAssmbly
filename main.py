"""protassem 装配流水线命令行入口。

用法:
  python main.py <data_dir>
      在目录内自动查找 .mrc / 结构文件(.pdb/.cif) / resolution.txt / contour_level.txt
  python main.py <mrc> <struct_dir> <res> <cont> [out_dir]
      直接给定密度图、结构目录、分辨率、contour（可选输出目录）

选项可位于位置参数之前、之间或之后（parse_intermixed_args）；数值选项的名称与
默认值与 README「参数说明」保持一致。
"""

import argparse
import os
import sys

# 重依赖（NumPy/Torch）必须在 RuntimeConfig.apply_thread_env() 之后再导入，
# 否则线程限制对已加载的 BLAS 无效 —— 见 main() 内部的两段式导入。


def build_parser():
    """构建 CLI 解析器：参数名、默认值、开关语义与历史版本完全一致。"""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="protassem 装配流水线：体素化 -> 点云采样 -> PARENet 拟合 -> 组装 -> 精修",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="位置参数：<data_dir>，或 <density.mrc> <struct_dir> <resolution> "
               "<contour> [output_dir]")
    parser.add_argument("paths", nargs="*", metavar="PATH",
                        help="数据目录；或 mrc 结构目录 分辨率 contour [输出目录]")
    parser.add_argument("--log", action="store_true",
                        help="写日志文件到输出目录（pipeline_<时间戳>.log）")
    parser.add_argument("--log-file", metavar="PATH",
                        help="指定日志文件路径（优先于 --log）")
    # 整链拟合接受阈值：链级 cc_mask >= 此值才作为整链直接接受
    parser.add_argument("--chain-threshold", type=float, default=0.45, metavar="CC")
    # 结构域第 1 轮接受阈值（随轮次向 --domain-min-cc 衰减）
    parser.add_argument("--domain-threshold", type=float, default=0.45, metavar="CC")
    # 结构域阈值下限/地板：轮次衰减与同源宽松后不低于此值
    parser.add_argument("--domain-min-cc", type=float, default=0.35, metavar="CC")
    # 多链复合物模板整体拟合的接受阈值（无复合物模板时不触发）
    parser.add_argument("--complex-threshold", type=float, default=0.35, metavar="CC")
    # 最终复合物的结构域过滤阈值：cc 低于此值的域不写入 assembled_complex.cif
    parser.add_argument("--complex-min-cc", type=float, default=0.25, metavar="CC")
    # 同源判定的 TM-score 阈值（装配阶段的同源分组）
    parser.add_argument("--similarity-threshold", type=float, default=0.85, metavar="TM")
    # Step4 精修枚举时判定同源域的 TM 阈值
    parser.add_argument("--refine-tm", type=float, default=0.75, metavar="TM")
    parser.add_argument("--num-processes", type=int, default=8, metavar="N",
                        help="并行进程数（CC 计算 / 局部优化）")
    parser.add_argument("--batch-size", type=int, default=8, metavar="N",
                        help="监控循环每攒多少 pred 做一次评估（每批立刻逐个局部优化）")
    # PARENet 掩码参数：掩码半径 = 回转半径 × 该因子；最小点间距 = 掩码半径 × 该因子
    parser.add_argument("--mask-radius-factor", type=float, default=1.35,
                        metavar="F", help="PARENet 掩码半径因子（回转半径的倍数）")
    parser.add_argument("--min-point-distance-factor", type=float, default=0.32,
                        metavar="F", help="掩码内最小点间距因子（掩码半径的倍数）")
    parser.add_argument("--complex-domain-opt", action="store_true",
                        help="复合物域优化：拆内部链 -> 域分割 -> 逐域微调 -> 按链合并比较 CC")
    parser.add_argument("--no-improve-accepted", dest="improve_accepted",
                        action="store_false",
                        help="关闭已接受链的逐域微调（默认开启）")
    parser.add_argument("--no-domain-opt", dest="domain_opt", action="store_false",
                        help="关闭链失败时的域优化兜底（默认开启）")
    parser.add_argument("--save-all-attempts", action="store_true",
                        help="保存所有拟合尝试到 all_attempts/（调试用）")
    parser.add_argument("--cleanup", action="store_true",
                        help="结束后删除 work/ 下的临时目录")
    parser.add_argument("--no-refine", dest="do_refine", action="store_false",
                        help="关闭 Step4 同源域精修（默认开启）")
    parser.add_argument("--homo-chain-refine", action="store_true",
                        help="Step5 同源链精修：残基保护 + 域补回 + 用好链模板修复差链")
    parser.add_argument("--no-pre-screen", dest="pre_screen", action="store_false",
                        help="关闭装配前的原始位姿并行预筛（默认开启）")
    parser.add_argument("--pre-screen-by-domain", action="store_true",
                        help="预筛按结构域而非整链接受；未命中域的链仍走常规流程")
    parser.add_argument("--no-domain-split", metavar="IDS", default="",
                        help="不做结构域拆分的链 ID，逗号分隔，如 A,B")
    parser.add_argument("--runtime-config", metavar="JSON", default=None,
                        help="运行配置 JSON（线程数/种子等；未给出的键取默认值）")
    return parser


def parse_args(argv, parser=None):
    """解析 argv（不含程序名）。

    Returns:
        (parser, args)；当没有位置参数时打印帮助并返回 (parser, None)。

    Raises:
        SystemExit(2): 未知选项、选项缺值、数值非法、位置参数个数不是 1/4/5。
    """
    parser = parser or build_parser()
    args = parser.parse_intermixed_args(argv)
    if not args.paths:
        parser.print_help()
        return parser, None
    if len(args.paths) not in (1, 4, 5):
        parser.error("expected 1 or 4-5 positional arguments, got %d"
                     % len(args.paths))
    return parser, args


def assembly_kwargs_from(args):
    """把解析结果映射为 AssemblyOrchestrator 的关键字参数。"""
    return {
        "chain_threshold": args.chain_threshold,
        "initial_domain_threshold": args.domain_threshold,
        "min_domain_threshold": args.domain_min_cc,
        "similarity_threshold": args.similarity_threshold,
        "complex_threshold": args.complex_threshold,
        "complex_domain_opt": args.complex_domain_opt,
        "improve_accepted": args.improve_accepted,
        "domain_opt": args.domain_opt,
        "save_all_attempts": args.save_all_attempts,
        "cleanup": args.cleanup,
        "do_refine": args.do_refine,
        "complex_min_cc": args.complex_min_cc,
        "refine_tm": args.refine_tm,
        "num_processes": args.num_processes,
        "batch_size": args.batch_size,
        "mask_radius_factor": args.mask_radius_factor,
        "min_point_distance_factor": args.min_point_distance_factor,
        "homo_chain_refine": args.homo_chain_refine,
        "pre_screen": args.pre_screen,
        "pre_screen_by_domain": args.pre_screen_by_domain,
        "no_domain_split_chains": frozenset(
            item.strip() for item in args.no_domain_split.split(",") if item.strip()),
    }


def resolve_log_file(args):
    """--log-file 优先于 --log；返回 True（自动命名）、路径字符串或 None。"""
    if args.log_file:
        return args.log_file
    if args.log:
        return True
    return None


def main(argv=None):
    """CLI 入口；返回进程退出码。"""
    parser, args = parse_args(sys.argv[1:] if argv is None else argv)
    if args is None:
        return 0

    from protassem.runtime.config import RuntimeConfig
    runtime_config = (RuntimeConfig.from_json(args.runtime_config)
                      if args.runtime_config else RuntimeConfig())
    runtime_config.apply_thread_env()   # 必须在导入 NumPy/Torch 之前

    from protassem.core.io import find_files
    from protassem.pipeline import run_pipeline

    kwargs = assembly_kwargs_from(args)
    log_file = resolve_log_file(args)

    if len(args.paths) == 1:
        data_dir = args.paths[0]
        mrcs = find_files(data_dir, ".mrc")
        if not mrcs:
            parser.error("%s: no .mrc file found" % data_dir)
        run_pipeline(
            density_mrc=mrcs[0],
            structure_files=find_files(data_dir, ".pdb", ".cif"),
            resolution=_read_param(parser, os.path.join(data_dir, "resolution.txt")),
            contour=_read_param(parser, os.path.join(data_dir, "contour_level.txt")),
            log_file=log_file,
            assembly_kwargs=kwargs,
            runtime_config=runtime_config,
        )
        return 0

    density_mrc, struct_dir = args.paths[0], args.paths[1]
    try:
        resolution = float(args.paths[2])
        contour = float(args.paths[3])
    except ValueError as exc:
        parser.error("resolution and contour must be numbers: %s" % exc)
    run_pipeline(
        density_mrc=density_mrc,
        structure_files=find_files(struct_dir, ".pdb", ".cif"),
        resolution=resolution,
        contour=contour,
        output_dir=args.paths[4] if len(args.paths) == 5 else None,
        log_file=log_file,
        assembly_kwargs=kwargs,
        runtime_config=runtime_config,
    )
    return 0


def _read_param(parser, path):
    """读取自动模式下的数值参数文件；缺失时报出具体路径。"""
    from protassem.core.io import read_param_file
    if not os.path.isfile(path):
        parser.error("missing parameter file: %s" % path)
    return read_param_file(path)


if __name__ == "__main__":
    raise SystemExit(main())
