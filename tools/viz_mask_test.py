#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""滑动球形掩码 —— 可视化 / A-B 对比工具。

做三件事：
  1. 采样密度图为点云（复用 sampling.sampler.sample_density_map）
  2. 切换三种中心选取模式（greedy / fps / kpconv）生成球形掩码并导出中心
  3. 把采样 .txt 点云转成 CA 原子 PDB，便于在 PyMOL / ChimeraX 里直接可视化

设计：可读 + 模块化 + 复用现有函数（不重复造轮子）。
参考风格：cryoAlign/source/test.py 的 save_coordinates_to_pdb。

用法：
  # 采样目标 + 用三种模式生成掩码 + 全部导出 PDB
  python tools/viz_mask_test.py <target.mrc> <source.txt|source.mrc> --mode all --out mask_viz

  # 只跑一种模式
  python tools/viz_mask_test.py <target.mrc> <source.txt> --mode fps

  # 只把一个已有 .txt 点云转成可视化 PDB
  python tools/viz_mask_test.py --viz-txt path/to/points.txt --out mask_viz
"""
import os
import sys
import argparse

import numpy as np

# 让脚本在任意目录下都能 import 到 protassem
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from protassem.core.io import load_sample_points
from protassem.sampling.sampler import sample_density_map
from protassem.fitting.sw_mask import generate_spherical_masks

MODES = ["greedy", "fps", "kpconv"]


# ----------------------------------------------------------------------
# 可视化基元：点云 -> CA 原子 PDB
# ----------------------------------------------------------------------
def points_to_pdb(points, pdb_path, chain="A", resname="ALA"):
    """把 (N,3) 点云写成 CA 原子的 PDB，PyMOL / ChimeraX 可直接打开。"""
    points = np.asarray(points).reshape(-1, 3)
    out_dir = os.path.dirname(os.path.abspath(pdb_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(pdb_path, "w") as f:
        f.write("MODEL\n")
        for i, (x, y, z) in enumerate(points, 1):
            f.write("ATOM  %5d  CA  %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f"
                    "          C\n"
                    % (i % 100000, resname, chain, (i - 1) % 9999 + 1,
                       x, y, z, 1.0, 1.0))
        f.write("ENDMDL\n")
    print("  wrote %d points -> %s" % (len(points), pdb_path))
    return pdb_path


def visualize_txt(txt_file, pdb_path=None):
    """把采样 .txt 点云转成可视化 PDB（解决 .txt 不便可视化的问题）。"""
    points, _ = load_sample_points(txt_file)
    if pdb_path is None:
        pdb_path = os.path.splitext(txt_file)[0] + "_viz.pdb"
    return points_to_pdb(points, pdb_path)


# ----------------------------------------------------------------------
# 采样：输入若为 .mrc 先采样成 .txt；若已是 .txt 直接用
# ----------------------------------------------------------------------
def resolve_txt(path, contour, voxel, out_dir):
    if path.lower().endswith(".txt"):
        return path
    _, _, txt = sample_density_map(path, contour=contour, voxel_size=voxel,
                                   output_dir=out_dir)
    return txt


# ----------------------------------------------------------------------
# 单个模式：生成掩码 + 导出中心（及可选每个掩码点云）
# ----------------------------------------------------------------------
def run_mode(target_txt, source_txt, mode, out_dir, dump_masks=False, **mask_kw):
    print("\n==== 模式: %s ====" % mode)
    masks = generate_spherical_masks(source_txt, target_txt,
                                     center_mode=mode, verbose=True, **mask_kw)
    mdir = os.path.join(out_dir, "mode_%s" % mode)
    os.makedirs(mdir, exist_ok=True)

    # masks[0] 是“完整数据”掩码，局部球形掩码从 [1:] 开始
    locals_ = masks[1:]
    centers = np.array([m.center for m in locals_]) if locals_ else np.empty((0, 3))
    points_to_pdb(centers, os.path.join(mdir, "centers_%s.pdb" % mode))

    if dump_masks:
        for m in locals_:
            points_to_pdb(m.point_cloud_data["point"],
                          os.path.join(mdir, "mask_%04d.pdb" % m.id))

    print("[%s] 局部掩码数 = %d；中心 PDB = %s"
          % (mode, len(locals_), os.path.join(mdir, "centers_%s.pdb" % mode)))
    return masks


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="滑动球形掩码可视化 / A-B 对比")
    ap.add_argument("target", nargs="?", help="目标密度图 .mrc 或已采样 .txt")
    ap.add_argument("source", nargs="?", help="源结构点云 .txt 或其模拟 .mrc")
    ap.add_argument("--mode", default="all",
                    help="中心模式: greedy | fps | kpconv | all（默认 all）")
    ap.add_argument("--contour", type=float, default=None,
                    help="目标 contour（缺省自动 3σ）")
    ap.add_argument("--voxel", type=float, default=2.0, help="采样体素 (默认 2.0)")
    ap.add_argument("--out", default="mask_viz", help="输出目录")
    ap.add_argument("--dump-masks", action="store_true",
                    help="额外导出每个局部掩码的点云 PDB")
    ap.add_argument("--viz-txt", default=None,
                    help="只把这个 .txt 点云转成可视化 PDB，然后退出")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # 仅可视化某个 .txt
    if args.viz_txt:
        name = os.path.splitext(os.path.basename(args.viz_txt))[0]
        visualize_txt(args.viz_txt, os.path.join(args.out, name + "_viz.pdb"))
        return

    if not args.target or not args.source:
        ap.error("需要 target 和 source（或用 --viz-txt 单独可视化一个 .txt）")

    target_txt = resolve_txt(args.target, args.contour, args.voxel, args.out)
    source_txt = resolve_txt(args.source, 0.0, args.voxel, args.out)

    # 目标点云本身先导出可视化
    visualize_txt(target_txt, os.path.join(args.out, "target_viz.pdb"))

    modes = MODES if args.mode == "all" else [args.mode]
    summary = []
    for mode in modes:
        masks = run_mode(target_txt, source_txt, mode, args.out,
                         dump_masks=args.dump_masks)
        summary.append((mode, len(masks) - 1))

    print("\n==== 对比汇总（局部掩码中心数）====")
    for mode, n in summary:
        print("  %-8s : %d" % (mode, n))
    print("可视化：把 %s/target_viz.pdb 和各 mode_*/centers_*.pdb 一起拖进 "
          "ChimeraX/PyMOL 对比中心分布。" % args.out)


if __name__ == "__main__":
    main()
