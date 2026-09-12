#!/usr/bin/env python3
"""计算结构与密度图的 CC_mask（独立小工具）。

用法:
    python compute_cc_mask.py <structure> <density.mrc> <resolution> [contour]

参数:
    structure    结构文件 (.pdb 或 .cif)
    density.mrc  密度图文件 (.mrc)
    resolution   分辨率 (Å)
    contour      密度阈值 (可选，默认 0.0；低于此值的密度被忽略)

示例:
    python compute_cc_mask.py model.pdb EMD-8436.mrc 5.6 0.04

也可作为模块导入:
    from compute_cc_mask import compute
    cc = compute("model.pdb", "map.mrc", 5.6, 0.04)
"""
import os
import sys
import argparse

# 让脚本在任意目录下都能找到 protassem 包
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from protassem.core.scoring import calculate_cc_mask


def compute(structure, density_mrc, resolution, contour=0.0):
    """计算结构与密度图的 CC_mask。

    参数:
        structure:   结构文件路径 (.pdb 或 .cif)
        density_mrc: 密度图文件路径 (.mrc)
        resolution:  分辨率 (Å)
        contour:     密度阈值 (默认 0.0)
    返回:
        cc_mask (float)
    """
    return calculate_cc_mask(density_mrc, structure, resolution, contour)


def main():
    p = argparse.ArgumentParser(
        description="计算结构与密度图的 CC_mask",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("structure", help="结构文件 (.pdb 或 .cif)")
    p.add_argument("density_mrc", help="密度图文件 (.mrc)")
    p.add_argument("resolution", type=float, help="分辨率 (Å)")
    p.add_argument("contour", type=float, nargs="?", default=0.0,
                   help="密度阈值 (默认 0.0)")
    a = p.parse_args()

    for f in (a.structure, a.density_mrc):
        if not os.path.exists(f):
            print(f"错误: 文件不存在 - {f}")
            return 1

    cc = compute(a.structure, a.density_mrc, a.resolution, a.contour)
    print(f"CC_mask = {cc:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
