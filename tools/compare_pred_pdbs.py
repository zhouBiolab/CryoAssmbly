"""对比两套配准产物目录里的 `pred_*.pdb` 坐标（统一验收：PDB 坐标以 1e-3 Å 检查）。

用法：
    python tools/compare_pred_pdbs.py --base <bench_out_a> --new <bench_out_b> [--report out.md]

对每个同名 `pred_*.pdb` 逐原子比较坐标（ATOM/HETATM 行按顺序），报告：
  - 逐文件最大坐标偏差（Å）；
  - 超过阈值的文件清单；
  - 全局最大偏差与文件数统计。

说明：T06 的单侧编码与联合布局在 float32 下存在 kernel 求和顺序差异（≤4e-6 的位姿差），
写出时 `%.3f` 舍入会让部分文件的字节不同（md5 不等），因此按任务卡要求用**坐标容差**判定。
"""

import argparse
import glob
import os


def read_coordinates(path):
    coordinates = []
    with open(path, "r", errors="ignore") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                coordinates.append((float(line[30:38]), float(line[38:46]),
                                     float(line[46:54])))
    return coordinates


def max_delta(left_path, right_path):
    left = read_coordinates(left_path)
    right = read_coordinates(right_path)
    if len(left) != len(right):
        return None, len(left), len(right)
    worst = 0.0
    for (lx, ly, lz), (rx, ry, rz) in zip(left, right):
        worst = max(worst, abs(lx - rx), abs(ly - ry), abs(lz - rz))
    return worst, len(left), len(right)


def main(argv=None):
    parser = argparse.ArgumentParser(description="pred_*.pdb 坐标对比（统一验收 1e-3 Å）")
    parser.add_argument("--base", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)

    base_files = {os.path.basename(path): path
                  for path in glob.glob(os.path.join(args.base, "target_*", "pred_*.pdb"))}
    new_files = {os.path.basename(path): path
                 for path in glob.glob(os.path.join(args.new, "target_*", "pred_*.pdb"))}
    shared = sorted(set(base_files) & set(new_files))
    only_base = sorted(set(base_files) - set(new_files))
    only_new = sorted(set(new_files) - set(base_files))

    lines = ["# pred_*.pdb 坐标对比（容差 %.1e Å；PDB 写出精度 %.3f，恰好一个量化步长判为容差内）"
             % (args.tolerance, 1e-3), "",
             "- base：`%s`（%d 个预测）" % (os.path.abspath(args.base), len(base_files)),
             "- new ：`%s`（%d 个预测）" % (os.path.abspath(args.new), len(new_files)),
             "- 同名文件 %d 个；仅 base 有 %s；仅 new 有 %s"
             % (len(shared), only_base or "无", only_new or "无"), ""]

    worst_overall = 0.0
    offenders = []
    shape_mismatch = []
    # PDB 写出精度是 %.3f，因此"恰好一个量化步长"(1e-3) 的差异属于舍入边界，判为容差内
    threshold = args.tolerance * (1.0 + 1e-6)
    for name in shared:
        delta, base_atoms, new_atoms = max_delta(base_files[name], new_files[name])
        if delta is None:
            shape_mismatch.append((name, base_atoms, new_atoms))
            continue
        worst_overall = max(worst_overall, delta)
        if delta > threshold:
            offenders.append((name, delta))

    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    lines.append("| 全局最大坐标偏差 | %.3g Å |" % worst_overall)
    lines.append("| 超出容差的文件数 | %d |" % len(offenders))
    lines.append("| 原子数不一致的文件数 | %d |" % len(shape_mismatch))
    lines.append("")
    if offenders:
        lines.append("超出容差的文件：")
        for name, delta in sorted(offenders, key=lambda item: -item[1])[:20]:
            lines.append("- `%s`：%.3g Å" % (name, delta))
        lines.append("")
    if shape_mismatch:
        lines.append("原子数不一致：%s" % shape_mismatch)
        lines.append("")
    passed = not offenders and not shape_mismatch and not only_base and not only_new
    lines.append("判定：%s" % ("通过（坐标均在容差内且文件集合一致）" if passed
                              else "**未通过**"))
    text = "\n".join(lines) + "\n"
    print(text)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            handle.write(text)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
