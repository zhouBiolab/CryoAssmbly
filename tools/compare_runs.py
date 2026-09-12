"""比较两次运行的装配结果（等价性检查工具，不做结论判断）。

用法：
    python tools/compare_runs.py <baseline_output_dir> <new_output_dir> [--report out.md]

比较内容：
  - final_results/ 下三个 CIF 的 md5（assembled_complex / assembled_complex_all / refined_complex）
  - CIF 不同时，列出每个链的残基数
  - assembly_summary.txt 中的组件行、CC_mask 行与排除域行
只输出差异事实，是否"等价"由使用者和方案第 5 节的验收标准判断。
"""

import argparse
import hashlib
import os
import re

from Bio.PDB import MMCIFParser

CIF_NAMES = ("assembled_complex.cif", "assembled_complex_all.cif", "refined_complex.cif")
SUMMARY_NAME = "assembly_summary.txt"

_SUMMARY_KEEP = re.compile(r"^(\d+\.\s*\[|.*CC_mask|.*cc_mask=|.*domain \d+:|.*Domains:)")


def final_results(output_dir):
    return os.path.join(output_dir, "assembly", "final_results")


def md5(path):
    with open(path, "rb") as handle:
        return hashlib.md5(handle.read()).hexdigest()


def chain_residue_counts(cif_path):
    """返回 {链号: 残基数}（只统计第一个 model）。"""
    structure = MMCIFParser(QUIET=True).get_structure("s", cif_path)
    model = next(structure.get_models())
    return {chain.id: sum(1 for _ in chain) for chain in model}


def summary_lines(path):
    """提取摘要中与结果有关的行。"""
    kept = []
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            text = raw.strip()
            if text and _SUMMARY_KEEP.match(text):
                kept.append(text)
    return kept


def compare(baseline_dir, new_dir):
    """返回差异描述列表。"""
    lines = []
    for name in CIF_NAMES:
        old_path = os.path.join(baseline_dir, name)
        new_path = os.path.join(new_dir, name)
        if not os.path.exists(old_path) or not os.path.exists(new_path):
            lines.append("%-24s 存在性不一致：baseline=%s new=%s"
                         % (name, os.path.exists(old_path), os.path.exists(new_path)))
            continue
        old_md5, new_md5 = md5(old_path), md5(new_path)
        if old_md5 == new_md5:
            lines.append("%-24s IDENTICAL  md5=%s" % (name, old_md5[:12]))
            continue
        lines.append("%-24s DIFFERENT  baseline=%s new=%s"
                     % (name, old_md5[:12], new_md5[:12]))
        old_counts = chain_residue_counts(old_path)
        new_counts = chain_residue_counts(new_path)
        if old_counts != new_counts:
            lines.append("    链/残基数：baseline=%s" % sorted(old_counts.items()))
            lines.append("               new     =%s" % sorted(new_counts.items()))

    old_summary = os.path.join(baseline_dir, SUMMARY_NAME)
    new_summary = os.path.join(new_dir, SUMMARY_NAME)
    if os.path.exists(old_summary) and os.path.exists(new_summary):
        old_lines = summary_lines(old_summary)
        new_lines = summary_lines(new_summary)
        if old_lines == new_lines:
            lines.append("%-24s IDENTICAL（%d 行）" % (SUMMARY_NAME, len(old_lines)))
        else:
            lines.append("%-24s DIFFERENT" % SUMMARY_NAME)
            for text in old_lines:
                if text not in new_lines:
                    lines.append("    baseline only: %s" % text)
            for text in new_lines:
                if text not in old_lines:
                    lines.append("    new only     : %s" % text)
    else:
        lines.append("%-24s 存在性不一致" % SUMMARY_NAME)
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description="比较两次运行的装配结果")
    parser.add_argument("baseline_output_dir")
    parser.add_argument("new_output_dir")
    parser.add_argument("--report", default=None, help="可选：把结果写入 Markdown 文件")
    args = parser.parse_args(argv)

    lines = compare(final_results(args.baseline_output_dir),
                    final_results(args.new_output_dir))
    for line in lines:
        print(line)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            handle.write("# 运行对比\n\n")
            handle.write("- baseline: `%s`\n- new: `%s`\n\n" % (args.baseline_output_dir,
                                                                args.new_output_dir))
            handle.write("```text\n" + "\n".join(lines) + "\n```\n")
        print("report written: %s" % args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
