"""固定配准微基准 A/B 对比（任务卡 T03，可复用于 T10 的逐项比较）。

用法：
    python tools/compare_registration_runs.py --base <bench_out_a> --new <bench_out_b> \
        [--report out.md]

对比内容：
  1) 结果一致性：每对预测数、overlap 列表、pred_*.pdb 内容哈希；
  2) 每对 wall（t00_report.json）与汇总；
  3) 服务端阶段累计耗时（冷 = 第 1 次重放，暖 = 第 2 次重放），只在一侧出现的阶段单独标注；
  4) 显存轨迹（server_mem_after / server_mem_peak 的 allocated 与 reserved）。

两侧必须来自同一 manifest；工具只读产物，不重跑推理。
"""

import argparse
import glob
import hashlib
import json
import os


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_run(bench_dir):
    report = json.load(open(os.path.join(bench_dir, "t00_report.json"), encoding="utf-8"))
    pairs = {}
    for timing_path in sorted(glob.glob(os.path.join(bench_dir, "target_*",
                                                      "server_timing.jsonl"))):
        pair_dir = os.path.dirname(timing_path)
        name = os.path.basename(pair_dir)
        rows = load_jsonl(timing_path)
        predictions = {}
        for path in sorted(glob.glob(os.path.join(pair_dir, "pred_*.pdb"))):
            predictions[os.path.basename(path)] = sha256(path)
        pairs[name] = {"rows": rows, "predictions": predictions}
    return {"dir": os.path.abspath(bench_dir), "report": report, "pairs": pairs}


def stage_totals(pairs, repeat):
    """按重放次序统计阶段累计耗时；重放次数 = 目录名 target_<order>_run_<repeat>。"""
    totals = {}
    for name, pair in pairs.items():
        if int(name.rsplit("_", 1)[1]) != repeat:
            continue
        for row in pair["rows"]:
            entry = totals.setdefault(row["stage"], {"count": 0, "seconds": 0.0})
            entry["count"] += 1
            entry["seconds"] += row["elapsed_s"]
    return totals


def memory_trace(pairs, stages=("server_mem_after", "server_mem_peak")):
    """返回按 (target, repeat) 排序的显存轨迹：[(name, allocated, reserved, source)]。"""
    trace = []
    for name in sorted(pairs, key=pair_sort_key):
        allocated = reserved = None
        source = "—"
        for row in pairs[name]["rows"]:
            if row["stage"] in stages:
                allocated = row.get("allocated", allocated)
                reserved = row.get("reserved", reserved)
                source = row["stage"].replace("server_mem_", "")
        if allocated is not None:
            trace.append((name, allocated, reserved, source))
    return trace


def peak_memory(pairs):
    """前向结束时的峰值（server_mem_peak；取所有对的最大值）。"""
    peak = {}
    for pair in pairs.values():
        for row in pair["rows"]:
            if row["stage"] != "server_mem_peak":
                continue
            for key in ("allocated", "reserved", "max_allocated", "max_reserved"):
                if key in row:
                    peak[key] = max(peak.get(key, 0), row[key])
    return peak


def after_sequence(pairs):
    """逐次配准结束时的 allocated 序列（按写入顺序），用于检查是否持续增长。"""
    values = []
    for name in sorted(pairs, key=pair_sort_key):
        for row in pairs[name]["rows"]:
            if row["stage"] == "server_mem_after" and "allocated" in row:
                values.append(row["allocated"] / 1048576.0)
    return values


def pair_sort_key(name):
    parts = name.split("_")
    return (int(parts[1]), int(parts[3]))


def record_key(record):
    return (record["target_order"], record["repeat"])


def wall_table(base_run, new_run, lines):
    base_records = {record_key(item): item for item in base_run["report"]["records"]}
    new_records = {record_key(item): item for item in new_run["report"]["records"]}
    lines.append("## 二、每对 wall（t00_report.json，秒）")
    lines.append("")
    lines.append("| target | repeat | base | new | delta | delta% |")
    lines.append("|---|---|---|---|---|---|")
    base_total = new_total = 0.0
    for key in sorted(set(base_records) | set(new_records)):
        base_wall = base_records.get(key, {}).get("wall_s")
        new_wall = new_records.get(key, {}).get("wall_s")
        if base_wall is None or new_wall is None:
            lines.append("| %s | %s | %s | %s | — | — |"
                         % (key[0], key[1], base_wall, new_wall))
            continue
        base_total += base_wall
        new_total += new_wall
        delta = new_wall - base_wall
        lines.append("| %s | %s | %.2f | %.2f | %+.2f | %+.1f%% |"
                     % (key[0], key[1], base_wall, new_wall, delta,
                        100.0 * delta / base_wall if base_wall else 0.0))
    lines.append("")
    if base_total:
        lines.append("- 全部重放合计：base %.2f s → new %.2f s（%+.2f s，%+.1f%%）"
                     % (base_total, new_total, new_total - base_total,
                        100.0 * (new_total - base_total) / base_total))
    lines.append("")
    return base_total, new_total


def identity_section(base_run, new_run, lines):
    lines.append("## 一、结果一致性")
    lines.append("")
    mismatched_counts = []
    mismatched_overlaps = []
    file_same = 0
    name_mismatch = []
    for key in sorted(set(base_run["pairs"]) | set(new_run["pairs"])):
        base_pair = base_run["pairs"].get(key)
        new_pair = new_run["pairs"].get(key)
        if base_pair is None or new_pair is None:
            name_mismatch.append(key)
            continue
        base_pred = base_pair["predictions"]
        new_pred = new_pair["predictions"]
        if len(base_pred) != len(new_pred):
            mismatched_counts.append((key, len(base_pred), len(new_pred)))
        if sorted(base_pred) != sorted(new_pred):
            name_mismatch.append(key)
        file_same += sum(1 for name, digest in base_pred.items()
                         if new_pred.get(name) == digest)
    base_overlaps = {record_key(item): sorted(item["overlaps"])
                     for item in base_run["report"]["records"]}
    new_overlaps = {record_key(item): sorted(item["overlaps"])
                    for item in new_run["report"]["records"]}
    for key in sorted(set(base_overlaps) | set(new_overlaps)):
        if base_overlaps.get(key) != new_overlaps.get(key):
            mismatched_overlaps.append(key)
    lines.append("- 预测文件数：base %d 个 / new %d 个；内容哈希一致 %d 个"
                 % (sum(len(pair["predictions"]) for pair in base_run["pairs"].values()),
                    sum(len(pair["predictions"]) for pair in new_run["pairs"].values()),
                    file_same))
    lines.append("- 逐对预测数不一致：%s" % (mismatched_counts or "无"))
    lines.append("- 预测文件名集合不一致：%s" % (name_mismatch or "无"))
    lines.append("- 逐对 overlap 列表不一致：%s" % (mismatched_overlaps or "无"))
    lines.append("")
    return not mismatched_counts and not name_mismatch and not mismatched_overlaps


def stage_section(base_run, new_run, lines):
    base_cold = stage_totals(base_run["pairs"], 1)
    new_cold = stage_totals(new_run["pairs"], 1)
    base_warm = stage_totals(base_run["pairs"], 2)
    new_warm = stage_totals(new_run["pairs"], 2)
    lines.append("## 三、服务端阶段累计（冷 = 第 1 次重放，暖 = 第 2 次重放；秒）")
    lines.append("")
    lines.append("| 阶段 | base 冷 | new 冷 | Δ冷 | base 暖 | new 暖 | Δ暖 |")
    lines.append("|---|---|---|---|---|---|---|")
    stages = sorted(set(base_cold) | set(new_cold) | set(base_warm) | set(new_warm),
                    key=lambda name: -(new_cold.get(name, {}).get("seconds", 0.0)
                                       + base_cold.get(name, {}).get("seconds", 0.0)))

    def cell(table, stage):
        return "%.2f" % table[stage]["seconds"] if stage in table else "—"

    def delta(base_table, new_table, stage):
        if stage in base_table and stage in new_table:
            return "%+.2f" % (new_table[stage]["seconds"] - base_table[stage]["seconds"])
        return "—"

    for stage in stages:
        lines.append("| %s | %s | %s | %s | %s | %s | %s |"
                     % (stage, cell(base_cold, stage), cell(new_cold, stage),
                        delta(base_cold, new_cold, stage),
                        cell(base_warm, stage), cell(new_warm, stage),
                        delta(base_warm, new_warm, stage)))
    lines.append("")
    lines.append("> 只在一侧出现的阶段（探针或本卡新增/移除的埋点）：%s"
                 % (sorted((set(base_cold) | set(base_warm))
                           ^ (set(new_cold) | set(new_warm))) or "无"))
    lines.append("> `server_forward` 是模型调用的主机侧 wall，包含模型内 CUDA 事件阶段，两者不可相加。")
    lines.append("")
    return base_cold, new_cold, base_warm, new_warm


def memory_section(base_run, new_run, lines):
    base_trace = {name: (allocated, reserved, source)
                  for name, allocated, reserved, source in memory_trace(base_run["pairs"])}
    new_trace = {name: (allocated, reserved, source)
                 for name, allocated, reserved, source in memory_trace(new_run["pairs"])}
    lines.append("## 四、显存轨迹（MiB；`server_mem_after` = 每对结束，旧产物可能只有 `server_mem_peak` = 前向结束）")
    lines.append("")
    lines.append("| 序号 | 对 | base allocated | new allocated | base reserved | new reserved |")
    lines.append("|---|---|---|---|---|---|")
    for index, name in enumerate(sorted(set(base_trace) | set(new_trace), key=pair_sort_key)):
        base_item = base_trace.get(name)
        new_item = new_trace.get(name)
        lines.append("| %d | %s | %s | %s | %s | %s |"
                     % (index, name,
                        "%.1f" % (base_item[0] / 1048576.0) if base_item else "—",
                        "%.1f" % (new_item[0] / 1048576.0) if new_item else "—",
                        "%.1f" % (base_item[1] / 1048576.0) if base_item else "—",
                        "%.1f" % (new_item[1] / 1048576.0) if new_item else "—"))
    lines.append("")
    for label, trace in (("base", base_trace), ("new", new_trace)):
        if not trace:
            continue
        values = [item[0] for item in trace.values()]
        sources = sorted({item[2] for item in trace.values()})
        lines.append("- %s：allocated 首/最小/最大/末 = %.1f / %.1f / %.1f / %.1f MiB（来源：%s）"
                     % (label, values[0] / 1048576.0, min(values) / 1048576.0,
                        max(values) / 1048576.0, values[-1] / 1048576.0, ", ".join(sources)))
    lines.append("")
    base_peak = peak_memory(base_run["pairs"])
    new_peak = peak_memory(new_run["pairs"])
    lines.append("### 前向峰值（`server_mem_peak`，MiB）")
    lines.append("")
    lines.append("| 指标 | base | new | delta |")
    lines.append("|---|---|---|---|")
    for key in ("allocated", "reserved", "max_allocated", "max_reserved"):
        base_value = base_peak.get(key, 0) / 1048576.0
        new_value = new_peak.get(key, 0) / 1048576.0
        lines.append("| %s | %.1f | %.1f | %+.1f |" % (key, base_value, new_value,
                                                      new_value - base_value))
    lines.append("")
    lines.append("### 逐次配准结束时的 allocated（检查是否持续增长）")
    lines.append("")
    for label, pairs in (("base", base_run["pairs"]), ("new", new_run["pairs"])):
        values = after_sequence(pairs)
        if not values:
            lines.append("- %s：无 `server_mem_after` 记录" % label)
            continue
        head = ", ".join("%.1f" % value for value in values[:6])
        tail = ", ".join("%.1f" % value for value in values[-6:])
        lines.append("- %s：共 %d 次配准；首/最小/最大/末 = %.1f / %.1f / %.1f / %.1f MiB；"
                     "末/首 = %.2f" % (label, len(values), values[0], min(values),
                                       max(values), values[-1], values[-1] / values[0]))
        lines.append("  - 前 6 次：%s" % head)
        lines.append("  - 末 6 次：%s" % tail)
    lines.append("")
    return base_trace, new_trace


def main(argv=None):
    parser = argparse.ArgumentParser(description="固定配准微基准 A/B 对比（T03/T10）")
    parser.add_argument("--base", required=True, help="基线产物目录")
    parser.add_argument("--new", required=True, help="候选产物目录")
    parser.add_argument("--report", default=None, help="可选：把结果写成 markdown")
    args = parser.parse_args(argv)

    base_run = load_run(args.base)
    new_run = load_run(args.new)

    lines = []
    lines.append("# 固定配准微基准 A/B 对比")
    lines.append("")
    lines.append("- base：`%s`" % base_run["dir"])
    lines.append("- new：`%s`" % new_run["dir"])
    base_manifest = base_run["report"]["manifest"]
    new_manifest = new_run["report"]["manifest"]
    base_digest = sha256(base_manifest) if os.path.exists(base_manifest) else "缺失"
    new_digest = sha256(new_manifest) if os.path.exists(new_manifest) else "缺失"
    lines.append("- manifest sha256：base `%s`（`%s`）" % (base_digest[:16], base_manifest))
    lines.append("- manifest sha256：new `%s`（`%s`）" % (new_digest[:16], new_manifest))
    if base_digest != new_digest:
        lines.append("")
        lines.append("> ⚠️ 两侧 manifest 内容不同（sha256 不一致），对比结论不成立。")
    lines.append("")
    identity_ok = identity_section(base_run, new_run, lines)
    wall_table(base_run, new_run, lines)
    stage_section(base_run, new_run, lines)
    memory_section(base_run, new_run, lines)
    lines.append("## 五、判定")
    lines.append("")
    lines.append("- 结果一致性：%s" % ("通过（预测数/文件名/overlap 全部一致）" if identity_ok
                                      else "**不通过**，见第一节"))
    text = "\n".join(lines) + "\n"

    print(text)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            handle.write(text)
        print("report written: %s" % args.report)
    return 0 if identity_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
