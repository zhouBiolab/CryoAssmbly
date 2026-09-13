"""T02 汇总：从微基准产物里算出冷/暖耗时、源侧可复用比例、几何重复率、scale 命中机会与显存峰值。

用法：
    python tools/summarize_registration.py <bench_out_dir> [--report out.md]

数据来源：
  - <bench_out>/t00_report.json          每个 (target, repeat) 的 wall / 预测数 / best overlap
  - <bench_out>/target_*/server_timing.jsonl   服务端阶段（含模型内 CUDA 事件阶段与显存峰值）
"""

import argparse
import glob
import json
import os

# 源侧可复用的"候选"阶段：几何/采样/分区/联合编码（backbone 同时编码两侧，T04/T06 才能拆分）
SOURCE_SIDE_STAGES = ("server_collate", "server_neighbors", "model_backbone")
GEOMETRY_STAGES = ("server_collate", "server_neighbors")


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def collect(out_dir):
    """返回 [(pair_name, repeat, rows)]，pair_name = target_<order>_run_<n>。"""
    pairs = []
    for timing_path in sorted(glob.glob(os.path.join(out_dir, "target_*", "server_timing.jsonl"))):
        pair_dir = os.path.dirname(timing_path)
        name = os.path.basename(pair_dir)
        repeat = int(name.rsplit("_", 1)[1])
        pairs.append((name, repeat, load_jsonl(timing_path)))
    return pairs


def aggregate(pairs, repeat):
    table = {}
    for _, pair_repeat, rows in pairs:
        if pair_repeat != repeat:
            continue
        for row in rows:
            entry = table.setdefault(row["stage"], {"count": 0, "seconds": 0.0})
            entry["count"] += 1
            entry["seconds"] += row["elapsed_s"]
    return table


def main():
    parser = argparse.ArgumentParser(description="T02 固定配准汇总")
    parser.add_argument("bench_dir")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    run_report = json.load(open(os.path.join(args.bench_dir, "t00_report.json"), encoding="utf-8"))
    pairs = collect(args.bench_dir)
    cold = aggregate(pairs, 1)
    warm = aggregate(pairs, 2) if any(repeat == 2 for _, repeat, _ in pairs) else None

    pools = sorted({json.dumps(row) for _, _, rows in pairs for row in rows
                    if row["stage"] == "server_pair_info"})
    pair_info = [json.loads(item) for item in pools]
    scales = {info["scale_bits"] for info in pair_info}
    total_pairs = len(pair_info)
    src_points = pair_info[0]["src_points"] if pair_info else None

    memory = {}
    for _, _, rows in pairs:
        for row in rows:
            if row["stage"] == "server_mem_peak":
                for key in ("allocated", "reserved", "max_allocated", "max_reserved"):
                    memory[key] = max(memory.get(key, 0), row.get(key, 0))

    source_side = sum(cold.get(stage, {}).get("seconds", 0.0) for stage in SOURCE_SIDE_STAGES)
    geometry = sum(cold.get(stage, {}).get("seconds", 0.0) for stage in GEOMETRY_STAGES)
    total_model = cold.get("model_backbone", {}).get("seconds", 0.0) + \
        cold.get("model_transformer", {}).get("seconds", 0.0) + \
        cold.get("model_lgr", {}).get("seconds", 0.0)
    pair_wall = sum(rec["wall_s"] for rec in run_report["records"] if rec["repeat"] == 1)

    lines = []
    lines.append("# T02：固定配准热点与源侧可复用比例")
    lines.append("")
    lines.append("输入：%d 个 (目标掩码 × 重复) 组合；源点云 %s 点；目标点数 %s"
                 % (total_pairs, src_points,
                    sorted(info["tgt_points"] for info in pair_info)))
    lines.append("")
    lines.append("## 一、服务端阶段（冷启动 = 第 1 次；单位秒，累计）")
    lines.append("")
    lines.append("| 阶段 | 冷 count | 冷 seconds | 暖 seconds |")
    lines.append("|---|---|---|---|")
    for stage, item in sorted(cold.items(), key=lambda kv: -kv[1]["seconds"]):
        warm_seconds = "%.2f" % warm[stage]["seconds"] if warm and stage in warm else "—"
        lines.append("| %s | %d | %.2f | %s |" % (stage, item["count"], item["seconds"], warm_seconds))
    lines.append("")
    lines.append("## 二、可复用比例（按冷启动口径）")
    lines.append("")
    lines.append("- 每对配准 wall（冷）合计：**%.2f s**" % pair_wall)
    lines.append("- 几何+分区（`server_collate` + `server_neighbors`）：%.2f s（%.1f%%）"
                 % (geometry, 100.0 * geometry / pair_wall if pair_wall else 0.0))
    lines.append("- 加上联合编码 `model_backbone`（源侧 + 目标侧一起算）：%.2f s（%.1f%%）"
                 % (source_side, 100.0 * source_side / pair_wall if pair_wall else 0.0))
    lines.append("- 模型三阶段合计（backbone+transformer+lgr）：%.2f s（%.1f%%）"
                 % (total_model, 100.0 * total_model / pair_wall if pair_wall else 0.0))
    lines.append("")
    lines.append("> 说明：`model_backbone` 同时编码两侧，**不能整体算作源侧可复用**；"
                 "只有 T04（单侧几何拆分）与 T06/T07（编码拆分与缓存）之后才能给出源侧真实占比。"
                 "当前数字是源侧可复用比例的**上界**。")
    lines.append("")
    lines.append("## 三、重复率与 scale 命中机会")
    lines.append("")
    lines.append("- 源点云在本次基准里被重建 **%d** 次（每个目标 × 每次重复各一次）→ 几何重复率 %.1f%%"
                 % (total_pairs, 100.0 * (1 - 1.0 / total_pairs) if total_pairs else 0.0))
    lines.append("- 精确 scale 的**不同取值**：%d 个（共 %d 对）→ scale 编码缓存的理论命中机会 %.1f%%"
                 % (len(scales), total_pairs,
                    100.0 * (1 - len(scales) / total_pairs) if total_pairs else 0.0))
    if scales:
        lines.append("- scale 位模式样本：%s" % ", ".join(sorted(scales)[:4]))
    lines.append("")
    lines.append("## 四、显存峰值（MIG 7g.80gb 切片内，进程级）")
    lines.append("")
    lines.append("| 指标 | 峰值 (MiB) |")
    lines.append("|---|---|")
    for key in ("allocated", "reserved", "max_allocated", "max_reserved"):
        lines.append("| %s | %.1f |" % (key, memory.get(key, 0) / 1048576.0))
    lines.append("")
    lines.append("## 五、有效参数与标签对照")
    lines.append("")
    configs = sorted({info["config_id"] for info in pair_info})
    samplings = sorted({info["sampling"] for info in pair_info})
    lines.append("- 实际执行的 config_id：%s（label `configs=all`）" % configs)
    lines.append("- 实际执行的采样方法：%s" % samplings)
    lines.append("- 每对共享同一 scale（见上）与同一源点云；目标点数不同：%s"
                 % sorted(info["tgt_points"] for info in pair_info))
    lines.append("")
    lines.append("## 六、结论")
    lines.append("")
    lines.append("1. 热点集中在 `model_backbone` / `model_transformer` / `model_lgr` 与 `server_postprocess`；")
    lines.append("2. 源几何（collate+neighbors）占比见上表，是**无条件可缓存**的部分；")
    lines.append("3. scale 命中机会由上面的不同取值数给出，T07 的收益上界即此；")
    lines.append("4. 冷/暖差异按同表两列对照（暖 = 同进程第 2 次），不要把冷暖差异当算子加速。")
    text = "\n".join(lines) + "\n"

    print(text)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            handle.write(text)
        print("report written: %s" % args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
