"""汇总一次运行的时间账（P2 细化）。

用法：
    python tools/summarize_timing.py <output_dir>

输出：
  - 客户端各阶段（fit_request 内部细分）
  - 服务端（PARENet 常驻进程）各阶段：写在每个请求输出目录的 server_timing.jsonl
  - 进程池生命周期（pool_start / pool_close，按调用点分组）
  - fit_request 的未归因时间 = fit_request 合计 - 已识别子阶段之和

注意：父子进程的计时区间会重叠（服务端阶段发生在客户端的 gpu_wait 窗口内），
pool_* 事件又包含在上层阶段里，因此**只做减法一次**、其余分列展示，不能相加。
"""

import argparse
import glob
import json
import os

CLIENT_SUBSTAGES = ("gpu_wait", "cc_batch", "cc_candidate_initial", "cc_verify",
                    "local_optimize", "final_select", "save_result",
                    "analyze_sources")
# 与上表**区间重叠**（candidate_stream 覆盖 gpu_wait/cc_batch/local_optimize/final_select），
# 只分列展示，不计入"已识别子阶段合计"，否则未归因会算成负数。
CLIENT_OVERLAP_STAGES = ("candidate_stream",)
SERVER_STAGES = ("server_queue_wait", "server_request_total", "server_preprocess",
                 "server_masks", "server_mask_preprocess", "server_to_gpu",
                 "server_forward", "server_postprocess", "server_write_pred")


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def totals(rows):
    result = {}
    for row in rows:
        entry = result.setdefault(row["stage"], {"count": 0, "seconds": 0.0})
        entry["count"] += 1
        entry["seconds"] += row["elapsed_s"]
    return result


def print_table(title, table, only=None):
    print("== %s ==" % title)
    items = [(stage, item) for stage, item in table.items()
             if only is None or stage in only]
    if not items:
        print("  (无)")
        return
    for stage, item in sorted(items, key=lambda kv: -kv[1]["seconds"]):
        print("  %-24s count=%-4d seconds=%9.2f" % (stage, item["count"], item["seconds"]))


def print_memory(server_rows):
    """服务端显存轨迹（T03）：按时间戳排序，检查 allocated 是否随请求数增长。"""
    rows = [row for row in server_rows
            if row["stage"] in ("server_mem_after", "server_mem_peak")]
    if not rows:
        print("== 服务端显存 ==\n  (无 server_mem_* 记录)")
        return
    rows.sort(key=lambda row: row.get("timestamp", 0.0))
    after = [row for row in rows if row["stage"] == "server_mem_after"] or rows
    allocated = [row.get("allocated", 0) / 1048576.0 for row in after]
    reserved = [row.get("reserved", 0) / 1048576.0 for row in after]
    peak_allocated = max(row.get("max_allocated", 0) for row in rows) / 1048576.0
    peak_reserved = max(row.get("max_reserved", 0) for row in rows) / 1048576.0
    print()
    print("== 服务端显存（按时间戳顺序；来源 %s，%d 条）=="
          % (after[0]["stage"], len(after)))
    print("  allocated 首/最小/最大/末  %8.1f /%8.1f /%8.1f /%8.1f MiB"
          % (allocated[0], min(allocated), max(allocated), allocated[-1]))
    print("  reserved  首/最小/最大/末  %8.1f /%8.1f /%8.1f /%8.1f MiB"
          % (reserved[0], min(reserved), max(reserved), reserved[-1]))
    print("  进程高水位 max_allocated  %8.1f MiB" % peak_allocated)
    print("  进程高水位 max_reserved   %8.1f MiB" % peak_reserved)
    print("  allocated 前 5 次：%s"
          % ", ".join("%.1f" % value for value in allocated[:5]))
    print("  allocated 末 5 次：%s"
          % ", ".join("%.1f" % value for value in allocated[-5:]))


def print_cache(server_rows):
    """几何缓存（T05）：累计统计 + 逐次配准命中。"""
    stats = [row for row in server_rows if row["stage"] == "server_cache_stats"]
    hits = [row for row in server_rows if row["stage"] == "server_cache_hit"]
    if not stats and not hits:
        print("== 几何缓存 ==\n  (无 server_cache_* 记录)")
        return
    print()
    print("== 几何缓存（T05）==")
    if hits:
        total = len(hits)
        source_hits = sum(1 for row in hits if row.get("src_hit"))
        target_hits = sum(1 for row in hits if row.get("tgt_hit"))
        print("  逐次配准 %d 次：源命中 %d（%.1f%%）、目标命中 %d（%.1f%%）"
              % (total, source_hits, 100.0 * source_hits / total,
                 target_hits, 100.0 * target_hits / total))
        print("  命中搬设备合计 %.2f s；未命中构建 %.2f s；写缓存 %.2f s"
              % (sum(row["elapsed_s"] for row in hits),
                 sum(row["elapsed_s"] for row in server_rows
                     if row["stage"] in ("server_collate", "server_neighbors")),
                 sum(row["elapsed_s"] for row in server_rows
                     if row["stage"] == "server_cache_store")))
    if stats:
        last = sorted(stats, key=lambda row: row.get("timestamp", 0.0))[-1]
        print("  累计（最后一次请求后）：capacity=%d MiB entries=%d bytes=%.2f MiB "
              "peak=%.2f MiB hits=%d misses=%d evictions=%d rejected_too_large=%d hit_rate=%s"
              % (last["capacity_bytes"] // (1024 * 1024), last.get("entries", 0),
                 last.get("bytes", 0) / 1048576.0, last.get("peak_bytes", 0) / 1048576.0,
                 last.get("hits", 0), last.get("misses", 0), last.get("evictions", 0),
                 last.get("rejected_too_large", 0), last.get("hit_rate")))


def main():
    parser = argparse.ArgumentParser(description="汇总一次运行的客户端/服务端时间账")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    out = args.output_dir

    client_rows = load_jsonl(os.path.join(out, "metrics", "performance.jsonl"))
    client = totals(client_rows)
    first_pred = [row["elapsed_s"] for row in client_rows if row["stage"] == "first_pred"]

    server_rows = []
    server_files = sorted(glob.glob(os.path.join(out, "**", "server_timing.jsonl"),
                                    recursive=True))
    for path in server_files:
        server_rows.extend(load_jsonl(path))

    print_table("客户端阶段（全部）", client)
    print()
    print_table("客户端：fit_request 的细分", client, only=CLIENT_SUBSTAGES)
    print()
    print("first_pred 延迟（相对监测循环开始，非累加）: %s"
          % ", ".join("%.1f s" % value for value in first_pred) or "  (无)")
    print()
    print_table("服务端阶段（%d 个 server_timing.jsonl）" % len(server_files),
                totals(server_rows), only=SERVER_STAGES)
    print()
    print_table("进程池生命周期（包含在上层阶段内，不重复计入）",
                totals(client_rows), only=("pool_start", "pool_close"))
    print_memory(server_rows)
    print_cache(server_rows)
    print()

    print_table("客户端：候选流（覆盖上表多个阶段，不重复计入）",
                totals(client_rows), only=CLIENT_OVERLAP_STAGES)
    print()
    fit = client.get("fit_request", {}).get("seconds", 0.0)
    identified = sum(client.get(stage, {}).get("seconds", 0.0)
                     for stage in CLIENT_SUBSTAGES)
    print("== 时间账 ==")
    print("  fit_request 合计        %9.2f s" % fit)
    print("  已识别子阶段合计        %9.2f s" % identified)
    print("  未归因                  %9.2f s" % (fit - identified))
    wall = client.get("pipeline_total", {}).get("seconds", 0.0)
    print("  pipeline_total（墙钟）  %9.2f s" % wall)
    pool_created = client.get("pool_start", {}).get("count", 0)
    print("  池创建次数              %d" % pool_created)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
