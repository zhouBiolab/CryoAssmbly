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
                    "local_optimize", "candidate_scan", "final_select", "save_result",
                    "analyze_sources")
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
