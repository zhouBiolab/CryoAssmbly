"""P4 评分缓存探针：同一条评分路径在缓存关/开下的读取次数、命中率、内存与耗时。

用法：
    python tools/score_cache_probe.py --mrc <map.mrc> --structure <chain.pdb> \
        --resolution 5.6 --contour 0.04 --calls 20 [--budget-mb 128]

输出（stdout 为 JSON，便于归档）：
    cache_off / cache_on 两组：耗时、每次调用的读取次数（= density miss 数）、
    命中率、缓存占用与峰值字节、最终 CC（两组必须逐位相同）。
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protassem.core import scoring  # noqa: E402
from protassem.core.scoring import calculate_cc_mask, score_cache_snapshot  # noqa: E402


def _measure(mrc, structure, resolution, contour, calls, budget_mb):
    scoring.configure_score_cache(budget_mb)
    per_call = []
    values = []
    for _ in range(calls):
        started = time.perf_counter()
        values.append(calculate_cc_mask(mrc, structure, resolution, contour))
        per_call.append(time.perf_counter() - started)
    elapsed = sum(per_call)
    rest = sorted(per_call[1:]) if len(per_call) > 1 else [0.0]
    snapshot = score_cache_snapshot()
    density = snapshot["density"]
    return {
        "budget_mb": budget_mb,
        "calls": calls,
        "elapsed_s": round(elapsed, 6),
        "ms_per_call": round(1000.0 * elapsed / calls, 4),
        "first_call_s": round(per_call[0], 6),
        "median_after_first_ms": round(1000.0 * rest[len(rest) // 2], 4),
        "density_reads": density["misses"],
        "hits": density["hits"],
        "hit_rate": density["hit_rate"],
        "entries": density["entries"],
        "bytes": density["bytes"],
        "peak_bytes": density["peak_bytes"],
        "structure_reads": snapshot["structure"]["misses"],
        "cc_first": values[0],
        "cc_all_equal": len(set(values)) == 1,
    }


def main():
    parser = argparse.ArgumentParser(description="P4 评分缓存探针")
    parser.add_argument("--mrc", required=True)
    parser.add_argument("--structure", required=True)
    parser.add_argument("--resolution", type=float, required=True)
    parser.add_argument("--contour", type=float, required=True)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--budget-mb", type=int, default=scoring.DEFAULT_SCORE_CACHE_MB)
    args = parser.parse_args()

    off = _measure(args.mrc, args.structure, args.resolution, args.contour,
                   args.calls, 0)
    on = _measure(args.mrc, args.structure, args.resolution, args.contour,
                  args.calls, args.budget_mb)
    result = {"cache_off": off, "cache_on": on,
              "same_result": off["cc_first"] == on["cc_first"]}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
