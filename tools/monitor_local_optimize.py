#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""局部优化日志分析器（离线；只读日志，不给运行加任何开销）。

为什么是离线脚本
----------------
局部优化的耗时与 CC 提升此前依赖生产代码里的埋点（performance.jsonl /
server_timing.jsonl）。那套埋点已按用户要求整体删除；本脚本改为**解析
pipeline 日志**，因此不碰任何生产代码、不增加运行开销。

被解析的日志契约
----------------
链拟合是串行的、`local_optimize` 在父进程内执行，所以同一时刻只有一个调用，
日志可按 start/done 严格配对：

    local_optimize start: cc=0.3473
      copy step=1.2 density=0.0286 cc=0.4184     <- 6 条，6 个并行副本全部结束后一次性打印
      copy step=3.0 ...                         <- 共 6 条，步长固定 1.2/3.0/3.5/4.5/5.5/6.0
    fine: 0.4184 -> 0.4185
    local_optimize done: 0.3473 -> 0.4185 (+0.0712)

行格式 `%(asctime)s %(levelname)s [%(name)s] %(message)s`，
时间戳 `%Y-%m-%d %H:%M:%S,mmm`（毫秒用逗号）。

阶段划分
--------
    多副本阶段 : start           -> 最后一条 copy step
    scipy 回退 : 最后一条 copy   -> scipy cc=（仅旧实现、且条件触发时才有）
    精修阶段   : 上一段末        -> fine:
    收尾       : fine:           -> done（选择/回退/copy2/清理临时文件）

用法
----
    # 每份日志的汇总
    python tools/monitor_local_optimize.py <log> [<log> ...]

    # 附每次调用的明细
    python tools/monitor_local_optimize.py --verbose <log>

    # 多份横向对比；日志数为偶数时，前半 = A 组、后半 = B 组，另给配对差值
    python tools/monitor_local_optimize.py --compare <logA1> <logA2> <logA3> \
                                                       <logB1> <logB2> <logB3>

    # 导出机器可读结果
    python tools/monitor_local_optimize.py --json out.json <log> ...

退出码：0 = 正常；1 = 日志里存在未闭合的 start（说明运行被中断）。
"""

import argparse
import collections
import datetime
import json
import os
import re
import statistics
import sys

TS_FMT = "%Y-%m-%d %H:%M:%S,%f"

LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s+"
    r"(?P<lvl>[A-Z]+)\s+\[(?P<logger>[^\]]+)\]\s(?P<msg>.*)$")

START_RE = re.compile(r"^local_optimize start: cc=(?P<cc>-?[\d.]+(?:[eE][-+]?\d+)?)$")
DONE_RE = re.compile(
    r"^local_optimize done: (?P<before>-?[\d.]+(?:[eE][-+]?\d+)?) -> "
    r"(?P<after>-?[\d.]+(?:[eE][-+]?\d+)?) "
    r"\((?P<delta>[+-][\d.]+(?:[eE][-+]?\d+)?)\)$")
COPY_RE = re.compile(
    r"^copy step=(?P<step>[\d.]+) density=(?P<density>-?[\d.]+(?:[eE][-+]?\d+)?) "
    r"cc=(?P<cc>-?[\d.]+(?:[eE][-+]?\d+)?)$")
FINE_RE = re.compile(
    r"^fine: (?P<before>-?[\d.]+(?:[eE][-+]?\d+)?) -> "
    r"(?P<after>-?[\d.]+(?:[eE][-+]?\d+)?)$")
SCIPY_HEAD_RE = re.compile(r"^density best CC .+ -> scipy CC opt$")
SCIPY_CC_RE = re.compile(r"^scipy cc=(?P<cc>-?[\d.]+(?:[eE][-+]?\d+)?)$")
FAILED_RE = re.compile(r"^local_optimize failed: (?P<err>.*)$")
COPY_FAIL_RE = re.compile(r"^density copy \(step=(?P<step>[\d.]+)\) failed: (?P<err>.*)$")
CAND_RE = re.compile(
    r"^Candidate #(?P<id>\d+): local optimization on (?P<path>\S+) "
    r"\(cc=(?P<cc>-?[\d.]+(?:[eE][-+]?\d+)?)\)$")

N_COPIES_EXPECTED = 6


def _parse_ts(text):
    return datetime.datetime.strptime(text, TS_FMT)


def _secs(a, b):
    """b - a 的秒数。"""
    return (b - a).total_seconds()


class Call:
    """一次 local_optimize 调用的完整记录。"""

    __slots__ = ("index", "candidate", "t_start", "t_copy_end", "t_scipy_end",
                 "t_fine", "t_done", "cc_before", "cc_after", "copies",
                 "scipy_triggered", "scipy_cc", "failed", "error", "copy_failures")

    def __init__(self, index, t_start, cc_before):
        self.index = index
        self.candidate = None
        self.t_start = t_start
        self.t_copy_end = None
        self.t_scipy_end = None
        self.t_fine = None
        self.t_done = None
        self.cc_before = cc_before
        self.cc_after = None
        self.copies = []              # [(step, cc), ...]
        self.scipy_triggered = False
        self.scipy_cc = None
        self.failed = False
        self.error = None
        self.copy_failures = []

    # ---- 阶段计时（秒）----
    @property
    def total(self):
        if self.t_done is None:
            return None
        return _secs(self.t_start, self.t_done)

    @property
    def copies_s(self):
        return _secs(self.t_start, self.t_copy_end) if self.t_copy_end else None

    @property
    def scipy_s(self):
        if not self.t_scipy_end:
            return None
        base = self.t_copy_end or self.t_start
        return _secs(base, self.t_scipy_end)

    @property
    def fine_s(self):
        if not self.t_fine:
            return None
        base = self.t_scipy_end or self.t_copy_end or self.t_start
        return _secs(base, self.t_fine)

    @property
    def tail_s(self):
        if not (self.t_fine and self.t_done):
            return None
        return _secs(self.t_fine, self.t_done)

    @property
    def delta_cc(self):
        if self.cc_before is None or self.cc_after is None:
            return None
        return self.cc_after - self.cc_before

    @property
    def best_copy_cc(self):
        if not self.copies:
            return None
        return max(cc for _step, cc in self.copies)

    def as_dict(self):
        return {
            "index": self.index,
            "candidate": self.candidate,
            "cc_before": self.cc_before,
            "cc_after": self.cc_after,
            "delta_cc": self.delta_cc,
            "total_s": self.total,
            "copies_s": self.copies_s,
            "scipy_s": self.scipy_s,
            "fine_s": self.fine_s,
            "tail_s": self.tail_s,
            "n_copies": len(self.copies),
            "copies": [{"step": s, "cc": c} for s, c in self.copies],
            "best_copy_cc": self.best_copy_cc,
            "scipy_triggered": self.scipy_triggered,
            "scipy_cc": self.scipy_cc,
            "failed": self.failed,
            "error": self.error,
            "copy_failures": list(self.copy_failures),
        }


class LogReport:
    """一份日志的解析结果。"""

    def __init__(self, path):
        self.path = path
        self.calls = []
        self.overlaps = 0             # 检测到的交叠（start 未闭合又来一个 start）
        self.unclosed = 0             # 结束时仍未闭合的 start
        self.copy_failures = 0
        self.log_start = None
        self.log_end = None
        self.n_lines = 0

    # ---- 汇总 ----
    def summary(self):
        done = [c for c in self.calls if c.t_done is not None and c.total is not None]
        totals = [c.total for c in done]
        deltas = [c.delta_cc for c in done if c.delta_cc is not None]
        out = {
            "log": self.path,
            "n_calls": len(self.calls),
            "n_completed": len(done),
            "n_failed": sum(1 for c in self.calls if c.failed),
            "overlaps": self.overlaps,
            "unclosed": self.unclosed,
            "copy_failures": self.copy_failures,
        }
        if totals:
            out.update({
                "total_s": sum(totals),
                "mean_s": statistics.mean(totals),
                "median_s": statistics.median(totals),
                "p90_s": _percentile(totals, 90),
                "max_s": max(totals),
                "min_s": min(totals),
                "copies_s": _sum_opt(c.copies_s for c in done),
                "scipy_s": _sum_opt(c.scipy_s for c in done),
                "fine_s": _sum_opt(c.fine_s for c in done),
                "tail_s": _sum_opt(c.tail_s for c in done),
                "scipy_calls": sum(1 for c in done if c.scipy_triggered),
                "copy_count_total": sum(len(c.copies) for c in done),
            })
            if deltas:
                out.update({
                    "delta_cc_sum": sum(deltas),
                    "delta_cc_mean": statistics.mean(deltas),
                    "delta_cc_median": statistics.median(deltas),
                    "improved": sum(1 for d in deltas if d > 1e-9),
                    "unchanged": sum(1 for d in deltas if abs(d) <= 1e-9),
                    "degraded": sum(1 for d in deltas if d < -1e-9),
                })
        if self.log_start and self.log_end:
            span = _secs(self.log_start, self.log_end)
            out["pipeline_span_s"] = span
            if totals and span > 0:
                out["share_of_pipeline"] = sum(totals) / span
        return out


def _percentile(values, pct):
    values = sorted(values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def _sum_opt(values):
    values = [v for v in values if v is not None]
    return sum(values) if values else None


def parse_log(path):
    """解析一份 pipeline 日志，返回 LogReport。"""
    report = LogReport(path)
    stack = []                    # 未闭合的 Call
    pending_candidate = None      # 最近一条 "Candidate #N: local optimization on ..."
    with open(path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            report.n_lines += 1
            line = raw.rstrip("\n")
            m = LINE_RE.match(line)
            if not m:
                continue
            ts = _parse_ts(m.group("ts"))
            if report.log_start is None:
                report.log_start = ts
            report.log_end = ts
            msg = m.group("msg").strip()
            logger = m.group("logger")

            if "local_optimizer" not in logger and "fitting.pipeline" not in logger:
                continue

            m2 = CAND_RE.match(msg)
            if m2:
                pending_candidate = "%s (cc=%s)" % (
                    os.path.basename(m2.group("path")), m2.group("cc"))
                continue

            m2 = COPY_FAIL_RE.match(msg)
            if m2:
                report.copy_failures += 1
                if stack:
                    stack[-1].copy_failures.append(
                        {"step": float(m2.group("step")), "error": m2.group("err")})
                continue

            m2 = START_RE.match(msg)
            if m2:
                if stack:
                    report.overlaps += 1
                call = Call(len(report.calls) + 1, ts, float(m2.group("cc")))
                call.candidate = pending_candidate
                pending_candidate = None
                report.calls.append(call)
                stack.append(call)
                continue

            m2 = COPY_RE.match(msg)
            if m2:
                target = stack[-1] if stack else (report.calls[-1] if report.calls else None)
                if target is not None:
                    target.copies.append((float(m2.group("step")), float(m2.group("cc"))))
                    target.t_copy_end = ts
                continue

            m2 = SCIPY_HEAD_RE.match(msg)
            if m2:
                if stack:
                    stack[-1].scipy_triggered = True
                    stack[-1].t_copy_end = stack[-1].t_copy_end or ts
                continue

            m2 = SCIPY_CC_RE.match(msg)
            if m2:
                if stack:
                    stack[-1].scipy_triggered = True
                    stack[-1].scipy_cc = float(m2.group("cc"))
                    stack[-1].t_scipy_end = ts
                continue

            m2 = FINE_RE.match(msg)
            if m2:
                if stack:
                    stack[-1].t_fine = ts
                continue

            m2 = FAILED_RE.match(msg)
            if m2:
                if stack:
                    call = stack.pop()
                    call.failed = True
                    call.error = m2.group("err")
                    call.t_done = ts
                continue

            m2 = DONE_RE.match(msg)
            if m2:
                if stack:
                    call = stack.pop()
                    call.t_done = ts
                    call.cc_after = float(m2.group("after"))
                continue

    report.unclosed = len(stack)
    for call in stack:
        call.failed = True
        call.error = call.error or "未闭合（运行被中断？）"
    return report


# ===========================================================================
# 输出
# ===========================================================================
def _fmt(value, spec="%.3f"):
    if value is None:
        return "-"
    return spec % value


def print_summary(summary, verbose_calls=None):
    path = summary["log"]
    print("=" * 78)
    print("日志: %s" % path)
    print("=" * 78)
    if summary["n_calls"] == 0:
        print()
        print("  未发现任何 local_optimize 调用。")
        print("  （若期望有调用，检查：接受阈值是否让链被早早接受、")
        print("    或 LOCAL_OPT_CC_FLOOR 是否把候选全滤掉）")
        return
    print()
    print("调用次数        : %d   （完成 %d，失败 %d）"
          % (summary["n_calls"], summary["n_completed"], summary["n_failed"]))
    if summary.get("total_s") is not None:
        print("局部优化墙钟合计: %.3f s" % summary["total_s"])
        print("  单次 均值/中位/p90/最长/最短: %.3f / %.3f / %.3f / %.3f / %.3f s"
              % (summary["mean_s"], summary["median_s"], summary["p90_s"],
                 summary["max_s"], summary["min_s"]))
        print()
        print("阶段分解（合计秒 / 占局部优化总时长的比例）")
        total = summary["total_s"]
        for label, key in (("多副本", "copies_s"), ("scipy 回退", "scipy_s"),
                           ("精修", "fine_s"), ("收尾", "tail_s")):
            val = summary.get(key)
            if val is None:
                print("  %-12s : -" % label)
            else:
                print("  %-12s : %10.3f s   %5.1f%%" % (label, val, 100.0 * val / total))
        print()
        print("副本行总数      : %d   （期望 %d 次调用 × %d = %d）"
              % (summary["copy_count_total"], summary["n_completed"],
                 N_COPIES_EXPECTED, summary["n_completed"] * N_COPIES_EXPECTED))
        print("scipy 回退触发  : %d 次" % summary["scipy_calls"])
        if summary.get("delta_cc_sum") is not None:
            print()
            print("CC 提升")
            print("  合计 %.6f   均值 %+.6f   中位 %+.6f"
                  % (summary["delta_cc_sum"], summary["delta_cc_mean"],
                     summary["delta_cc_median"]))
            print("  改善 %d / 持平 %d / **变差 %d**"
                  % (summary["improved"], summary["unchanged"], summary["degraded"]))
    if summary.get("pipeline_span_s"):
        print()
        print("整个 pipeline 墙钟: %.1f s" % summary["pipeline_span_s"])
        if summary.get("share_of_pipeline") is not None:
            print("局部优化占比      : %.1f%%" % (100.0 * summary["share_of_pipeline"]))
    print()
    flags = []
    if summary["overlaps"]:
        flags.append("检测到 %d 次调用交叠（可能开了 --homo-chain-refine 并行分支）"
                     % summary["overlaps"])
    if summary["unclosed"]:
        flags.append("%d 次调用未闭合（运行被中断？）" % summary["unclosed"])
    if summary["copy_failures"]:
        flags.append("%d 个密度副本失败" % summary["copy_failures"])
    if summary["n_failed"]:
        flags.append("%d 次调用失败" % summary["n_failed"])
    if flags:
        print("异常:")
        for f in flags:
            print("  - %s" % f)
    else:
        print("异常: 无")

    if verbose_calls:
        print()
        print("-" * 78)
        print("每次调用明细")
        print("-" * 78)
        print("  %-4s %-38s %-8s %-8s %-9s %-8s %-8s %-8s"
              % ("#", "候选", "cc前", "cc后", "Δcc", "总(s)", "副本(s)", "精修(s)"))
        for call in verbose_calls:
            print("  %-4d %-38s %-8.4f %-8s %-+9.4f %-8s %-8s %-8s"
                  % (call.index,
                     (call.candidate or "-")[:38],
                     call.cc_before if call.cc_before is not None else float("nan"),
                     _fmt(call.cc_after, "%.4f"),
                     call.delta_cc if call.delta_cc is not None else float("nan"),
                     _fmt(call.total), _fmt(call.copies_s), _fmt(call.fine_s)))


COMPARE_COLS = [
    ("n_calls", "调用次数", "%d"),
    ("total_s", "总时长(s)", "%.1f"),
    ("mean_s", "均值(s)", "%.3f"),
    ("median_s", "中位(s)", "%.3f"),
    ("p90_s", "p90(s)", "%.3f"),
    ("max_s", "最长(s)", "%.3f"),
    ("copies_s", "副本(s)", "%.1f"),
    ("scipy_s", "scipy(s)", "%.1f"),
    ("fine_s", "精修(s)", "%.1f"),
    ("scipy_calls", "scipy次数", "%d"),
    ("delta_cc_sum", "Δcc合计", "%+.4f"),
    ("delta_cc_mean", "Δcc均值", "%+.5f"),
    ("degraded", "变差次数", "%d"),
]


def print_compare(summaries, pair=False):
    print("=" * 78)
    print("横向对比（%d 份日志）" % len(summaries))
    print("=" * 78)
    names = [os.path.basename(os.path.dirname(s["log"])) or os.path.basename(s["log"])
             for s in summaries]
    head = "%-12s" % "指标" + "".join("%-14s" % n[:13] for n in names)
    print()
    print(head)
    print("-" * len(head))
    for key, label, spec in COMPARE_COLS:
        row = "%-12s" % label
        for s in summaries:
            val = s.get(key)
            row += "%-14s" % ("-" if val is None else spec % val)
        print(row)

    if pair and len(summaries) % 2 == 0:
        half = len(summaries) // 2
        group_a, group_b = summaries[:half], summaries[half:]
        print()
        print("-" * 78)
        print("配对差值（A 组第 i 份 vs B 组第 i 份）")
        print("-" * 78)
        print("%-8s %-16s %-16s %-12s" % ("对", "Δ总时长(s)", "ΔΔcc合计", "Δ均值(s)"))
        for i in range(half):
            a, b = group_a[i], group_b[i]
            d_total = _diff(a.get("total_s"), b.get("total_s"))
            d_cc = _diff(a.get("delta_cc_sum"), b.get("delta_cc_sum"))
            d_mean = _diff(a.get("mean_s"), b.get("mean_s"))
            print("%-8s %-16s %-16s %-12s"
                  % ("%d" % (i + 1), _fmt_signed(d_total), _fmt_signed(d_cc),
                     _fmt_signed(d_mean)))
        print()
        print("A 组均值 total=%.1f s   B 组均值 total=%.1f s"
              % (_mean_opt([s.get("total_s") for s in group_a]),
                 _mean_opt([s.get("total_s") for s in group_b])))


def _diff(a, b):
    if a is None or b is None:
        return None
    return b - a


def _fmt_signed(v, spec="%+.3f"):
    return "-" if v is None else spec % v


def _mean_opt(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else float("nan")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="局部优化日志分析器（离线，只读 pipeline 日志）")
    parser.add_argument("logs", nargs="+", help="pipeline_*.log 路径")
    parser.add_argument("--verbose", action="store_true", help="附每次调用明细")
    parser.add_argument("--compare", action="store_true",
                        help="多份横向对比；日志数为偶数时前半=A组、后半=B组并给配对差值")
    parser.add_argument("--json", metavar="OUT", default=None,
                        help="把解析结果写成 JSON")
    args = parser.parse_args(argv)

    missing = [p for p in args.logs if not os.path.isfile(p)]
    if missing:
        for p in missing:
            sys.stderr.write("ERROR: no such file: %s\n" % p)
        return 2

    reports = [parse_log(p) for p in args.logs]
    summaries = [r.summary() for r in reports]

    if args.compare:
        print_compare(summaries, pair=True)
    else:
        for report, summary in zip(reports, summaries):
            print_summary(summary, verbose_calls=report.calls if args.verbose else None)
            print()

    if args.json:
        payload = {
            "summaries": summaries,
            "calls": [[c.as_dict() for c in r.calls] for r in reports],
        }
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print("JSON: %s" % args.json)

    return 1 if any(s["unclosed"] for s in summaries) else 0


if __name__ == "__main__":
    raise SystemExit(main())
