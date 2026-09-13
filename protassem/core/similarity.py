"""USalign wrapper for TM-score / sequence-identity calculation.

带进程内缓存：同一对 pdb 的 TM-score 只用 USalign 算一次（轮内/轮间反复比对不再重复跑）。
prefill_tm_cache 可在准备阶段并行预填一批 pdb 对。
"""

import os
import re
import subprocess

USALIGN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "USalign")

_TM_CACHE = {}


def _pair_key(p1, p2):
    a = os.path.abspath(str(p1))
    b = os.path.abspath(str(p2))
    return (a, b) if a <= b else (b, a)


def _run_usalign_tm(pdb1, pdb2, usalign_path=None):
    path = usalign_path or USALIGN_PATH
    try:
        result = subprocess.run(
            [path, str(pdb1), str(pdb2), "-TMscore", "7", "-ter", "0"],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            return 0.0
        tm1 = re.search(r"TM-score=\s*([\d.]+) \(normalized by length of Structure_1", result.stdout)
        tm2 = re.search(r"TM-score=\s*([\d.]+) \(normalized by length of Structure_2", result.stdout)
        if tm1 and tm2:
            return min(float(tm1.group(1)), float(tm2.group(1)))
        return 0.0
    except Exception:
        return 0.0


def calculate_tm_score(pdb1, pdb2, usalign_path=None):
    """Calculate TM-score between two structures (min of both directions). 带缓存。"""
    k = _pair_key(pdb1, pdb2)
    v = _TM_CACHE.get(k)
    if v is not None:
        return v
    v = _run_usalign_tm(pdb1, pdb2, usalign_path)
    _TM_CACHE[k] = v
    return v


def _tm_worker(pair):
    p1, p2 = pair
    return (_pair_key(p1, p2), _run_usalign_tm(p1, p2))


def prefill_tm_cache(pairs, context):
    """用运行级池预填一批 (pdb1, pdb2) 的 TM-score 进缓存。返回实际计算的对数。"""
    todo, seen = [], set()
    for p1, p2 in pairs:
        k = _pair_key(p1, p2)
        if k in _TM_CACHE or k in seen:
            continue
        seen.add(k)
        todo.append((p1, p2))
    if not todo:
        return 0
    for key, value in context.map(_tm_worker, todo):
        _TM_CACHE[key] = value
    return len(todo)


def calculate_seqid(pdb1, pdb2, usalign_path=None):
    """Sequence identity over the USalign-aligned region (n_identical/n_aligned).

    同源判定用它而非结构 TM：同一蛋白的拷贝即使被组装坏(结构 TM 很低)，序列同一性仍≈1。
    """
    path = usalign_path or USALIGN_PATH
    try:
        result = subprocess.run(
            [path, str(pdb1), str(pdb2), "-TMscore", "7", "-ter", "0"],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            return 0.0
        m = re.search(r"Seq_ID=n_identical/n_aligned=\s*([\d.]+)", result.stdout)
        return float(m.group(1)) if m else 0.0
    except Exception:
        return 0.0
