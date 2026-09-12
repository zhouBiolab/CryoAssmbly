# -*- coding: utf-8 -*-
"""组装优化辅助：参数集中 + clash/CA helper + 轮末候选选择。供 orchestrator/unified_queue 复用。

- AssemblyOptConfig：组装可调参数，集中一处调参。
- ca_count / ca_overlap：CA 原子数 / 链间 CA 重叠比（clash，复用 check_clash 的 numpy 实现）。
- select_round_end_candidate：轮末无人达标时选 cc 最高者；若有候选 cc 与最高相差 ≤ ca_margin，
  则在这批近似候选里偏向 CA 原子数更多的一个（只接受一个）。
"""
import os
import sys
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))          # demo_reg 根（check_clash 在此）
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@dataclass
class AssemblyOptConfig:
    """组装可调参数（集中调参处）。"""
    chain_similar_relax: float = 0.03      # 同源组已有接受时，链阈放宽
    domain_similar_relax: float = 0.025    # 相似域已接受时，域阈放宽
    clash_overlap_thr: float = 0.10        # CA 重叠比 > 此值算 clash（保 CA 多者）


def ca_count(pdb):
    """PDB 里 CA/P 原子数（残基数近似）。"""
    n = 0
    try:
        with open(pdb) as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() in ("CA", "P"):
                    n += 1
    except Exception:
        pass
    return n


def ca_overlap(pdb_a, pdb_b):
    """两结构 CA 重叠比（复用 check_clash 的 numpy 实现，惰性 import 避免引入 torch）。"""
    import check_clash
    return check_clash.calculate_overlap_ratio_numpy(pdb_a, pdb_b, clash_distance=3.0)


def select_round_end_candidate(candidates, ca_margin):
    """轮末候选选择（单选）。

    candidates: [{"cc": float, "pdb": pdb_path, "drec": ..., "key": ...}, ...]。
    规则：取 cc 最高者 best；在与 best 的 cc 相差 ≤ ca_margin 的近似候选里，选 CA 原子数
    最多的一个（cc 接近时偏向更完整的结构）。返回单个 candidate；candidates 为空返回 None。
    """
    if not candidates:
        return None
    best_cc = max(c["cc"] for c in candidates)
    near = [c for c in candidates if best_cc - c["cc"] <= ca_margin]
    return max(near, key=lambda c: ca_count(c["pdb"]))
