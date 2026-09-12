#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""geo_test.py — 对称 refine 的评测脚手架（仅测试用，不进主流程）。

复合物(final.cif / symmetrized.cif)是「整个复合物」。两套指标：

1) 主指标 —— 复合物级 USalign 多链对齐(-mm 1)：单一坐标系全局叠合，给出整体
   TM-score 与 RMSD。**纯刚体移动也会改变它**，所以对 Mode A(刚体对称收紧)敏感，
   对 Mode B(同源替换)同样敏感。这是判定 refine 是否改善的依据（TM↑ 或 RMSD↓）。
   代价：大复合物单次 ~100s。

2) 诊断指标 —— per-chain：把复合物拆成单链，与 native 各链做一一对应(匈牙利分配，
   不再贪心塌缩)，逐链 TM/RMSD。注意单链是各自最优叠合，**对刚体位移不敏感**，
   仅用于看哪条链 fold/构象差（如 Mode B 的差拷贝）。

依赖：protassem/core/USalign、scipy(linear_sum_assignment)、Bio.PDB。

用法：
  python geo_test.py geo_case_2                       # 基线
  python geo_test.py geo_case_2 --after sym.cif       # 对比 refine 前后
  python geo_test.py --complex a.cif --native 7ptt.pdb [--after b.cif]
  python geo_test.py geo_case_2 --no-perchain         # 只算复合物级主指标(快)
"""
import os
import re
import sys
import glob
import copy
import shutil
import argparse
import subprocess

import numpy as np
from scipy.optimize import linear_sum_assignment
from Bio.PDB import MMCIFParser, PDBParser, PDBIO, Select, Structure, Model

try:
    import check_clash
    _HAS_CLASH = True
except Exception:
    _HAS_CLASH = False

_HERE = os.path.dirname(os.path.abspath(__file__))
USALIGN = os.path.join(_HERE, "protassem", "core", "USalign")

AA3 = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE",
       "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}


# ----------------------------------------------------------------------
# 主指标：复合物级多链对齐(-mm)
# ----------------------------------------------------------------------
def complex_align(complex_file, native_file, fast=False):
    """USalign -mm 1：整复合物 vs native，返回整体 TM/RMSD。

    TM-score 取 normalized by Structure_2(= native 参考长度)。
    """
    cmd = [USALIGN, complex_file, native_file, "-mm", "1", "-ter", "0"]
    if fast:
        cmd.append("-fast")
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=1800)
    except Exception as e:
        sys.stderr.write("USalign -mm failed: %s\n" % e)
        return None
    m_al = re.search(r"Aligned length=\s*(\d+),\s*RMSD=\s*([\d.]+)", out)
    m_t1 = re.search(r"TM-score=\s*([\d.]+) \(normalized by length of Structure_1", out)
    m_t2 = re.search(r"TM-score=\s*([\d.]+) \(normalized by length of Structure_2", out)
    if not (m_al and m_t1 and m_t2):
        return None
    return {"aligned": int(m_al.group(1)), "rmsd": float(m_al.group(2)),
            "tm1": float(m_t1.group(1)), "tm2": float(m_t2.group(1))}


# ----------------------------------------------------------------------
# 拆链（诊断指标用）
# ----------------------------------------------------------------------
def _parser_for(path):
    return MMCIFParser(QUIET=True) if path.lower().endswith(".cif") else PDBParser(QUIET=True)


class _ProteinSelect(Select):
    def accept_residue(self, residue):
        return residue.get_resname() in AA3 and residue.id[0] == " "


def split_chains(structure_file, out_dir):
    """拆成单链 PDB（只保留含 CA 的蛋白链）。返回 [(chain_id, path), ...]。"""
    os.makedirs(out_dir, exist_ok=True)
    struct = _parser_for(structure_file).get_structure("s", structure_file)
    model = next(iter(struct))
    out = []
    for chain in model:
        n_ca = sum(1 for res in chain
                   if res.get_resname() in AA3 and ("CA" in res) and res.id[0] == " ")
        if n_ca < 3:
            continue
        cid = (chain.id or "X").strip() or "X"
        safe = re.sub(r"[^A-Za-z0-9_]", "_", cid)
        path = os.path.join(out_dir, "chain_%s.pdb" % safe)
        ch = copy.deepcopy(chain)
        ch.detach_parent()
        ch.id = "A"
        st = Structure.Structure("x")
        md = Model.Model(0)
        md.add(ch)
        st.add(md)
        io = PDBIO()
        io.set_structure(st)
        io.save(path, select=_ProteinSelect())
        out.append((cid, path))
    return out


def pair_tm(s1, s2):
    """单链对 TM/RMSD：返回 dict(rmsd, tm1, tm2, tm=min)，失败 None。"""
    try:
        out = subprocess.check_output(
            [USALIGN, s1, s2, "-TMscore", "7", "-ter", "0"],
            stderr=subprocess.STDOUT, text=True, timeout=180)
    except Exception:
        return None
    m_rmsd = re.search(r"RMSD=\s*([\d.]+)", out)
    m_tm1 = re.search(r"TM-score=\s*([\d.]+) \(normalized by length of Structure_1", out)
    m_tm2 = re.search(r"TM-score=\s*([\d.]+) \(normalized by length of Structure_2", out)
    if not (m_tm1 and m_tm2):
        return None
    tm1, tm2 = float(m_tm1.group(1)), float(m_tm2.group(1))
    return {"rmsd": float(m_rmsd.group(1)) if m_rmsd else None,
            "tm1": tm1, "tm2": tm2, "tm": min(tm1, tm2)}


def perchain_eval(complex_file, native_file, workdir, tag):
    """每条复合物链 → 一一对应(匈牙利, 最大化总 TM)的 native 链。"""
    qchains = split_chains(complex_file, os.path.join(workdir, tag + "_chains"))
    nchains = split_chains(native_file, os.path.join(workdir, "native_chains"))
    if not qchains or not nchains:
        return [], len(nchains)
    Q, N = len(qchains), len(nchains)
    res = [[None] * N for _ in range(Q)]
    tmmat = np.zeros((Q, N))
    for i, (_, cpath) in enumerate(qchains):
        for j, (_, npath) in enumerate(nchains):
            r = pair_tm(cpath, npath)
            res[i][j] = r
            tmmat[i, j] = r["tm"] if r else 0.0
    rows = []
    if Q and N:
        ri, ci = linear_sum_assignment(-tmmat)  # 最大化总 TM 的一一对应
        amap = {int(i): int(j) for i, j in zip(ri, ci)}
    else:
        amap = {}
    for i, (cid, _) in enumerate(qchains):
        if i in amap:
            j = amap[i]
            best = dict(res[i][j] or {})
            best["native"] = nchains[j][0]
        else:
            best = None
        rows.append({"chain": cid, "best": best})
    return rows, N


def _agg(rows):
    tms = [r["best"]["tm"] for r in rows if r["best"]]
    rmsds = [r["best"]["rmsd"] for r in rows if r["best"] and r["best"]["rmsd"] is not None]
    return {"mean_tm": sum(tms) / len(tms) if tms else 0.0,
            "mean_rmsd": sum(rmsds) / len(rmsds) if rmsds else 0.0}


def _print_perchain(title, rows, n_native):
    print("-" * 72)
    print("[per-chain 诊断] %s   (复合物链=%d, native链=%d)" % (title, len(rows), n_native))
    print("%-10s %-10s %8s %8s" % ("chain", "->native", "TM", "RMSD"))
    for r in rows:
        b = r["best"]
        if not b:
            print("%-10s %-10s %8s" % (r["chain"], "-", "FAIL"))
            continue
        print("%-10s %-10s %8.4f %8.3f" % (r["chain"], b["native"], b["tm"], b["rmsd"] or -1))
    a = _agg(rows)
    print("  mean_TM=%.4f  mean_RMSD=%.3f" % (a["mean_tm"], a["mean_rmsd"]))
    return a


def _print_complex(title, cx):
    print("=" * 72)
    print("[复合物级 -mm 主指标] %s" % title)
    if not cx:
        print("  对齐失败")
        return
    print("  TM-score(vs native)=%.4f   RMSD=%.3f   aligned=%d"
          % (cx["tm2"], cx["rmsd"], cx["aligned"]))


def complex_clash(complex_file, workdir, tag):
    """复合物内部链间 clash：拆链后两两 CA 重叠比，返回总和/最大/重叠对数。"""
    if not _HAS_CLASH:
        return None
    chains = split_chains(complex_file, os.path.join(workdir, tag + "_clash_chains"))
    paths = [p for _, p in chains]
    total, mx, npair = 0.0, 0.0, 0
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            a = check_clash.calculate_overlap_ratio_numpy(paths[i], paths[j])
            b = check_clash.calculate_overlap_ratio_numpy(paths[j], paths[i])
            o = max(a, b)
            total += o
            mx = max(mx, o)
            if o > 0.1:
                npair += 1
    return {"total": total, "max": mx, "pairs": npair}


def _print_clash(title, cl):
    if cl is None:
        return
    print("[clash 内部链间] %s  总重叠=%.3f  最大对=%.3f  重叠对数(>0.1)=%d"
          % (title, cl["total"], cl["max"], cl["pairs"]))


# ----------------------------------------------------------------------
# case 目录解析
# ----------------------------------------------------------------------
_GROUP_RE = re.compile(r"^[CDTOI]\d*$")


def _resolve_case(case_dir):
    final = None
    for ext in ("cif", "pdb"):
        cand = os.path.join(case_dir, "final." + ext)
        if os.path.exists(cand):
            final = cand
            break
    if not final:
        raise FileNotFoundError("找不到 final.cif/final.pdb in %s" % case_dir)
    finalname = os.path.basename(final)
    structs = [f for f in glob.glob(os.path.join(case_dir, "*.cif")) +
               glob.glob(os.path.join(case_dir, "*.pdb"))
               if os.path.basename(f) != finalname
               and not os.path.basename(f).startswith("symmetrized")]
    native = structs[0] if structs else None
    marker = None
    for f in os.listdir(case_dir):
        if _GROUP_RE.match(f) and os.path.isfile(os.path.join(case_dir, f)):
            marker = f
            break
    return final, native, marker


def main():
    p = argparse.ArgumentParser(description="symmetry refine eval: complex(-mm) + per-chain vs native")
    p.add_argument("case_dir", nargs="?")
    p.add_argument("--complex", dest="cx")
    p.add_argument("--native")
    p.add_argument("--after", help="refine 后复合物，用于对比 Δ")
    p.add_argument("--workdir")
    p.add_argument("--no-perchain", action="store_true", help="只算复合物级主指标")
    p.add_argument("--fast", action="store_true", help="USalign -mm 加 -fast(更快略糙)")
    a = p.parse_args()

    marker = None
    if a.case_dir:
        final, native, marker = _resolve_case(a.case_dir)
        complex_file = a.cx or final
        native = a.native or native
        workdir = a.workdir or os.path.join(a.case_dir, "geo_test_work")
    else:
        if not (a.cx and a.native):
            p.error("without case_dir you must give --complex and --native")
        complex_file, native = a.cx, a.native
        workdir = a.workdir or os.path.join(os.path.dirname(os.path.abspath(complex_file)),
                                            "geo_test_work")
    if not native:
        p.error("native not found")
    if not os.path.exists(USALIGN):
        p.error("USalign not found: %s" % USALIGN)

    if os.path.isdir(workdir):
        shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir, exist_ok=True)

    print("complex: %s" % complex_file)
    print("native : %s" % native)
    print("group  : %s\n" % (marker or "?"))

    cx_before = complex_align(complex_file, native, fast=a.fast)
    _print_complex("baseline final vs native", cx_before)
    cl_before = complex_clash(complex_file, workdir, "before")
    _print_clash("baseline", cl_before)
    pc_before = None
    if not a.no_perchain:
        pc_before, n_nat = perchain_eval(complex_file, native, workdir, "before")
        _print_perchain("baseline", pc_before, n_nat)

    if a.after:
        print()
        cx_after = complex_align(a.after, native, fast=a.fast)
        _print_complex("after-refine vs native", cx_after)
        cl_after = complex_clash(a.after, workdir, "after")
        _print_clash("after-refine", cl_after)
        if not a.no_perchain:
            pc_after, n_nat = perchain_eval(a.after, native, workdir, "after")
            _print_perchain("after-refine", pc_after, n_nat)

        print("=" * 72)
        print("Δ (after - before)：TM↑ / RMSD↓ / clash↓ 为改善")
        if cx_before and cx_after:
            dtm = cx_after["tm2"] - cx_before["tm2"]
            drmsd = cx_after["rmsd"] - cx_before["rmsd"]
            verdict = "IMPROVED" if (dtm > 1e-4 or drmsd < -1e-3) else "no-gain"
            print("  [复合物级] ΔTM=%+.4f  ΔRMSD=%+.3f  -> %s" % (dtm, drmsd, verdict))
        if cl_before and cl_after:
            print("  [clash] 总重叠 %.3f -> %.3f  (Δ=%+.3f)"
                  % (cl_before["total"], cl_after["total"],
                     cl_after["total"] - cl_before["total"]))


if __name__ == "__main__":
    main()
