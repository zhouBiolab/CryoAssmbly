#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""geo_sym_refine.py — 同源 refine（纯同源，无对称信息）。

用拟合好的同源链当模板，序列叠合搬到拟合差的链当前位姿，再密度优化，clash 门控选最优。

流程：
  1. 拆单链；Seq_ID 同源分组。
  2. 逐链 cc_mask（并行）。
  3. 组内 cc 降序：组内最好链当模板；cc 与最好链相差很小的链视为“好”，保留不动。
     差链候选 = orig + 每个 cc 更高的 donor(叠到本帧 + 密度优化)。
  4. 候选生成（align + local_optimize）跨“差链×donor” **多进程并行**（每个 local_optimize
     用 num_processes=1，外层 Pool 并行，避免嵌套）。
  5. 顺序做 clash 门控：与已定链不显著重叠的候选里 cc 最高；优于 orig+eps 才替换。
  6. 重组输出。

提速：初始 cc_mask、同源分组 USalign、候选优化 三处都并行。
"""
import os
import re
import sys
import glob
import copy
import argparse
import contextlib
import subprocess
import io as _io
from multiprocessing import Pool

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from Bio.PDB import (MMCIFParser, PDBParser, PDBIO, MMCIFIO, Select,
                     Structure, Model)
from protassem.core.scoring import calculate_cc_mask
from protassem.fitting.local_optimizer import local_optimize
from protassem.assembly.refine.tr_rmsd import calculate_and_align_with_sequence
import check_clash

USALIGN = os.path.join(_HERE, "protassem", "core", "USalign")
AA3 = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE",
       "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}


class _ProteinSelect(Select):
    def accept_residue(self, residue):
        return residue.get_resname() in AA3 and residue.id[0] == " "


def _parser_for(path):
    return MMCIFParser(QUIET=True) if path.lower().endswith(".cif") else PDBParser(QUIET=True)


def split_chains(structure_file, out_dir):
    """拆单链 PDB（保留原链号）。返回有序 [(chain_id, pdb_path), ...]。"""
    os.makedirs(out_dir, exist_ok=True)
    struct = _parser_for(structure_file).get_structure("s", structure_file)
    model = next(iter(struct))
    out = []
    for chain in model:
        n_ca = sum(1 for r in chain
                   if r.get_resname() in AA3 and ("CA" in r) and r.id[0] == " ")
        if n_ca < 3:
            continue
        cid = (chain.id or "X").strip() or "X"
        ch = copy.deepcopy(chain)
        ch.detach_parent()
        ch.id = "A"
        st = Structure.Structure("x")
        md = Model.Model(0)
        md.add(ch)
        st.add(md)
        path = os.path.join(out_dir, "chain_%s.pdb" % re.sub(r"[^A-Za-z0-9_]", "_", cid))
        io = PDBIO()
        io.set_structure(st)
        io.save(path, select=_ProteinSelect())
        out.append((cid, path))
    return out


# ----------------------------------------------------------------------
# 同源判定（Seq_ID）+ 并行分组
# ----------------------------------------------------------------------
def _usalign_seqid(p1, p2):
    try:
        out = subprocess.check_output([USALIGN, p1, p2, "-TMscore", "7", "-ter", "0"],
                                      stderr=subprocess.STDOUT, text=True, timeout=180)
    except Exception:
        return 0.0
    m = re.search(r"Seq_ID=n_identical/n_aligned=\s*([\d.]+)", out)
    return float(m.group(1)) if m else 0.0


def _seqid_worker(args):
    return _usalign_seqid(args[0], args[1])


def homolog_groups(chains, seqid_thr=0.9, nproc=8):
    n = len(chains)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    args = [(chains[i][1], chains[j][1]) for (i, j) in pairs]
    if nproc > 1 and len(args) > 1:
        with Pool(min(nproc, len(args))) as pool:
            sids = pool.map(_seqid_worker, args)
    else:
        sids = [_seqid_worker(a) for a in args]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (i, j), sid in zip(pairs, sids):
        if sid >= seqid_thr:
            parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


# ----------------------------------------------------------------------
# 并行 cc_mask
# ----------------------------------------------------------------------
def _cc_worker(args):
    mrc, path, res, cont = args
    return calculate_cc_mask(mrc, path, res, cont)


def parallel_cc(paths, mrc, res, cont, nproc=8):
    args = [(mrc, p, res, cont) for p in paths]
    if nproc > 1 and len(paths) > 1:
        with Pool(min(nproc, len(paths))) as pool:
            return pool.map(_cc_worker, args)
    return [_cc_worker(a) for a in args]


# ----------------------------------------------------------------------
# 对齐
# ----------------------------------------------------------------------
def _quiet_align(ref_pdb, mob_pdb, out_pdb):
    """tr_rmsd 序列叠合：把 mob 搬到 ref 的位姿，输出 out。静默。"""
    try:
        with contextlib.redirect_stdout(_io.StringIO()):
            calculate_and_align_with_sequence(ref_pdb, mob_pdb, out_pdb)
        return os.path.exists(out_pdb)
    except Exception as e:
        sys.stderr.write("align failed: %s\n" % e)
        return False


# ----------------------------------------------------------------------
# 候选生成 worker（并行）：donor 叠到 target 帧 + 密度优化
#   每个 worker 内 local_optimize 用 num_processes=1（不开内层 Pool，避免嵌套）。
# ----------------------------------------------------------------------
def _cand_worker(arg):
    cid, donor_cid, target_path, donor_path, density, res, cont, out_pdb = arg
    seed = out_pdb + ".seed.pdb"
    if not _quiet_align(target_path, donor_path, seed):
        return (cid, donor_cid, None, -1.0)
    try:
        seed_cc = calculate_cc_mask(density, seed, res, cont)
        ok, pk, ck = local_optimize(seed, density, out_pdb, res, cont,
                                    context=None, initial_cc=seed_cc)
        return (cid, donor_cid, (pk if ok else seed), (ck if ok else seed_cc))
    except Exception as e:
        sys.stderr.write("cand %s<-%s failed: %s\n" % (cid, donor_cid, e))
        return (cid, donor_cid, None, -1.0)


# ----------------------------------------------------------------------
# 残基覆盖检测 + 缺失域补回
# ----------------------------------------------------------------------
def _get_resids(pdb_path):
    resids = set()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                try:
                    resids.add(int(line[22:26].strip()))
                except ValueError:
                    pass
    return resids


def _merge_missing(target_pdb, donor_pdb, out_pdb, missing_resids):
    t_struct = PDBParser(QUIET=True).get_structure("t", target_pdb)
    d_struct = PDBParser(QUIET=True).get_structure("d", donor_pdb)
    t_chain = next(iter(next(iter(t_struct))))
    d_chain = next(iter(next(iter(d_struct))))
    added = 0
    for res in list(d_chain):
        if res.id[1] in missing_resids and res.get_resname() in AA3 and res.id[0] == " ":
            r = copy.deepcopy(res)
            r.detach_parent()
            try:
                t_chain.add(r)
                added += 1
            except Exception:
                pass
    io = PDBIO()
    io.set_structure(t_struct)
    io.save(out_pdb, select=_ProteinSelect())
    return added


def _fill_worker(arg):
    cid, dcid, target_path, donor_path, density, res, cont, fill_dir, missing = arg
    os.makedirs(fill_dir, exist_ok=True)
    aligned = os.path.join(fill_dir, "%s_aligned_%s.pdb" % (cid, dcid))
    if not _quiet_align(target_path, donor_path, aligned):
        return (cid, None, -1.0, 0)
    merged = os.path.join(fill_dir, "%s_merged.pdb" % cid)
    n_added = _merge_missing(target_path, aligned, merged, missing)
    if n_added == 0:
        return (cid, None, -1.0, 0)
    try:
        merged_cc = calculate_cc_mask(density, merged, res, cont)
    except Exception:
        return (cid, merged, -1.0, n_added)
    try:
        opt_out = os.path.join(fill_dir, "%s_opt.pdb" % cid)
        ok, pk, ck = local_optimize(merged, density, opt_out, res, cont,
                                    context=None, initial_cc=merged_cc)
        return (cid, pk if ok else merged, ck if ok else merged_cc, n_added)
    except Exception:
        return (cid, merged, merged_cc, n_added)


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def refine(complex_file, density_mrc, resolution, contour, out_cif,
           workdir, group="?", seqid_thr=0.9, eps=0.003, nproc=8,
           clash_thr=0.1, good_margin=0.03):
    chains = split_chains(complex_file, os.path.join(workdir, "chains"))
    print("链数: %d  标记: %s" % (len(chains), group))
    groups = homolog_groups(chains, seqid_thr, nproc=nproc)
    print("同源分组: %s" % [[chains[i][0] for i in g] for g in groups])

    paths = [p for _, p in chains]
    cc_list = parallel_cc(paths, density_mrc, resolution, contour, nproc=nproc)
    cc = {chains[i][0]: cc_list[i] for i in range(len(chains))}
    print("初始 cc_mask: %s" % {cid: round(cc[cid], 4) for cid, _ in chains})

    id2path = {cid: path for cid, path in chains}
    opt_dir = os.path.join(workdir, "opt")
    os.makedirs(opt_dir, exist_ok=True)
    orig_nres = {cid: len(_get_resids(id2path[cid])) for cid, _ in chains}

    # ---- 1. 决定好链/差链；为差链组装 (差链 × cc更高donor) 任务 ----
    is_good = {}
    jobs = []
    for g in groups:
        members = sorted(g, key=lambda i: cc[chains[i][0]], reverse=True)
        best_cc = cc[chains[members[0]][0]]
        for i in members:
            cid = chains[i][0]
            if best_cc - cc[cid] < good_margin:
                is_good[cid] = True                       # 好链：保留不动
                continue
            is_good[cid] = False
            for j in members:                              # 所有 cc 更高的 donor
                dcid = chains[j][0]
                if cc[dcid] > cc[cid]:
                    out_pdb = os.path.join(opt_dir, "%s_from_%s.pdb" % (cid, dcid))
                    jobs.append((cid, dcid, id2path[cid], id2path[dcid],
                                 density_mrc, resolution, contour, out_pdb))

    # ---- 2. 并行生成所有候选（align + local_optimize, 每个 1 进程）----
    print("候选优化任务数: %d (并行 %d)" % (len(jobs), nproc))
    if jobs:
        with Pool(min(nproc, len(jobs))) as pool:
            results = pool.map(_cand_worker, jobs)
    else:
        results = []
    cand_of = {}                                           # cid -> [(donor, cc, path)]
    for cid, dcid, pth, ck in results:
        if pth:
            cand_of.setdefault(cid, []).append((dcid, ck, pth))

    # ---- 3. 顺序 clash 门控选择 ----
    best_path = dict(id2path)
    accepted = []

    def clash_of(path):
        return sum(check_clash.calculate_overlap_ratio_numpy(path, ap)
                   for _, ap in accepted)

    for g in groups:
        for i in sorted(g, key=lambda i: cc[chains[i][0]], reverse=True):  # cc 高在前
            cid = chains[i][0]
            if is_good.get(cid):
                accepted.append((cid, best_path[cid]))
                print("  链 %s: cc %.4f 保留(好链)" % (cid, cc[cid]))
                continue
            cand = [("orig", cc[cid], id2path[cid])]
            for dcid, ck, pth in cand_of.get(cid, []):
                cand_nr = len(_get_resids(pth))
                if cand_nr < orig_nres[cid] * 0.9:
                    print("    候选 from_%s 丢残基 (%d->%d), 跳过" % (dcid, orig_nres[cid], cand_nr))
                    continue
                cand.append(("from_%s" % dcid, ck, pth))
            scored = [(lab, c, pth, clash_of(pth)) for (lab, c, pth) in cand]
            valid = [s for s in scored if s[3] <= clash_thr]
            if valid:
                lab, bcc, bpath, bclash = max(valid, key=lambda s: s[1])
            else:
                lab, bcc, bpath, bclash = min(scored, key=lambda s: s[3])
            if bcc > cc[cid] + eps:
                best_path[cid] = bpath
                print("  链 %s: cc %.4f -> %.4f  [%s, clash=%.3f]" %
                      (cid, cc[cid], bcc, lab, bclash))
            else:
                print("  链 %s: cc %.4f 保留(最佳候选 %s=%.4f, clash=%.3f)" %
                      (cid, cc[cid], lab, bcc, bclash))
            accepted.append((cid, best_path[cid]))

    # ---- 3.5. 缺失域补回：从同源完整链补回丢失残基 ----
    resid_map = {cid: _get_resids(best_path[cid]) for cid, _ in chains}
    fill_jobs = []
    fill_dir = os.path.join(workdir, "fill")
    for g in groups:
        if len(g) < 2:
            continue
        cids = [chains[i][0] for i in g]
        full_set = set()
        for cid in cids:
            full_set |= resid_map[cid]
        for cid in cids:
            missing = full_set - resid_map[cid]
            if not missing:
                continue
            best_donor, best_cov = None, 0
            for dcid in cids:
                if dcid == cid:
                    continue
                cov = len(resid_map[dcid] & missing)
                if cov > best_cov:
                    best_cov = cov
                    best_donor = dcid
            if best_donor and best_cov > 0:
                fill_jobs.append((cid, best_donor, best_path[cid], best_path[best_donor],
                                  density_mrc, resolution, contour, fill_dir, missing))

    if fill_jobs:
        os.makedirs(fill_dir, exist_ok=True)
        print("域补回任务: %d" % len(fill_jobs))
        if nproc > 1 and len(fill_jobs) > 1:
            with Pool(min(nproc, len(fill_jobs))) as pool:
                fill_results = pool.map(_fill_worker, fill_jobs)
        else:
            fill_results = [_fill_worker(j) for j in fill_jobs]
        for cid, fpath, fcc, n_added in fill_results:
            if fpath is None:
                print("  链 %s: 补回失败" % cid)
                continue
            orig_cc = cc.get(cid, 0.0)
            drop = orig_cc - fcc
            cl = sum(check_clash.calculate_overlap_ratio_numpy(fpath, best_path[oc])
                     for oc, _ in chains if oc != cid)
            print("  链 %s: +%d 残基, cc %.4f -> %.4f, clash=%.3f" %
                  (cid, n_added, orig_cc, fcc, cl))
            if cl <= clash_thr:
                best_path[cid] = fpath
                cc[cid] = fcc
                print("    -> 接受 (补回完整)")
            else:
                print("    -> 拒绝 (clash %.3f > %.2f)" % (cl, clash_thr))

    # ---- 4. 重组 ----
    merged = Structure.Structure("sym")
    mm = Model.Model(0)
    merged.add(mm)
    for cid, _ in chains:
        s = PDBParser(QUIET=True).get_structure("c", best_path[cid])
        ch = copy.deepcopy(next(iter(next(iter(s)))))
        ch.detach_parent()
        ch.id = cid
        mm.add(ch)
    io = MMCIFIO()
    io.set_structure(merged)
    io.save(out_cif)
    print("已写出: %s" % out_cif)
    return out_cif


_GROUP_RE = re.compile(r"^[CDTOI]\d*$")


def _resolve_case(case_dir):
    final = None
    for ext in ("cif", "pdb"):
        cand = os.path.join(case_dir, "final." + ext)
        if os.path.exists(cand):
            final = cand
            break
    mrc = glob.glob(os.path.join(case_dir, "*.mrc"))
    res_f = os.path.join(case_dir, "resolution.txt")
    cont_f = os.path.join(case_dir, "contour_level.txt")
    marker = "?"
    for f in os.listdir(case_dir):
        if _GROUP_RE.match(f) and os.path.isfile(os.path.join(case_dir, f)):
            marker = f
            break
    resolution = float(open(res_f).read().strip()) if os.path.exists(res_f) else None
    contour = float(open(cont_f).read().strip()) if os.path.exists(cont_f) else 0.0
    return final, (mrc[0] if mrc else None), resolution, contour, marker


def main():
    p = argparse.ArgumentParser()
    p.add_argument("case_dir", nargs="?")
    p.add_argument("--complex", dest="cx")
    p.add_argument("--density")
    p.add_argument("--resolution", type=float)
    p.add_argument("--contour", type=float, default=0.0)
    p.add_argument("--group")
    p.add_argument("--out")
    p.add_argument("--seqid-threshold", dest="seqid_thr", type=float, default=0.9)
    p.add_argument("--eps", type=float, default=0.003)
    p.add_argument("--good-margin", dest="good_margin", type=float, default=0.03)
    p.add_argument("--nproc", type=int, default=8)
    p.add_argument("--clash-thr", dest="clash_thr", type=float, default=0.1)
    a = p.parse_args()

    if a.case_dir:
        final, mrc, res, cont, marker = _resolve_case(a.case_dir)
        cx = a.cx or final
        density = a.density or mrc
        resolution = a.resolution or res
        contour = a.contour if a.contour else cont
        group = a.group or marker
        out = a.out or os.path.join(a.case_dir, "symmetrized_complex.cif")
        workdir = os.path.join(a.case_dir, "geo_sym_work")
    else:
        cx, density, resolution, contour = a.cx, a.density, a.resolution, a.contour
        group = a.group or "?"
        out = a.out or "symmetrized_complex.cif"
        workdir = "geo_sym_work"

    for name, val in [("complex", cx), ("density", density), ("resolution", resolution)]:
        if not val:
            p.error("missing %s" % name)
    os.makedirs(workdir, exist_ok=True)
    print("complex=%s density=%s res=%s contour=%s" % (cx, density, resolution, contour))
    refine(cx, density, resolution, contour, out, workdir, group=group,
           seqid_thr=a.seqid_thr, eps=a.eps, nproc=a.nproc,
           clash_thr=a.clash_thr, good_margin=a.good_margin)


if __name__ == "__main__":
    main()
