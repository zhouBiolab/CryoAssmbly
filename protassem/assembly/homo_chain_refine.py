#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 5：同源链精修（homo-chain-refine）。

在 Step4（同源域精修）之后运行：用拟合好的同源链当模板，修拟合差的链
（结构域缺失 / 结构域方位错 → 表现为 cc 低）。

触发判定（两级）：
  1. 必须存在同源链（Seq_ID 分组）。
  2. 同源组内链 cc 有分化：链 cc 差 ≤ chain_eps(0.02) → 该组跳过；
     否则比较“最好链 vs 较差链”的各结构域 cc，确认差链确有结构域变差才修。

修复：差链用同源组里 cc 更高的链当模板（序列叠合搬到本帧 + 密度优化），多进程并行生成候选，
clash 门控选最优，优于原链才替换。好链固定不动。重建复合物。

复用项目已有：core.scoring.calculate_cc_mask、core.similarity.calculate_seqid、
fitting.local_optimizer.local_optimize、assembly.refine.tr_rmsd.calculate_and_align_with_sequence、
check_clash.calculate_overlap_ratio_numpy（clash）。
"""
import os
import re
import sys
import copy
import argparse
import contextlib
import io as _io
from multiprocessing import Pool

from Bio.PDB import (MMCIFParser, PDBParser, PDBIO, MMCIFIO, Select,
                     Structure, Model, Chain)

from protassem.core.scoring import calculate_cc_mask
from protassem.core.similarity import calculate_seqid
from protassem.fitting.local_optimizer import local_optimize
from protassem.assembly.refine.tr_rmsd import calculate_and_align_with_sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))      # demo_reg 根（check_clash 在此）
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
USALIGN = os.path.join(os.path.dirname(_HERE), "core", "USalign")
AA3 = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE",
       "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}


class _ProteinSelect(Select):
    def accept_residue(self, residue):
        return residue.get_resname() in AA3 and residue.id[0] == " "


def _parser_for(path):
    return MMCIFParser(QUIET=True) if path.lower().endswith(".cif") else PDBParser(QUIET=True)


def split_chains(structure_file, out_dir):
    """拆单链 PDB（保留原链号）。返回 [(chain_id, pdb_path), ...]。"""
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


def _slice_domain(chain_pdb, ranges, out_pdb):
    """按残基范围 ranges=[(s,e),...] 从链结构切出一个域，写 out_pdb。"""
    s = PDBParser(QUIET=True).get_structure("c", chain_pdb)
    ch = Chain.Chain("A")
    for model in s:
        for chain in model:
            for res in chain:
                rid = res.id[1]
                if any(a <= rid <= b for (a, b) in ranges):
                    ch.add(res.copy())
        break
    if len(list(ch)) == 0:
        return False
    st = Structure.Structure("d")
    md = Model.Model(0)
    md.add(ch)
    st.add(md)
    io = PDBIO()
    io.set_structure(st)
    io.save(out_pdb)
    return True


def _count_breaks(chain_pdb, ranges, workdir, loose_max=10.0):
    # 按 domain_ranges 切域后用 Step4 连接评分统计断开(无效连接)数；断开=接缝 CA-CA>loose_max(默认10A)。
    # 无 ranges 或不足 2 域时返回 0。
    if not ranges:
        return 0
    os.makedirs(workdir, exist_ok=True)
    dom_files = []
    for dnum, segs in sorted(ranges.items()):
        dp = os.path.join(workdir, "brk_d%s.pdb" % dnum)
        if _slice_domain(chain_pdb, segs, dp):
            dom_files.append(dp)
    if len(dom_files) < 2:
        return 0
    try:
        from protassem.assembly.refine.refine_energy import calculate_chain_connection_score
        with contextlib.redirect_stdout(_io.StringIO()):
            _, details = calculate_chain_connection_score(
                dom_files, ideal_distance=3.8, tight_tolerance=1.5,
                loose_min=5.3, loose_max=loose_max)
        return int(details.get("no_score_connections", 0))
    except Exception as e:
        log.warning("breaks calc failed: %s", e)
        return 0


# ----------------------------------------------------------------------
# 并行：Seq_ID 分组 / cc_mask
# ----------------------------------------------------------------------
def _seqid_worker(a):
    return calculate_seqid(a[0], a[1], USALIGN)


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


def _cc_worker(a):
    mrc, path, res, cont = a
    return calculate_cc_mask(mrc, path, res, cont)


def parallel_cc(paths, mrc, res, cont, nproc=8):
    args = [(mrc, p, res, cont) for p in paths]
    if nproc > 1 and len(paths) > 1:
        with Pool(min(nproc, len(paths))) as pool:
            return pool.map(_cc_worker, args)
    return [_cc_worker(a) for a in args]


def _quiet_align(ref_pdb, mob_pdb, out_pdb):
    try:
        with contextlib.redirect_stdout(_io.StringIO()):
            calculate_and_align_with_sequence(ref_pdb, mob_pdb, out_pdb)
        return os.path.exists(out_pdb)
    except Exception as e:
        sys.stderr.write("align failed: %s\n" % e)
        return False


def _clash_overlap(pdb_a, pdb_b):
    """复用 check_clash 的 numpy CA 重叠比（惰性 import，避免 fork 前引入 torch）。"""
    import check_clash
    return check_clash.calculate_overlap_ratio_numpy(pdb_a, pdb_b, clash_distance=3.0)


# ----------------------------------------------------------------------
# 候选 worker（并行）：donor 叠到 target 帧 + 密度优化（内部 1 进程，避免嵌套 Pool）
# ----------------------------------------------------------------------
def _cand_worker(arg):
    cid, dcid, target_path, donor_path, density, res, cont, out_pdb = arg
    seed = out_pdb + ".seed.pdb"
    if not _quiet_align(target_path, donor_path, seed):
        return (cid, dcid, None, -1.0)
    try:
        seed_cc = calculate_cc_mask(density, seed, res, cont)
        ok, pk, ck = local_optimize(seed, density, out_pdb, res, cont,
                                    num_processes=1, initial_cc=seed_cc)
        return (cid, dcid, (pk if ok else seed), (ck if ok else seed_cc))
    except Exception as e:
        sys.stderr.write("cand %s<-%s failed: %s\n" % (cid, dcid, e))
        return (cid, dcid, None, -1.0)


def _anchor_cand_worker(arg):
    # 逐域锚定：把模板 donor(C) 锚到目标链第 dnum 域上 + 密度优化
    cid, dnum, segs, target_path, donor_path, density, res, cont, out_pdb = arg
    ref = out_pdb + ".ref.pdb"
    if not _slice_domain(target_path, segs, ref):
        return (cid, dnum, None, -1.0)
    seed = out_pdb + ".seed.pdb"
    if not _quiet_align(ref, donor_path, seed):
        return (cid, dnum, None, -1.0)
    try:
        seed_cc = calculate_cc_mask(density, seed, res, cont)
        ok, pk, ck = local_optimize(seed, density, out_pdb, res, cont,
                                    num_processes=1, initial_cc=seed_cc)
        return (cid, dnum, (pk if ok else seed), (ck if ok else seed_cc))
    except Exception as e:
        sys.stderr.write("anchor %s d%s failed: %s" % (cid, dnum, e))
        return (cid, dnum, None, -1.0)


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
                                    num_processes=1, initial_cc=merged_cc)
        return (cid, pk if ok else merged, ck if ok else merged_cc, n_added)
    except Exception:
        return (cid, merged, merged_cc, n_added)


# ----------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------
def run_homo_chain_refine(complex_file, density_mrc, resolution, contour, out_cif,
                          workdir, domain_ranges=None, nproc=8, seqid_thr=0.9,
                          chain_eps=0.02, domain_eps=0.02, clash_thr=0.1,
                          eps=0.003, log=print):
    """对复合物做同源链精修。domain_ranges: {chain_id: {dnum: [(s,e),...]}}（可选，用于域级判定）。

    返回 out_cif（有改动）或 None（无需/未改动）。
    """
    os.makedirs(workdir, exist_ok=True)
    chains = split_chains(complex_file, os.path.join(workdir, "chains"))
    if len(chains) < 2:
        log("homo-chain-refine: 链数 <2，跳过")
        return None
    groups = homolog_groups(chains, seqid_thr, nproc=nproc)
    log("homo-chain-refine 同源分组: %s" % [[chains[i][0] for i in g] for g in groups])
    if all(len(g) < 2 for g in groups):
        log("homo-chain-refine: 无同源链，跳过")
        return None

    paths = [p for _, p in chains]
    cc_list = parallel_cc(paths, density_mrc, resolution, contour, nproc=nproc)
    cc = {chains[i][0]: cc_list[i] for i in range(len(chains))}
    id2path = {cid: p for cid, p in chains}
    log("homo-chain-refine 链 cc: %s" % {c: round(cc[c], 4) for c, _ in chains})

    opt_dir = os.path.join(workdir, "opt")
    os.makedirs(opt_dir, exist_ok=True)
    brk_dir = os.path.join(workdir, "breaks")

    # ---- 每条链断开数（按 domain_ranges 切域，Step4 连接评分，断开阈值 10A）----
    breaks = {}
    for cid, path in chains:
        r = domain_ranges.get(cid) if domain_ranges else None
        breaks[cid] = _count_breaks(path, r, os.path.join(brk_dir, cid))
    log("homo-chain-refine 各链断开数: %s" % {c: breaks[c] for c, _ in chains})

    # ---- 每组选最好链作模板：先连续(断开少) 再 cc 高 ----
    group_best = {}
    for g in groups:
        cids = [chains[i][0] for i in g]
        b = min(cids, key=lambda c: (breaks[c], -cc[c]))
        for c in cids:
            group_best[c] = b

    # ---- 触发：比最好链断开更多，或有更差的域 ----
    to_refine = set()
    for g in groups:
        if len(g) < 2:
            continue
        cids = [chains[i][0] for i in g]
        b = group_best[cids[0]]
        for cid in cids:
            if cid == b:
                continue
            if breaks[cid] > breaks[b]:
                to_refine.add(cid)
            elif (domain_ranges and cid in domain_ranges and b in domain_ranges
                  and _domain_worse(cid, b, id2path, domain_ranges, density_mrc,
                                    resolution, contour, opt_dir, domain_eps, nproc, log)):
                to_refine.add(cid)

    best_path = dict(id2path)
    orig_nres = {cid: len(_get_resids(id2path[cid])) for cid, _ in chains}
    changed = False
    if not to_refine:
        log("homo-chain-refine: 无需精修的差链")

    # ---- 候选：逐域锚定（把模板 C 分别锚到目标链每个同源域上）----
    jobs = []
    for cid in to_refine:
        b = group_best[cid]
        if b == cid:
            continue
        ranges_c = domain_ranges.get(cid, {}) if domain_ranges else {}
        for dnum, segs in sorted(ranges_c.items()):
            out_pdb = os.path.join(opt_dir, "%s_from_%s_anchor%s.pdb" % (cid, b, dnum))
            jobs.append((cid, dnum, segs, id2path[cid], id2path[b],
                         density_mrc, resolution, contour, out_pdb))
    log("homo-chain-refine 锚定候选任务数: %d (并行 %d)" % (len(jobs), nproc))
    if jobs:
        with Pool(min(nproc, len(jobs))) as pool:
            results = pool.map(_anchor_cand_worker, jobs)
    else:
        results = []
    cand_of = {}
    for cid, dnum, pth, ck in results:
        if pth:
            cand_of.setdefault(cid, []).append((dnum, ck, pth))

    # ---- 取舍/防重叠：cc 高占位、cc 低让位换槽，clash 为硬门 ----
    # 模板为刚体放置，断开数不随锚定改变（= 模板自身断开数）
    tmpl_breaks = {cid: breaks.get(group_best[cid], 0) for cid in to_refine}
    accepted = []

    def clash_of(path):
        return sum(_clash_overlap(path, ap) for _, ap in accepted)

    # 不修的链（含模板 C）按原位姿放入作 clash 基线
    for cid, _ in chains:
        if cid not in to_refine:
            accepted.append((cid, best_path[cid]))

    # 备好每条要修链的候选：orig + 各锚定落点
    cand_pool = {}
    for cid in to_refine:
        cands = [{"label": "orig", "path": id2path[cid],
                  "cc": cc[cid], "breaks": breaks[cid]}]
        for dnum, ck, pth in cand_of.get(cid, []):
            if len(_get_resids(pth)) < orig_nres[cid] * 0.9:
                log("    %s 锚定域%s 丢残基, 跳过" % (cid, dnum))
                continue
            cands.append({"label": "anchor%s" % dnum, "path": pth,
                          "cc": ck, "breaks": tmpl_breaks[cid]})
        cand_pool[cid] = cands

    def _best_anchor_cc(cid):
        a = [c["cc"] for c in cand_pool[cid] if c["label"] != "orig"]
        return max(a) if a else cand_pool[cid][0]["cc"]

    # 贪心：最佳锚定 cc 高的链先占位，cc 低的让位换槽
    for cid in sorted(to_refine, key=_best_anchor_cc, reverse=True):
        cands = cand_pool[cid]
        for c in cands:
            c["clash"] = clash_of(c["path"])
        feasible = [c for c in cands if c["clash"] <= clash_thr]
        if feasible:
            winner = min(feasible, key=lambda c: (c["breaks"], -round(c["cc"] / max(eps, 1e-9))))
        else:
            winner = min(cands, key=lambda c: c["clash"])
        o = cands[0]
        if winner is not o:
            best_path[cid] = winner["path"]
            changed = True
            log("  链 %s: 断开%d->%d cc%.4f->%.4f clash%.3f->%.3f [%s]" %
                (cid, o["breaks"], winner["breaks"], o["cc"], winner["cc"],
                 o["clash"], winner["clash"], winner["label"]))
        else:
            log("  链 %s: 保留原位姿 (断开%d cc%.4f clash%.3f)" %
                (cid, o["breaks"], o["cc"], o["clash"]))
        accepted.append((cid, best_path[cid]))

    # ---- 缺失域补回 ----
    resid_map = {cid: _get_resids(best_path[cid]) for cid, _ in chains}
    fill_dir = os.path.join(workdir, "fill")
    fill_jobs = []
    for g in groups:
        if len(g) < 2:
            continue
        cids_g = [chains[i][0] for i in g]
        full_set = set()
        for c in cids_g:
            full_set |= resid_map[c]
        for c in cids_g:
            missing = full_set - resid_map[c]
            if not missing:
                continue
            best_donor, best_cov = None, 0
            for dc in cids_g:
                if dc == c:
                    continue
                cov = len(resid_map[dc] & missing)
                if cov > best_cov:
                    best_cov = cov
                    best_donor = dc
            if best_donor and best_cov > 0:
                fill_jobs.append((c, best_donor, best_path[c], best_path[best_donor],
                                  density_mrc, resolution, contour, fill_dir, missing))
    if fill_jobs:
        os.makedirs(fill_dir, exist_ok=True)
        log("域补回任务: %d" % len(fill_jobs))
        if nproc > 1 and len(fill_jobs) > 1:
            with Pool(min(nproc, len(fill_jobs))) as pool:
                fill_results = pool.map(_fill_worker, fill_jobs)
        else:
            fill_results = [_fill_worker(j) for j in fill_jobs]
        for c, fpath, fcc, n_added in fill_results:
            if fpath is None:
                log("  链 %s: 补回失败" % c)
                continue
            orig_cc = cc.get(c, 0.0)
            cl = sum(_clash_overlap(fpath, best_path[oc])
                     for oc, _ in chains if oc != c)
            log("  链 %s: +%d 残基, cc %.4f -> %.4f, clash=%.3f" %
                (c, n_added, orig_cc, fcc, cl))
            if cl <= clash_thr:
                best_path[c] = fpath
                cc[c] = fcc
                changed = True
                log("    -> 接受 (补回完整)")
            else:
                log("    -> 拒绝 (clash %.3f > %.2f)" % (cl, clash_thr))

    if not changed:
        log("homo-chain-refine: 无变更，跳过输出")
        return None

    # ---- 重建复合物 ----
    merged = Structure.Structure("homo")
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
    log("homo-chain-refine 已写出: %s" % out_cif)
    return out_cif


def _domain_worse(cid, best_cid, id2path, domain_ranges, mrc, res, cont,
                  opt_dir, domain_eps, nproc, log):
    """比较差链 cid 与最好链 best_cid 的各域 cc；差链某域明显更差则返回 True。"""
    ranges = domain_ranges.get(cid) or {}
    if not ranges:
        return True
    bad_paths, good_paths, dnums = [], [], []
    for dnum, segs in ranges.items():
        bp = os.path.join(opt_dir, "%s_d%s.pdb" % (cid, dnum))
        gp = os.path.join(opt_dir, "%s_d%s.pdb" % (best_cid, dnum))
        if _slice_domain(id2path[cid], segs, bp) and \
           _slice_domain(id2path[best_cid], segs, gp):
            bad_paths.append(bp)
            good_paths.append(gp)
            dnums.append(dnum)
    if not dnums:
        return True
    bad_cc = parallel_cc(bad_paths, mrc, res, cont, nproc=nproc)
    good_cc = parallel_cc(good_paths, mrc, res, cont, nproc=nproc)
    worse = False
    for dnum, bc, gc in zip(dnums, bad_cc, good_cc):
        if gc - bc > domain_eps:
            log("    链 %s 域%s cc=%.3f < 最好链 %s 域 cc=%.3f -> 该域差" %
                (cid, dnum, bc, best_cid, gc))
            worse = True
    return worse


# ----------------------------------------------------------------------
# 独立域补回（orchestrator 无条件调用，不需要 --homo-chain-refine）
# ----------------------------------------------------------------------
def fill_domain_gaps(complex_file, density_mrc, resolution, contour, out_cif,
                     workdir, nproc=8, seqid_thr=0.9, clash_thr=0.1, log=print):
    os.makedirs(workdir, exist_ok=True)
    chains = split_chains(complex_file, os.path.join(workdir, "gap_chains"))
    if len(chains) < 2:
        return None
    groups = homolog_groups(chains, seqid_thr, nproc=nproc)
    if all(len(g) < 2 for g in groups):
        return None

    id2path = {cid: p for cid, p in chains}
    resid_map = {cid: _get_resids(p) for cid, p in chains}

    fill_jobs = []
    fill_dir = os.path.join(workdir, "fill")
    for g in groups:
        if len(g) < 2:
            continue
        cids = [chains[i][0] for i in g]
        full_set = set()
        for c in cids:
            full_set |= resid_map[c]
        for c in cids:
            missing = full_set - resid_map[c]
            if not missing:
                continue
            best_donor, best_cov = None, 0
            for dc in cids:
                if dc == c:
                    continue
                cov = len(resid_map[dc] & missing)
                if cov > best_cov:
                    best_cov = cov
                    best_donor = dc
            if best_donor and best_cov > 0:
                fill_jobs.append((c, best_donor, id2path[c], id2path[best_donor],
                                  density_mrc, resolution, contour, fill_dir, missing))
    if not fill_jobs:
        return None

    os.makedirs(fill_dir, exist_ok=True)
    log("域补回: %d 条链缺失残基" % len(fill_jobs))
    paths = [p for _, p in chains]
    cc_list = parallel_cc(paths, density_mrc, resolution, contour, nproc=nproc)
    cc = {chains[i][0]: cc_list[i] for i in range(len(chains))}

    if nproc > 1 and len(fill_jobs) > 1:
        with Pool(min(nproc, len(fill_jobs))) as pool:
            fill_results = pool.map(_fill_worker, fill_jobs)
    else:
        fill_results = [_fill_worker(j) for j in fill_jobs]

    best_path = dict(id2path)
    changed = False
    for c, fpath, fcc, n_added in fill_results:
        if fpath is None:
            log("  链 %s: 补回失败" % c)
            continue
        orig_cc = cc.get(c, 0.0)
        cl = sum(_clash_overlap(fpath, best_path[oc]) for oc, _ in chains if oc != c)
        log("  链 %s: +%d 残基, cc %.4f -> %.4f, clash=%.3f" %
            (c, n_added, orig_cc, fcc, cl))
        if cl <= clash_thr:
            best_path[c] = fpath
            cc[c] = fcc
            changed = True
            log("    -> 接受")
        else:
            log("    -> 拒绝 (clash %.3f > %.2f)" % (cl, clash_thr))

    if not changed:
        return None

    merged = Structure.Structure("filled")
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
    log("域补回写出: %s" % out_cif)
    return out_cif


# ----------------------------------------------------------------------
# 单测入口
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("complex")
    p.add_argument("density")
    p.add_argument("resolution", type=float)
    p.add_argument("contour", type=float, nargs="?", default=0.0)
    p.add_argument("--out", default="homo_chain_refined.cif")
    p.add_argument("--workdir", default="homo_chain_work")
    p.add_argument("--nproc", type=int, default=8)
    a = p.parse_args()
    run_homo_chain_refine(a.complex, a.density, a.resolution, a.contour, a.out,
                          a.workdir, nproc=a.nproc)


if __name__ == "__main__":
    main()
