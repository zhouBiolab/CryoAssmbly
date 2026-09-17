"""Chain/complex fitting logic for the unified assembly queue."""

import os
import shutil
import logging
from collections import defaultdict

from protassem.core.structure import (
    pdb_to_cif, align_by_resid,
)
from protassem.core.scoring import calculate_cc_mask
from protassem.core.similarity import calculate_tm_score
from protassem.fitting.pipeline import run_fitting
from protassem.fitting.local_optimizer import local_optimize

log = logging.getLogger(__name__)


class ChainFitState:
    """Tracks similarity groups and accepted chains during chain fitting."""

    def __init__(self):
        self.similar_groups = defaultdict(list)
        self.accepted_groups = defaultdict(list)
        self.accepted_chain_pdbs = []
        self.group_fail_count = defaultdict(int)   # 链同源组 -> 链拟合失败次数
        self.accepted_group_ids = set()            # 已接受过链/部分域的同源组


def fit_chain_item(orch, rec, state):
    """Fit a single chain/complex.

    Returns list of domain records to add to queue, or None.
    """
    cid = rec["chain_id"]
    pdb = rec["pdb_file"]
    is_complex = rec.get("is_complex", False)

    # 相似分组：优先用预计算 group_id（与 _precompute_similarity 一致），缺失才回退运行时 USalign。
    group_id = rec.get("group_id")
    if group_id is None:
        group_id = orch._find_similar_group(pdb, state.similar_groups)
        if group_id is None:
            group_id = cid
    state.similar_groups[group_id].append(rec)

    hgid = group_id
    # 同源链组 2 次保底：本组还没接受过任何链时最多试 2 条；2 条都没接受
    # (链不达标且链域交织也没收到域) -> 跳过该组其余相似链，域进池(域拟合阶段再跑)。
    if (hgid not in state.accepted_group_ids
            and state.group_fail_count.get(hgid, 0) >= 2):
        rec["status"] = "skipped_similar_failed"
        domains = [d for d in orch.domain_records.get(cid, [])
                   if d["status"] == "available"]
        if len(domains) >= 1:
            if cid not in orch.needs_domain_assembly:
                orch.needs_domain_assembly.append(cid)
            log.info("Chain %s: 同源组(%s)链拟合已失败 2 次 -> 跳过链拟合，域进池(域拟合再跑)",
                     cid, hgid)
            return domains
        rec["status"] = "rejected_no_domain"
        return None

    if not orch._target_has_points():
        return None

    orch._ensure_work_files()
    orch._chain_iter += 1
    log.info("Chain fitting %d: %s%s", orch._chain_iter, cid,
             " (complex)" if is_complex else "")
    temp_chain_dir = orch.work_dir / "temp_chain" / ("temp_chain_%d" % orch._chain_iter)
    if temp_chain_dir.exists():
        shutil.rmtree(temp_chain_dir)
    os.makedirs(temp_chain_dir)
    shutil.copy2(rec["pdb_file"], temp_chain_dir)
    shutil.copy2(rec["txt_file"], temp_chain_dir)

    base_threshold = orch.complex_threshold if is_complex else orch.chain_threshold

    fit_dir = orch.work_dir / "chain_fit" / ("chain_fit_%d" % orch._chain_iter)
    fit_result = run_fitting(
        orch.current_target_txt, str(temp_chain_dir),
        orch.current_density_mrc, orch.resolution, orch.contour,
        str(fit_dir), mode="chain",
        early_stop_threshold=base_threshold,
        original_density_mrc=orch.original_density_mrc,
        num_processes=orch.num_processes, batch_size=orch.batch_size,
        context=orch.context)

    if not fit_result["success"] or not fit_result.get("final_pdb"):
        rec["status"] = "failed"
        orch.failed_chain_pdbs.append(pdb)
        state.group_fail_count[hgid] += 1
        domains = [d for d in orch.domain_records.get(cid, [])
                   if d["status"] == "available"]
        if len(domains) >= 1:
            orch.needs_domain_assembly.append(cid)
            return domains
        rec["status"] = "rejected_no_domain"
        log.info("Chain %s: fitting failed, no domains available", cid)
        return None

    final_pdb = fit_result["final_pdb"]
    cc = fit_result["cc_mask"]
    rec["cc_mask"] = cc
    rec["fitted_pdb"] = final_pdb

    threshold = base_threshold
    has_accepted_similar = len(state.accepted_groups.get(group_id, [])) > 0
    if has_accepted_similar:
        threshold = base_threshold - orch.chain_similar_relax

    orch._save_attempt("chain_%s_iter%d" % (cid, orch._chain_iter), rec, cc, final_pdb)

    if cc >= threshold:
        log.info("Chain %s accepted (cc=%.4f >= %.3f)", cid, cc, threshold)
        if orch.improve_accepted:
            final_pdb, cc = try_improve_chain_with_domains(orch, rec, final_pdb, cc)
            rec["cc_mask"] = cc
        orch._accept_chain(rec, final_pdb, cc)
        orch._mask_region(final_pdb)
        state.accepted_groups[group_id].append(rec)
        state.accepted_chain_pdbs.append(pdb)
        state.accepted_group_ids.add(hgid)
        for d in orch.domain_records.get(cid, []):
            d["status"] = "rejected"
        return None
    else:
        log.info("Chain %s rejected (cc=%.4f < %.3f)", cid, cc, threshold)
        domains = orch.domain_records.get(cid, [])
        has_domains = len(domains) >= 1
        multi_domain = len(domains) > 1
        success = False
        if multi_domain and orch.domain_opt:
            success = try_domains_via_chain_pose(orch, rec, final_pdb)
        if has_domains:
            if success:
                # 链姿态降域接受了部分域；剩下没达标的域仍要走域拟合
                rec["status"] = "accepted_via_domains"
                state.accepted_groups[group_id].append(rec)
                state.accepted_chain_pdbs.append(pdb)
                state.accepted_group_ids.add(hgid)
            else:
                rec["status"] = "needs_domain_assembly"
                orch.failed_chain_pdbs.append(pdb)
                state.group_fail_count[hgid] += 1
            orch.needs_domain_assembly.append(cid)
            available = [d for d in domains if d["status"] == "available"]
            return available
        else:
            rec["status"] = "rejected_no_domain"
            orch.failed_chain_pdbs.append(pdb)
            log.info("Chain %s: no domains available, skipping", cid)
            return None


def try_domains_via_chain_pose(orch, chain_rec, fitted_chain_pdb):
    """When chain fails, transform its domains by chain pose and optimize."""
    cid = chain_rec["chain_id"]
    domains = orch.domain_records.get(cid, [])
    if not domains:
        return False

    transform_dir = orch.work_dir / ("chain_domain_transform_%s" % cid)
    os.makedirs(transform_dir, exist_ok=True)
    success_count = 0

    for drec in domains:
        if drec["status"] in ("accepted", "rejected"):
            continue
        dnum = drec["domain_num"]
        transformed = str(transform_dir / ("transformed_d%d.pdb" % dnum))
        if not align_by_resid(fitted_chain_pdb, drec["pdb_file"], transformed):
            continue

        init_cc = calculate_cc_mask(orch.current_density_mrc, transformed,
                                    orch.resolution, orch.contour)
        if init_cc <= 0.10:
            continue

        opt_pdb = str(transform_dir / ("optimized_d%d.pdb" % dnum))
        ok, opt_path, opt_cc = local_optimize(
            transformed, orch.current_density_mrc, opt_pdb,
            orch.resolution, orch.contour,
            initial_cc=init_cc, context=orch.context)
        if ok and opt_path:
            final_pdb, final_cc = opt_path, opt_cc
        else:
            final_pdb, final_cc = transformed, init_cc

        dthr = orch.initial_domain_threshold
        if orch._is_homolog_accepted_domain(drec):
            dthr = max(orch.min_domain_threshold,
                       orch.initial_domain_threshold - orch.domain_similar_relax)
        if final_cc >= dthr:
            orch._domain_iter += 1
            orch._accept_domain(drec, final_pdb, final_cc)
            orch._mask_region(final_pdb)
            success_count += 1
        elif final_cc > 0:
            drec["chain_pose_cc"] = final_cc
            drec["chain_pose_pdb"] = final_pdb

    return success_count > 0


def try_improve_chain_with_domains(orch, chain_rec, fitted_pdb, original_cc):
    """Try to improve an accepted chain by optimizing its domains individually."""
    cid = chain_rec["chain_id"]
    domains = orch.domain_records.get(cid, [])
    if not domains or len(domains) <= 1:
        return fitted_pdb, original_cc

    improve_dir = orch.work_dir / ("chain_improve_%s" % cid)
    os.makedirs(improve_dir, exist_ok=True)
    temp_fitted = []

    is_complex = chain_rec.get("is_complex", False)

    for drec in domains:
        dnum = drec["domain_num"]
        transformed = str(improve_dir / ("transformed_d%d.pdb" % dnum))
        if not align_by_resid(fitted_pdb, drec["pdb_file"], transformed):
            continue

        transformed_cc = calculate_cc_mask(orch.current_density_mrc, transformed,
                                           orch.resolution, orch.contour)
        opt_pdb = str(improve_dir / ("optimized_d%d.pdb" % dnum))
        ok, opt_path, opt_cc = local_optimize(
            transformed, orch.current_density_mrc, opt_pdb,
            orch.resolution, orch.contour,
            initial_cc=transformed_cc,
            context=orch.context)
        if ok and opt_path and opt_cc > transformed_cc:
            final = opt_path
        else:
            final = transformed

        src_cid = drec.get("source_chain_id")
        chain_id_for_cif = src_cid if (is_complex and src_cid) else cid
        cif = str(improve_dir / ("domain_%d.cif" % dnum))
        pdb_to_cif(final, cif, chain_id=chain_id_for_cif)
        temp_fitted.append({"domain_num": dnum, "fitted_cif": cif,
                            "chain_id": chain_id_for_cif,
                            "source_chain_id": src_cid})

    if not temp_fitted:
        return fitted_pdb, original_cc
    chain_rec["domain_cifs"] = temp_fitted

    from protassem.assembly.domain_assembler import merge_domains
    ranges, _ = orch.domain_adjacency.get(cid, ({}, {}))
    assembled_cif = merge_domains(orch, cid, temp_fitted, ranges,
                                  is_complex=is_complex)
    if not assembled_cif:
        return fitted_pdb, original_cc

    assembled_cc = calculate_cc_mask(orch.original_density_mrc, assembled_cif,
                                     orch.resolution, orch.contour)
    if assembled_cc > original_cc:
        log.info("Chain %s improved by domain reassembly: %.4f -> %.4f",
                 cid, original_cc, assembled_cc)
        from Bio.PDB import MMCIFParser, PDBIO as PDB_IO
        parser = MMCIFParser(QUIET=True)
        s = parser.get_structure("a", assembled_cif)
        improved_pdb = str(improve_dir / ("improved_%s.pdb" % cid))
        io = PDB_IO()
        io.set_structure(s)
        io.save(improved_pdb)
        return improved_pdb, assembled_cc

    return fitted_pdb, original_cc
