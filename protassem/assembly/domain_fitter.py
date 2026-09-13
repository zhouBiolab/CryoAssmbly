"""Domain fitting logic for the unified assembly queue."""

import logging

from protassem.fitting.pipeline import run_fitting

log = logging.getLogger(__name__)


def fit_domain_once(orch, drec, threshold, round_num):
    """全局轮次里拟合单个域一次。返回 (status, cc, fitted_pdb)。

    status: 'accepted'(达标->已接受+掩码) / 'candidate'(没达标,留作轮末候选) / 'failed'(无结果)。
    同源宽松：相似域已接受 -> 有效阈值 = threshold - domain_similar_relax。
    """
    cid = drec["chain_id"]
    dnum = drec["domain_num"]
    dkey = "%s_d%d" % (cid, dnum)
    orch._ensure_work_files()
    orch._domain_iter += 1
    fit_dir = (orch.work_dir / "domain_fit" / ("round%02d" % round_num)
               / ("%s_d%d" % (cid, dnum)))
    log.info("  [域拟合] %s | round %d | thr=%.3f", dkey, round_num, threshold)
    fit_result = run_fitting(
        orch.current_target_txt, drec["txt_file"],
        orch.current_density_mrc, orch.resolution, orch.contour,
        str(fit_dir), mode="domain", chain_pdb=drec["pdb_file"],
        early_stop_threshold=threshold,
        original_density_mrc=orch.original_density_mrc,
        num_processes=orch.num_processes, batch_size=orch.batch_size,
        metrics=orch.metrics)
    if not fit_result["success"] or not fit_result.get("final_pdb"):
        drec["status"] = "rejected"
        orch.failed_domain_pdbs.append(drec["pdb_file"])
        log.info("    %s 无拟合结果 -> rejected", dkey)
        return ("failed", 0.0, None)
    cc = fit_result["cc_mask"]
    fp = fit_result["final_pdb"]
    eff = threshold
    if orch._is_homolog_accepted_domain(drec):
        eff = max(orch.min_domain_threshold, threshold - orch.domain_similar_relax)
        log.info("    %s 同源宽松 -> eff_thr=%.3f", dkey, eff)
    if cc >= eff:
        log.info("    %s 达标 cc=%.4f >= %.4f -> 接受+掩码", dkey, cc, eff)
        orch._accept_domain(drec, fp, cc)
        orch._mask_region(fp)
        return ("accepted", cc, fp)
    # 平台期：连续多轮 cc 无明显提升 -> 判定到顶。
    # 接受"历史最优 cc 且不与已接受结构过度 clash"的 pose（cc 均对原始密度算，可跨轮比较）。
    attempts = drec.setdefault("_attempts", [])
    attempts.append((cc, fp))
    min_att = getattr(orch, "plateau_min_attempts", 3)
    eps = getattr(orch, "plateau_eps", 0.02)
    if len(attempts) >= min_att:
        recent = [a for a, _ in attempts[-min_att:]]
        if max(recent) - min(recent) < eps:
            chosen = next(((a, f) for a, f in sorted(attempts, key=lambda t: t[0], reverse=True)
                           if not orch._clashes_with_accepted(f)), None)
            if chosen is not None:
                bcc, bfp = chosen
                log.info("    %s 连续 %d 轮 cc 平台(Δ=%.4f<%.3f) -> 到顶，接受历史最优 cc=%.4f(当前 %.4f，无过度 clash)",
                         dkey, min_att, max(recent) - min(recent), eps, bcc, cc)
                orch._accept_domain(drec, bfp, bcc)
                orch._mask_region(bfp)
                return ("accepted", bcc, bfp)
            log.info("    %s 平台到顶但历史 pose 均与已接受结构 clash -> 继续作候选", dkey)
            return ("candidate", cc, fp)
    log.info("    %s 未达标 cc=%.4f < %.4f -> 轮末候选(已试 %d 轮)", dkey, cc, eff, len(attempts))
    return ("candidate", cc, fp)
