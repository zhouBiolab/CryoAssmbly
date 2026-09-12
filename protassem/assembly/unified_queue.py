"""Unified GLOBAL-ROUND assembly.

链和域统一按回旋半径排序、全局分轮：
- 一轮 = 把当前所有待处理项(pending 链 + available 域)按回旋半径走一遍。
- cc >= 当前阈值（同源相似已接受 -> 阈值 - domain_similar_relax）-> 立即接受 + 掩码。
- 同源组（还没接受过）：本轮最多试 2 个（怕首次没拟好），其余留到下一轮。
- 本轮没人达标 -> 在没达标候选里选 cc 最高（cc 相差 <= CA_TIEBREAK_MARGIN 时偏向 CA 多的一个）+ 降阈值。
- 阈值 0.45 -> 0.35（每轮 -0.015 前3 / -0.02）。
"""
import logging

from protassem.core.similarity import calculate_tm_score
from protassem.assembly.assembly_opt import select_round_end_candidate

log = logging.getLogger(__name__)

CA_TIEBREAK_MARGIN = 0.03   # 轮末无人达标：cc 相差 <= 此值的候选里偏向 CA 多的


def _accept_count(orch):
    return len(orch.accepted_chains) + len(orch.accepted_domain_pdbs)


def _active_items(orch, recent_pdbs=None):
    """当前待处理项：pending 链 + available 域。
    排序：与上一轮接受域相似的域优先(轮间优先) -> 再按回旋半径降序。"""
    items = []
    pending_chain_ids = set()
    for rec in orch.chain_records:
        if rec["status"] == "pending":
            items.append(("chain", rec))
            pending_chain_ids.add(rec["chain_id"])
    # 域只有在其父链已被拟过(非 pending)后才进池：避免某链的域(或其同源链的域)
    # 在该链自己链拟合之前就被拟。
    for doms in orch.domain_records.values():
        for d in doms:
            if d["status"] == "available" and d["chain_id"] not in pending_chain_ids:
                items.append(("domain", d))
    recent_pdbs = recent_pdbs or []
    _gof = getattr(orch, "domain_group_of", None) or {}
    recent_gids = set()
    for rp in recent_pdbs:
        g = _gof.get(rp)
        if g is not None:
            recent_gids.add(g)

    def _sim_to_recent(rec):
        gid = rec.get("group_id")
        if gid is not None and recent_gids:
            return 1 if gid in recent_gids else 0
        for rp in recent_pdbs:
            try:
                if calculate_tm_score(rec["pdb_file"], rp) >= orch.similarity_threshold:
                    return 1
            except Exception:
                pass
        return 0

    def keyf(x):
        kind, rec = x
        sim = _sim_to_recent(rec) if (kind == "domain" and recent_pdbs) else 0
        return (-sim, -rec["gyration_radius"])

    items.sort(key=keyf)
    return items


def run_unified_assembly(orch):
    from protassem.assembly.chain_fitter import fit_chain_item, ChainFitState
    from protassem.assembly.domain_fitter import fit_domain_once

    log.info("=" * 60)
    log.info("Unified GLOBAL-ROUND assembly (domain thr %.3f -> %.3f)",
             orch.initial_domain_threshold, orch.min_domain_threshold)
    log.info("=" * 60)

    chain_state = ChainFitState()
    # 预筛选已接受的链 -> 登记到相似组
    for rec in orch.chain_records:
        if rec["status"] == "accepted_as_chain":
            gid = rec.get("group_id", rec["chain_id"])
            chain_state.similar_groups[gid].append(rec)
            chain_state.accepted_groups[gid].append(rec)
            chain_state.accepted_chain_pdbs.append(rec["pdb_file"])

    # 单域链：直接当域处理（不单独链拟合、不域优化）
    for rec in orch.chain_records:
        if (rec["status"] == "pending" and not rec.get("is_complex")
                and len(orch.domain_records.get(rec["chain_id"], [])) == 1):
            rec["status"] = "routed_to_domain"
            if rec["chain_id"] not in orch.needs_domain_assembly:
                orch.needs_domain_assembly.append(rec["chain_id"])
            for d in orch.domain_records[rec["chain_id"]]:
                d["status"] = "available"
            log.info("链 %s 单域 -> 直接域拟合", rec["chain_id"])

    threshold = orch.initial_domain_threshold
    decreases = 0
    round_num = 0
    last_accepted_pdbs = []   # 上一轮接受的域 pdb（轮间优先：其相似域下轮前排）

    while orch._target_has_points():
        items = _active_items(orch, last_accepted_pdbs)
        if not items:
            break
        round_num += 1
        dom_before = len(orch.accepted_domain_pdbs)
        labels = []
        for t, r in items:
            if t == "chain":
                labels.append("C:%s" % r["chain_id"])
            else:
                labels.append("D:%s_d%d" % (r["chain_id"], r["domain_num"]))
        log.info("-" * 50)
        log.info("[Round %d] domain_thr=%.3f | %d 项: %s",
                 round_num, threshold, len(items), labels)

        any_accepted = False
        candidates = []          # 本轮没达标的域 drec（带 _rcc/_rpdb）
        group_reps = []          # 本轮"未接受同源组"的代表 pdb
        group_count = {}         # rep -> 本轮已试个数（2-tries）
        skipped_count = 0        # 本轮因 2-tries 被跳过(仍 available, 待下一轮)
        accepted_gids = set(orch.accepted_domain_groups)
        processed = 0            # 本轮实际处理的项数（链拟合/域拟合）

        for kind, rec in items:
            if not orch._target_has_points():
                log.info("[Round %d] 目标点云耗尽，停止", round_num)
                break

            if kind == "chain":
                if rec["status"] != "pending":
                    continue
                before = _accept_count(orch)
                fit_chain_item(orch, rec, chain_state)
                processed += 1
                if _accept_count(orch) > before:
                    any_accepted = True
                    log.info("[Round %d] 链 %s 达标接受 -> 本轮结束, 下一轮",
                             round_num, rec["chain_id"])
                    break
                # 失败链已分域(available)，下一轮 _active_items 会捡到（按回旋半径交织）
                continue

            # ---------- domain ----------
            drec = rec
            if drec["status"] != "available":
                continue
            pdb = drec["pdb_file"]
            dkey = "%s_d%d" % (drec["chain_id"], drec["domain_num"])

            # 同源"未接受组"：本轮最多试 2 个（怕首次没拟好），其余留下一轮
            gid = drec.get("group_id")
            if gid is not None:
                if gid not in accepted_gids:
                    if group_count.get(gid, 0) >= 2:
                        log.info("  %s: 同源组(%s)本轮已试 2 个，跳过(留下一轮)", dkey, gid)
                        skipped_count += 1
                        continue
                    group_count[gid] = group_count.get(gid, 0) + 1
            elif not orch._is_similar_to_any(pdb, orch.accepted_domain_src_pdbs):
                rep = None
                for rp in group_reps:
                    if calculate_tm_score(pdb, rp) >= orch.similarity_threshold:
                        rep = rp
                        break
                if rep is not None and group_count.get(rep, 0) >= 2:
                    log.info("  %s: 同源组本轮已试 2 个，跳过(留下一轮)", dkey)
                    skipped_count += 1
                    continue
                if rep is None:
                    rep = pdb
                    group_reps.append(rep)
                    group_count[rep] = 0
                group_count[rep] += 1

            status, cc, fp = fit_domain_once(orch, drec, threshold, round_num)
            processed += 1
            if status == "accepted":
                any_accepted = True
                log.info("[Round %d] 域 %s 达标接受 -> 本轮结束, 下一轮",
                         round_num, dkey)
                break
            elif status == "candidate":
                drec["_rcc"] = cc
                drec["_rpdb"] = fp
                candidates.append(drec)

        # ---------- 轮末 ----------
        if any_accepted:
            log.info("[Round %d] 有达标，进入下一轮（阈值不变 %.3f）", round_num, threshold)
        elif candidates:
            cands = [{"cc": d["_rcc"], "pdb": d["_rpdb"], "drec": d,
                      "key": "%s_d%d" % (d["chain_id"], d["domain_num"])}
                     for d in candidates]
            chosen = select_round_end_candidate(cands, CA_TIEBREAK_MARGIN)
            log.info("[Round %d] 没人达标 -> 轮末选最高(cc≈则偏 CA 多): %s cc=%.4f",
                     round_num, chosen["key"], chosen["cc"])
            orch._accept_domain(chosen["drec"], chosen["pdb"], chosen["cc"])
            orch._mask_region(chosen["pdb"])
            old = threshold
            dec = 0.015 if decreases < 3 else 0.02
            threshold = max(orch.min_domain_threshold, threshold - dec)
            decreases += 1
            log.info("[Round %d] 降阈值 %.3f -> %.3f", round_num, old, threshold)
        elif skipped_count > 0:
            log.info("[Round %d] 无达标无候选，但有 %d 个被跳过(2-tries) -> 下一轮重试",
                     round_num, skipped_count)
            # 不 break、不降阈值；下一轮 group_count 重置后会拟这些被跳过的
        elif processed > 0:
            log.info("[Round %d] 本轮处理 %d 项均失败/被拒(无接受无候选) -> 进下一轮"
                     "(失败链的域已入池)", round_num, processed)
            # 不 break：失败链已分域、被拒域已剔除，while 会重新评估池
        else:
            log.info("[Round %d] 无可处理项，结束", round_num)
            break

        for d in candidates:
            d.pop("_rcc", None)
            d.pop("_rpdb", None)

        last_accepted_pdbs = orch.accepted_domain_src_pdbs[dom_before:]
        if last_accepted_pdbs:
            log.info("[Round %d] 本轮接受 %d 域 -> 下一轮其相似域优先前排",
                     round_num, len(last_accepted_pdbs))

    log.info("=" * 60)
    log.info("Unified assembly done: %d 轮, accepted=%d 组件",
             round_num, _accept_count(orch))
