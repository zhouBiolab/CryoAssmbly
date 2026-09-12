"""Step 4: complex refinement via homologous-domain enumeration.

Runs the vendored ChainEnumerator (refine/chain_enumerator.py) over
work/fitted_domains/ to enumerate homologous-domain substitutions across
chains and emit a globally optimal complex (refined_complex.cif).

Two responsibilities live here so the orchestrator stays lean:
  1. gating   - decide whether Step 4 should run at all
  2. backfill - make sure EVERY accepted chain is represented as domain CIFs
                under fitted_domains/<chain>/ (chains accepted as a whole never
                went through _accept_domain, so their dir would be empty and the
                enumerator could not build a complete complex).
Then it imports and runs ChainEnumerator (pure CPU, fork-safe; PARENet lives in
a separate server process so the main process never initialised CUDA).
"""

import os
import glob
import shutil
import logging

from protassem.core.structure import align_by_resid, pdb_to_cif

log = logging.getLogger("protassem.refine")

_HERE = os.path.dirname(os.path.abspath(__file__))
USALIGN = os.path.join(os.path.dirname(_HERE), "core", "USalign")


def maybe_refine(orch):
    """Entry point called from AssemblyOrchestrator.run().

    Returns the refined complex CIF path, or None if Step 4 was skipped/failed.
    """
    if not getattr(orch, "do_refine", True):
        log.info("Step 4 (refine) disabled (--no-refine)")
        return None

    reason = _skip_reason(orch)
    if reason:
        log.info("Step 4 (refine) skipped: %s", reason)
        return None

    fitted_domains = orch.work_dir / "fitted_domains"
    log.info("=" * 60)
    log.info("Step 4: Complex refinement (homologous-domain enumeration)")
    log.info("=" * 60)

    n = _backfill_chains_as_domains(orch, fitted_domains)
    log.info("Backfilled %d whole-chain(s) into fitted_domains", n)

    enum_root = _filter_fitted_domains(orch, fitted_domains)
    final_cif = _run_enumerator(orch, enum_root)
    if not final_cif or not os.path.exists(final_cif):
        log.warning("Step 4 produced no complex; keeping assembled_complex.cif")
        return None

    refined = orch.final_dir / "refined_complex.cif"
    shutil.copy2(final_cif, str(refined))
    log.info("Step 4 complete. Primary: %s", refined)
    log.info("Backup (un-refined): %s", orch.final_dir / "assembled_complex.cif")
    return str(refined)


# ----------------------------------------------------------------------
# gating
# ----------------------------------------------------------------------

def _skip_reason(orch):
    """Return a reason string if Step 4 should NOT run, else None."""
    if not any(orch.domain_records.values()):
        return "no domains were split"
    if not orch.accepted_domain_pdbs:
        return "no domain fitting was performed"
    if not _has_homologous_chains(orch):
        return "no homologous chains"
    return None


def _has_homologous_chains(orch):
    """True if >=2 accepted chains are mutually similar (TM >= threshold)."""
    templates = []
    for rec in orch.accepted_chains:
        cr = _chain_record(orch, rec["chain_id"])
        if cr and cr.get("pdb_file"):
            templates.append(cr["pdb_file"])
    for i, pdb in enumerate(templates):
        if orch._is_similar_to_any(pdb, templates[:i]):
            return True
    return False


# ----------------------------------------------------------------------
# backfill
# ----------------------------------------------------------------------

def _backfill_chains_as_domains(orch, fitted_domains):
    """Populate fitted_domains/<chain>/ for every accepted whole-chain.

    Reuses the per-domain best already chosen by _try_improve_chain_with_domains
    (stored on the chain record as 'domain_cifs'), so that chain-pose-vs-domain
    cc_mask comparison is preserved rather than recomputed. Otherwise slices the
    accepted chain pose into domains via align_by_resid (same logic the
    chain-pose fallback already uses).
    """
    count = 0
    for rec in orch.accepted_chains:
        if rec.get("type") != "chain":
            continue
        cid = rec["chain_id"]
        out_dir = fitted_domains / cid
        if out_dir.exists() and glob.glob(os.path.join(str(out_dir), "*.cif")):
            continue  # already populated by real domain fitting
        os.makedirs(str(out_dir), exist_ok=True)

        cr = _chain_record(orch, cid)
        chain_cif = rec.get("fitted_cif")
        domain_cifs = cr.get("domain_cifs") if cr else None

        if domain_cifs:
            for d in domain_cifs:
                src = d.get("fitted_cif")
                if src and os.path.exists(src):
                    shutil.copy2(src, os.path.join(
                        str(out_dir), "domain_%s.cif" % d["domain_num"]))
            count += 1
            continue

        domains = orch.domain_records.get(cid, [])
        if domains and chain_cif and os.path.exists(chain_cif):
            for drec in domains:
                dnum = drec["domain_num"]
                tmp_pdb = os.path.join(str(out_dir), "_tmp_d%s.pdb" % dnum)
                if align_by_resid(chain_cif, drec["pdb_file"], tmp_pdb):
                    pdb_to_cif(tmp_pdb, os.path.join(
                        str(out_dir), "domain_%s.cif" % dnum), chain_id=cid)
                    os.remove(tmp_pdb)
            count += 1
        elif chain_cif and os.path.exists(chain_cif):
            shutil.copy2(chain_cif, os.path.join(str(out_dir), "domain_1.cif"))
            count += 1
    return count


def _chain_record(orch, chain_id):
    for cr in orch.chain_records:
        if cr.get("chain_id") == chain_id:
            return cr
    return None


# ----------------------------------------------------------------------
# enumerator invocation
# ----------------------------------------------------------------------

def _filter_fitted_domains(orch, fitted_domains):
    """按 complex_min_cc 过滤 Step4 的输入域集。

    fitted_domains/<chain>/*.cif 中 cc_mask < complex_min_cc 的已拟合域被剔除，
    其余拷到 fitted_domains_filtered/ 作为枚举器输入（整链补回的域无 cc 记录，保留）。
    无域被剔除时直接返回原目录。
    """
    thr = getattr(orch, "complex_min_cc", 0.0)
    cc_by_path = {}
    for drecs in orch.domain_records.values():
        for d in drecs:
            fc = d.get("fitted_cif")
            if fc:
                cc_by_path[os.path.abspath(fc)] = d.get("cc_mask", 0.0)

    filtered = orch.work_dir / "fitted_domains_filtered"
    if filtered.exists():
        shutil.rmtree(str(filtered), ignore_errors=True)

    kept = dropped = 0
    for cid_dir in sorted(glob.glob(os.path.join(str(fitted_domains), "*"))):
        if not os.path.isdir(cid_dir):
            continue
        cid = os.path.basename(cid_dir)
        if cid == "optimization_workspace" or cid.endswith("_filtered"):
            continue
        for cif in glob.glob(os.path.join(cid_dir, "*.cif")):
            cc = cc_by_path.get(os.path.abspath(cif))
            if cc is not None and cc < thr:
                log.info("Step4 filter: drop %s/%s (cc=%.4f < %.3f)",
                         cid, os.path.basename(cif), cc, thr)
                dropped += 1
                continue
            dst = filtered / cid
            os.makedirs(str(dst), exist_ok=True)
            shutil.copy2(cif, os.path.join(str(dst), os.path.basename(cif)))
            kept += 1

    log.info("Step4 input filtered by complex_min_cc=%.3f: kept %d, dropped %d",
             thr, kept, dropped)
    return str(filtered) if dropped else str(fitted_domains)


def _run_enumerator(orch, root_dir):
    """Import and run the vendored ChainEnumerator; return final.cif path."""
    from protassem.assembly.refine.chain_enumerator import ChainEnumerator

    enumerator = ChainEnumerator(
        root_dir=root_dir,
        usalign_path=USALIGN,
        tm_threshold=getattr(orch, "refine_tm", 0.75),
        n_processes=orch.num_processes,
    )
    enumerator.run_enumeration()
    final_cif = os.path.join(root_dir, "optimization_workspace",
                             "final_results", "final.cif")
    return final_cif if os.path.exists(final_cif) else None
