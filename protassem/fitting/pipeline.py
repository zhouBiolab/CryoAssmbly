"""Unified fitting pipeline — replaces demo_main.py + demo_domain_main.py.

Eliminates ~1500 lines of duplicated code. Calls demo_mask.py (GPU inference)
via subprocess, uses core/scoring.py for CC evaluation (direct, no subprocess),
uses local_optimizer.py for candidate optimization.

Modes:
  chain  — iterate over source files (sorted by point count), up to 8 attempts
  domain — single source file, with optional low-resolution threshold adjustment
"""

import os
import sys
import re
import glob
import time
import shutil
import logging
import threading
import subprocess
from protassem.runtime.metrics import Metrics, worker_count
from protassem.runtime.execution import ExecutionContext

from protassem.core.scoring import calculate_cc_mask
from protassem.fitting.local_optimizer import local_optimize
from protassem.fitting.parenet_client import start_request

log = logging.getLogger(__name__)

# process count for parallel CC / local-optimize copies (set by run_fitting)
_NUM_PROCESSES = 1
# how many new pred files to accumulate before a monitor evaluation (set by run_fitting)
_BATCH_SIZE = 20
_METRICS = None


def _cc_worker(arg):
    """Compute CC_mask for one pred file (worker for parallel batch CC).

    失败不再伪装成低分（原实现返回 cc=None）：带文件名抛出真实错误。
    """
    pdb_file, density_mrc, resolution, contour = arg
    try:
        cc = calculate_cc_mask(density_mrc, pdb_file, resolution, contour)
    except Exception as exc:
        raise RuntimeError("CC_mask failed for %s: %s: %s"
                           % (os.path.basename(pdb_file), type(exc).__name__, exc)) from exc
    return {"pdb_file": pdb_file, "cc_mask": cc, "overlap": _extract_overlap(pdb_file)}

DEMO_MASK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_mask.py")
DEMO_MASK_CWD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ======================================================================
# Public API
# ======================================================================

def run_fitting(target_txt, source, density_mrc, resolution, contour,
                output_dir, mode="chain", chain_pdb=None,
                early_stop_threshold=0.54, original_density_mrc=None,
                num_processes=1, batch_size=20, metrics=None, context=None):
    """Unified entry point for chain and domain fitting.

    Args:
        target_txt: target point cloud (.txt)
        source: directory of source files (chain) or single .txt (domain)
        density_mrc: current density map (may be masked)
        resolution: map resolution in angstroms
        contour: density contour level
        output_dir: output directory
        mode: "chain" or "domain"
        chain_pdb: parent chain PDB (required for domain mode)
        early_stop_threshold: optimized CC threshold for early stop
        original_density_mrc: original unmasked density map for final eval
        metrics: 运行级 Metrics（P2）；None 时在本请求输出目录下自建

    Returns:
        dict: success (bool), final_pdb (str|None), cc_mask (float)
    """
    global _NUM_PROCESSES, _BATCH_SIZE, _METRICS
    _NUM_PROCESSES = worker_count(num_processes)
    _BATCH_SIZE = batch_size
    os.makedirs(output_dir, exist_ok=True)
    orig_mrc = original_density_mrc or density_mrc
    _METRICS = metrics or Metrics(os.path.join(output_dir, "metrics"))
    # P3：优先复用运行级共享池；独立调用 run_fitting 时自建并在返回前释放。
    owns_context = context is None
    exec_context = context or ExecutionContext(metrics=_METRICS,
                                               pool_workers=_NUM_PROCESSES)
    stage_context = _METRICS.stage("fit_request", mode=mode)
    try:
        with stage_context:
            if mode == "chain":
                return _fit_chain(target_txt, source, density_mrc, resolution,
                                  contour, output_dir, early_stop_threshold,
                                  orig_mrc, exec_context)
            return _fit_domain(target_txt, source, density_mrc, resolution,
                               contour, output_dir, chain_pdb,
                               early_stop_threshold, orig_mrc, exec_context)
    finally:
        if owns_context:
            exec_context.close()
        _METRICS.write_summary()


# ======================================================================
# Chain mode: iterate over source files
# ======================================================================

def _fit_chain(target_txt, source_dir, density_mrc, resolution, contour,
               output_dir, stop_threshold, orig_mrc, context):
    """Chain fitting: iterate over source files sorted by point count."""
    with _METRICS.stage("analyze_sources"):
        candidates = _analyze_source_files(source_dir)
    if not candidates:
        log.error("No source files found in %s", source_dir)
        return _fail()

    cc_threshold = 0.20
    max_attempts = min(8, len(candidates))
    log.info("Chain fitting: %d candidates, max %d attempts", len(candidates), max_attempts)

    for attempt_idx in range(max_attempts):
        cand = candidates[attempt_idx]
        pdb_path = _find_pdb_for_txt(cand["file_path"], source_dir)
        if not pdb_path:
            continue

        log.info("Attempt %d/%d: %s (%d points)",
                 attempt_idx + 1, max_attempts, cand["filename"], cand["point_count"])

        attempt_dir = os.path.join(output_dir, f"mask_mode_attempt_{attempt_idx + 1}")
        result = _fit_single(
            target_txt, cand["file_path"], pdb_path, attempt_dir,
            density_mrc, resolution, contour, pdb_path,
            cc_threshold, stop_threshold, orig_mrc, context)

        if result["success"]:
            with _METRICS.stage("save_result"):
                final_pdb = _save_final_result(result, attempt_dir, pdb_path)
            cc = _verify_cc(final_pdb, orig_mrc, resolution, contour)
            return {"success": True, "final_pdb": final_pdb, "cc_mask": cc}

    log.warning("All %d chain fitting attempts produced no result", max_attempts)
    return _fail()


# ======================================================================
# Domain mode: single source file
# ======================================================================

def _fit_domain(target_txt, source_txt, density_mrc, resolution, contour,
                output_dir, chain_pdb, stop_threshold, orig_mrc, context):
    """Domain fitting: single source file."""
    pdb_path = chain_pdb or _find_pdb_for_txt(source_txt, os.path.dirname(source_txt))
    if not pdb_path:
        log.error("No chain PDB found for domain fitting")
        return _fail()

    # local-optimization trigger threshold (flat 0.25 for domains).
    # The early-stop threshold (stop_threshold) is NOT resolution-adjusted.
    cc_threshold = 0.25

    result = _fit_single(
        target_txt, source_txt, pdb_path, output_dir,
        density_mrc, resolution, contour, pdb_path,
        cc_threshold, stop_threshold, orig_mrc, context)

    if result["success"]:
        with _METRICS.stage("save_result"):
            final_pdb = _save_final_result(result, output_dir, pdb_path)
        cc = _verify_cc(final_pdb, orig_mrc, resolution, contour)
        return {"success": True, "final_pdb": final_pdb, "cc_mask": cc}

    return _fail()


# ======================================================================
# Core: run one PARENet session + monitor + evaluate
# ======================================================================

def _fit_single(target_txt, source_txt, chain_pdb, output_dir,
                density_mrc, resolution, contour, pdb_for_naming,
                cc_threshold, stop_threshold, orig_mrc, context):
    """Run PARENet + monitoring + optimization for one source file."""
    reg_dir = os.path.join(output_dir, "registration")
    temp_dir = os.path.join(output_dir, "temp_candidates")
    os.makedirs(reg_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    proc = _start_parenet(target_txt, source_txt, chain_pdb, reg_dir)

    monitor_result = _monitor_and_evaluate(
        reg_dir, density_mrc, resolution, contour,
        chain_pdb, cc_threshold, stop_threshold, temp_dir, proc,
        context=context)

    best = monitor_result.get("best_result")
    if best and best.get("success"):
        return best
    return {"success": False}


def _start_parenet(target, source, chain_pdb, output_dir):
    """Send a fitting request to the persistent PARENet server.

    Returns a request handle (poll()/terminate()) compatible with the monitor.
    推理路径由服务端配置（`parenet_client.configure_inference_mode`，来自 RuntimeConfig）；
    这里不额外覆盖。
    """
    task_id = os.path.basename(output_dir)
    with _METRICS.stage("request_submit", task_id=task_id):
        return start_request(target, source, chain_pdb, output_dir,
                             use_mask=True, configs="all",
                             mask_radius_factor=1.35,
                             min_point_distance_factor=0.32)


# ======================================================================
# Monitoring loop
# ======================================================================

def _monitor_and_evaluate(reg_dir, density_mrc, resolution, contour,
                          chain_pdb, cc_threshold, stop_threshold,
                          temp_dir, proc, batch_size=None, context=None):
    """Watch for pred_*.pdb files, evaluate CC, optimize candidates."""
    if batch_size is None:
        batch_size = _BATCH_SIZE
    processed = set()
    all_results = []
    candidates = []
    best_result = None
    early_stop = False
    gpu_wait_s = 0.0
    scan_s = 0.0
    first_pred_s = None
    task_id = os.path.basename(os.path.dirname(reg_dir))
    loop_started = time.perf_counter()

    while True:
        running = proc.poll() is None
        _scan_started = time.perf_counter()
        valid = [f for f in sorted(glob.glob(os.path.join(reg_dir, "pred_*.pdb")))
                 if _is_valid_pred(f)]
        scan_s += time.perf_counter() - _scan_started
        if first_pred_s is None and valid:
            first_pred_s = time.perf_counter() - loop_started
            _METRICS.record("first_pred", first_pred_s, task_id=task_id)
        new_files = [f for f in valid if f not in processed]

        if running and len(new_files) >= batch_size:
            batch = new_files[:batch_size]
            for f in batch:
                processed.add(f)

            with _METRICS.stage("cc_batch", candidate_count=len(batch)):
                batch_results = _batch_cc(batch, density_mrc, resolution, contour,
                                          context)
            all_results.extend(batch_results)

            high = sorted(
                [r for r in batch_results if r["cc_mask"] is not None and r["cc_mask"] > cc_threshold],
                key=lambda r: r["cc_mask"], reverse=True)

            for r in high:
                with _METRICS.stage("local_optimize",
                                    candidate_count=len(candidates) + 1):
                    opt = _optimize_candidate(
                        r["pdb_file"], density_mrc, resolution, contour,
                        temp_dir, len(candidates) + 1, known_cc=r["cc_mask"],
                        context=context)
                if opt["success"]:
                    candidates.append(opt)
                    if opt["optimized_cc"] >= stop_threshold:
                        best_result = opt
                        early_stop = True
                        _kill(proc)
                        break
            if early_stop:
                break

        if not running:
            break
        time.sleep(2.5)
        gpu_wait_s += 2.5

    _METRICS.record("gpu_wait", gpu_wait_s, task_id=task_id)

    # process remaining files
    _scan_started = time.perf_counter()
    remaining = [f for f in sorted(glob.glob(os.path.join(reg_dir, "pred_*.pdb")))
                 if _is_valid_pred(f) and f not in processed]
    scan_s += time.perf_counter() - _scan_started
    _METRICS.record("candidate_scan", scan_s, task_id=task_id)
    if remaining:
        for f in remaining:
            processed.add(f)
        with _METRICS.stage("cc_batch", candidate_count=len(remaining)):
            batch_results = _batch_cc(remaining, density_mrc, resolution, contour,
                                      context)
        all_results.extend(batch_results)

    # final strategy selection
    if not early_stop:
        with _METRICS.stage("final_select", task_id=task_id):
            best_result = _select_final_result(
                all_results, candidates, density_mrc, resolution, contour,
                stop_threshold, temp_dir, context=context)

    if best_result is None and all_results:
        valid_r = [r for r in all_results if r["cc_mask"] is not None]
        if valid_r:
            raw_best = max(valid_r, key=lambda r: r["cc_mask"])
            best_result = {
                "success": True, "optimized_pdb": raw_best["pdb_file"],
                "optimized_cc": raw_best["cc_mask"], "is_unoptimized": True}

    return {"best_result": best_result, "all_results": all_results,
            "candidates": candidates, "early_stop": early_stop}


def _hybrid_score(cc, overlap):
    """Hybrid score with a single CC_mask reliability split at 0.2.

    CC_mask >= 0.2 is considered reliable enough -> full weight; below 0.2 it is
    less reliable, so its weight is halved and the ranking leans more on overlap.
    Overlap weight is constant.
    """
    cc = cc or 0.0
    overlap = overlap or 0.0
    cc_w = 1.0 if cc >= 0.2 else 0.5
    overlap_w = 1.0
    return cc_w * cc + overlap_w * overlap


def _select_final_result(all_results, candidates, density_mrc, resolution,
                         contour, stop_threshold, temp_dir, context=None):
    """No early stop: among files not yet optimized, optimize the top 5 by CC
    plus the top 5 by hybrid score (non-overlapping), then pick the best overall.
    """
    valid_r = [r for r in all_results if r["cc_mask"] is not None]
    if not valid_r:
        return None

    optimized_files = {c.get("pdb_file") for c in candidates}
    unopt = [r for r in valid_r if r["pdb_file"] not in optimized_files]
    for r in unopt:
        r["hybrid"] = _hybrid_score(r["cc_mask"], r.get("overlap", 0))

    # 1) top 5 by CC_mask
    by_cc = sorted(unopt, key=lambda r: r["cc_mask"], reverse=True)[:5]
    chosen = {r["pdb_file"] for r in by_cc}
    # 2) top 5 by hybrid score, excluding files already picked by CC
    by_hybrid = [r for r in sorted(unopt, key=lambda r: r["hybrid"], reverse=True)
                 if r["pdb_file"] not in chosen][:5]
    picks = by_cc + by_hybrid
    log.info("Final optimization: %d by CC + %d by hybrid (non-overlapping)",
             len(by_cc), len(by_hybrid))

    best, new_c = _optimize_top_n(picks, density_mrc, resolution, contour,
                                  temp_dir, candidates, stop_threshold,
                                  max_n=len(picks), context=context)
    candidates.extend(new_c)
    if best:
        return best
    if candidates:
        return max(candidates, key=lambda c: c["optimized_cc"])
    raw_best = max(valid_r, key=lambda r: r["cc_mask"])
    return {"success": True, "optimized_pdb": raw_best["pdb_file"],
            "optimized_cc": raw_best["cc_mask"], "is_unoptimized": True}


# ======================================================================
# Optimization
# ======================================================================

def _optimize_candidate(pdb_file, density_mrc, resolution, contour,
                        temp_dir, candidate_id, known_cc=None, context=None):
    """Optimize a candidate via local fitting; reuse known_cc (no recompute)."""
    work = os.path.join(temp_dir, f"candidate_{candidate_id}")
    os.makedirs(work, exist_ok=True)
    out_pdb = os.path.join(work, "optimized.pdb")

    # original CC: reuse the value already computed in _batch_cc when available
    if known_cc is not None:
        orig_cc = known_cc
    else:
        try:
            with _METRICS.stage("cc_candidate_initial", task_id=str(candidate_id)):
                orig_cc = calculate_cc_mask(density_mrc, pdb_file, resolution, contour)
        except Exception:
            orig_cc = 0.0

    # record the exact structure that met the local-opt condition and is now
    # being optimized (full path -> locate the source pred file when debugging)
    log.info("Candidate #%d: local optimization on %s (cc=%.4f)",
             candidate_id, pdb_file, orig_cc)

    # local_optimize returns the final CC -> no recompute here
    ok, opt_pdb, opt_cc = local_optimize(pdb_file, density_mrc, out_pdb,
                                         resolution, contour,
                                         initial_cc=orig_cc,
                                         metrics=_METRICS, context=context)
    if not ok or not opt_pdb or not os.path.exists(opt_pdb):
        log.warning("Candidate #%d [src: %s]: local optimization failed",
                    candidate_id, os.path.basename(pdb_file))
        return {"success": False, "candidate_id": candidate_id}

    log.info("Candidate #%d [src: %s]: CC %.4f -> %.4f (improvement %.4f)",
             candidate_id, os.path.basename(pdb_file),
             orig_cc, opt_cc, opt_cc - orig_cc)

    return {"success": True, "candidate_id": candidate_id,
            "original_cc": orig_cc, "optimized_cc": opt_cc,
            "optimized_pdb": opt_pdb, "pdb_file": pdb_file,
            "improvement": opt_cc - orig_cc}


def _optimize_top_n(results, density_mrc, resolution, contour,
                    temp_dir, existing, threshold, max_n=5, context=None):
    """Optimize top N candidates, stop early if threshold is met."""
    best = None
    new_candidates = []
    for r in results[:max_n]:
        cid = len(existing) + len(new_candidates) + 1
        opt = _optimize_candidate(r["pdb_file"], density_mrc, resolution,
                                  contour, temp_dir, cid, known_cc=r.get("cc_mask"),
                                  context=context)
        if opt["success"]:
            new_candidates.append(opt)
            if opt["optimized_cc"] >= threshold:
                best = opt
                break
    return best, new_candidates


# ======================================================================
# Result saving
# ======================================================================

def _save_final_result(best_result, output_dir, chain_pdb_path):
    """Save best result PDB to results/ directory with proper naming."""
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    src = best_result.get("optimized_pdb") or best_result.get("pdb_file")
    if not src or not os.path.exists(src):
        return None

    basename = os.path.basename(chain_pdb_path)
    if basename.startswith("chain_"):
        out_name = "pred_" + basename[6:]
    else:
        out_name = "pred_" + basename

    dst = os.path.join(results_dir, out_name)
    shutil.copy2(src, dst)
    origin = os.path.basename(best_result.get("pdb_file") or src)
    log.info("Result saved: %s (source: %s, cc_mask=%.4f)",
             dst, origin, best_result.get("optimized_cc", 0.0))
    return dst


def _verify_cc(pdb_file, density_mrc, resolution, contour):
    """Calculate CC_mask on original density for final verification."""
    if not pdb_file or not os.path.exists(pdb_file):
        return 0.0
    try:
        with _METRICS.stage("cc_verify"):
            cc = calculate_cc_mask(density_mrc, pdb_file, resolution, contour)
        log.info("Verified CC_mask (original density): %.6f", cc)
        return cc
    except Exception as e:
        log.warning("CC verification failed: %s", e)
        return 0.0


# ======================================================================
# Utilities
# ======================================================================

def _batch_cc(pdb_files, density_mrc, resolution, contour, context):
    """Compute CC_mask for a batch of PDB files（复用运行级池，按输入顺序归并）。"""
    args = [(f, density_mrc, resolution, contour) for f in pdb_files]
    return context.map(_cc_worker, args)


def _is_valid_pred(pdb_file):
    return bool(re.match(r"pred_.*\d+\.\d+\.pdb$", os.path.basename(pdb_file)))


def _extract_overlap(pdb_file):
    m = re.search(r"(\d+\.\d+)\.pdb$", os.path.basename(pdb_file))
    return float(m.group(1)) if m else 0.0


def _count_points(txt_file):
    """Count points in a sample txt file (fast, no parsing)."""
    try:
        with open(txt_file) as f:
            lines = f.readlines()
        return max(0, (len(lines) - 5) // 2) if len(lines) > 5 else 0
    except Exception:
        return 0


def _analyze_source_files(source_dir):
    """List source txt files sorted by point count (descending)."""
    txt_files = glob.glob(os.path.join(source_dir, "*.txt"))
    txt_files = [f for f in txt_files
                 if not os.path.basename(f).upper().startswith("EMD")
                 and ("chain" in os.path.basename(f).lower()
                      or "complex" in os.path.basename(f).lower())]
    info = [{"file_path": f, "filename": os.path.basename(f),
             "point_count": _count_points(f)} for f in txt_files]
    info.sort(key=lambda x: x["point_count"], reverse=True)
    for i, item in enumerate(info[:10], 1):
        log.info("  %d. %s: %d points", i, item["filename"], item["point_count"])
    return info


def _find_pdb_for_txt(txt_file, search_dir):
    """Find the PDB file corresponding to a source txt file."""
    base = os.path.basename(txt_file).replace(".txt", "")
    clean = re.sub(r"mol.*$|_\d+\.\d+$|_(sample|pred).*", "", base)
    pdb_name = clean + ".pdb"

    for d in [search_dir, os.path.dirname(txt_file)]:
        path = os.path.join(d, pdb_name)
        if os.path.exists(path):
            return path

    for root, _, files in os.walk(search_dir):
        if pdb_name in files:
            return os.path.join(root, pdb_name)
    return None


def _kill(proc):
    # request handle: signal the server to stop the current request early
    try:
        proc.terminate()
    except Exception:
        pass


def _fail():
    return {"success": False, "final_pdb": None, "cc_mask": 0.0}
