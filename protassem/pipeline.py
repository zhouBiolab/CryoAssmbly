"""Three-step protein assembly pipeline.

Step 1: Voxelization  (PDB/CIF -> simulated MRC)
Step 2: Sampling      (MRC -> point cloud TXT)
Step 3: Assembly      (point cloud registration + fitting + assembly)
"""

import os
import sys
import shutil
import logging
import math
from datetime import datetime

from protassem.core.io import find_files, read_param_file
from protassem.core.mrc_origin import normalize_density_map
from protassem.runtime.config import apply_seed
from protassem.runtime.execution import ExecutionContext
from protassem.core.structure import read_chain_ids, split_structure_to_chains
from protassem.voxelize.mol_to_mrc import pdb2vol
from protassem.sampling.sampler import sample_density_map
from protassem.assembly.orchestrator import run_assembly

log = logging.getLogger(__name__)


def setup_logging(output_dir, log_file=None):
    """Configure logging to console and optionally to file."""
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file is True:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(output_dir, f"pipeline_{timestamp}.log")

    if isinstance(log_file, str):
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=handlers,
        force=True,
    )

    if isinstance(log_file, str):
        log.info("Log file: %s", log_file)


def _validate_inputs(density_mrc, structure_files, resolution, contour, voxel_size):
    """校验流水线输入；只在此边界检查，内部函数不再重复校验。

    contour 必须是有限数值：core/scoring.calculate_cc_mask 直接执行
    ``exp_map > contour``，None 会在评分阶段以 TypeError 暴露。采样器的
    3*sigma 兜底只服务显式传入 None 的直接调用方，不属于本入口契约。
    """
    if not os.path.isfile(density_mrc):
        raise FileNotFoundError("density map not found: %s" % density_mrc)
    if not str(density_mrc).lower().endswith(".mrc"):
        raise ValueError("density map must be a .mrc file: %s" % density_mrc)
    if not structure_files:
        raise ValueError("no structure files (.pdb/.cif) given")
    missing = [f for f in structure_files if not os.path.isfile(f)]
    if missing:
        raise FileNotFoundError("structure file(s) not found: %s"
                                % ", ".join(missing))
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be a positive number, got %r" % (resolution,))
    if contour is None or not math.isfinite(contour):
        raise ValueError("contour must be a finite number, got %r" % (contour,))
    if not math.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be a positive number, got %r" % (voxel_size,))


def _normalize_density(density_mrc, output_dir):
    """把实验密度图的 header 原点化成与读法约定无关的形式，返回下游应使用的路径。

    输入已是规范形式（``nstart`` 全为 0）时原样返回，不复制文件。口径、为什么
    统一在文件侧而不改 `Sample` 或 `core.scoring`、以及"origin 停在显示位置"
    这条规则的理由与残余假设，见 `protassem.core.mrc_origin`。
    """
    result = normalize_density_map(
        density_mrc,
        os.path.join(output_dir, "density", os.path.basename(density_mrc)))
    if result.path != density_mrc:
        # 只报事实：Sample 的锚点从旧值移到显示原点，nstart 归零；图的渲染位置不变
        log.info("Density origin normalized: sample anchor %s -> %s, "
                 "nstart %s -> 0 (display origin %s unchanged)",
                 tuple(round(float(v), 3) for v in result.sample_anchor),
                 tuple(round(float(v), 3) for v in result.origin),
                 tuple(int(v) for v in result.nstart),
                 tuple(round(float(v), 3) for v in result.previous_origin))
    return result.path


def run_pipeline(density_mrc, structure_files, resolution, contour,
                 output_dir=None, voxel_size=2.0, log_file=None,
                 assembly_kwargs=None, runtime_config=None):
    """Run the full three-step pipeline.

    Args:
        density_mrc: path to experimental density map (.mrc). Its header origin is
            normalized (``nstart`` folded into ``origin``) before Step 1; when that
            changes anything, the copy under ``<output_dir>/density/`` is used from
            then on and the original file is left untouched.
        structure_files: list of structure file paths (.pdb/.cif)
        resolution: map resolution in angstroms
        contour: density contour level
        output_dir: output directory (default: <mrc_dir>/output)
        voxel_size: sampling voxel size (default 2.0)
        log_file: True for auto log file, or str path, or None
        runtime_config: RuntimeConfig（阶段二）；提供时记录实测生效线程数
        assembly_kwargs: dict of thresholds passed to AssemblyOrchestrator
            chain_threshold, initial_domain_threshold,
            min_domain_threshold, similarity_threshold

    Returns:
        dict with target_txt, source_txts, output_dir, complex_cif

    Raises:
        FileNotFoundError: density map or a structure file does not exist.
        ValueError: no structure files, or resolution/contour/voxel_size invalid.
    """
    _validate_inputs(density_mrc, structure_files, resolution, contour, voxel_size)

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(density_mrc)), "output")
    os.makedirs(output_dir, exist_ok=True)

    setup_logging(output_dir, log_file)

    if runtime_config is not None:
        log.info("Runtime config: blas_threads=%d seed=%d geometry_cache_mb=%d "
                 "encoding_cache_mb=%d inference_mode=%s allow_tf32=%s(生效 %s) "
                 "hypothesis_chunk=%d tail_pipeline=%s score_cache_mb=%d tm_cache=%s",
                 runtime_config.blas_threads, runtime_config.seed,
                 runtime_config.geometry_cache_mb, runtime_config.encoding_cache_mb,
                 runtime_config.inference_mode,
                 runtime_config.allow_tf32, runtime_config.tf32(),
                 runtime_config.hypothesis_chunk, runtime_config.tail_pipeline,
                 runtime_config.score_cache_mb, runtime_config.tm_cache)
        log.info("Effective threads: %s",
                 runtime_config.describe_effective_threads())
        # T05–T09：缓存容量、推理路径、TF32 策略、假设分块与尾部流水线透传给 PARENet 常驻服务进程
        # （必须在服务启动前设定；服务已在运行时只告警不重启）
        from protassem.fitting.parenet_client import (configure_allow_tf32,
                                                     configure_encoding_cache,
                                                     configure_geometry_cache,
                                                     configure_hypothesis_chunk,
                                                     configure_inference_mode,
                                                     configure_tail_pipeline)
        configure_geometry_cache(runtime_config.geometry_cache_mb)
        configure_encoding_cache(runtime_config.encoding_cache_mb)
        configure_allow_tf32(runtime_config.allow_tf32)
        configure_inference_mode(runtime_config.inference_mode)
        configure_hypothesis_chunk(runtime_config.hypothesis_chunk)
        configure_tail_pipeline(runtime_config.tail_pipeline)
        # 老卡收口（O6 复查项）：父进程随机源由配置固定 → 回退型局部优化不再随运行漂移
        log.info("Parent RNG seeded: %s", apply_seed(runtime_config.seed))
        # 老卡收口 P4：评分缓存预算（密度上下文 + 结构坐标；worker 通过环境变量继承）
        from protassem.core.scoring import apply_score_cache
        apply_score_cache(runtime_config.score_cache_mb)
        log.info("Score cache budget: %d MiB", runtime_config.score_cache_mb)
        # 老卡收口 P5：TM 缓存（SQLite；父进程查询/写入，worker 只跑 USalign）
        from protassem.core.similarity import configure_tm_cache
        configure_tm_cache(runtime_config.tm_cache)
        log.info("TM cache mode: %s", runtime_config.tm_cache)

    log.info("Density map : %s", density_mrc)
    log.info("Structures  : %d files", len(structure_files))
    log.info("Resolution  : %s, Contour: %s, Voxel: %.2f", resolution, contour, voxel_size)

    # Step 0 之前的唯一 MRC 处理：把 nstart 折进 origin，使采样端与评分端同框。
    # 必须晚于 _validate_inputs / os.makedirs（非法输入不建输出目录），早于 Step 1/2/3。
    density_mrc = _normalize_density(density_mrc, output_dir)

    # ---- Step 0: Standardize input structures ----
    # Read chain IDs from inside each file; keep multi-chain files as whole units.
    # If chain IDs collide across inputs, remap collisions to unique IDs (keep
    # first occurrence; assign next free from a-z/A-Z/aa..ZZ) and write as CIF.
    from protassem.core.structure import (
        cif_to_pdb, logical_chain_ids, write_structure_with_chain_map)
    std_dir = os.path.join(output_dir, "standardized")
    os.makedirs(std_dir, exist_ok=True)

    infos = []
    all_ids = []
    for f in structure_files:
        cids = read_chain_ids(f)
        infos.append({"f": f, "ext": os.path.splitext(f)[1].lower(),
                      "cids": cids, "is_complex": len(cids) > 1})
        all_ids.extend(cids)

    has_dup = len(all_ids) != len(set(all_ids))
    if has_dup:
        dups = sorted({c for c in all_ids if all_ids.count(c) > 1})
        log.info("Duplicate chain IDs across inputs %s -> remapping to unique", dups)
        used = set()
        _pool = logical_chain_ids()

        def _next_free():
            for c in _pool:
                if c not in used:
                    return c
            raise RuntimeError("chain ID pool exhausted")

        multichar = False
        for info in infos:
            m = {}
            for cid in info["cids"]:
                if cid not in used:
                    used.add(cid)
                    m[cid] = cid
                else:
                    nc = _next_free()
                    used.add(nc)
                    m[cid] = nc
                    if len(nc) > 1:
                        multichar = True
            info["map"] = m
            info["remapped"] = any(v != k for k, v in m.items())
        if multichar:
            log.warning("Remap produced multi-char chain IDs (>52 chains); "
                        "PDB-based internal steps may not preserve them")
    else:
        for info in infos:
            info["map"] = None
            info["remapped"] = False

    standardized_files = []
    chain_counter = {}
    complex_counter = 0
    for info in infos:
        f, ext, cids = info["f"], info["ext"], info["cids"]
        m, remapped = info["map"], info["remapped"]
        new_cids = [m[c] for c in cids] if m else cids
        if info["is_complex"]:
            complex_counter += 1
            cids_str = "+".join(new_cids)
            if remapped:
                std_path = os.path.join(std_dir, f"complex_{cids_str}_{complex_counter}.cif")
                write_structure_with_chain_map(f, m, std_path)
            else:
                std_path = os.path.join(std_dir, f"complex_{cids_str}_{complex_counter}{ext}")
                shutil.copy2(f, std_path)
            standardized_files.append(std_path)
            log.info("  %s -> %s (complex: chains %s)",
                     os.path.basename(f), os.path.basename(std_path), cids_str)
        else:
            new_cid = new_cids[0] if new_cids else "A"
            chain_counter[new_cid] = chain_counter.get(new_cid, 0) + 1
            if remapped:
                std_path = os.path.join(std_dir, f"chain_{new_cid}_{chain_counter[new_cid]}.cif")
                write_structure_with_chain_map(f, m, std_path)
            else:
                std_path = os.path.join(std_dir, f"chain_{new_cid}_{chain_counter[new_cid]}{ext}")
                shutil.copy2(f, std_path)
            standardized_files.append(std_path)
            log.info("  %s -> %s (chain %s)", os.path.basename(f),
                     os.path.basename(std_path), new_cid)
    structure_files = standardized_files

    # ---- Step 1: Voxelize ----
    log.info("=" * 60)
    log.info("Step 1: Voxelization")
    vox_dir = os.path.join(output_dir, "voxelized")
    os.makedirs(vox_dir, exist_ok=True)
    sim_mrcs = []
    for f in structure_files:
        name = os.path.splitext(os.path.basename(f))[0]
        out = os.path.join(vox_dir, f"{name}.mrc")
        log.info("  %s -> %s.mrc", os.path.basename(f), name)
        pdb2vol(f, resolution, output_mrc=out)
        sim_mrcs.append(out)

    # ---- Step 2: Sampling ----
    log.info("=" * 60)
    log.info("Step 2: Sampling")

    sample_dir = os.path.join(output_dir, "sampled")
    os.makedirs(sample_dir, exist_ok=True)
    shutil.copy2(os.path.abspath(density_mrc), sample_dir)
    pts, norms, target_txt = sample_density_map(
        density_mrc, contour, voxel_size, output_dir=sample_dir)
    log.info("  Target: %d points", len(pts))

    src_dir = os.path.join(output_dir, "sampled_sources")
    os.makedirs(src_dir, exist_ok=True)
    source_txts = []
    for i, mrc in enumerate(sim_mrcs):
        _, _, txt = sample_density_map(mrc, voxel_size=voxel_size,
                                       output_dir=src_dir)
        source_txts.append(txt)
        shutil.copy2(structure_files[i], src_dir)

    log.info("Step 1 & 2 done.")

    # ---- Step 3: Assembly ----
    log.info("=" * 60)
    log.info("Step 3: Assembly")

    kw = assembly_kwargs or {}
    log.info("  chain_threshold=%.3f  domain_threshold=%.3f  "
             "min_domain=%.3f  similarity=%.2f",
             kw.get("chain_threshold", 0.40),
             kw.get("initial_domain_threshold", 0.40),
             kw.get("min_domain_threshold", 0.25),
             kw.get("similarity_threshold", 0.85))

    assembly_dir = os.path.join(output_dir, "assembly")
    pool_workers = int(kw.get("num_processes", 1))
    start_method = runtime_config.pool_start_method if runtime_config else None
    context = ExecutionContext(pool_workers=pool_workers,
                               start_method=start_method)
    try:
        complex_cif = run_assembly(
            target_txt=target_txt,
            source_dir=src_dir,
            density_mrc=os.path.abspath(density_mrc),
            resolution=resolution,
            contour=contour,
            output_dir=assembly_dir,
            context=context,
            **kw,
        )
    finally:
        context.close()
    log.info("=" * 60)
    if complex_cif:
        log.info("Assembly complete: %s", complex_cif)
    else:
        log.info("Assembly finished (no complex produced)")

    return {
        "target_txt": target_txt,
        "source_txts": source_txts,
        "output_dir": output_dir,
        "complex_cif": complex_cif,
    }
