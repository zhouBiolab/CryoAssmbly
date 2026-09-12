"""Three-step protein assembly pipeline.

Step 1: Voxelization  (PDB/CIF -> simulated MRC)
Step 2: Sampling      (MRC -> point cloud TXT)
Step 3: Assembly      (point cloud registration + fitting + assembly)
"""

import os
import sys
import shutil
import logging
from datetime import datetime

from protassem.core.io import find_files, read_param_file
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


def run_pipeline(density_mrc, structure_files, resolution, contour,
                 output_dir=None, voxel_size=2.0, log_file=None,
                 assembly_kwargs=None):
    """Run the full three-step pipeline.

    Args:
        density_mrc: path to experimental density map (.mrc)
        structure_files: list of structure file paths (.pdb/.cif)
        resolution: map resolution in angstroms
        contour: density contour level
        output_dir: output directory (default: <mrc_dir>/output)
        voxel_size: sampling voxel size (default 2.0)
        log_file: True for auto log file, or str path, or None
        assembly_kwargs: dict of thresholds passed to AssemblyOrchestrator
            chain_threshold, initial_domain_threshold,
            min_domain_threshold, similarity_threshold

    Returns:
        dict with target_txt, source_txts, output_dir, complex_cif
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(density_mrc)), "output")
    os.makedirs(output_dir, exist_ok=True)

    setup_logging(output_dir, log_file)

    log.info("Density map : %s", density_mrc)
    log.info("Structures  : %d files", len(structure_files))
    log.info("Resolution  : %s, Contour: %s, Voxel: %.2f", resolution, contour, voxel_size)

    # ---- Step 0: Standardize input structures ----
    # Read chain IDs from inside each file; keep multi-chain files as whole units.
    # If chain IDs collide across inputs, remap collisions to unique IDs (keep
    # first occurrence; assign next free from a-z/A-Z/aa..ZZ) and write as CIF.
    from protassem.core.structure import (
        cif_to_pdb, chain_id_pool, write_structure_with_chain_map)
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
        _pool = chain_id_pool()

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
    complex_cif = run_assembly(
        target_txt=target_txt,
        source_dir=src_dir,
        density_mrc=os.path.abspath(density_mrc),
        resolution=resolution,
        contour=contour,
        output_dir=assembly_dir,
        **kw,
    )
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
