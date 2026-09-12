"""Shared fitting utilities — extracted from demo_mask.py.

Functions here are reusable across fitting modules.
No GPU/model dependencies.
"""

import os
import re
import glob
import logging
import numpy as np
from numba import jit
from typing import Union, Tuple, List

log = logging.getLogger(__name__)


# ======================================================================
# Point cloud operations
# ======================================================================

def compute_overlap(src, tgt, search_voxel_size):
    """Compute mutual overlap between two point clouds using KDTree.

    Returns (has_corr_src, has_corr_tgt, src_tgt_corr).
    """
    import open3d as o3d

    if isinstance(src, np.ndarray):
        src_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src))
        src_xyz = src
    else:
        src_pcd = src
        src_xyz = np.asarray(src.points)

    if isinstance(tgt, np.ndarray):
        tgt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt))
        tgt_xyz = tgt
    else:
        tgt_pcd = tgt
        tgt_xyz = np.asarray(tgt.points)

    try:
        tgt_corr = np.full(tgt_xyz.shape[0], -1, dtype=int)
        src_tree = o3d.geometry.KDTreeFlann(src_pcd)
        for i, t in enumerate(tgt_xyz):
            [k, idx, _] = src_tree.search_radius_vector_3d(t, search_voxel_size)
            if k > 0:
                tgt_corr[i] = idx[0]

        src_corr = np.full(src_xyz.shape[0], -1, dtype=int)
        tgt_tree = o3d.geometry.KDTreeFlann(tgt_pcd)
        for i, s in enumerate(src_xyz):
            [k, idx, _] = tgt_tree.search_radius_vector_3d(s, search_voxel_size)
            if k > 0:
                src_corr[i] = idx[0]

        src_indices = np.arange(len(src_corr))
        valid_src = src_corr >= 0
        mutual = np.zeros(len(src_corr), dtype=bool)
        mutual[valid_src] = (tgt_corr[src_corr[valid_src]] == src_indices[valid_src])
        src_tgt_corr = np.stack([np.nonzero(mutual)[0], src_corr[mutual]])
        has_corr_src = src_corr >= 0
        has_corr_tgt = tgt_corr >= 0
        return has_corr_src, has_corr_tgt, src_tgt_corr
    except Exception as e:
        log.warning("compute_overlap error: %s", e)
        return None, None, None


def farthest_point_sampling(points, num_samples):
    """FPS using numpy."""
    N = points.shape[0]
    if N <= num_samples:
        return np.arange(N)
    sampled_idx = np.zeros(num_samples, dtype=int)
    distances = np.full(N, np.inf)
    farthest = np.random.randint(0, N)
    for i in range(num_samples):
        sampled_idx[i] = farthest
        current = points[farthest, :3]
        dist = np.linalg.norm(points[:, :3] - current, axis=1)
        distances = np.minimum(distances, dist)
        farthest = np.argmax(distances)
    return sampled_idx


@jit(nopython=True)
def farthest_point_sampling_numba(points, num_samples, seed=0):
    """FPS with numba JIT — deterministic via fixed start point."""
    N, D = points.shape
    if N <= num_samples:
        return np.arange(N, dtype=np.int32)
    sampled_idx = np.zeros(num_samples, dtype=np.int32)
    distances = np.full(N, np.inf, dtype=np.float32)
    farthest = (43333) % N
    for i in range(num_samples):
        sampled_idx[i] = farthest
        for j in range(N):
            dist = 0.0
            for k in range(D):
                diff = points[j, k] - points[farthest, k]
                dist += diff * diff
            dist = np.sqrt(dist)
            if dist < distances[j]:
                distances[j] = dist
        max_dist = -1.0
        for j in range(N):
            if distances[j] > max_dist:
                max_dist = distances[j]
                farthest = j
    return sampled_idx


def apply_transformation(points, R, t):
    """Apply rotation and translation to point cloud."""
    return (R @ points.T).T + t


# ======================================================================
# PDB transformation
# ======================================================================

def transform_pdb(input_pdb, R, t, output_pdb):
    """Read PDB, apply rotation+translation to all ATOM coords, write PDB.

    Uses local_optimizer.StructureData for robust PDB read/write.
    """
    from protassem.fitting.local_optimizer import StructureData

    structure = StructureData(input_pdb)
    structure.apply_transformation(R, t)
    structure.write_pdb(output_pdb)


# ======================================================================
# Output path generation
# ======================================================================

SAMPLING_METHODS = {"voxel": "v", "fps": "f"}


def generate_output_pdb_path(source_file_path, config_id, sampling_method,
                             output_dir=None, mask_suffix=None):
    """Generate output PDB path from source file + config + sampling."""
    if output_dir is None:
        output_dir = os.path.dirname(source_file_path)
    base = os.path.splitext(os.path.basename(source_file_path))[0][:8]
    tag = SAMPLING_METHODS[sampling_method]
    if mask_suffix:
        name = f"pred_{base}_{config_id}{tag}_{mask_suffix}.pdb"
    else:
        name = f"pred_{base}_{config_id}{tag}.pdb"
    return os.path.abspath(os.path.join(output_dir, name))


def generate_ranked_pdb_path(original_path, rank, overlap_score):
    """Generate PDB path with rank and overlap score."""
    d = os.path.dirname(original_path)
    base = os.path.splitext(os.path.basename(original_path))[0]
    return os.path.join(d, f"{base}_{overlap_score:.4f}_top{rank:02d}.pdb")


def generate_optimal_pdb_path(original_path, overlap_score, mask_name):
    """Generate optimal PDB path for a mask result."""
    d = os.path.dirname(original_path)
    mask_simple = os.path.splitext(mask_name)[0]
    if mask_simple.startswith("mask"):
        mask_simple = mask_simple[4:]
    mask_simple = mask_simple.lstrip("_")
    mask_simple = re.sub(r"_\d+$", "", mask_simple)[:12]

    source_base = os.path.splitext(os.path.basename(original_path))[0]
    if source_base.startswith("pred_"):
        m = re.search(r"_\d+[fv]", source_base)
        prefix = source_base[5:m.start()] if m else source_base[5:13]
    else:
        prefix = source_base[:8]
    prefix = prefix.rstrip("_")
    return os.path.join(d, f"pred_{prefix}_{mask_simple}_{overlap_score:.6f}.pdb")


# ======================================================================
# Mask file utilities
# ======================================================================

def find_mask_files(temp_dir):
    """Find mask .txt files in a temp directory."""
    if not os.path.exists(temp_dir):
        return []
    skip = {"all_mask_centers.txt", "masks_report.txt"}
    mask_files = []
    for f in glob.glob(os.path.join(temp_dir, "*.txt")):
        name = os.path.basename(f)
        if name in skip:
            continue
        if "mask" in name.lower():
            mask_files.append(f)
    return mask_files


def rename_pdb_files_by_ranking(successful_results):
    """Rename pred PDB files by overlap score ranking."""
    results_with_pdb = [r for r in successful_results
                        if r.get("pred_pdb_path") and os.path.exists(r["pred_pdb_path"])]
    results_with_pdb.sort(key=lambda x: x.get("overlap", 0), reverse=True)

    for rank, result in enumerate(results_with_pdb, 1):
        old_path = result["pred_pdb_path"]
        new_path = generate_ranked_pdb_path(old_path, rank, result.get("overlap", 0))
        try:
            os.rename(old_path, new_path)
            result["pred_pdb_path"] = new_path
            result["rank"] = rank
        except Exception as e:
            log.warning("Failed to rename %s: %s", old_path, e)
            result["rank"] = rank

    return successful_results


def process_single_mask_optimally(mask_results):
    """Select the best result for a single mask and rename its PDB."""
    valid = [r for r in mask_results
             if not r.get("error") and r.get("overlap") is not None
             and r.get("pred_pdb_path") and os.path.exists(r.get("pred_pdb_path", ""))]
    if not valid:
        return None

    best = max(valid, key=lambda r: r["overlap"])
    mask_name = os.path.basename(best.get("target_file", "unknown"))
    old_path = best["pred_pdb_path"]
    new_path = generate_optimal_pdb_path(old_path, best["overlap"], mask_name)

    try:
        os.rename(old_path, new_path)
        best["pred_pdb_path"] = new_path
    except Exception as e:
        log.warning("Failed to rename optimal PDB: %s", e)

    # remove non-optimal PDBs for this mask
    for r in valid:
        if r is not best and r.get("pred_pdb_path") and os.path.exists(r["pred_pdb_path"]):
            try:
                os.remove(r["pred_pdb_path"])
            except Exception:
                pass

    return best