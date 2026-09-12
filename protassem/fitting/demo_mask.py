"""PARENet inference engine — GPU point cloud registration.

Core logic: loads PARENet model, runs inference on source/target point clouds,
generates predicted PDB structures with overlap scoring.

Dependencies:
  - config, model, pareconv: loaded via sys.path from original project
  - sw_mask: spherical mask generator (local copy)
  - fitting/utils: shared utilities (compute_overlap, FPS, PDB transform, etc.)
"""

import os
import sys
import logging
import numpy as np
import random
import glob
import torch
import argparse
import warnings
from tqdm import tqdm
from typing import Union, Tuple, List
import re
import json

# Add project root so protassem package is importable when run as subprocess
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pareconv.utils.data_mask import registration_collate_fn_stack_mode, precompute_neibors
import pareconv.utils.data_mask as data_mask_module
from pareconv.utils.torch import to_cuda, release_cuda
from pareconv.modules.ops.transformation import apply_transform
from protassem.fitting.parenet.config import make_cfg
from protassem.fitting.parenet.model import create_model
from protassem.core.points_txt import read_point_cloud_file

from protassem.fitting.utils import (
    compute_overlap,
    farthest_point_sampling_numba,
    apply_transformation,
    transform_pdb,
    generate_output_pdb_path,
    generate_optimal_pdb_path,
    find_mask_files,
    rename_pdb_files_by_ranking,
    process_single_mask_optimally,
    SAMPLING_METHODS,
)

try:
    from protassem.fitting.sw_mask import (
        generate_spherical_masks, save_masks, SphericalMaskGenerator
    )
    MASK_MODULE_AVAILABLE = True
except ImportError:
    MASK_MODULE_AVAILABLE = False
    logging.warning("sw_mask not available, mask mode disabled")

warnings.filterwarnings("ignore", category=UserWarning)

log = logging.getLogger(__name__)

# ======================================================================
# Voxel size configurations
# ======================================================================

VOXEL_SIZE_CONFIGS_NORMAL = {
    0: [2, 3.6, 6.48, 11.664],
    1: [1, 1, 3.6, 6.48],
    2: [1, 1, 4, 12],
    3: [1, 2, 4, 11],
    4: [1, 3.6, 6.48, 11.664],
    5: [1, 1, 4, 8],
}

VOXEL_SIZE_CONFIGS_MASK = {
    0: [2, 3.6, 6.48, 11.664],
    1: [1, 1, 3.6, 6.48],
    2: [1, 1, 4, 12],
    3: [1, 2, 4, 11],
    4: [1, 3.6, 6.48, 11.664],
    5: [1, 1, 4, 8],
    6: [1, 1, 2, 4],
}


# ======================================================================
# Point cloud I/O and preprocessing
# ======================================================================

def load_sample_points(file_path):
    """读取点云文件，返回 index/point/vector/density 结构化数组。

    格式契约与解析规则见 protassem.core.points_txt（单一实现）。
    """
    return read_point_cloud_file(file_path)


def load_mask_points(file_path):
    """Load mask file, return (N,3) coordinate array."""
    points = []
    with open(file_path, "r") as f:
        lines = f.readlines()
    for i in range(5, len(lines)):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            continue
        rel = i - 5 + 1
        if rel % 2 == 1:
            parts = line.split()
            if len(parts) >= 4:
                points.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not points:
        raise ValueError(f"No points in mask file: {file_path}")
    return np.array(points, dtype=np.float32)


class PreprocessedData:
    """Container for preprocessed point cloud data."""

    def __init__(self, points, vectors, density, indices, centroid, is_mask_file=False):
        self.points = points
        self.vectors = vectors
        self.density = density
        self.indices = indices
        self.centroid = centroid
        self.is_mask_file = is_mask_file


def preprocess_point_cloud_data(file_path, point_limit=65000, is_mask_file=False):
    """Load and preprocess point cloud data."""
    if is_mask_file:
        pts = load_mask_points(file_path)
        return PreprocessedData(
            pts, np.ones((len(pts), 3), dtype=np.float32),
            np.ones(len(pts), dtype=np.float32),
            np.arange(len(pts), dtype=np.int32),
            pts.mean(axis=0), True)

    data = load_sample_points(file_path)
    if point_limit and len(data) > point_limit:
        idx = farthest_point_sampling_numba(data["point"], point_limit)
        data = data[idx]
    pts = data["point"]
    return PreprocessedData(
        pts, data["vector"], data["density"], data["index"],
        pts.mean(axis=0), False)


# ======================================================================
# Core inference
# ======================================================================

def process_single_pair(src_data, tgt_data, source_path, target_path,
                        chain_pdb_path, model, cfg, config_id, sampling_method,
                        output_dir=None, use_mask=False, masks=None,
                        masks_save_path=None, mask_suffix=None,
                        original_target_data=None):
    """Run PARENet inference on one source-target pair."""
    result = {
        "source_file": os.path.basename(source_path),
        "target_file": os.path.basename(target_path),
        "chain_pdb": os.path.basename(chain_pdb_path) if chain_pdb_path else "None",
        "config_id": config_id,
        "sampling_method": sampling_method,
        "error": "",
        "overlap": None,
        "ref_points": None,
        "src_points": None,
        "pred_pdb_path": None,
        "use_mask": use_mask,
        "masks_generated": 0,
        "is_mask_file": tgt_data.is_mask_file,
        "mask_suffix": mask_suffix,
    }

    try:
        if use_mask and not tgt_data.is_mask_file:
            if masks is None:
                result["error"] = "Mask mode but no masks"
                return result
            result["masks_generated"] = len(masks)
            result["masks_save_path"] = masks_save_path
            result.update({"overlap": 0.0, "ref_points": len(tgt_data.points),
                          "src_points": len(src_data.points)})
            return result

        # normalize
        c_ref = tgt_data.centroid
        c_src = src_data.centroid
        ref_norm = tgt_data.points.copy()
        src_norm = src_data.points.copy()
        ref_norm[:, :3] -= c_ref
        src_norm[:, :3] -= c_src

        if original_target_data is not None and tgt_data.is_mask_file:
            ref_for_ov = original_target_data.points.copy()
            ref_for_ov[:, :3] -= c_ref
            src_for_ov = src_norm.copy()
        else:
            ref_for_ov = ref_norm.copy()
            src_for_ov = src_norm.copy()

        scale = max(np.linalg.norm(ref_norm, axis=1).max(),
                    np.linalg.norm(src_norm, axis=1).max()).astype(np.float32)

        data_dict = {
            "ref_points": ref_norm.astype(np.float32),
            "src_points": src_norm.astype(np.float32),
            "ref_feats": np.ones((ref_norm.shape[0], 1), dtype=np.float32),
            "src_feats": np.ones((src_norm.shape[0], 1), dtype=np.float32),
            "transform": torch.from_numpy(np.eye(4, dtype=np.float32)),
            "scale": scale,
        }

        data_dict = registration_collate_fn_stack_mode(
            [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size,
            cfg.backbone.num_neighbors, cfg.backbone.subsample_ratio)
        data_dict = to_cuda(data_dict)

        nbr = precompute_neibors(data_dict["points"], data_dict["lengths"],
                                 cfg.backbone.num_stages, cfg.backbone.num_neighbors)
        data_dict.update(nbr)
        output_dict = model(data_dict)

        T_est = output_dict["estimated_transform"]
        pred_R = T_est[:3, :3].cpu().numpy()
        pred_t = T_est[:3, 3].cpu().numpy()

        src4 = apply_transformation(src_for_ov, pred_R, pred_t)
        _, _, corr = compute_overlap(ref_for_ov, src4, 1.5)
        # 分母用完整（未掩码）目标点云 ref_for_ov，而非源点云
        overlap = corr.shape[1] / len(ref_for_ov) if corr is not None and corr.size else 0.0

        if chain_pdb_path and os.path.exists(chain_pdb_path):
            try:
                pred_pdb = generate_output_pdb_path(
                    source_path, config_id, sampling_method, output_dir, mask_suffix)
                t_corrected = pred_t + (c_ref.astype(np.float32) - c_src.astype(np.float32))
                transform_pdb(chain_pdb_path, pred_R, t_corrected, pred_pdb)
                result["pred_pdb_path"] = pred_pdb
            except Exception as e:
                log.warning("PDB transform failed: %s", e)

        result.update({"overlap": overlap,
                      "ref_points": len(output_dict["ref_points"]),
                      "src_points": len(output_dict["src_points"])})

        data_dict = release_cuda(data_dict)
        output_dict = release_cuda(output_dict)
        torch.cuda.empty_cache()

    except Exception as e:
        result["error"] = str(e)
        log.error("process_single_pair error: %s", e)
        torch.cuda.empty_cache()

    return result


# ======================================================================
# Mask generation
# ======================================================================

def generate_masks_once(src_points, ref_points, mask_params, output_dir,
                        source_path, target_path):
    """Generate spherical masks (called once, reused for all configs)."""
    if not MASK_MODULE_AVAILABLE:
        raise ImportError("sw_mask not available")

    masks = generate_spherical_masks(
        source_data=source_path, target_data=target_path,
        mask_radius_factor=mask_params.get("mask_radius_factor", 1.35),
        min_coverage=mask_params.get("min_coverage", 0.15),
        min_point_distance_factor=mask_params.get("min_point_distance_factor", 0.32),
        verbose=mask_params.get("verbose", True))

    if len(masks) == 0:
        raise ValueError("No valid masks generated")

    masks_save_path = None
    if output_dir and mask_params.get("save_masks", False):
        temp_dir = os.path.join(output_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        save_masks(masks, temp_dir, save_individual_files=True)
        masks_save_path = temp_dir

    log.info("Generated %d spherical masks", len(masks))
    return masks, masks_save_path


# ======================================================================
# CLI
# ======================================================================

def make_parser():
    p = argparse.ArgumentParser(description="PARENet point cloud registration")
    p.add_argument("--target", "-t", default=None)
    p.add_argument("--source", "-s", default=None)
    p.add_argument("--chain_pdb", "-c", default=None)
    p.add_argument("--weights", "-w",
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "parenet", "weights", "epoch-18.pth.tar"))
    p.add_argument("--output_dir", "-o", default=None)
    p.add_argument("--seed", type=int, default=100000)
    p.add_argument("--configs", default="all")
    p.add_argument("--use_mask", action="store_true")
    p.add_argument("--mask_radius_factor", type=float, default=1.35)
    p.add_argument("--min_coverage", type=float, default=0.15)
    p.add_argument("--min_point_distance_factor", type=float, default=0.32)
    p.add_argument("--save_masks", action="store_true")
    p.add_argument("--server", action="store_true",
                   help="persistent server mode: load model once, serve stdin requests")
    return p


# ======================================================================
# Model singleton (load once, reuse across requests)
# ======================================================================

_MODEL = None
_CFG = None


def _get_model(weights):
    """Load the PARENet model once and cache it (model, cfg)."""
    global _MODEL, _CFG
    if _MODEL is None:
        _CFG = make_cfg()
        _MODEL = create_model(_CFG).cuda()
        state = torch.load(weights)
        _MODEL.load_state_dict(state["model"])
        _MODEL.eval()
        log.info("Model loaded (cached)")
    return _MODEL, _CFG


# ======================================================================
# Inference (one source->target fitting); uses the cached model
# ======================================================================

def run_inference(target, source, chain_pdb, output_dir, weights,
                  use_mask=True, configs="all", seed=100000,
                  mask_radius_factor=1.35, min_coverage=0.15,
                  min_point_distance_factor=0.32, stop_file=None):
    """Run PARENet inference for one source/target pair.

    Algorithm identical to the original main(); only parameterized so the model
    can be reused and an optional stop_file allows early termination (the
    in-process equivalent of killing the old subprocess).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if use_mask and not MASK_MODULE_AVAILABLE:
        log.error("Mask mode requested but sw_mask not available")
        return

    model_net, cfg = _get_model(weights)
    os.makedirs(output_dir, exist_ok=True)

    CONFIGS = VOXEL_SIZE_CONFIGS_MASK if use_mask else VOXEL_SIZE_CONFIGS_NORMAL
    point_limit = 70000 if use_mask else 70000
    if configs.lower() == "all":
        config_ids = list(CONFIGS.keys())
    else:
        config_ids = [int(x) for x in configs.split(",")]

    # preprocess
    src_data = preprocess_point_cloud_data(source, point_limit=point_limit)
    tgt_data = preprocess_point_cloud_data(target, point_limit=point_limit)

    # masks
    masks, masks_save_path, mask_data_list = None, None, []
    if use_mask:
        mask_params = {
            "mask_radius_factor": mask_radius_factor,
            "min_coverage": min_coverage,
            "min_point_distance_factor": min_point_distance_factor,
            "verbose": True, "save_masks": True,
        }
        masks, masks_save_path = generate_masks_once(
            source, target, mask_params, output_dir, source, target)
        for mf in find_mask_files(os.path.join(output_dir, "temp")):
            try:
                md = preprocess_point_cloud_data(mf, point_limit=None, is_mask_file=True)
                mask_data_list.append((mf, md))
            except Exception:
                continue

    def _stopped():
        return stop_file is not None and os.path.exists(stop_file)

    # inference
    all_results = []

    if use_mask and mask_data_list:
        optimal_results = []
        for mask_file, mask_data in mask_data_list:
            if _stopped():
                break
            mask_name = os.path.basename(mask_file)
            mask_results = []
            for cid in config_ids:
                if _stopped():
                    break
                for sm in ["voxel", "fps"]:
                    try:
                        suffix = os.path.splitext(mask_name)[0]
                        if suffix.startswith("mask"):
                            suffix = suffix[4:]
                        suffix = suffix[:10]
                        r = process_single_pair(
                            src_data, mask_data, source, mask_file,
                            chain_pdb, model_net, cfg, cid, sm,
                            output_dir, use_mask=False,
                            mask_suffix=suffix, original_target_data=tgt_data)
                        mask_results.append(r)
                        all_results.append(r)
                    except Exception as e:
                        log.error("Config %d-%s crashed: %s", cid, sm, e)
            best = process_single_mask_optimally(mask_results)
            if best:
                optimal_results.append(best)

        optimal_results.sort(key=lambda x: x.get("overlap") or 0, reverse=True)
        log.info("Optimal results per mask:")
        for i, r in enumerate(optimal_results, 1):
            pdb = os.path.basename(r.get("pred_pdb_path", "N/A"))
            log.info("  %d. %s overlap=%.6f", i, pdb, r.get("overlap", 0))
    else:
        for cid in config_ids:
            if _stopped():
                break
            for sm in ["voxel", "fps"]:
                try:
                    r = process_single_pair(
                        src_data, tgt_data, source, target,
                        chain_pdb, model_net, cfg, cid, sm,
                        output_dir, use_mask=use_mask,
                        masks=masks, masks_save_path=masks_save_path)
                    all_results.append(r)
                    if not r["error"]:
                        log.info("Config %d-%s: overlap=%.6f", cid, sm, r.get("overlap", 0))
                except Exception as e:
                    log.error("Config %d-%s crashed: %s", cid, sm, e)

        successful = [r for r in all_results if not r.get("error") and r.get("overlap") is not None]
        if successful:
            rename_pdb_files_by_ranking(successful)

    # summary
    ok = [r for r in all_results if not r.get("error") and r.get("overlap") is not None]
    fail = [r for r in all_results if r.get("error")]
    log.info("Total: %d, Success: %d, Failed: %d", len(all_results), len(ok), len(fail))
    if ok:
        best = max(ok, key=lambda r: r["overlap"])
        pred = best.get("pred_pdb_path")
        log.info("Best overlap: %.6f (%s)", best["overlap"],
                 os.path.basename(pred) if pred else "N/A")


# ======================================================================
# Persistent server: load model once, serve stdin JSON requests
# ======================================================================

def _server_loop(weights):
    """Read one JSON request per stdin line; signal completion via _DONE file.

    Request keys: target, source, chain_pdb, output_dir, use_mask, configs,
                  mask_radius_factor, min_point_distance_factor
    Completion / early-stop are communicated through files in output_dir
    (_PARENET_DONE / _PARENET_STOP) — stdout/stderr are NOT used for IPC,
    so the client never blocks on a full pipe.
    """
    _get_model(weights)  # preload once
    log.info("PARENet server ready (pid=%d)", os.getpid())
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            log.error("bad request: %s", e)
            continue
        output_dir = req["output_dir"]
        done_file = os.path.join(output_dir, "_PARENET_DONE")
        stop_file = os.path.join(output_dir, "_PARENET_STOP")
        try:
            run_inference(
                target=req["target"], source=req["source"],
                chain_pdb=req.get("chain_pdb"), output_dir=output_dir,
                weights=weights, use_mask=req.get("use_mask", True),
                configs=req.get("configs", "all"),
                seed=req.get("seed", 100000),
                mask_radius_factor=req.get("mask_radius_factor", 1.35),
                min_coverage=req.get("min_coverage", 0.15),
                min_point_distance_factor=req.get("min_point_distance_factor", 0.32),
                stop_file=stop_file)
        except Exception as e:
            log.error("request failed: %s", e)
        finally:
            try:
                with open(done_file, "w") as f:
                    f.write("done\n")
            except Exception:
                pass


def main():
    args = make_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    if args.server:
        _server_loop(args.weights)
        return

    # single-run CLI (backward compatible)
    if not args.target or not args.source:
        log.error("--target and --source are required (or use --server)")
        return
    output_dir = args.output_dir or os.path.dirname(args.source)
    run_inference(
        target=args.target, source=args.source, chain_pdb=args.chain_pdb,
        output_dir=output_dir, weights=args.weights, use_mask=args.use_mask,
        configs=args.configs, seed=args.seed,
        mask_radius_factor=args.mask_radius_factor, min_coverage=args.min_coverage,
        min_point_distance_factor=args.min_point_distance_factor)


if __name__ == "__main__":
    main()
