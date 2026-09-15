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
import time

# Add project root so protassem package is importable when run as subprocess
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pareconv.utils.data_mask as data_mask_module
from protassem.fitting.parenet.config import make_cfg
from protassem.fitting.parenet.model import create_model, INFERENCE_OUTPUT_FIELDS
from protassem.core.points_txt import read_point_cloud_file
from protassem.fitting.candidate_ledger import (CandidateLedgerWriter,
                                                MASK_ERROR_REASON)
from protassem.fitting.cloud_encoding import (acquire_geometry, join_geometries,
                                              EncodingCache, GeometryCache)
from protassem.fitting.parenet.model import model_fingerprint
from protassem.runtime.config import (DEFAULT_ALLOW_TF32, DEFAULT_ENCODING_CACHE_MB,
                                      DEFAULT_GEOMETRY_CACHE_MB, DEFAULT_HYPOTHESIS_CHUNK,
                                      DEFAULT_INFERENCE_MODE, DEFAULT_TAIL_PIPELINE,
                                      INFERENCE_MODES, apply_tf32_policy,
                                      effective_allow_tf32)
from protassem.runtime.tail_pipeline import TailPipeline

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

def effective_sampling_config():
    """当前真正生效的多尺度配置（体素尺寸列表 + 最后一层采样方式）。

    pareconv 的 `precompute_subsample` 从模块级全局变量读配置；本仓库没有任何地方设置它们，
    因此实际始终是 config 0 + voxel（T02 实测：标签 configs=all 时 6 个配置的输入完全相同）。
    T04 起单侧几何构建显式传参，这里只把生效值读出来，不修改全局状态。
    """
    config_id, sampling_method = data_mask_module.get_current_config()
    voxel_sizes = data_mask_module.VOXEL_SIZE_CONFIGS.get(
        config_id, data_mask_module.VOXEL_SIZE_CONFIGS[0])
    return [float(value) for value in voxel_sizes], sampling_method


@torch.no_grad()   # T03：推理全程关闭梯度，不建计算图（权重/数值路径不变）
def process_single_pair(src_data, tgt_data, source_path, target_path,
                        chain_pdb_path, model, cfg, config_id, sampling_method,
                        output_dir=None, use_mask=False, masks=None,
                        masks_save_path=None, mask_suffix=None,
                        original_target_data=None, geometry_cache=None,
                        inference_mode=DEFAULT_INFERENCE_MODE, encoding_cache=None,
                        tail_pipeline=None):
    """Run PARENet inference on one source-target pair.

    只有本函数调用模型；@torch.no_grad() 覆盖"单侧几何构建 → 编码 → 配准 → 后处理"全路径。
    geometry_cache：T05 的单侧几何缓存（None = 关闭；由服务进程或调用方显式拥有）。
    inference_mode：joint = 联合布局 + `forward`（默认，旧数值）；split = 单侧编码 + 双侧配准。
    encoding_cache：T07 的源编码缓存（仅 split 模式使用；None = 关闭）。
    tail_pipeline：T09 的 CPU 尾部流水线（None = 就地执行后处理与写盘）。
    """
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

        # T04：源/目标各自构建单侧几何（多尺度点 + 邻居），再由适配层拼成联合布局。
        # T05：单侧几何走有界 CPU 缓存（cache=None 或 fps 采样时不查不存，构建代码同一份）。
        # 生效采样配置来自 pareconv 的模块级全局变量（T02 实测：标签 configs=all 实际只跑
        # config 0 + voxel），这里显式读出、传参并记录，不再依赖隐式全局。
        voxel_sizes, effective_sampling = effective_sampling_config()
        num_neighbors = cfg.backbone.num_neighbors
        ref_features = torch.ones((ref_norm.shape[0], 1), dtype=torch.float32)
        src_features = torch.ones((src_norm.shape[0], 1), dtype=torch.float32)

        ref_acquired = acquire_geometry(
            geometry_cache, torch.from_numpy(ref_norm.astype(np.float32)), ref_features,
            voxel_sizes, effective_sampling, num_neighbors, centroid=c_ref)
        src_acquired = acquire_geometry(
            geometry_cache, torch.from_numpy(src_norm.astype(np.float32)), src_features,
            voxel_sizes, effective_sampling, num_neighbors, centroid=c_src)
        ref_geometry = ref_acquired.geometry
        src_geometry = src_acquired.geometry

        _record_timing(output_dir, "server_collate",
                       ref_acquired.collate_seconds + src_acquired.collate_seconds,
                       config_id=config_id, sampling=sampling_method)
        _record_timing(output_dir, "server_neighbors",
                       ref_acquired.neighbors_seconds + src_acquired.neighbors_seconds)
        _record_timing(output_dir, "server_cache_hit",
                       ref_acquired.hit_seconds + src_acquired.hit_seconds,
                       tgt_hit=ref_acquired.hit, src_hit=src_acquired.hit,
                       cacheable=ref_acquired.cacheable)
        _record_timing(output_dir, "server_cache_store",
                       ref_acquired.store_seconds + src_acquired.store_seconds)

        # T06：inference_mode="split" 时走单侧编码（可缓存）+ 双侧配准；"joint" 时走旧联合布局 + forward
        timing_sink = lambda stage, seconds: _record_timing(  # noqa: E731
            output_dir, stage, seconds, config_id=config_id, sampling=sampling_method)
        joint_reference = inference_mode == "joint"
        data_dict = None
        if joint_reference:
            _t_stage = time.perf_counter()
            data_dict = join_geometries(
                ref_geometry, src_geometry, scale=scale,
                transform=torch.from_numpy(np.eye(4, dtype=np.float32)))
            _record_timing(output_dir, "server_join", time.perf_counter() - _t_stage)

        # T02：输入指纹与有效参数（用于判断缓存可复用比例）
        import struct
        _record_timing(output_dir, "server_pair_info", 0.0,
                       config_id=config_id, sampling=sampling_method,
                       effective_sampling=effective_sampling,
                       voxel_sizes=[float(value) for value in voxel_sizes],
                       src_points=int(src_data.points.shape[0]),
                       tgt_points=int(tgt_data.points.shape[0]),
                       ref_radius=float(np.linalg.norm(ref_norm, axis=1).max()),
                       src_radius=float(np.linalg.norm(src_norm, axis=1).max()),
                       scale=float(scale),
                       scale_bits=struct.pack(">f", float(scale)).hex(),
                       inference_mode=inference_mode,
                       collate_stage_points=[int(item.shape[0]) for item in
                                             ref_geometry.points])

        if joint_reference:
            _t_stage = time.perf_counter()
            output_dict = model(data_dict, output_fields=INFERENCE_OUTPUT_FIELDS,
                                timing=timing_sink)
            _record_timing(output_dir, "server_forward", time.perf_counter() - _t_stage)
        else:
            _t_stage = time.perf_counter()
            ref_encoded = model.encode_cloud(ref_geometry, scale, timing=timing_sink)
            _record_timing(output_dir, "server_encode_tgt", time.perf_counter() - _t_stage,
                           config_id=config_id, sampling=sampling_method)
            # T07：源编码命中则跳过源 backbone（目标编码不缓存，只保留当前掩码结果）
            src_encoded, src_hit = None, False
            if encoding_cache is not None and encoding_cache.enabled:
                _t_stage = time.perf_counter()
                src_encoded = encoding_cache.get(src_geometry, scale)
                src_hit = src_encoded is not None
                _record_timing(output_dir, "server_encode_cache",
                               time.perf_counter() - _t_stage, src_hit=src_hit)
            if src_encoded is None:
                _t_stage = time.perf_counter()
                src_encoded = model.encode_cloud(src_geometry, scale, timing=timing_sink)
                _record_timing(output_dir, "server_encode_src",
                               time.perf_counter() - _t_stage,
                               config_id=config_id, sampling=sampling_method,
                               cached=False)
                if encoding_cache is not None and encoding_cache.enabled:
                    _t_stage = time.perf_counter()
                    encoding_cache.put(src_geometry, scale, src_encoded)
                    _record_timing(output_dir, "server_encode_store",
                                   time.perf_counter() - _t_stage)
            _t_stage = time.perf_counter()
            output_dict = model.register_pair(
                ref_encoded, src_encoded, output_fields=INFERENCE_OUTPUT_FIELDS,
                timing=timing_sink)
            _record_timing(output_dir, "server_register", time.perf_counter() - _t_stage,
                           src_cache_hit=src_hit)

        # 后处理计时必须从模型结束处开始，否则会与 server_forward 重叠相加（T02 偏差处理）
        _t_stage = time.perf_counter()
        _record_timing(output_dir, "server_mem_peak", 0.0,
                       allocated=torch.cuda.memory_allocated(),
                       reserved=torch.cuda.memory_reserved(),
                       max_allocated=torch.cuda.max_memory_allocated(),
                       max_reserved=torch.cuda.max_memory_reserved())

        T_est = output_dict["estimated_transform"]
        pred_R = T_est[:3, :3].cpu().numpy()
        pred_t = T_est[:3, 3].cpu().numpy()
        # 只保留下游必要的 CPU 结果（GPU 张量随 output_dict 在本函数返回后释放）
        ref_count = len(output_dict["ref_points"])
        src_count = len(output_dict["src_points"])

        def finish_pair():
            """T09 尾部任务：后处理 + 写盘（确定性，不消耗随机数）。"""
            _t_tail = time.perf_counter()
            src4 = apply_transformation(src_for_ov, pred_R, pred_t)
            _, _, corr = compute_overlap(ref_for_ov, src4, 1.5)
            # 分母用完整（未掩码）目标点云 ref_for_ov，而非源点云
            overlap_value = corr.shape[1] / len(ref_for_ov) \
                if corr is not None and corr.size else 0.0
            _record_timing(output_dir, "server_postprocess",
                           time.perf_counter() - _t_tail)

            _t_tail = time.perf_counter()
            if chain_pdb_path and os.path.exists(chain_pdb_path):
                try:
                    pred_pdb = generate_output_pdb_path(
                        source_path, config_id, sampling_method, output_dir, mask_suffix)
                    # 位姿在点云质心系求解：写出时以 c_src 为旋转中心、并把平移补到 c_ref
                    t_corrected = pred_t + (c_ref.astype(np.float32) - c_src.astype(np.float32))
                    transform_pdb(chain_pdb_path, pred_R, t_corrected, pred_pdb,
                                  center=c_src.astype(np.float32))
                    result["pred_pdb_path"] = pred_pdb
                except Exception as e:
                    log.warning("PDB transform failed: %s", e)
            _record_timing(output_dir, "server_write_pred",
                           time.perf_counter() - _t_tail)
            result.update({"overlap": overlap_value,
                           "ref_points": ref_count, "src_points": src_count})

        # T09：尾部交给预取 worker 与下一次配准的 GPU 工作重叠；tail_pipeline=None 时就地执行
        if tail_pipeline is not None:
            tail_pipeline.submit(finish_pair)
        else:
            finish_pair()

        # T03：不再递归 release_cuda（把每个张量都拷成 numpy）也不再逐对 empty_cache——
        # data_dict/output_dict 是本函数局部变量，返回即结束引用，显存由缓存分配器复用。
        _record_timing(output_dir, "server_mem_after", 0.0,
                       allocated=torch.cuda.memory_allocated(),
                       reserved=torch.cuda.memory_reserved(),
                       config_id=config_id, sampling=sampling_method)

    except Exception as e:
        result["error"] = str(e)
        log.error("process_single_pair error: %s", e)

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
    p.add_argument("--geometry-cache-mb", type=int, default=DEFAULT_GEOMETRY_CACHE_MB,
                   help="单侧几何 CPU 缓存容量（MiB，0 = 关闭；默认 %d）"
                        % DEFAULT_GEOMETRY_CACHE_MB)
    p.add_argument("--tail-pipeline", dest="tail_pipeline", action="store_true",
                   default=DEFAULT_TAIL_PIPELINE,
                   help="启用 CPU 尾部流水线（T09；后处理/写盘与下一次配准的 GPU 工作重叠）")
    p.add_argument("--no-tail-pipeline", dest="tail_pipeline", action="store_false",
                   help="关闭 CPU 尾部流水线（就地执行后处理与写盘）")
    p.add_argument("--hypothesis-chunk", type=int, default=DEFAULT_HYPOTHESIS_CHUNK,
                   help="位姿假设评分分块大小（T08；0 = 原整批路径，默认 %d）"
                        % DEFAULT_HYPOTHESIS_CHUNK)
    p.add_argument("--encoding-cache-mb", type=int, default=DEFAULT_ENCODING_CACHE_MB,
                   help="源编码缓存 GPU 预算（MiB，0 = 关闭；仅 split 模式；默认 %d）"
                        % DEFAULT_ENCODING_CACHE_MB)
    p.add_argument("--inference-mode", choices=INFERENCE_MODES, default=DEFAULT_INFERENCE_MODE,
                   help="推理路径：joint = 联合布局 + forward（默认，旧数值）；"
                        "split = 单侧编码 + 双侧配准（要求关闭 TF32）")
    p.add_argument("--allow-tf32", dest="allow_tf32", action="store_true", default=DEFAULT_ALLOW_TF32,
                   help="允许 TF32（默认跟随推理模式：joint 允许、split 不允许）")
    p.add_argument("--no-allow-tf32", dest="allow_tf32", action="store_false",
                   help="关闭 TF32（数值与张量形状无关；split 模式必须）")
    p.add_argument("--server", action="store_true",
                   help="persistent server mode: load model once, serve stdin requests")
    return p


# ======================================================================
# Model singleton (load once, reuse across requests)
# ======================================================================

_MODEL = None
_CFG = None


def _get_model(weights, allow_tf32, hypothesis_chunk=DEFAULT_HYPOTHESIS_CHUNK):
    """Load the PARENet model once and cache it (model, cfg).

    TF32 策略与 T08 假设分块都在**构造模型之前**设定（早于任何前向）；
    取值由调用方按 RuntimeConfig 解析，本函数只负责应用。
    """
    global _MODEL, _CFG
    if _MODEL is None:
        policy = apply_tf32_policy(allow_tf32)
        _CFG = make_cfg()
        _MODEL = create_model(_CFG, hypothesis_chunk=int(hypothesis_chunk)).cuda()
        state = torch.load(weights)
        _MODEL.load_state_dict(state["model"])
        _MODEL.eval()
        log.info("Model loaded (cached); TF32 policy: %s; hypothesis_chunk=%d",
                 policy, int(hypothesis_chunk))
    return _MODEL, _CFG


# ======================================================================
# Inference (one source->target fitting); uses the cached model
# ======================================================================

def run_inference(target, source, chain_pdb, output_dir, weights,
                  use_mask=True, configs="all", seed=100000,
                  mask_radius_factor=1.35, min_coverage=0.15,
                  min_point_distance_factor=0.32, stop_file=None,
                  geometry_cache=None, allow_tf32=DEFAULT_ALLOW_TF32,
                  inference_mode=DEFAULT_INFERENCE_MODE, encoding_cache=None,
                  hypothesis_chunk=DEFAULT_HYPOTHESIS_CHUNK,
                  tail_pipeline_enabled=DEFAULT_TAIL_PIPELINE, request_id=None):
    """Run PARENet inference for one source/target pair.

    Algorithm identical to the original main(); only parameterized so the model
    can be reused and an optional stop_file allows early termination (the
    in-process equivalent of killing the old subprocess).

    geometry_cache：T05 单侧几何缓存（None = 关闭）；由调用方显式传入并拥有。
    allow_tf32    ：TF32 策略（None = 跟随 inference_mode；由调用方解析）。
    inference_mode："joint"（联合布局 + forward，默认）或 "split"（单侧编码 + 双侧配准）。
    encoding_cache：T07 源编码缓存（None = 关闭；仅 split 模式使用）。
    hypothesis_chunk：T08 假设评分分块（0 = 原路径；模型构造前应用，需在首次加载时给出）。
    tail_pipeline_enabled：T09 CPU 尾部流水线开关（默认关闭；结果与顺序必须一致）。
    request_id    ：O6 候选台账的请求身份（客户端生成；缺省时服务端自造一个）。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if use_mask and not MASK_MODULE_AVAILABLE:
        log.error("Mask mode requested but sw_mask not available")
        return

    model_net, cfg = _get_model(weights, effective_allow_tf32(inference_mode, allow_tf32),
                                hypothesis_chunk)
    os.makedirs(output_dir, exist_ok=True)
    ledger = CandidateLedgerWriter(
        output_dir, request_id or "server-%d" % os.getpid())

    CONFIGS = VOXEL_SIZE_CONFIGS_MASK if use_mask else VOXEL_SIZE_CONFIGS_NORMAL
    point_limit = 70000 if use_mask else 70000
    if configs.lower() == "all":
        config_ids = list(CONFIGS.keys())
    else:
        config_ids = [int(x) for x in configs.split(",")]

    # preprocess
    _t_stage = time.perf_counter()
    src_data = preprocess_point_cloud_data(source, point_limit=point_limit)
    tgt_data = preprocess_point_cloud_data(target, point_limit=point_limit)
    _record_timing(output_dir, "server_preprocess", time.perf_counter() - _t_stage)

    # masks
    masks, masks_save_path, mask_data_list = None, None, []
    if use_mask:
        mask_params = {
            "mask_radius_factor": mask_radius_factor,
            "min_coverage": min_coverage,
            "min_point_distance_factor": min_point_distance_factor,
            "verbose": True, "save_masks": True,
        }
        _t_stage = time.perf_counter()
        masks, masks_save_path = generate_masks_once(
            source, target, mask_params, output_dir, source, target)
        _record_timing(output_dir, "server_masks", time.perf_counter() - _t_stage)
        _t_stage = time.perf_counter()
        for mf in find_mask_files(os.path.join(output_dir, "temp")):
            try:
                md = preprocess_point_cloud_data(mf, point_limit=None, is_mask_file=True)
                mask_data_list.append((mf, md))
            except Exception:
                continue
        _record_timing(output_dir, "server_mask_preprocess",
                       time.perf_counter() - _t_stage)

    def _stopped():
        return stop_file is not None and os.path.exists(stop_file)

    # inference
    all_results = []
    # T09：尾部流水线（深度 1）。所有"消费结果"的位置必须先 wait()，见下面各点
    tail_pipeline = TailPipeline(enabled=tail_pipeline_enabled, name="parenet-tail")
    tail_wait_seconds = 0.0

    def _drain_tail(stage):
        nonlocal tail_wait_seconds
        _t_wait = time.perf_counter()
        tail_pipeline.wait()
        waited = time.perf_counter() - _t_wait
        tail_wait_seconds += waited
        if waited > 0.001:
            _record_timing(output_dir, "server_tail_wait", waited, where=stage)

    ledger_errors = {"count": 0, "first": None}
    skipped_mask_errors = 0

    def _publish(candidate_id, results, name=None, overlap=None, source=None,
                 mask_level=False):
        """发布一个候选的终态（O6）：ok / filtered / error 三态必须显式。

        **有成功候选就发布 ok** —— 同一个 mask 内个别评估抛异常不再掩盖已经得到的成功
        结果；失败记录仍保留在结果集里参与统计。没有成功候选且有执行错误时发布 error：
        `mask_level=True`（掩码级评估）带 `reason=MASK_ERROR_REASON`，客户端跳过它且不会
        让请求级结束状态变成 error；其余执行失败计入 `ledger_errors`，请求级仍报 error。
        """
        nonlocal skipped_mask_errors
        errors = [r.get("error") for r in results if r.get("error")]
        if name and os.path.exists(name):
            ledger.publish(candidate_id, "ok", name=name, overlap=overlap,
                           source=source)
            return
        if errors:
            if mask_level:
                skipped_mask_errors += 1
                ledger.publish(candidate_id, "error", source=source,
                               error=errors[0], reason=MASK_ERROR_REASON)
            else:
                ledger_errors["count"] += 1
                if ledger_errors["first"] is None:
                    ledger_errors["first"] = errors[0]
                ledger.publish(candidate_id, "error", source=source,
                               error=errors[0])
            return
        ledger.publish(candidate_id, "filtered", source=source,
                       reason="no valid prediction")

    failure = None
    try:
        if use_mask and mask_data_list:
            optimal_results = []
            for mask_index, (mask_file, mask_data) in enumerate(mask_data_list):
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
                                mask_suffix=suffix, original_target_data=tgt_data,
                                geometry_cache=geometry_cache,
                                inference_mode=inference_mode,
                                encoding_cache=encoding_cache,
                                tail_pipeline=tail_pipeline)
                            mask_results.append(r)
                            all_results.append(r)
                        except Exception as e:
                            log.error("Config %d-%s crashed: %s", cid, sm, e)
                            # 异常在掩码结果集与全局结果集**各记一次**（不重复追加）：
                            # 台账里的候选终态由 _publish 统一决定。
                            failure_record = {"error": str(e), "config_id": cid,
                                              "sampling_method": sm}
                            mask_results.append(failure_record)
                            all_results.append(failure_record)
                _drain_tail("mask_best")      # 消费本掩码结果前必须等尾部完成
                best = process_single_mask_optimally(mask_results)
                # O6：id = 掩码在 mask_data_list 中的序号（稳定生成顺序）；
                # 改名/删除已在上一步完成，此处发布的名字不会再变
                _publish(mask_index, mask_results,
                         name=(best or {}).get("pred_pdb_path"),
                         overlap=(best or {}).get("overlap"),
                         source={"mask": mask_name,
                                 "config": (best or {}).get("config_id"),
                                 "sampling": (best or {}).get("sampling_method")},
                         mask_level=True)
                if best:
                    optimal_results.append(best)

            _drain_tail("sort")
            optimal_results.sort(key=lambda x: x.get("overlap") or 0, reverse=True)
            log.info("Optimal results per mask:")
            for i, r in enumerate(optimal_results, 1):
                pdb = os.path.basename(r.get("pred_pdb_path", "N/A"))
                log.info("  %d. %s overlap=%.6f", i, pdb, r.get("overlap", 0))
        else:
            logged = 0
            pending = []          # (source, result) 按生成顺序；**改名之后**才发布
            for cid in config_ids:
                if _stopped():
                    break
                for sm in ["voxel", "fps"]:
                    try:
                        r = process_single_pair(
                            src_data, tgt_data, source, target,
                            chain_pdb, model_net, cfg, cid, sm,
                            output_dir, use_mask=use_mask,
                            masks=masks, masks_save_path=masks_save_path,
                            geometry_cache=geometry_cache,
                            inference_mode=inference_mode,
                            encoding_cache=encoding_cache,
                            tail_pipeline=tail_pipeline)
                        all_results.append(r)
                    except Exception as e:
                        log.error("Config %d-%s crashed: %s", cid, sm, e)
                        r = {"error": str(e)}
                    # O6：无掩码分支用自己的稳定生成顺序（config 外层、sampling 内层）编号；
                    # P1-4：这里只登记，发布推迟到 rename 之后（见循环外）
                    pending.append(({"config": cid, "sampling": sm}, r))
                _drain_tail("ranking")        # 排名与日志都要读结果
                while logged < len(all_results):
                    done = all_results[logged]
                    logged += 1
                    if not done.get("error") and done.get("overlap") is not None:
                        log.info("Config %s-%s: overlap=%.6f", done.get("config_id"),
                                 done.get("sampling_method"), done["overlap"])

            _drain_tail("rename")
            successful = [r for r in all_results if not r.get("error")
                          and r.get("overlap") is not None]
            if successful:
                rename_pdb_files_by_ranking(successful)   # 会回写 pred_pdb_path
            # P1-4：发布必须发生在**尾部任务完成且最终命名之后** —— 台账里的名字此后不再变化；
            # 旧实现边算边发布，改名后台账指向失效文件，tail 开启时还可能把未写出的候选记成 filtered。
            for candidate_id, (candidate_source, result) in enumerate(pending):
                _publish(candidate_id, [result], name=result.get("pred_pdb_path"),
                         overlap=result.get("overlap"), source=candidate_source)

        # summary（读结果前必须等尾部完成）
        _drain_tail("summary")
    except BaseException as exc:
        failure = exc
        raise
    finally:
        try:
            tail_pipeline.close()
        except Exception as e:
            log.error("尾部流水线关闭失败：%s", e)
        # O6：结束记录是唯一权威的结束标记；失败必须显式，不能被当成正常结束。
        # 优先级（明确写死，不靠默认）：
        #   1) 请求本身抛异常               -> error
        #   2) 客户端主动早停               -> cancelled（客户端已停止消费，其后候选不再有意义）
        #   3) 存在**非掩码级**候选执行失败 -> error（请求跑完了但产出不完整）
        #   4) 其余                         -> ok
        # 掩码级评估失败在 _publish 里带 reason=MASK_ERROR_REASON 发布，客户端跳过它，
        # 因此不进入 ledger_errors，也不改变本状态。
        try:
            if failure is not None:
                ledger.finish("error", error="%s: %s" % (type(failure).__name__, failure))
            elif _stopped():
                ledger.finish("cancelled")
            elif ledger_errors["count"]:
                ledger.finish("error",
                              error="%d 个候选执行失败，首个：%s"
                                    % (ledger_errors["count"], ledger_errors["first"]))
            else:
                ledger.finish("ok")
        except Exception as e:
            log.error("候选台账结束记录写入失败：%s", e)
    if tail_pipeline.enabled:
        _record_timing(output_dir, "server_tail_stats", 0.0,
                       submitted=tail_pipeline.submitted, completed=tail_pipeline.completed,
                       waited_seconds=round(tail_wait_seconds, 6))
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

def _record_timing(output_dir, stage, elapsed_s, **fields):
    """把服务端阶段耗时追加到请求输出目录的 server_timing.jsonl。

    父进程（客户端）另有一套计时，两者覆盖的时间区间会重叠，报告中分别展示、
    不做相加。写计时失败不影响推理本身。
    """
    row = {"pid": os.getpid(), "stage": stage,
           "elapsed_s": round(float(elapsed_s), 6), "timestamp": time.time()}
    row.update(fields)
    try:
        with open(os.path.join(output_dir, "server_timing.jsonl"), "a",
                  encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
    except OSError as exc:
        log.warning("server timing write failed: %s", exc)


def _server_loop(weights, geometry_cache_mb=DEFAULT_GEOMETRY_CACHE_MB,
                 inference_mode=DEFAULT_INFERENCE_MODE, allow_tf32=DEFAULT_ALLOW_TF32,
                 encoding_cache_mb=DEFAULT_ENCODING_CACHE_MB,
                 hypothesis_chunk=DEFAULT_HYPOTHESIS_CHUNK,
                 tail_pipeline=DEFAULT_TAIL_PIPELINE):
    """Read one JSON request per stdin line; signal completion via _DONE file.

    Request keys: target, source, chain_pdb, output_dir, use_mask, configs,
                  mask_radius_factor, min_point_distance_factor
    Completion / early-stop are communicated through files in output_dir
    (_PARENET_DONE / _PARENET_STOP) — stdout/stderr are NOT used for IPC,
    so the client never blocks on a full pipe.

    T05：几何缓存由**本服务进程拥有**，跨请求复用（容量以 MiB 计，0 = 关闭）。
    T06：推理路径与 TF32 策略在加载模型时设定（默认 joint + 框架默认精度，即旧数值）。
    T07：源编码缓存（GPU，仅 split 模式）同样由本进程拥有、跨请求复用。
    """
    resolved_tf32 = effective_allow_tf32(inference_mode, allow_tf32)
    model, _cfg = _get_model(weights, resolved_tf32, hypothesis_chunk)  # preload once
    geometry_cache = GeometryCache(int(geometry_cache_mb) * 1024 * 1024) \
        if geometry_cache_mb else None
    encoding_cache = None
    if inference_mode == "split" and encoding_cache_mb:
        fingerprint = model_fingerprint(model)
        encoding_cache = EncodingCache(int(encoding_cache_mb) * 1024 * 1024, fingerprint)
        log.info("Encoding cache enabled: %d MiB, model fingerprint %s…",
                 encoding_cache_mb, fingerprint[:12])
    log.info("PARENet server ready (pid=%d, geometry_cache_mb=%s, inference_mode=%s, "
             "allow_tf32=%s, encoding_cache_mb=%s, hypothesis_chunk=%d)",
             os.getpid(), geometry_cache_mb, inference_mode, resolved_tf32,
             encoding_cache_mb if encoding_cache is not None else 0, int(hypothesis_chunk))
    _last_request_end = time.perf_counter()
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
        _request_started = time.perf_counter()
        _record_timing(output_dir, "server_queue_wait",
                       _request_started - _last_request_end)
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
                stop_file=stop_file, geometry_cache=geometry_cache,
                allow_tf32=resolved_tf32,
                inference_mode=req.get("inference_mode") or inference_mode,
                encoding_cache=encoding_cache, hypothesis_chunk=hypothesis_chunk,
                tail_pipeline_enabled=tail_pipeline,
                request_id=req.get("request_id"))
        except Exception as e:
            log.error("request failed: %s", e)
        finally:
            try:
                with open(done_file, "w") as f:
                    f.write("done\n")
            except Exception:
                pass
        _last_request_end = time.perf_counter()
        _record_timing(output_dir, "server_request_total",
                       _last_request_end - _request_started)
        if geometry_cache is not None:
            _record_timing(output_dir, "server_cache_stats", 0.0,
                           **geometry_cache.snapshot())
        if encoding_cache is not None:
            _record_timing(output_dir, "server_encoding_cache_stats", 0.0,
                           **encoding_cache.snapshot())


def main():
    args = make_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    if args.server:
        _server_loop(args.weights, args.geometry_cache_mb, args.inference_mode,
                     args.allow_tf32, args.encoding_cache_mb, args.hypothesis_chunk,
                     args.tail_pipeline)
        return

    # single-run CLI (backward compatible)
    if not args.target or not args.source:
        log.error("--target and --source are required (or use --server)")
        return
    output_dir = args.output_dir or os.path.dirname(args.source)
    geometry_cache = GeometryCache(args.geometry_cache_mb * 1024 * 1024) \
        if args.geometry_cache_mb else None
    encoding_cache = None
    if args.inference_mode == "split" and args.encoding_cache_mb:
        resolved_tf32 = effective_allow_tf32(args.inference_mode, args.allow_tf32)
        model_net, _cfg = _get_model(args.weights, resolved_tf32, args.hypothesis_chunk)
        encoding_cache = EncodingCache(args.encoding_cache_mb * 1024 * 1024,
                                       model_fingerprint(model_net))
    run_inference(
        target=args.target, source=args.source, chain_pdb=args.chain_pdb,
        output_dir=output_dir, weights=args.weights, use_mask=args.use_mask,
        configs=args.configs, seed=args.seed,
        mask_radius_factor=args.mask_radius_factor, min_coverage=args.min_coverage,
        min_point_distance_factor=args.min_point_distance_factor,
        geometry_cache=geometry_cache, allow_tf32=args.allow_tf32,
        inference_mode=args.inference_mode, encoding_cache=encoding_cache,
        hypothesis_chunk=args.hypothesis_chunk,
        tail_pipeline_enabled=args.tail_pipeline)


if __name__ == "__main__":
    main()
