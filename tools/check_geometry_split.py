"""T04 验收工具：单侧几何构建（cloud_encoding）与旧联合入口逐阶段逐张量按位比较。

用法：
    # 真实输入（基准 manifest 里的一个目标掩码）
    python tools/check_geometry_split.py --manifest tests/cases/registration_manifest.json \
        --target-order 0

    # 合成小点云 / 不同点数（默认三组，自定义体素与邻居数）
    python tools/check_geometry_split.py --synthetic

比较范围：points / lengths / features / neighbors / subsampling / upsampling /
每侧节点分区（masks、knn_indices、knn_masks）/ transform 设备与 dtype。

说明：旧联合入口（`registration_collate_fn_stack_mode` + `precompute_neibors`）从 pareconv
的**模块级全局变量**读体素配置，而单侧路径显式传参。为了让两条路径用同一份配置，本工具在
对比期间临时写入 `data_mask.VOXEL_SIZE_CONFIGS[98]` 与 `CURRENT_CONFIG_ID`，结束后恢复原值；
生产路径（demo_mask）不碰这些全局变量。
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

from pareconv.modules.ops import point_to_node_partition
from pareconv.utils import data_mask
from pareconv.utils.data_mask import (registration_collate_fn_stack_mode,
                                      precompute_neibors)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from protassem.fitting.cloud_encoding import (build_geometry, join_geometries,
                                              attach_node_partition)
from protassem.fitting.parenet.config import make_cfg

TEMP_CONFIG_ID = 98      # 仅对比实验使用的临时体素配置槽位


def parse_cases(text):
    cases = []
    for item in text.split(","):
        ref, src = item.lower().split("x")
        cases.append((int(ref), int(src)))
    return cases


def same(left, right):
    if tuple(left.shape) != tuple(right.shape) or left.dtype != right.dtype:
        return False
    return left.detach().cpu().numpy().tobytes() == right.detach().cpu().numpy().tobytes()


def legacy_joint(ref_points, src_points, ref_feats, src_feats, transform, scale,
                 num_stages, voxel_sizes, sampling_method, num_neighbors):
    """旧联合入口：collate（CPU 下采样）+ to_cuda + precompute_neibors（CUDA k-NN）。"""
    saved_voxels = data_mask.VOXEL_SIZE_CONFIGS.get(TEMP_CONFIG_ID)
    saved_config = data_mask.CURRENT_CONFIG_ID
    saved_sampling = data_mask.CURRENT_SAMPLING_METHOD
    data_mask.VOXEL_SIZE_CONFIGS[TEMP_CONFIG_ID] = list(voxel_sizes)
    data_mask.CURRENT_CONFIG_ID = TEMP_CONFIG_ID
    data_mask.CURRENT_SAMPLING_METHOD = sampling_method
    try:
        data_dict = {
            "ref_points": ref_points.cpu().numpy(),
            "src_points": src_points.cpu().numpy(),
            "ref_feats": ref_feats.cpu().numpy(),
            "src_feats": src_feats.cpu().numpy(),
            "transform": transform.cpu(),
            "scale": scale,
        }
        joint = registration_collate_fn_stack_mode([data_dict], num_stages, 1.0,
                                                   num_neighbors, 1.8)
        joint = {key: ([item.cuda() for item in value] if isinstance(value, list)
                       else (value.cuda() if isinstance(value, torch.Tensor) else value))
                 for key, value in joint.items()}
        joint.update(precompute_neibors(joint["points"], joint["lengths"],
                                        num_stages, num_neighbors))
        return joint
    finally:
        if saved_voxels is None:
            data_mask.VOXEL_SIZE_CONFIGS.pop(TEMP_CONFIG_ID, None)
        else:
            data_mask.VOXEL_SIZE_CONFIGS[TEMP_CONFIG_ID] = saved_voxels
        data_mask.CURRENT_CONFIG_ID = saved_config
        data_mask.CURRENT_SAMPLING_METHOD = saved_sampling


def split_joint(ref_points, src_points, ref_feats, src_feats, transform, scale,
                voxel_sizes, sampling_method, num_neighbors, patch):
    ref_geometry = build_geometry(ref_points, ref_feats, voxel_sizes, sampling_method,
                                 num_neighbors)
    src_geometry = build_geometry(src_points, src_feats, voxel_sizes, sampling_method,
                                 num_neighbors)
    data_dict = join_geometries(ref_geometry, src_geometry, scale=scale, transform=transform)
    attach_node_partition(ref_geometry, patch)
    attach_node_partition(src_geometry, patch)
    return data_dict, {"ref": ref_geometry, "src": src_geometry}


def compare(label, joint, data_dict, geometries, num_stages, num_neighbors, patch):
    """逐项比较，返回 (通过?, 行列表)。"""
    lines = []
    ok = True
    ref_lens = [int(item[0].item()) for item in joint["lengths"]]

    def report(name, verdict, detail=""):
        nonlocal ok
        ok = ok and verdict
        lines.append("   %-28s %s%s" % (name, "一致" if verdict else "**不一致**",
                                        ("  " + detail) if detail else ""))

    report("features", same(joint["features"], data_dict["features"]))
    report("transform 设备/dtype",
           joint["transform"].device == data_dict["transform"].device
           and joint["transform"].dtype == data_dict["transform"].dtype,
           "%s/%s vs %s/%s" % (joint["transform"].device, joint["transform"].dtype,
                               data_dict["transform"].device, data_dict["transform"].dtype))
    for stage in range(num_stages):
        report("points[%d]" % stage, same(joint["points"][stage], data_dict["points"][stage]))
        report("lengths[%d]" % stage, same(joint["lengths"][stage], data_dict["lengths"][stage]))
        report("neighbors[%d]" % stage,
               same(joint["neighbors"][stage], data_dict["neighbors"][stage]))
        if stage < num_stages - 1:
            report("subsampling[%d]" % stage,
                   same(joint["subsampling"][stage], data_dict["subsampling"][stage]))
        # pareconv 的 upsampling 列表元素 j 对应 stage j+1
        if 0 < stage < num_stages - 1:
            report("upsampling[%d]" % (stage - 1),
                   same(joint["upsampling"][stage - 1], data_dict["upsampling"][stage - 1]))

    # 节点分区：单侧构建 vs 联合切片
    for side, key in (("ref", 0), ("src", 1)):
        fine = joint["points"][1][:, :3]
        coarse = joint["points"][-1][:, :3]
        ref_count = ref_lens[1]
        ref_coarse = ref_lens[-1]
        fine_slice = fine[:ref_count] if side == "ref" else fine[ref_count:]
        coarse_slice = coarse[:ref_coarse] if side == "ref" else coarse[ref_coarse:]
        _, masks, knn_indices, knn_masks = point_to_node_partition(
            fine_slice.contiguous(), coarse_slice.contiguous(), patch)
        partition = geometries[side].node_partition
        report("%s 分区 masks" % side, same(masks, partition.masks))
        report("%s 分区 knn_indices" % side, same(knn_indices, partition.knn_indices))
        report("%s 分区 knn_masks" % side, same(knn_masks, partition.knn_masks))

    lines.append("   阶段点数（ref/src，联合入口）：%s"
                 % [(int(joint["lengths"][i][0]), int(joint["lengths"][i][1]))
                    for i in range(num_stages)])
    lines.append("   阶段点数（ref/src，单侧构建）：%s"
                 % [(geometries["ref"].stage_counts[i], geometries["src"].stage_counts[i])
                    for i in range(num_stages)])
    padding_risk = [(side, stage, geometries[side].stage_counts[stage], num_neighbors[stage])
                    for side in ("ref", "src")
                    for stage in range(num_stages)
                    if geometries[side].stage_counts[stage] < num_neighbors[stage]]
    lines.append("   哨兵风险（阶段点数 < 邻居数 时 pointops 会 0 填充尾部槽位）：%s"
                 % (padding_risk or "无"))
    return ok, lines


def run_case(label, ref_points, src_points, num_neighbors, patch, voxel_sizes,
             sampling_method, num_stages):
    ref_feats = torch.ones((ref_points.shape[0], 1), dtype=torch.float32)
    src_feats = torch.ones((src_points.shape[0], 1), dtype=torch.float32)
    transform = torch.from_numpy(np.eye(4, dtype=np.float32))
    scale = float(max(ref_points[:, :3].norm(dim=1).max(), src_points[:, :3].norm(dim=1).max()))
    joint = legacy_joint(ref_points, src_points, ref_feats, src_feats, transform, scale,
                         num_stages, voxel_sizes, sampling_method, num_neighbors)
    data_dict, geometries = split_joint(ref_points, src_points, ref_feats, src_feats,
                                        transform, scale, voxel_sizes, sampling_method,
                                        num_neighbors, patch)
    ok, lines = compare(label, joint, data_dict, geometries, num_stages, num_neighbors, patch)
    print("== %s ==" % label)
    print("\n".join(lines))
    print("   判定：%s" % ("通过" if ok else "**未通过**"))
    print()
    return ok


def real_case(manifest_path, target_order, num_neighbors, patch, num_stages):
    from protassem.fitting.demo_mask import preprocess_point_cloud_data
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    target = [item for item in manifest["targets"] if item["order"] == target_order][0]
    src = preprocess_point_cloud_data(manifest["source"]["txt"], point_limit=70000)
    tgt = preprocess_point_cloud_data(target["txt"], point_limit=70000)
    ref_norm, src_norm = tgt.points.copy(), src.points.copy()
    ref_norm[:, :3] -= tgt.centroid
    src_norm[:, :3] -= src.centroid
    config_id, sampling = data_mask.get_current_config()
    voxel_sizes = data_mask.VOXEL_SIZE_CONFIGS.get(config_id,
                                                   data_mask.VOXEL_SIZE_CONFIGS[0])
    print("真实输入：%s（%d 点） / %s（%d 点）；生效配置 config_id=%d sampling=%s voxels=%s"
          % (os.path.basename(manifest["source"]["txt"]), ref_norm.shape[0],
             os.path.basename(target["txt"]), src_norm.shape[0], config_id, sampling,
             voxel_sizes))
    return run_case("manifest target %d" % target_order,
                    torch.from_numpy(ref_norm.astype(np.float32)),
                    torch.from_numpy(src_norm.astype(np.float32)),
                    num_neighbors, patch, voxel_sizes, sampling, num_stages)


def synthetic_case(ref_count, src_count, num_neighbors, patch, num_stages):
    torch.manual_seed(11)
    ref_points = (torch.rand(ref_count, 3) * 20.0 - 10.0)
    src_points = (torch.rand(src_count, 3) * 20.0 - 10.0)
    voxel_sizes = [0.5, 1.0, 2.0, 4.0][:num_stages]
    return run_case("合成 ref=%d src=%d" % (ref_count, src_count), ref_points, src_points,
                    num_neighbors, patch, voxel_sizes, "voxel", num_stages)


def main(argv=None):
    parser = argparse.ArgumentParser(description="T04 几何拆分等价性检查")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--target-order", type=int, default=0)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--cases", default="300x120,2048x96,96x2048")
    parser.add_argument("--num-neighbors", default=None,
                        help="逗号分隔的每阶段邻居数（默认取模型配置）")
    parser.add_argument("--patch", type=int, default=None,
                        help="节点分区每节点点数（默认取模型配置）")
    args = parser.parse_args(argv)

    cfg = make_cfg()
    num_stages = cfg.backbone.num_stages
    num_neighbors = ([int(item) for item in args.num_neighbors.split(",")]
                     if args.num_neighbors else list(cfg.backbone.num_neighbors))
    patch = args.patch if args.patch else cfg.model.num_points_in_patch
    if len(num_neighbors) != num_stages:
        raise SystemExit("num-neighbors 需要 %d 个值" % num_stages)

    torch.cuda.set_device(0)
    results = []
    if args.manifest:
        results.append(real_case(args.manifest, args.target_order, num_neighbors,
                                 patch, num_stages))
    if args.synthetic or not args.manifest:
        for ref_count, src_count in parse_cases(args.cases):
            results.append(synthetic_case(ref_count, src_count, num_neighbors, patch,
                                          num_stages))
    print("总判定：%s（%d/%d 组通过）"
          % ("通过" if all(results) else "**未通过**", sum(results), len(results)))
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
