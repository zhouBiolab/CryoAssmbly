"""T06 验收工具：单侧编码 + 双侧配准 vs 旧 `forward`（联合布局），逐层比较。

用法：
    # 真实输入（基准 manifest 的一个目标掩码）
    python tools/check_encoding_split.py --manifest tests/cases/registration_manifest.json \
        --target-order 0

    # 合成输入（不同点数，覆盖邻域/分区规模变化）
    python tools/check_encoding_split.py --synthetic

比较方式：
  1. 参考路径 = `PARE_Net.forward(join_geometries(ref, src))`（联合布局，原实现）；
  2. 拆分路径 = `encode_cloud()` ×2 + `register_pair()`；
  3. 对每个共有字段给出**按位是否一致**与 `max|Δ|`，按统一验收容差（atol=1e-6、rtol=1e-5）
     判定是否通过；只在一侧出现的字段单独列出（例如参考路径多出的输入 `transform`）；
  4. 另外把 `EncodedCloud` 的每个字段与参考路径中对应的输出字段逐项比较（编码层证据）。

位姿（`estimated_transform`）与候选相关量（`hypotheses`/`corr_scores`/`*_corr_points`）
位姿要求"一致"：工具对它们同时报出按位结果与最大偏差，由报告判定。
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from protassem.fitting.cloud_encoding import (build_geometry, join_geometries)
from protassem.fitting.parenet.model import create_model
from protassem.fitting.parenet.config import make_cfg
from protassem.runtime.config import DEFAULT_ALLOW_TF32, apply_tf32_policy

ATOL = 1e-6
RTOL = 1e-5
# 编码层字段 ↔ 参考 forward 输出字段（ref 侧；src 侧前缀不同）
ENCODING_FIELDS = (("points", "ref_points"), ("points_f", "ref_points_f"),
                   ("points_c", "ref_points_c"),
                   ("node_knn_indices", "ref_node_knn_indices"),
                   ("re_feats_c", "ref_feats_c_re"),
                   ("feats_f", "ref_feats_f"), ("re_feats_f", "re_ref_feats_f"),
                   ("m_scores", "m_ref_scores"))


def max_abs_delta(left, right):
    if tuple(left.shape) != tuple(right.shape) or left.dtype != right.dtype:
        return None
    if left.numel() == 0:
        return 0.0
    if left.dtype.is_floating_point:
        return float((left.double() - right.double()).abs().max().item())
    return float((left.long() - right.long()).abs().max().item()) if left.numel() else 0.0


def within_tolerance(delta, left, right=None):
    """统一验收容差：atol=1e-6、rtol=1e-5（逐元素 allclose 语义）。"""
    if delta is None or right is None:
        return False
    return bool(torch.allclose(left.float(), right.float(), atol=ATOL, rtol=RTOL))


def compare_fields(lines, title, reference, candidate):
    lines.append("## %s" % title)
    lines.append("")
    lines.append("| 字段 | 形状 | 按位一致 | max\\|Δ\\| | 容差内 |")
    lines.append("|---|---|---|---|---|")
    only_reference = sorted(set(reference) - set(candidate))
    only_candidate = sorted(set(candidate) - set(reference))
    passed = []
    for name in sorted(set(reference) & set(candidate)):
        left, right = reference[name], candidate[name]
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            continue
        same = tuple(left.shape) == tuple(right.shape) and \
            left.detach().cpu().numpy().tobytes() == right.detach().cpu().numpy().tobytes()
        delta = max_abs_delta(left, right)
        ok = within_tolerance(delta, left, right)
        passed.append((name, same, ok, delta))
        lines.append("| `%s` | %s | %s | %s | %s |"
                     % (name, tuple(left.shape), "是" if same else "否",
                        "—" if delta is None else "%.3g" % delta, "是" if ok else "否"))
    lines.append("")
    if only_reference:
        lines.append("- 仅参考路径有：%s" % ", ".join("`%s`" % item for item in only_reference))
    if only_candidate:
        lines.append("- 仅拆分路径有：%s" % ", ".join("`%s`" % item for item in only_candidate))
    exact = sum(1 for _, same, _, _ in passed if same)
    tolerant = sum(1 for _, _, ok, _ in passed if ok)
    pose = [(name, same, delta) for name, same, ok, delta in passed
            if name in ("estimated_transform", "hypotheses", "corr_scores",
                        "ref_corr_points", "src_corr_points")]
    lines.append("- 共有字段 %d 个：按位一致 %d 个，容差内 %d 个"
                 % (len(passed), exact, tolerant))
    for name, same, delta in pose:
        lines.append("- 位姿/候选 `%s`：按位一致=%s，max|Δ|=%s"
                     % (name, "是" if same else "否", "—" if delta is None else "%.3g" % delta))
    lines.append("")
    return tolerant == len(passed)


def abs_delta(left, right):
    if left is None or right is None:
        return float("nan")
    return float((left.double() - right.double()).abs().max().item())


def final_verdict(lines, reference, candidate):
    """最终判定（统一验收口径）：位姿按 1e-3（PDB 写出精度）、候选索引/掩码/原始点逐位一致。

    与"初始诊断容差"（特征 atol=1e-6/rtol=1e-5）分开表述：后者本节前面的表格已给出。
    """
    transform_delta = abs_delta(reference.get("estimated_transform"),
                                candidate.get("estimated_transform"))
    identity_fields = ("ref_node_corr_indices", "src_node_corr_indices",
                       "ref_node_corr_knn_points", "src_node_corr_knn_points",
                       "ref_node_corr_knn_masks", "src_node_corr_knn_masks",
                       "ref_node_knn_indices", "src_node_knn_indices",
                       "ref_points", "src_points")
    identical = [name for name in identity_fields
                 if name in reference and name in candidate
                 and reference[name].detach().cpu().numpy().tobytes()
                 == candidate[name].detach().cpu().numpy().tobytes()]
    lines.append("## 最终判定（统一验收口径）")
    lines.append("")
    lines.append("- 位姿 `estimated_transform` max|Δ| = %.3g（PDB 写出精度 1e-3）→ %s"
                 % (transform_delta, "通过" if transform_delta < 1e-3 else "**不通过**"))
    lines.append("- 候选索引/掩码/原始点逐位一致：%d/%d → %s"
                 % (len(identical), len(identity_fields),
                    "通过" if len(identical) == len(identity_fields) else "**不通过**"))
    lines.append("")
    return transform_delta < 1e-3 and len(identical) == len(identity_fields)


def run_case(label, ref_points, src_points, model, cfg, num_neighbors, voxel_sizes):
    device = next(model.parameters()).device
    ref_features = torch.ones((ref_points.shape[0], 1), dtype=torch.float32)
    src_features = torch.ones((src_points.shape[0], 1), dtype=torch.float32)
    scale = float(max(ref_points[:, :3].norm(dim=1).max(),
                      src_points[:, :3].norm(dim=1).max()))

    ref_geometry = build_geometry(ref_points, ref_features, voxel_sizes, "voxel",
                                  num_neighbors, device=device)
    src_geometry = build_geometry(src_points, src_features, voxel_sizes, "voxel",
                                  num_neighbors, device=device)

    lines = ["== %s ==" % label,
             "阶段点数 ref=%s src=%s，scale=%.6f"
             % (ref_geometry.stage_counts, src_geometry.stage_counts, scale), ""]

    # 参考路径：联合布局 + 原 forward
    data_dict = join_geometries(ref_geometry, src_geometry, scale=scale,
                                transform=torch.eye(4))
    with torch.no_grad():
        reference = model(data_dict)

    # 拆分路径：单侧编码 + 双侧配准
    with torch.no_grad():
        target_encoded = model.encode_cloud(ref_geometry, scale)
        source_encoded = model.encode_cloud(src_geometry, scale)
        candidate = model.register_pair(target_encoded, source_encoded)

    ok = compare_fields(lines, "输出字段（参考 forward vs 拆分路径）", reference, candidate)
    ok = final_verdict(lines, reference, candidate) and ok

    encoding_lines = []
    for side, encoded, prefix in (("目标(ref)", target_encoded, "ref"),
                                  ("源(src)", source_encoded, "src")):
        pairs = {}
        for field, reference_field in ENCODING_FIELDS:
            name = reference_field.replace("ref_", "%s_" % prefix, 1)
            if name in reference:
                pairs[field] = reference[name]
        values = {field: getattr(encoded, field) for field in pairs}
        ok = compare_fields(encoding_lines, "编码层字段（%s，初始诊断容差）" % side,
                            pairs, values) and ok
    lines.extend(encoding_lines)

    print("\n".join(lines))
    return ok


def real_case(manifest_path, target_order, model, cfg):
    from protassem.fitting.demo_mask import preprocess_point_cloud_data
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    target = [item for item in manifest["targets"] if item["order"] == target_order][0]
    src = preprocess_point_cloud_data(manifest["source"]["txt"], point_limit=70000)
    tgt = preprocess_point_cloud_data(target["txt"], point_limit=70000)
    ref_norm, src_norm = tgt.points.copy(), src.points.copy()
    ref_norm[:, :3] -= tgt.centroid
    src_norm[:, :3] -= src.centroid
    from pareconv.utils import data_mask
    config_id, sampling = data_mask.get_current_config()
    voxel_sizes = data_mask.VOXEL_SIZE_CONFIGS.get(config_id, data_mask.VOXEL_SIZE_CONFIGS[0])
    print("真实输入：%s（%d 点）/ %s（%d 点）"
          % (os.path.basename(manifest["source"]["txt"]), ref_norm.shape[0],
             os.path.basename(target["txt"]), src_norm.shape[0]))
    print("生效配置 config_id=%d sampling=%s voxels=%s" % (config_id, sampling, voxel_sizes))
    return run_case("manifest target %d" % target_order,
                    torch.from_numpy(ref_norm.astype(np.float32)),
                    torch.from_numpy(src_norm.astype(np.float32)),
                    model, cfg, list(cfg.backbone.num_neighbors), voxel_sizes)


def synthetic_case(ref_count, src_count, model, cfg, voxel_sizes):
    torch.manual_seed(17)
    ref_points = torch.rand(ref_count, 3) * 30.0 - 15.0
    src_points = torch.rand(src_count, 3) * 30.0 - 15.0
    return run_case("合成 ref=%d src=%d" % (ref_count, src_count), ref_points, src_points,
                    model, cfg, list(cfg.backbone.num_neighbors), voxel_sizes)


def main(argv=None):
    parser = argparse.ArgumentParser(description="T06 单侧编码/双侧配准等价性检查")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--target-order", type=int, default=0)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--cases", default="3000x2000,2000x3000")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--allow-tf32", action="store_true", default=DEFAULT_ALLOW_TF32,
                        help="允许 TF32（默认关闭；开启时同一比较会明显变差，见 T06 报告）")
    args = parser.parse_args(argv)

    policy = apply_tf32_policy(args.allow_tf32)
    print("TF32 策略：%s（默认关闭：TF32 使数值依赖张量形状）" % policy)
    cfg = make_cfg()
    weights = args.weights or os.path.join(PROJECT_ROOT, "protassem", "fitting", "parenet",
                                           "weights", "epoch-18.pth.tar")
    model = create_model(cfg).cuda()
    model.load_state_dict(torch.load(weights)["model"])
    model.eval()
    voxel_sizes = [2, 3.6, 6.48, 11.664]

    results = []
    if args.manifest:
        results.append(real_case(args.manifest, args.target_order, model, cfg))
    if args.synthetic or not args.manifest:
        for item in args.cases.split(","):
            ref_count, src_count = (int(value) for value in item.lower().split("x"))
            results.append(synthetic_case(ref_count, src_count, model, cfg,
                                          [value / 4.0 for value in voxel_sizes]))
    print("总判定：%s（%d/%d 组通过）"
          % ("通过" if all(results) else "**未通过**", sum(results), len(results)))
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
