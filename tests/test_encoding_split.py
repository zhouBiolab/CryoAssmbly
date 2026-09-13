"""T06 单侧编码/双侧配准契约测试。

CPU 可跑：
  - `backbone_input()` 的键与 upsampling 紧凑约定（元素 j ↔ stage j+1）、scale 张量；
  - `apply_tf32_policy()` 设定并回读后端标志；
  - `node_partition()` 的阶段数契约（<2 阶段报错）。
CUDA 用例（需要模型权重，`torch.cuda.is_available()` 为假时跳过）：
  - 单侧节点分区与 pareconv `point_to_node_partition` 逐位一致；
  - `register_pair(encode_cloud(ref), encode_cloud(src))` 与旧 `forward(join(...))`
    在 TF32 关闭下等价：位姿 max|Δ| < 1e-3、候选索引/掩码/原始点逐位一致、分数在容差内。
    逐层完整比较由 `tools/check_encoding_split.py` 给出（本测试是防回归的最小版本）。
"""

import json
import os
import unittest

import torch

from protassem.fitting.cloud_encoding import (CloudGeometry, GeometryError,
                                              attach_node_partition, backbone_input,
                                              build_geometry, join_geometries, node_partition)
from protassem.fitting.parenet.model import create_model
from protassem.fitting.parenet.config import make_cfg
from protassem.runtime.config import (DEFAULT_ALLOW_TF32, RuntimeConfig, apply_tf32_policy,
                                      effective_allow_tf32)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "tests", "cases", "registration_manifest.json")
VOXELS = (0.5, 1.0, 2.0, 4.0)
NEIGHBORS = (8, 8, 8, 8)
PATCH = 4


def make_points(count, seed=5, extent=20.0):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(count, 3, generator=generator) * extent - extent / 2.0


def weights_path():
    with open(MANIFEST_PATH, encoding="utf-8") as handle:
        manifest = json.load(handle)
    return manifest["dependencies"]["weights_path"]


class BackboneInputTest(unittest.TestCase):
    def test_keys_and_upsampling_convention(self):
        # 手工几何（CPU）：stage 0/3 的 upsampling 为 None，只有 stage 1/2 有值
        stage_counts = (20, 10, 5, 3)
        geometry = CloudGeometry(
            points=[make_points(count, seed=index) for index, count in enumerate(stage_counts)],
            lengths=[torch.tensor([count]) for count in stage_counts],
            features=torch.ones((stage_counts[0], 1)), voxel_sizes=VOXELS,
            sampling_method="voxel",
            upsampling=[None, torch.zeros(10, 1, dtype=torch.long),
                        torch.zeros(5, 1, dtype=torch.long), None])
        payload = backbone_input(geometry, 7.5, device="cpu")
        self.assertEqual({"points", "neighbors", "subsampling", "upsampling", "scale"},
                         set(payload))
        # pareconv 约定：upsampling 元素 j 对应 stage j+1（stage 0 无）
        self.assertEqual(geometry.num_stages - 2, len(payload["upsampling"]))
        for index, item in enumerate(payload["upsampling"]):
            self.assertTrue(torch.equal(geometry.upsampling[index + 1], item))
        self.assertIsInstance(payload["scale"], torch.Tensor)
        self.assertAlmostEqual(7.5, float(payload["scale"].item()), places=5)

    def test_node_partition_requires_two_stages(self):
        geometry = CloudGeometry(points=[make_points(20)], lengths=[torch.tensor([20])],
                                 features=torch.ones((20, 1)), voxel_sizes=(1.0,),
                                 sampling_method="voxel", upsampling=[None])
        with self.assertRaises(GeometryError):
            node_partition(geometry, PATCH)


class Tf32PolicyTest(unittest.TestCase):
    def test_policy_sets_and_reports_flags(self):
        original = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
        try:
            disabled = apply_tf32_policy(False)
            self.assertEqual({"cudnn_allow_tf32": False, "matmul_allow_tf32": False}, disabled)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            enabled = apply_tf32_policy(True)
            self.assertEqual({"cudnn_allow_tf32": True, "matmul_allow_tf32": True}, enabled)
        finally:
            torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = original

    def test_effective_policy_follows_inference_mode(self):
        # None = 跟随推理模式：joint 保持框架默认（旧数值），split 强制关闭
        self.assertTrue(DEFAULT_ALLOW_TF32 is None)
        self.assertTrue(effective_allow_tf32("joint", None))
        self.assertFalse(effective_allow_tf32("split", None))
        self.assertFalse(effective_allow_tf32("joint", False))
        self.assertTrue(effective_allow_tf32("joint", True))

    def test_split_with_tf32_is_rejected(self):
        with self.assertRaises(ValueError):
            effective_allow_tf32("split", True)
        with self.assertRaises(ValueError):
            RuntimeConfig(inference_mode="split", allow_tf32=True)
        with self.assertRaises(ValueError):
            effective_allow_tf32("unknown", None)

    def test_runtime_config_defaults(self):
        config = RuntimeConfig()
        self.assertEqual("joint", config.inference_mode)
        self.assertTrue(config.tf32())
        split = RuntimeConfig(inference_mode="split")
        self.assertFalse(split.tf32())


@unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA 与模型权重")
class EncodingSplitEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = apply_tf32_policy(False)
        cfg = make_cfg()
        cls.cfg = cfg
        cls.model = create_model(cfg).cuda()
        cls.model.load_state_dict(torch.load(weights_path())["model"])
        cls.model.eval()

    def setUp(self):
        self.model.num_points_in_patch = PATCH
        self.ref = build_geometry(make_points(3000, seed=7), torch.ones((3000, 1)),
                                  VOXELS, "voxel", NEIGHBORS)
        self.src = build_geometry(make_points(2000, seed=9), torch.ones((2000, 1)),
                                  VOXELS, "voxel", NEIGHBORS)
        self.scale = float(max(self.ref.points[0][:, :3].norm(dim=1).max(),
                               self.src.points[0][:, :3].norm(dim=1).max()))

    def test_node_partition_matches_reference_helper(self):
        from pareconv.modules.ops import point_to_node_partition
        geometry = self.ref
        partition = node_partition(geometry, PATCH)
        _, masks, knn_indices, knn_masks = point_to_node_partition(
            geometry.points[1][:, :3].contiguous(), geometry.points[-1][:, :3].contiguous(),
            PATCH)
        self.assertTrue(torch.equal(masks, partition.masks))
        self.assertTrue(torch.equal(knn_indices, partition.knn_indices))
        self.assertTrue(torch.equal(knn_masks, partition.knn_masks))
        attached = attach_node_partition(geometry, PATCH)
        self.assertTrue(torch.equal(attached.node_partition.knn_indices, knn_indices))

    def test_split_matches_joint_forward(self):
        data_dict = join_geometries(self.ref, self.src, scale=self.scale, transform=torch.eye(4))
        with torch.no_grad():
            reference = self.model(data_dict)
            target = self.model.encode_cloud(self.ref, self.scale)
            source = self.model.encode_cloud(self.src, self.scale)
            candidate = self.model.register_pair(target, source)

        pose_delta = float((reference["estimated_transform"].double()
                            - candidate["estimated_transform"].double()).abs().max().item())
        self.assertLess(pose_delta, 1e-3)
        for name in ("ref_node_corr_indices", "src_node_corr_indices",
                     "ref_node_corr_knn_points", "src_node_corr_knn_points",
                     "ref_node_corr_knn_masks", "src_node_corr_knn_masks",
                     "ref_points", "src_points"):
            self.assertTrue(torch.equal(reference[name], candidate[name]),
                            "%s 应逐位一致" % name)
        self.assertTrue(torch.allclose(reference["corr_scores"], candidate["corr_scores"],
                                       atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(reference["matching_scores"],
                                       candidate["matching_scores"], atol=1e-6, rtol=1e-5))

    def test_register_pair_rejects_scale_mismatch(self):
        target = self.model.encode_cloud(self.ref, self.scale)
        source = self.model.encode_cloud(self.src, self.scale + 1.0)
        with self.assertRaises(ValueError):
            self.model.register_pair(target, source)

    def test_register_pair_rejects_training_mode(self):
        target = self.model.encode_cloud(self.ref, self.scale)
        source = self.model.encode_cloud(self.src, self.scale)
        self.model.train()
        try:
            with self.assertRaises(RuntimeError):
                self.model.register_pair(target, source)
        finally:
            self.model.eval()


if __name__ == "__main__":
    unittest.main()
