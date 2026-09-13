"""T08 假设评分分块契约测试。

CPU 可跑（分块选择器是纯 torch 运算）：
  - `select_best_hypothesis` 在 chunk=0/1/3/全部 下的 best_index 与内点掩码**逐位一致**；
  - tie-break：内点数相同时取**首个**下标（与整批 argmax 相同）；
  - 空假设集合报错。
CUDA 用例（需要模型权重）：
  - 同一输入下 chunk=0 与 chunk=64/1 的 `register_pair` 输出（位姿、hypotheses、
    corr_scores、候选索引）逐位一致；chunk<=0 时 LGR/HP 仍是上游类。
"""

import json
import os
import unittest

import torch

from protassem.fitting.chunked_registration import (ChunkedHypothesisProposer,
                                                   ChunkedLocalGlobalRegistration)
from protassem.fitting.hypothesis_scoring import select_best_hypothesis
from protassem.fitting.parenet.config import make_cfg
from protassem.fitting.parenet.model import create_model
from protassem.runtime.config import apply_tf32_policy

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
        return json.load(handle)["dependencies"]["weights_path"]


class SelectBestHypothesisTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.ref = make_points(40, seed=1, extent=4.0)
        self.src = make_points(40, seed=2, extent=4.0)
        # 30 个假设：绕 z 轴不同角度 + 小幅平移
        transforms = []
        for index in range(30):
            angle = 0.1 * index
            rotation = torch.tensor([[torch.cos(torch.tensor(angle)), -torch.sin(torch.tensor(angle)), 0.0],
                                     [torch.sin(torch.tensor(angle)), torch.cos(torch.tensor(angle)), 0.0],
                                     [0.0, 0.0, 1.0]])
            transform = torch.eye(4)
            transform[:3, :3] = rotation
            transform[:3, 3] = torch.tensor([0.02 * index, 0.0, 0.0])
            transforms.append(transform)
        self.transforms = torch.stack(transforms)

    def test_chunk_sizes_match_full_path(self):
        reference_index, reference_mask = select_best_hypothesis(
            self.ref, self.src, self.transforms, acceptance_radius=0.2, chunk_size=0)
        for chunk in (1, 2, 3, 7, 29, 30, 64):
            index, mask = select_best_hypothesis(self.ref, self.src, self.transforms,
                                                 acceptance_radius=0.2, chunk_size=chunk)
            self.assertEqual(reference_index, index, "chunk=%d 的最优假设不同" % chunk)
            self.assertTrue(torch.equal(reference_mask, mask), "chunk=%d 的掩码不同" % chunk)

    def test_tie_break_keeps_first_index(self):
        # 两个恒等假设 → 内点数相同；再放一个更差的
        identity = torch.eye(4).unsqueeze(0)
        worse = torch.eye(4).unsqueeze(0).clone()
        worse[0, :3, 3] = torch.tensor([100.0, 0.0, 0.0])
        transforms = torch.cat([identity, identity.clone(), worse], dim=0)
        index, _ = select_best_hypothesis(self.ref, self.src, transforms, 0.2, chunk_size=0)
        self.assertEqual(0, index)
        for chunk in (1, 2):
            chunked, _ = select_best_hypothesis(self.ref, self.src, transforms, 0.2,
                                                chunk_size=chunk)
            self.assertEqual(index, chunked)

    def test_empty_transforms_rejected(self):
        with self.assertRaises(ValueError):
            select_best_hypothesis(self.ref, self.src, torch.zeros(0, 4, 4), 0.2, chunk_size=0)

    def test_chunked_classes_are_used_only_when_enabled(self):
        model_zero = create_model(make_cfg(), hypothesis_chunk=0)
        model_chunked = create_model(make_cfg(), hypothesis_chunk=64)
        self.assertIs(type(model_zero.combienrefistration.lgr).__name__,
                      "LocalGlobalRegistration")
        self.assertIs(type(model_zero.combienrefistration.hp).__name__, "HypothesisProposer")
        self.assertIsInstance(model_chunked.combienrefistration.lgr,
                              ChunkedLocalGlobalRegistration)
        self.assertIsInstance(model_chunked.combienrefistration.hp,
                              ChunkedHypothesisProposer)
        self.assertEqual(64, model_chunked.combienrefistration.lgr.chunk_size)
        self.assertEqual(64, model_chunked.combienrefistration.hp.chunk_size)
        # 共享 cfg 不被污染（make_cfg() 返回模块级单例）
        self.assertFalse(hasattr(make_cfg().fine_matching, "hypothesis_chunk"))


@unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA 与模型权重")
class ChunkedRegistrationEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        apply_tf32_policy(False)
        cls.cfg = make_cfg()
        cls.models = {}
        for chunk in (0, 64, 1):
            model = create_model(cls.cfg, hypothesis_chunk=chunk).cuda()
            model.load_state_dict(torch.load(weights_path())["model"])
            model.eval()
            model.num_points_in_patch = PATCH
            cls.models[chunk] = model

    def setUp(self):
        from protassem.fitting.cloud_encoding import build_geometry
        self.ref = build_geometry(make_points(3000, seed=7), torch.ones((3000, 1)),
                                  VOXELS, "voxel", NEIGHBORS)
        self.src = build_geometry(make_points(2000, seed=9), torch.ones((2000, 1)),
                                  VOXELS, "voxel", NEIGHBORS)
        self.scale = float(max(self.ref.points[0][:, :3].norm(dim=1).max(),
                               self.src.points[0][:, :3].norm(dim=1).max()))

    def _register(self, chunk):
        model = self.models[chunk]
        with torch.no_grad():
            target = model.encode_cloud(self.ref, self.scale)
            source = model.encode_cloud(self.src, self.scale)
            return model.register_pair(target, source)

    def test_chunked_outputs_are_bitwise_identical(self):
        reference = self._register(0)
        for chunk in (64, 1):
            candidate = self._register(chunk)
            for name in ("estimated_transform", "hypotheses", "corr_scores",
                         "ref_corr_points", "src_corr_points",
                         "ref_node_corr_indices", "src_node_corr_indices"):
                self.assertTrue(torch.equal(reference[name], candidate[name]),
                                "chunk=%d 的 %s 不一致" % (chunk, name))


if __name__ == "__main__":
    unittest.main()
