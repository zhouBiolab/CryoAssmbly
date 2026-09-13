"""T07 源编码缓存契约测试。

CPU 可跑：
  - `encoding_cache_key()` 的成分：几何指纹、精确 scale 位模式、权重/配置指纹、dtype、编码版本；
  - `object_tensor_bytes()` 的递归字节计费；
  - `EncodingCache` 的命中/失效/容量语义（基于 T05 的 `ByteLruCache`）。
CUDA 用例（需要模型权重）：
  - 同 scale 命中、不同 scale 失效、旋转不改缓存（换点集 → 失效，且结果仍与重算一致）；
  - 缓存命中与未命中的 `register_pair` 输出逐位一致（候选与最终输出等价）。
"""

import json
import os
import unittest
from dataclasses import dataclass

import torch

from protassem.fitting.cloud_encoding import (EncodingCache, build_geometry,
                                              build_stage_points, encoding_cache_key,
                                              object_tensor_bytes)
from protassem.fitting.parenet.config import make_cfg
from protassem.fitting.parenet.model import create_model, model_fingerprint
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
        manifest = json.load(handle)
    return manifest["dependencies"]["weights_path"]


@dataclass
class _FakeEncoded:
    features: torch.Tensor
    scores: torch.Tensor


class KeyCompositionTest(unittest.TestCase):
    def key(self, geometry="g0", scale=1.5, model="m0", dtype="torch.float32", version=1):
        return encoding_cache_key(geometry, scale, model, dtype, version)

    def test_same_inputs_same_key(self):
        self.assertEqual(self.key(), self.key())

    def test_each_component_invalidates(self):
        base = self.key()
        self.assertNotEqual(base, self.key(geometry="g1"))
        self.assertNotEqual(base, self.key(scale=1.5000001))
        self.assertNotEqual(base, self.key(model="m1"))
        self.assertNotEqual(base, self.key(dtype="torch.float64"))
        self.assertNotEqual(base, self.key(version=2))

    def test_exact_scale_bits(self):
        # 单精度下 1.5 与 1.5000001 的位模式不同 → 不同 key（无分桶、无近似）
        self.assertNotEqual(encoding_cache_key("g", 1.5, "m", "d", 1),
                            encoding_cache_key("g", 1.5000001, "m", "d", 1))
        # 同一 float32 值的两种写法（1.5 与 float32(1.5)）必须同 key
        self.assertEqual(encoding_cache_key("g", 1.5, "m", "d", 1),
                         encoding_cache_key("g", torch.tensor(1.5, dtype=torch.float32).item(),
                                            "m", "d", 1))

    def test_key_is_hex_sha256(self):
        digest = self.key()
        self.assertEqual(64, len(digest))
        int(digest, 16)


class ByteAccountingTest(unittest.TestCase):
    def test_dataclass_tensor_bytes(self):
        value = _FakeEncoded(features=torch.zeros(10, 4), scores=torch.zeros(10))
        self.assertEqual(10 * 4 * 4 + 10 * 4, object_tensor_bytes(value))

    def test_nested_and_empty(self):
        geometry = type("G", (), {})()
        self.assertEqual(0, object_tensor_bytes(geometry))
        self.assertEqual(0, object_tensor_bytes(None))
        self.assertEqual(8, object_tensor_bytes(torch.zeros(2, dtype=torch.float32)))


class EncodingCacheSemanticsTest(unittest.TestCase):
    def setUp(self):
        self.cache = EncodingCache(4096, model_fingerprint="m0", name="test")
        # 只测 key/计费/容量语义，不需要 k-NN → 用 CPU 多尺度点（build_stage_points）
        self.geometry = build_stage_points(make_points(200), torch.ones((200, 1)), VOXELS,
                                           "voxel", device="cpu")
        self.encoded = _FakeEncoded(features=torch.zeros(4, 4), scores=torch.zeros(4))

    def test_put_get_roundtrip(self):
        self.assertTrue(self.cache.enabled)
        self.assertTrue(self.cache.put(self.geometry, 2.0, self.encoded))
        self.assertIsNotNone(self.cache.get(self.geometry, 2.0))
        self.assertEqual(1, self.cache.snapshot()["hits"])

    def test_scale_and_geometry_invalidate(self):
        self.cache.put(self.geometry, 2.0, self.encoded)
        self.assertIsNone(self.cache.get(self.geometry, 2.5))       # 不同 scale
        other = build_stage_points(make_points(200, seed=6), torch.ones((200, 1)), VOXELS,
                                   "voxel", device="cpu")
        self.assertIsNone(self.cache.get(other, 2.0))               # 不同点集

    def test_rotated_geometry_is_not_a_hit(self):
        """旋转不改缓存：点集变了（旋转后坐标不同）→ 必须失效，不能假命中。"""
        theta = 0.7
        matrix = torch.tensor([[torch.cos(torch.tensor(theta)), -torch.sin(torch.tensor(theta)), 0.0],
                               [torch.sin(torch.tensor(theta)), torch.cos(torch.tensor(theta)), 0.0],
                               [0.0, 0.0, 1.0]])
        rotated = make_points(200) @ matrix.T
        geometry = build_stage_points(rotated, torch.ones((200, 1)), VOXELS, "voxel",
                                      device="cpu")
        self.cache.put(self.geometry, 2.0, self.encoded)
        self.assertIsNone(self.cache.get(geometry, 2.0))

    def test_zero_capacity_is_disabled(self):
        cache = EncodingCache(0, model_fingerprint="m0")
        self.assertFalse(cache.enabled)
        self.assertFalse(cache.put(self.geometry, 2.0, self.encoded))
        self.assertIsNone(cache.get(self.geometry, 2.0))

    def test_oversized_entry_not_stored(self):
        cache = EncodingCache(16, model_fingerprint="m0")
        self.assertFalse(cache.put(self.geometry, 2.0, self.encoded))
        self.assertEqual(1, cache.snapshot()["rejected_too_large"])


@unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA 与模型权重")
class EncodingCacheEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        apply_tf32_policy(False)
        cfg = make_cfg()
        cls.cfg = cfg
        cls.model = create_model(cfg).cuda()
        cls.model.load_state_dict(torch.load(weights_path())["model"])
        cls.model.eval()
        cls.fingerprint = model_fingerprint(cls.model)

    def setUp(self):
        self.model.num_points_in_patch = PATCH
        self.ref = build_geometry(make_points(3000, seed=7), torch.ones((3000, 1)),
                                  VOXELS, "voxel", NEIGHBORS)
        self.src = build_geometry(make_points(2000, seed=9), torch.ones((2000, 1)),
                                  VOXELS, "voxel", NEIGHBORS)
        self.scale = float(max(self.ref.points[0][:, :3].norm(dim=1).max(),
                               self.src.points[0][:, :3].norm(dim=1).max()))
        self.cache = EncodingCache(256 * 1024 * 1024, self.fingerprint)

    def _register(self, source_encoded):
        target = self.model.encode_cloud(self.ref, self.scale)
        with torch.no_grad():
            return self.model.register_pair(target, source_encoded)

    def test_hit_equals_miss_bitwise(self):
        with torch.no_grad():
            fresh = self.model.encode_cloud(self.src, self.scale)
            self.assertIsNone(self.cache.get(self.src, self.scale))     # 尚未缓存
            self.assertTrue(self.cache.put(self.src, self.scale, fresh))
            cached = self.cache.get(self.src, self.scale)
        self.assertIsNotNone(cached)
        self.assertTrue(torch.equal(fresh.feats_f, cached.feats_f))
        self.assertTrue(torch.equal(fresh.re_feats_c, cached.re_feats_c))
        self.assertTrue(torch.equal(fresh.node_knn_indices, cached.node_knn_indices))

        with_cache = self._register(cached)
        without_cache = self._register(fresh)
        self.assertTrue(torch.equal(with_cache["estimated_transform"],
                                    without_cache["estimated_transform"]))
        for name in ("ref_node_corr_indices", "src_node_corr_indices", "matching_scores",
                     "corr_scores", "hypotheses"):
            self.assertTrue(torch.equal(with_cache[name], without_cache[name]),
                            "%s 应逐位一致" % name)

    def test_scale_change_misses(self):
        with torch.no_grad():
            encoded = self.model.encode_cloud(self.src, self.scale)
        self.cache.put(self.src, self.scale, encoded)
        self.assertIsNotNone(self.cache.get(self.src, self.scale))
        self.assertIsNone(self.cache.get(self.src, self.scale + 0.5))

    def test_target_is_not_cached(self):
        """任务卡：目标仅保留当前掩码结果 —— 缓存只由调用方对源侧写入。"""
        with torch.no_grad():
            self.model.encode_cloud(self.ref, self.scale)
        self.assertIsNone(self.cache.get(self.ref, self.scale))


if __name__ == "__main__":
    unittest.main()
