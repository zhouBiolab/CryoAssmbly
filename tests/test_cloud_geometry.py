"""T04 单侧几何契约测试（CPU 可跑部分）。

覆盖：
  - `build_stage_points` 的多尺度点数与输入契约（CPU 张量、形状、非空）；
  - `offset_indices` 的哨兵规则（邻居不足时尾部槽位保持 0，不加偏移）；
  - `join_geometries` 的跨侧偏移：neighbors 按本阶段点数、subsampling 按本阶段点数、
    upsampling 按下一阶段点数（行与值的层次不同）；
  - `CloudGeometry.fingerprint` 的稳定性与敏感性。

CUDA 相关部分（k-NN、真实拆分等价性）由 `tools/check_geometry_split.py` 在服务器上验证。
"""

import unittest

import numpy as np
import torch

from protassem.fitting.cloud_encoding import (CloudGeometry, GeometryError, build_stage_points,
                                              join_geometries, offset_indices)

VOXELS = (0.5, 1.0, 2.0, 4.0)


def make_points(count, seed=3, extent=20.0):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(count, 3, generator=generator) * extent - extent / 2.0


def make_geometry(stage_counts, voxels=VOXELS):
    points = [make_points(count, seed=index + 1) for index, count in enumerate(stage_counts)]
    lengths = [torch.tensor([count], dtype=torch.long) for count in stage_counts]
    features = torch.ones((stage_counts[0], 1), dtype=torch.float32)
    return CloudGeometry(points=points, lengths=lengths, features=features,
                         voxel_sizes=tuple(voxels), sampling_method="voxel",
                         upsampling=[None] * len(stage_counts))


class BuildStagePointsTest(unittest.TestCase):
    def test_stage_counts_shrink_and_keep_first_stage(self):
        points = make_points(2000)
        features = torch.ones((2000, 1), dtype=torch.float32)
        geometry = build_stage_points(points, features, VOXELS, "voxel", device="cpu")
        self.assertEqual(4, geometry.num_stages)
        self.assertEqual(2000, geometry.point_count)
        self.assertTrue(torch.equal(points, geometry.points[0]))
        for index in range(1, geometry.num_stages):
            self.assertLessEqual(geometry.stage_counts[index], geometry.stage_counts[index - 1])
            self.assertGreaterEqual(geometry.stage_counts[index], 1)
        self.assertEqual(geometry.stage_counts,
                         [int(item[0]) for item in geometry.lengths])

    def test_deterministic(self):
        points = make_points(500)
        features = torch.ones((500, 1), dtype=torch.float32)
        first = build_stage_points(points, features, VOXELS, "voxel", device="cpu")
        second = build_stage_points(points, features, VOXELS, "voxel", device="cpu")
        self.assertEqual(first.fingerprint(), second.fingerprint())

    def test_rejects_cuda_input(self):
        points = make_points(64).cuda()
        features = torch.ones((64, 1), dtype=torch.float32)
        with self.assertRaises(GeometryError):
            build_stage_points(points, features, VOXELS, "voxel")

    def test_rejects_shape_and_config_mismatch(self):
        features = torch.ones((10, 1), dtype=torch.float32)
        with self.assertRaises(GeometryError):
            build_stage_points(torch.zeros(10), features, VOXELS, "voxel", device="cpu")
        with self.assertRaises(GeometryError):
            build_stage_points(torch.zeros(10, 3), torch.ones((9, 1)), VOXELS, "voxel",
                               device="cpu")
        with self.assertRaises(GeometryError):
            build_stage_points(torch.zeros(0, 3), torch.ones((0, 1)), VOXELS, "voxel",
                               device="cpu")
        with self.assertRaises(GeometryError):
            build_stage_points(torch.zeros(10, 2), features, VOXELS, "voxel", device="cpu")
        with self.assertRaises(GeometryError):
            build_stage_points(torch.zeros(10, 3), features, (), "voxel", device="cpu")
        with self.assertRaises(GeometryError):
            build_stage_points(torch.zeros(10, 3), features, VOXELS, "random", device="cpu")

    def test_num_neighbors_must_match_stages(self):
        points = make_points(200)
        geometry = build_stage_points(points, torch.ones((200, 1)), VOXELS, "voxel",
                                     device="cpu")
        from protassem.fitting.cloud_encoding import build_neighbors
        with self.assertRaises(GeometryError):
            build_neighbors(geometry, [8, 8])


class OffsetIndicesTest(unittest.TestCase):
    def test_uniform_offset_when_no_padding(self):
        indices = torch.tensor([[0, 3], [4, 5]], dtype=torch.long)
        shifted = offset_indices(indices, 100, source_count=6)
        self.assertTrue(torch.equal(indices + 100, shifted))

    def test_zero_offset_is_identity(self):
        indices = torch.tensor([[0, 1]], dtype=torch.long)
        self.assertIs(indices, offset_indices(indices, 0, source_count=1))

    def test_padding_slots_keep_sentinel(self):
        # source_count=2 < k=4 -> 后两列是 pointops 的 0 填充，不能加偏移
        indices = torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.long)
        shifted = offset_indices(indices, 10, source_count=2)
        expected = torch.tensor([[10, 11, 0, 0], [11, 10, 0, 0]], dtype=torch.long)
        self.assertTrue(torch.equal(expected, shifted))


class JoinGeometriesTest(unittest.TestCase):
    def setUp(self):
        self.ref = make_geometry([10, 4, 2])
        self.src = make_geometry([5, 3, 1])
        self.ref.neighbors = [torch.arange(count * 2).reshape(count, 2) % count
                              for count in (10, 4, 2)]
        self.src.neighbors = [torch.zeros(count, 2, dtype=torch.long) for count in (5, 3, 1)]
        self.ref.subsampling = [torch.zeros(4, 2, dtype=torch.long),
                                torch.zeros(2, 2, dtype=torch.long)]
        self.src.subsampling = [torch.zeros(3, 2, dtype=torch.long),
                                torch.zeros(1, 2, dtype=torch.long)]
        self.ref.upsampling = [None, torch.zeros(4, 1, dtype=torch.long)]
        self.src.upsampling = [None, torch.zeros(3, 1, dtype=torch.long)]
        self.transform = torch.eye(4)

    def test_offsets_by_stage(self):
        joined = join_geometries(self.ref, self.src, scale=1.0, transform=self.transform)
        self.assertEqual([15, 7, 3], [int(item.shape[0]) for item in joined["points"]])
        self.assertEqual([[10, 5], [4, 3], [2, 1]],
                         [[int(value) for value in item] for item in joined["lengths"]])
        # neighbors：src 侧加本阶段 ref 点数
        self.assertTrue(torch.equal(self.ref.neighbors[1],
                                    joined["neighbors"][1][:4]))
        self.assertTrue(torch.equal(self.src.neighbors[1] + 4,
                                    joined["neighbors"][1][4:]))
        # subsampling：行 = 下一阶段点数（ref 4 行），src 侧值加本阶段 ref 点数
        self.assertTrue(torch.equal(self.src.subsampling[0] + 10,
                                    joined["subsampling"][0][4:]))
        # upsampling：src 侧加下一阶段 ref 点数（值指向下一阶段点）
        self.assertTrue(torch.equal(self.src.upsampling[1] + 2,
                                    joined["upsampling"][0][4:]))
        self.assertEqual(2, len(joined["subsampling"]))
        self.assertEqual(1, len(joined["upsampling"]))
        self.assertEqual(15, joined["features"].shape[0])
        self.assertEqual(1, joined["batch_size"])

    def test_rejects_stage_or_device_mismatch(self):
        other = make_geometry([5, 3])
        with self.assertRaises(GeometryError):
            join_geometries(self.ref, other, scale=1.0, transform=self.transform)
        broken = make_geometry([5, 3, 1])
        broken.neighbors = broken.neighbors[:2]
        with self.assertRaises(GeometryError):
            join_geometries(self.ref, broken, scale=1.0, transform=self.transform)


class FingerprintTest(unittest.TestCase):
    def test_changes_with_points_and_config(self):
        points = make_points(200)
        features = torch.ones((200, 1), dtype=torch.float32)
        base = build_stage_points(points, features, VOXELS, "voxel", device="cpu")
        digest = base.fingerprint()
        moved = points.clone()
        moved[0, 0] += 0.001
        changed = build_stage_points(moved, features, VOXELS, "voxel", device="cpu")
        self.assertNotEqual(digest, changed.fingerprint())
        other_config = build_stage_points(points, features, (0.4, 1.0, 2.0, 4.0), "voxel",
                                          device="cpu")
        self.assertNotEqual(digest, other_config.fingerprint())

    def test_fingerprint_is_hex_sha256(self):
        points = make_points(50)
        geometry = build_stage_points(points, torch.ones((50, 1)), VOXELS, "voxel",
                                      device="cpu")
        digest = geometry.fingerprint()
        self.assertEqual(64, len(digest))
        int(digest, 16)
        self.assertTrue(np.asarray(geometry.points[0]).dtype == np.float32)


if __name__ == "__main__":
    unittest.main()
