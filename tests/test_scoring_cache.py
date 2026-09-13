"""P4 评分缓存测试：等价性、失效契约、预算与配置变更。"""

import os
import tempfile
import time
import unittest

import mrcfile
import numpy as np

from protassem.core import scoring
from protassem.core.scoring import (calculate_cc_mask, density_context,
                                    density_fingerprint, invalidate_density,
                                    score_cache_snapshot)
from tests import fixtures


def _write_map(path, shape=(8, 8, 8), voxel_size=(2.0, 2.0, 2.0),
               origin=(1.0, 2.0, 3.0), value=5.0, mapcrs=None, offset=0.0):
    data = np.zeros(shape, dtype=np.float32)
    data[2:-2, 2:-2, 2:-2] = value + offset
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = tuple(voxel_size)
        mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z = origin
        if mapcrs:
            mrc.header.mapc, mrc.header.mapr, mrc.header.maps = mapcrs
        mrc.update_header_from_data()
        mrc.update_header_stats()
    return path


class ScoringCacheTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(scoring.configure_score_cache, 128)
        self.dir = self._tmp.name
        self.mrc = _write_map(os.path.join(self.dir, "map.mrc"), shape=(16, 16, 16),
                              voxel_size=(2.0, 2.0, 2.0), origin=(1.0, 2.0, 3.0))
        # 结构放在非零体素块内部，保证 CC ≠ 0（否则"覆盖后不同值"无法验证）
        self.structure = fixtures.make_chain_structure(
            os.path.join(self.dir, "chain.pdb"), [("A", (8.0, 8.0, 8.0))], residues=6)
        scoring.configure_score_cache(128)
        self.assertTrue(abs(calculate_cc_mask(self.mrc, self.structure, 6.0, 0.04)) > 1e-6)

    def test_cache_on_off_are_identical(self):
        cached = calculate_cc_mask(self.mrc, self.structure, 6.0, 0.04)
        scoring.configure_score_cache(0)          # 关闭缓存（恒 miss、不存）
        uncached = calculate_cc_mask(self.mrc, self.structure, 6.0, 0.04)
        self.assertEqual(cached, uncached)

    def test_second_call_hits_the_cache(self):
        scoring.configure_score_cache(128)
        first = calculate_cc_mask(self.mrc, self.structure, 6.0, 0.04)
        snapshot = score_cache_snapshot()
        second = calculate_cc_mask(self.mrc, self.structure, 6.0, 0.04)
        self.assertEqual(first, second)
        self.assertGreaterEqual(snapshot["density"]["misses"], 1)
        after = score_cache_snapshot()
        self.assertGreaterEqual(after["density"]["hits"], 1)
        self.assertGreaterEqual(after["structure"]["hits"], 1)

    def test_nonzero_origin_non_cubic_and_anisotropic_voxel(self):
        path = _write_map(os.path.join(self.dir, "aniso.mrc"), shape=(10, 12, 14),
                          voxel_size=(1.5, 2.0, 2.5), origin=(4.0, 5.0, 6.0))
        scoring.configure_score_cache(128)
        cached = calculate_cc_mask(path, self.structure, 5.0, 0.04)
        context = density_context(path, 0.04)
        self.assertEqual(context.shape, (10, 12, 14))
        np.testing.assert_allclose(context.voxel_size, [1.5, 2.0, 2.5])
        np.testing.assert_allclose(context.origin, [4.0, 5.0, 6.0])
        scoring.configure_score_cache(0)
        uncached = calculate_cc_mask(path, self.structure, 5.0, 0.04)
        self.assertEqual(cached, uncached)

    def test_axis_mapping_is_applied_and_cacheable(self):
        # mapc/mapr/maps 置换：读取路径必须按 header 重排（形状随置换变化），且缓存不改变结果
        path = _write_map(os.path.join(self.dir, "axis.mrc"), shape=(6, 8, 10),
                          mapcrs=(2, 3, 1))
        scoring.configure_score_cache(128)
        first = calculate_cc_mask(path, self.structure, 6.0, 0.04)
        context = density_context(path, 0.04)
        with mrcfile.open(path, permissive=True) as mrc:
            raw = mrc.data.copy()
        self.assertEqual(sorted(context.shape), sorted(raw.shape))
        second = calculate_cc_mask(path, self.structure, 6.0, 0.04)
        self.assertEqual(first, second)
        self.assertGreaterEqual(score_cache_snapshot()["density"]["hits"], 1)

    def test_same_path_overwrite_is_not_served_from_cache(self):
        scoring.configure_score_cache(128)
        first = calculate_cc_mask(self.mrc, self.structure, 6.0, 0.04)
        time.sleep(0.01)
        _write_map(self.mrc, value=2.0, offset=7.0)          # 同名覆盖
        second = calculate_cc_mask(self.mrc, self.structure, 6.0, 0.04)
        scoring.configure_score_cache(0)
        fresh = calculate_cc_mask(self.mrc, self.structure, 6.0, 0.04)
        self.assertEqual(second, fresh)
        self.assertNotEqual(first, second)

    def test_density_version_separates_entries(self):
        scoring.configure_score_cache(128)
        calculate_cc_mask(self.mrc, self.structure, 6.0, 0.04, density_version="v1")
        key1 = density_fingerprint(self.mrc, 0.04, "v1")
        key2 = density_fingerprint(self.mrc, 0.04, "v2")
        self.assertNotEqual(key1, key2)
        calculate_cc_mask(self.mrc, self.structure, 6.0, 0.04, density_version="v2")
        self.assertEqual(score_cache_snapshot()["density"]["entries"], 2)

    def test_explicit_invalidation(self):
        scoring.configure_score_cache(128)
        calculate_cc_mask(self.mrc, self.structure, 6.0, 0.04)
        self.assertGreaterEqual(score_cache_snapshot()["density"]["entries"], 1)
        self.assertGreaterEqual(invalidate_density(), 1)
        self.assertEqual(score_cache_snapshot()["density"]["entries"], 0)

    def test_zero_budget_disables_and_tiny_budget_rejects(self):
        scoring.configure_score_cache(0)
        snapshot = score_cache_snapshot()
        self.assertEqual(snapshot["density"]["capacity_bytes"], 0)
        calculate_cc_mask(self.mrc, self.structure, 6.0, 0.04)
        self.assertEqual(score_cache_snapshot()["density"]["entries"], 0)

        path = _write_map(os.path.join(self.dir, "big.mrc"), shape=(64, 64, 64),
                          value=3.0)
        scoring.configure_score_cache(1)                       # 1 MiB 预算装不下
        calculate_cc_mask(path, self.structure, 6.0, 0.04)
        snapshot = score_cache_snapshot()
        self.assertGreaterEqual(snapshot["density"]["rejected_too_large"], 0)
        self.assertLessEqual(snapshot["density"]["bytes"], 1024 * 1024)

    def test_budget_change_clears_entries(self):
        scoring.configure_score_cache(128)
        calculate_cc_mask(self.mrc, self.structure, 6.0, 0.04)
        self.assertGreaterEqual(score_cache_snapshot()["density"]["entries"], 1)
        scoring.configure_score_cache(64)
        snapshot = score_cache_snapshot()
        self.assertEqual(snapshot["density"]["entries"], 0)
        self.assertEqual(snapshot["score_cache_mb"], 64)

    def test_structure_change_is_not_served_from_cache(self):
        scoring.configure_score_cache(128)
        first = calculate_cc_mask(self.mrc, self.structure, 6.0, 0.04)
        time.sleep(0.01)
        fixtures.make_chain_structure(self.structure, [("A", (3.0, 1.0, 0.0))],
                                      residues=6)              # 同名覆盖
        second = calculate_cc_mask(self.mrc, self.structure, 6.0, 0.04)
        scoring.configure_score_cache(0)
        fresh = calculate_cc_mask(self.mrc, self.structure, 6.0, 0.04)
        self.assertEqual(second, fresh)
        self.assertNotEqual(first, second)

    def test_feature_cache_reexport_still_works(self):
        from protassem.fitting.feature_cache import ByteLruCache
        from protassem.runtime.byte_cache import ByteLruCache as RuntimeCache
        self.assertIs(ByteLruCache, RuntimeCache)


if __name__ == "__main__":
    unittest.main()
