"""密度图原点规范化测试：nstart 归零、origin 停在显示位置，采样端与评分端同框。

背景与口径（含三种 origin/nstart 组合的冲突表、Sample 锚点的实测证据）见
protassem/core/mrc_origin.py。这里锁定五件事：
1. sample_anchor 与 display_origin 两条口径本身的取值；
2. nstart 非零时写出修正副本，**origin 停在显示位置**、数组与轴序逐位不变；
3. nstart 全零时不动文件（不产生副本）；
4. run_pipeline 的接线位置与落盘目录；
5. 修正 fce7a4b 的关键点：副本的 origin 不再被平移 nstart*voxel。
"""

import os
import tempfile
import unittest

import mrcfile
import numpy as np

from protassem import pipeline
from protassem.core import mrc_origin
from tests import fixtures


def _read_header(path):
    with mrcfile.open(path, permissive=True) as mrc:
        origin = np.array([mrc.header.origin.x, mrc.header.origin.y,
                           mrc.header.origin.z], dtype=np.float64)
        nstart = np.array([int(mrc.header.nxstart), int(mrc.header.nystart),
                           int(mrc.header.nzstart)])
        mapcrs = (int(mrc.header.mapc), int(mrc.header.mapr), int(mrc.header.maps))
        voxel = np.array([mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z])
    return origin, nstart, mapcrs, voxel


class SampleAnchorTest(unittest.TestCase):
    """采样器 Sample 的口径：nstart 与 origin 都计入。"""

    def test_folds_nstart_into_origin(self):
        got = mrc_origin.sample_anchor([27.898, 60.088, 47.212],
                                       [26, 56, 44],
                                       [1.073, 1.073, 1.073])
        self.assertTrue(np.allclose(got, [55.796, 120.176, 94.424], atol=1e-3))

    def test_zero_nstart_is_identity(self):
        origin = np.array([1.0, 2.0, 3.0])
        got = mrc_origin.sample_anchor(origin, [0, 0, 0], [2.0, 2.0, 2.0])
        self.assertTrue(np.allclose(got, origin))


class DisplayOriginTest(unittest.TestCase):
    """MRC 标准 / 评分端口径：origin 非零则用它，否则退回 nstart*voxel。"""

    def test_nonzero_origin_wins(self):
        got = mrc_origin.display_origin([10.0, 20.0, 30.0], [5, 7, 9],
                                        [2.0, 2.0, 2.0])
        self.assertTrue(np.allclose(got, [10.0, 20.0, 30.0]))

    def test_zero_origin_falls_back_to_nstart(self):
        got = mrc_origin.display_origin([0.0, 0.0, 0.0], [5, 7, 9],
                                        [2.0, 2.0, 2.0])
        self.assertTrue(np.allclose(got, [10.0, 14.0, 18.0]))

    def test_matches_sample_anchor_when_origin_is_zero(self):
        """origin 为零时两条口径同值 -> 本来就不需要规范化。"""
        o, ns, v = [0.0, 0.0, 0.0], [5, 7, 9], [2.0, 2.0, 2.0]
        self.assertTrue(np.allclose(mrc_origin.display_origin(o, ns, v),
                                    mrc_origin.sample_anchor(o, ns, v)))


class NormalizeDensityMapTest(unittest.TestCase):
    """规范化只改 header 的 4 个字段，数据按字节搬运，且 origin 不被平移。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

    def _dest(self, name="normalized.mrc"):
        return os.path.join(self.tmp, name)

    def test_shifted_map_gets_corrected_copy(self):
        src = fixtures.make_mrc(os.path.join(self.tmp, "shifted.mrc"),
                                shape=(10, 12, 14), voxel_size=1.5,
                                origin=(30.0, 60.0, 45.0), nstart=(20, 40, 30))
        dest = self._dest()
        result = mrc_origin.normalize_density_map(src, dest)

        self.assertEqual(result.path, dest)
        self.assertTrue(os.path.isfile(dest))
        # origin 停在显示位置（原值），nstart 归零；**不是** origin + nstart*voxel
        self.assertTrue(np.allclose(result.origin, [30.0, 60.0, 45.0]))
        self.assertFalse(np.allclose(result.origin, [60.0, 120.0, 90.0]),
                         "origin must NOT be shifted by nstart*voxel")
        self.assertTrue(np.allclose(result.previous_origin, [30.0, 60.0, 45.0]))
        self.assertTrue(np.allclose(result.nstart, [20, 40, 30]))
        # 规范化前的 Sample 锚点被记录下来，供日志说明"从哪变到哪"
        self.assertTrue(np.allclose(result.sample_anchor, [60.0, 120.0, 90.0]))

        origin, nstart, _, voxel = _read_header(dest)
        self.assertTrue(np.allclose(origin, [30.0, 60.0, 45.0]))
        self.assertTrue(np.array_equal(nstart, [0, 0, 0]))
        self.assertTrue(np.allclose(voxel, 1.5))

        # 数组数据逐位不变（不转置、不重采样、不转 dtype）
        with mrcfile.open(src, permissive=True) as a, mrcfile.open(dest, permissive=True) as b:
            self.assertEqual(a.data.shape, b.data.shape)
            self.assertEqual(a.data.dtype, b.data.dtype)
            self.assertTrue(np.array_equal(a.data, b.data))

    def test_zero_origin_with_nstart_is_also_normalized(self):
        """origin=0 + nstart 非零：规范化后 origin 取 nstart*voxel，两端同值。"""
        src = fixtures.make_mrc(os.path.join(self.tmp, "zeros.mrc"),
                                shape=(10, 12, 14), voxel_size=2.0,
                                origin=(0.0, 0.0, 0.0), nstart=(1, 2, 3))
        dest = self._dest("zeros_norm.mrc")
        result = mrc_origin.normalize_density_map(src, dest)

        self.assertEqual(result.path, dest)
        origin, nstart, _, _ = _read_header(dest)
        self.assertTrue(np.allclose(origin, [2.0, 4.0, 6.0]))
        self.assertTrue(np.array_equal(nstart, [0, 0, 0]))
        # origin 为零时两条口径本来就同值
        self.assertTrue(np.allclose(result.sample_anchor, result.origin))

    def test_axis_order_is_preserved(self):
        """不让 io.write_mrc 代劳的理由：它会重置 mapc/mapr/maps。

        非标准轴序（mapc,mapr,maps = 2,3,1）规范化后必须仍是 2,3,1，
        否则数据的轴解释被改变。
        """
        src = os.path.join(self.tmp, "axis.mrc")
        data = np.zeros((10, 12, 14), dtype=np.float32)
        data[2:-2, 2:-2, 2:-2] = 5.0
        with mrcfile.new(src, overwrite=True) as mrc:
            mrc.set_data(data)
            mrc.voxel_size = (2.0, 2.0, 2.0)
            mrc.update_header_from_data()
            mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart = 1, 2, 3
            mrc.header.mapc, mrc.header.mapr, mrc.header.maps = 2, 3, 1
            mrc.flush()
        dest = self._dest("axis_norm.mrc")
        mrc_origin.normalize_density_map(src, dest)

        _, _, mapcrs_src, _ = _read_header(src)
        _, _, mapcrs_dest, _ = _read_header(dest)
        self.assertEqual(mapcrs_src, (2, 3, 1))
        self.assertEqual(mapcrs_dest, (2, 3, 1))

    def test_canonical_map_is_left_alone(self):
        src = fixtures.make_mrc(os.path.join(self.tmp, "canon.mrc"),
                                origin=(0.0, 0.0, 0.0), nstart=(0, 0, 0))
        dest = self._dest("never_written.mrc")
        result = mrc_origin.normalize_density_map(src, dest)

        self.assertEqual(result.path, src)
        self.assertFalse(os.path.exists(dest), "canonical map must not be copied")
        self.assertTrue(np.allclose(result.origin, result.previous_origin))

    def test_nonzero_origin_with_zero_nstart_is_left_alone(self):
        """项目自己产出的图（pdb2vol）就是这种：origin 非零、nstart 全零。"""
        src = fixtures.make_mrc(os.path.join(self.tmp, "vox.mrc"),
                                origin=(12.0, 34.0, 56.0), nstart=(0, 0, 0))
        dest = self._dest("vox_norm.mrc")
        result = mrc_origin.normalize_density_map(src, dest)

        self.assertEqual(result.path, src)
        self.assertFalse(os.path.exists(dest))
        self.assertTrue(np.allclose(result.origin, [12.0, 34.0, 56.0]))

    def test_missing_source_raises(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            mrc_origin.normalize_density_map(
                os.path.join(self.tmp, "nope.mrc"), self._dest())
        self.assertIn("density map not found", str(ctx.exception))


class PipelineNormalizationTest(unittest.TestCase):
    """run_pipeline 的接线：产物落 <output_dir>/density/，规范图不落。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name
        self.out = os.path.join(self.tmp, "out")

    def test_shifted_map_is_written_under_output_density_dir(self):
        src = fixtures.make_mrc(os.path.join(self.tmp, "shifted.mrc"),
                                shape=(10, 12, 14), voxel_size=2.0,
                                origin=(4.0, 8.0, 12.0), nstart=(1, 2, 3))
        with self.assertLogs("protassem.pipeline", level="INFO") as captured:
            effective = pipeline._normalize_density(src, self.out)

        expected = os.path.join(self.out, "density", "shifted.mrc")
        self.assertEqual(effective, expected)
        self.assertTrue(os.path.isfile(expected))
        origin, nstart, _, _ = _read_header(expected)
        # origin 停在显示位置（原值 4/8/12），不是 4+2=6 / 8+4=12 / 12+6=18
        self.assertTrue(np.allclose(origin, [4.0, 8.0, 12.0]))
        self.assertTrue(np.array_equal(nstart, [0, 0, 0]))
        self.assertTrue(np.allclose(
            _read_header(src)[0], [4.0, 8.0, 12.0]), "source must stay untouched")

        logged = "\n".join(captured.output)
        self.assertIn("Density origin normalized", logged)
        self.assertIn("nstart (1, 2, 3) -> 0", logged)
        # 日志应同时给出 Sample 锚点的变化
        self.assertIn("sample anchor", logged)

    def test_canonical_map_creates_no_density_dir(self):
        src = fixtures.make_mrc(os.path.join(self.tmp, "canon.mrc"), nstart=(0, 0, 0))
        effective = pipeline._normalize_density(src, self.out)

        self.assertEqual(effective, src)
        self.assertFalse(os.path.exists(os.path.join(self.out, "density")))


if __name__ == "__main__":
    unittest.main()
