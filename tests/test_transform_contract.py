"""T01 变换契约测试：旋转中心、已知刚体变换、残差组合与原地污染。

关键结论（实测）：网络位姿在**点云质心**系求解，而 PDB 写出默认绕**原子质心**旋转；
两者相差 (I - R)(c_atom - c_src)。配准输出边界必须显式传 center=c_src。
"""

import os
import tempfile
import unittest

import numpy as np

from protassem.fitting.local_optimizer import StructureData
from protassem.fitting.utils import apply_transformation, transform_pdb

PDB_ROUNDING = 1e-3          # write_pdb 用 %.3f


def rotation_z(degrees):
    theta = np.radians(degrees)
    return np.array([[np.cos(theta), -np.sin(theta), 0.0],
                     [np.sin(theta), np.cos(theta), 0.0],
                     [0.0, 0.0, 1.0]])


def write_atoms(path, coords):
    with open(path, "w") as handle:
        for index, coord in enumerate(coords, 1):
            handle.write("ATOM  %5d  CA  ALA A%4d    %8.3f%8.3f%8.3f  1.00  0.00           C\n"
                         % (index, index, coord[0], coord[1], coord[2]))
    return path


ATOMS = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.0, 0.0],
                  [11.4, 0.0, 0.0], [15.2, 6.0, 9.0]])
CLOUD = np.array([[0.5, 0.2, -0.3], [4.0, 0.1, 0.2], [7.9, -0.4, 0.1],
                  [11.8, 0.3, 0.4], [15.0, 5.6, 9.4], [2.0, 1.0, 1.0],
                  [9.0, -1.0, -1.0]])


class TransformContractTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name
        self.pdb = write_atoms(os.path.join(self.tmp, "atoms.pdb"), ATOMS)
        self.c_atom = ATOMS.mean(axis=0)
        self.c_src = CLOUD.mean(axis=0)
        self.c_ref = self.c_src + np.array([2.0, -1.5, 3.0])
        self.R = rotation_z(37.0)
        self.t_est = np.array([0.7, -1.3, 2.1])

    def _written(self, center):
        out = os.path.join(self.tmp, "out_%s.pdb" % (center is None))
        t_corrected = self.t_est + (self.c_ref - self.c_src)
        transform_pdb(self.pdb, self.R, t_corrected, out, center=center)
        return StructureData(out).get_coordinates().astype(np.float64)

    def test_point_cloud_centered_pose_maps_atoms_correctly(self):
        """配准输出边界：center=c_src 时写出坐标 = R(x - c_src) + t' + c_ref。"""
        expected = apply_transformation(ATOMS - self.c_src, self.R, self.t_est) + self.c_ref
        produced = self._written(self.c_src)
        self.assertLess(np.abs(produced - expected).max(), PDB_ROUNDING + 1e-6)

    def test_default_center_is_atomic_centroid(self):
        """默认（局部优化沿用）绕原子质心旋转：R(x - c_atom) + c_atom + t。"""
        t = np.array([1.0, 2.0, 3.0])
        out = os.path.join(self.tmp, "default_center.pdb")
        transform_pdb(self.pdb, self.R, t, out)
        produced = StructureData(out).get_coordinates().astype(np.float64)
        expected = apply_transformation(ATOMS - self.c_atom, self.R, t) + self.c_atom
        self.assertLess(np.abs(produced - expected).max(), PDB_ROUNDING + 1e-6)

    def test_atomic_centroid_rotation_differs_by_known_offset(self):
        """取证：不传 center 时相对正确结果偏移 (I - R)(c_atom - c_src)。"""
        expected = apply_transformation(ATOMS - self.c_src, self.R, self.t_est) + self.c_ref
        produced = self._written(None)
        offset = (np.eye(3) - self.R) @ (self.c_atom - self.c_src)
        self.assertGreater(np.linalg.norm(offset), 0.1)      # 本用例的两质心确实不同
        self.assertLess(np.abs((produced - expected) - offset).max(),
                        PDB_ROUNDING + 1e-6)

    def test_centroids_equal_removes_the_discrepancy(self):
        """原子质心 == 点云质心时，两条路径一致（只剩 PDB 写出精度）。"""
        shift = self.c_src - self.c_atom
        atoms = ATOMS + shift
        pdb = write_atoms(os.path.join(self.tmp, "shifted.pdb"), atoms)
        expected = apply_transformation(atoms - self.c_src, self.R, self.t_est) + self.c_ref
        out = os.path.join(self.tmp, "shifted_out.pdb")
        transform_pdb(pdb, self.R, self.t_est + (self.c_ref - self.c_src), out)
        produced = StructureData(out).get_coordinates().astype(np.float64)
        self.assertLess(np.abs(produced - expected).max(), PDB_ROUNDING + 1e-6)

    def test_known_rigid_transform_round_trip(self):
        """已知刚体变换：先正变换再逆变换回到原坐标（限 PDB 写出精度）。"""
        out = os.path.join(self.tmp, "fwd.pdb")
        transform_pdb(self.pdb, self.R, self.t_est, out, center=self.c_src)
        back = os.path.join(self.tmp, "back.pdb")
        transform_pdb(out, self.R.T, -self.R.T @ self.t_est, back, center=self.c_src)
        produced = StructureData(back).get_coordinates().astype(np.float64)
        self.assertLess(np.abs(produced - ATOMS).max(), 2 * PDB_ROUNDING + 1e-6)

    def test_residual_composition_convention(self):
        """两阶段组合：T = (R2 R1, R2 t1 + t2)（与 T_residual @ T_initial 同约定）。"""
        R1, t1 = rotation_z(20.0), np.array([1.0, 0.0, -2.0])
        R2, t2 = rotation_z(-35.0), np.array([0.5, 2.5, 1.0])
        composed = apply_transformation(apply_transformation(ATOMS, R1, t1), R2, t2)
        single = apply_transformation(ATOMS, R2 @ R1, R2 @ t1 + t2)
        self.assertLess(np.abs(composed - single).max(), 1e-9)

    def test_apply_transformation_does_not_modify_input(self):
        """缓存/共享数组不得被旋转原地污染。"""
        points = CLOUD.copy()
        before = points.copy()
        rotated = apply_transformation(points, self.R, self.t_est)
        np.testing.assert_array_equal(points, before)
        self.assertFalse(np.shares_memory(rotated, points))

    def test_structure_data_transformation_does_not_touch_source_file(self):
        """PDB 变换只改内存副本，不写回源文件。"""
        with open(self.pdb, encoding="utf-8") as handle:
            before = handle.read()
        structure = StructureData(self.pdb)
        structure.apply_transformation(self.R, self.t_est)
        with open(self.pdb, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), before)


if __name__ == "__main__":
    unittest.main()
