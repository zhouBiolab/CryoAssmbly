"""局部优化器测试：P0 记账改造、P2 步长、P3 解析梯度、P4 回退删除的契约。

背景
----
本轮对 `protassem/fitting/local_optimizer.py` 做了四项改动：
  P0  每次改进不再 deepcopy 两份原子表，只存 6 个位姿参数（严格等价）
  P2  步长增长表达式由 min(×1.2, ×2) 澄清为 ×1.2（逐值相同，仅可读性）
  P3  梯度改为"预计算密度梯度场 + 解析 Euler 链式法则"
  P4  删除 ScipyFitter 回退

其中 P3 的判据必须分级：`rotation_derivatives` 是**解析的矩阵导数**，可以要求
逐元素一致；而"梯度场 + 链式法则"与旧的**有限差分目标梯度**并非同一个量
（∇[interp(ρ)] ≠ interp[∇ρ]），只能要求**方向高度一致**（方向余弦）。
"""

import copy
import os
import tempfile
import unittest

import mrcfile
import numpy as np
from scipy.spatial.transform import Rotation

from protassem.fitting import local_optimizer as lo
from tests import fixtures


def _write_atoms(path, coords, resname="ALA", chain="A"):
    """写一个最小 PDB（逐原子 CA），返回路径。"""
    with open(path, "w") as handle:
        for i, (x, y, z) in enumerate(coords, 1):
            handle.write(
                "ATOM  %5d  CA  %3s %1s%4d    %8.3f%8.3f%8.3f  1.00  0.00           C\n"
                % (i % 100000, resname, chain, i, x, y, z))
        handle.write("END\n")
    return path


def _gaussian_density(pos, center, sigma):
    d = pos - center
    return np.exp(-np.sum(d * d, axis=1) / (2.0 * sigma * sigma))


def _grad_gaussian(pos, center, sigma):
    d = pos - center
    return _gaussian_density(pos, center, sigma)[:, None] * (-d) / (sigma * sigma)


class _AnalyticGradientProbe:
    """用解析高斯密度替代 DensityMap，验证梯度的数学形式。

    只实现 DensityFitter 需要的方法，从而把"梯度公式是否正确"与
    "MRC 读取/插值"完全解耦。
    """

    def __init__(self, center, sigma):
        self.center = np.asarray(center, dtype=np.float64)
        self.sigma = float(sigma)
        self.calls = 0
        # 兼容 DensityFitter 可能访问的属性
        self.voxel_size = np.array([2.0, 2.0, 2.0])
        self.origin = np.zeros(3)

    def get_density_at_position(self, pos):
        self.calls += 1
        return _gaussian_density(np.asarray(pos, dtype=np.float64),
                                 self.center, self.sigma)

    def gradient_at_positions(self, pos):
        return _grad_gaussian(np.asarray(pos, dtype=np.float64),
                              self.center, self.sigma)


# ===========================================================================
# P3 · 旋转矩阵解析导数（这部分确实是解析的，可要求逐元素一致）
# ===========================================================================
class RotationDerivativesTest(unittest.TestCase):
    """`rotation_derivatives` 必须与 scipy 的矩阵中心差分逐元素一致。"""

    POSES = [(0.0, 0.0, 0.0), (0.3, -0.5, 0.8), (1.2, -0.9, 0.7),
             (1.5, 1.4, -1.5)]

    def test_matches_matrix_finite_difference(self):
        eps = 1e-6
        for theta in self.POSES:
            dRs = lo.rotation_derivatives(theta)
            self.assertEqual(3, len(dRs))
            for j in range(3):
                p = list(theta)
                p[j] += eps
                Rp = Rotation.from_euler("xyz", p).as_matrix()
                p[j] -= 2 * eps
                Rm = Rotation.from_euler("xyz", p).as_matrix()
                fd = (Rp - Rm) / (2 * eps)
                self.assertLess(
                    float(np.abs(dRs[j] - fd).max()), 1e-8,
                    "dR/dθ_%d 与矩阵中心差分不一致（theta=%s）" % (j, theta))


# ===========================================================================
# P3 · 梯度场 + 链式法则 vs 旧的全原子有限差分（只能要求方向一致）
# ===========================================================================
class AnalyticPoseGradientTest(unittest.TestCase):
    """解析 Euler 梯度与旧有限差分目标梯度的方向余弦，按姿态量级分组。"""

    SIGMA = 4.0

    def setUp(self):
        rng = np.random.default_rng(0)
        self.coords = rng.normal(0.0, 8.0, size=(400, 3))
        self.center = self.coords.mean(axis=0)
        self.dmap = _AnalyticGradientProbe(center=[1.0, 2.0, -1.5], sigma=self.SIGMA)
        self.t = np.array([0.7, -1.3, 2.1])

    # ---- 参照实现：旧的全原子中心差分 ----
    def _reference_rot_grad(self, theta, eps=1e-6):
        g = np.zeros(3)
        for j in range(3):
            p = list(theta)
            p[j] += eps
            Rp = Rotation.from_euler("xyz", p).as_matrix()
            dp = np.mean(_gaussian_density(
                np.dot(self.coords - self.center, Rp.T) + self.center + self.t,
                self.dmap.center, self.SIGMA))
            p[j] -= 2 * eps
            Rm = Rotation.from_euler("xyz", p).as_matrix()
            dm = np.mean(_gaussian_density(
                np.dot(self.coords - self.center, Rm.T) + self.center + self.t,
                self.dmap.center, self.SIGMA))
            g[j] = (dp - dm) / (2 * eps)
        return g

    def _analytic_rot_grad(self, theta):
        R = Rotation.from_euler("xyz", theta).as_matrix()
        tr = np.dot(self.coords - self.center, R.T) + self.center + self.t
        g = _grad_gaussian(tr, self.dmap.center, self.SIGMA)
        q = self.coords - self.center
        out = np.empty(3)
        for j, dR in enumerate(lo.rotation_derivatives(theta)):
            out[j] = np.mean(np.sum(g * (q @ dR.T), axis=1))
        return out

    def _torque_rot_grad(self, theta):
        """反例：用 torque 当旋转梯度（axis-angle 语义，与 Euler 参数不对应）。"""
        R = Rotation.from_euler("xyz", theta).as_matrix()
        tr = np.dot(self.coords - self.center, R.T) + self.center + self.t
        g = _grad_gaussian(tr, self.dmap.center, self.SIGMA)
        lever = tr - self.center            # 已含平移的力臂（错误做法）
        return np.mean(np.cross(lever, g), axis=0)

    @staticmethod
    def _cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-30 or nb < 1e-30:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _check_pose(self, deg, threshold):
        theta = np.deg2rad(np.asarray(deg, dtype=float))
        got = self._analytic_rot_grad(theta)
        ref = self._reference_rot_grad(theta)
        cos = self._cos(got, ref)
        self.assertGreaterEqual(cos, threshold,
                                "姿态 %s 的方向余弦 %.6f 低于 %.4f" % (deg, cos, threshold))
        return cos

    def test_near_zero_pose(self):
        self._check_pose((0, 0, 0), 0.9999)
        self._check_pose((20, 0, 0), 0.9999)

    def test_moderate_poses(self):
        self._check_pose((0, 35, 0), 0.999)
        self._check_pose((25, -30, 15), 0.999)

    def test_large_poses(self):
        self._check_pose((60, 0, 0), 0.99)
        self._check_pose((70, -50, 40), 0.99)
        self._check_pose((89, 80, -88), 0.99)

    def test_torque_is_not_a_valid_substitute(self):
        """锁定：不得用 torque 替代 Euler 梯度。

        torque 只在 b=c=0（纯绕 x 轴）时碰巧等于 Euler 梯度；一旦中间轴或末轴
        非零就开始偏离，接近万向锁时方向几乎正交甚至反号。这里断言"解析版在
        复合姿态下显著优于 torque 版"，防止以后图省事改回 torque。
        """
        deg = (70, -50, 40)
        theta = np.deg2rad(np.asarray(deg, dtype=float))
        ref = self._reference_rot_grad(theta)
        cos_analytic = self._cos(self._analytic_rot_grad(theta), ref)
        cos_torque = self._cos(self._torque_rot_grad(theta), ref)
        self.assertGreater(cos_analytic, 0.99)
        self.assertLess(cos_torque, 0.9,
                        "torque 版在 %s 的方向余弦 %.4f 意外地高，反例失效"
                        % (deg, cos_torque))
        self.assertGreater(cos_analytic - cos_torque, 0.1,
                           "解析版应显著优于 torque 版")

    def test_translation_gradient_is_mean_density_gradient(self):
        theta = np.deg2rad(np.array([25.0, -30.0, 15.0]))
        R = Rotation.from_euler("xyz", theta).as_matrix()
        tr = np.dot(self.coords - self.center, R.T) + self.center + self.t
        expected = _grad_gaussian(tr, self.dmap.center, self.SIGMA).mean(axis=0)
        got = self.dmap.gradient_at_positions(tr).mean(axis=0)
        self.assertTrue(np.allclose(got, expected, atol=1e-12))


# ===========================================================================
# P3 · DensityMap 的梯度场（真实 MRC）
# ===========================================================================
class DensityMapGradientTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

    def _gaussian_mrc(self, name="gauss.mrc", shape=(32, 32, 32), sigma=4.0,
                      voxel=1.0, center=None):
        """写入一个三维高斯团的 MRC（复用 fixtures.make_gaussian_mrc）。"""
        if center is None:
            center = tuple(s // 2 for s in shape)
        path = fixtures.make_gaussian_mrc(
            os.path.join(self.tmp, name), shape=shape, voxel_size=voxel,
            sigma=sigma, center=center)
        return path, np.asarray(center, dtype=float), sigma, voxel

    def test_gradient_field_is_computed_once(self):
        path, _c, _s, _v = self._gaussian_mrc()
        dmap = lo.DensityMap(path, None)
        self.assertIsNone(dmap._grad)
        dmap._gradient_fields()
        first = dmap._grad
        dmap._gradient_fields()
        self.assertIs(first, dmap._grad, "梯度场应惰性计算且只算一次")

    def test_gradient_points_toward_density_peak(self):
        """高斯团：中心附近的梯度应指向峰（即背离中心的方向为负）。"""
        path, center, _s, voxel = self._gaussian_mrc()
        dmap = lo.DensityMap(path, None)
        # 峰的一侧（索引空间减 3），世界坐标 = origin + (ix,iy,iz)*voxel
        probe_idx = center - 3.0
        pos = np.array([[probe_idx[2] * voxel, probe_idx[1] * voxel,
                         probe_idx[0] * voxel]])
        grad = dmap.gradient_at_positions(pos)[0]
        toward_peak = np.array([3.0 * voxel, 3.0 * voxel, 3.0 * voxel])
        cos = float(np.dot(grad, toward_peak)
                    / (np.linalg.norm(grad) * np.linalg.norm(toward_peak)))
        self.assertGreater(cos, 0.9, "梯度方向应指向密度峰，实测 cos=%.4f" % cos)

    def test_gradient_matches_finite_difference_of_interpolation(self):
        """梯度场插值值 ≈ 对插值函数做中心差分（两者本不严格相等，只比方向）。"""
        path, center, _s, voxel = self._gaussian_mrc()
        dmap = lo.DensityMap(path, None)
        pos = np.array([[center[2] * voxel + 1.3, center[1] * voxel - 0.7,
                         center[0] * voxel + 2.1]])
        eps = 1e-3
        fd = np.empty(3)
        for k in range(3):
            p, m = pos.copy(), pos.copy()
            p[0, k] += eps
            m[0, k] -= eps
            fd[k] = (dmap.get_density_at_position(p)[0]
                     - dmap.get_density_at_position(m)[0]) / (2 * eps)
        got = dmap.gradient_at_positions(pos)[0]
        cos = float(np.dot(got, fd) / (np.linalg.norm(got) * np.linalg.norm(fd)))
        # 注意：梯度场用的是体素中心差分，而 get_density_at_position 是分段线性
        # 插值，两者的导数本不严格相等（体素内折点效应）；小图上差异更明显。
        # 这里只要求方向高度一致。
        self.assertGreater(cos, 0.99, "与插值中心差分方向应高度一致，cos=%.6f" % cos)


# ===========================================================================
# P0 · 记账改造（严格等价）
# ===========================================================================
class PoseBookkeepingTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name
        rng = np.random.default_rng(7)
        self.coords = rng.normal(0.0, 6.0, size=(60, 3))
        self.pdb = _write_atoms(os.path.join(self.tmp, "chain.pdb"), self.coords)
        self.dmap = _AnalyticGradientProbe(center=[0.5, -0.5, 1.0], sigma=5.0)

    def test_best_score_matches_final_coordinates(self):
        """fit() 返回的 best_score 必须等于对最终坐标重新求值的均值。"""
        sd = lo.StructureData(self.pdb)
        fitter = lo.DensityFitter(sd, self.dmap)
        score = fitter.fit(max_iter=60, step_size=1.0)
        final = sd.get_coordinates()
        recheck = float(np.mean(self.dmap.get_density_at_position(
            final.astype(np.float64))))
        self.assertAlmostEqual(score, recheck, places=6)

    def test_fit_is_deterministic(self):
        digests = []
        for _ in range(2):
            sd = lo.StructureData(self.pdb)
            lo.DensityFitter(sd, self.dmap).fit(max_iter=60, step_size=1.0)
            digests.append(float(sd.get_coordinates().sum()))
        self.assertEqual(digests[0], digests[1], "两次 fit 结果应完全一致")

    def test_eval_has_no_side_effect_on_structure(self):
        """_eval 只探测、不改结构（记账改造的核心前提）。"""
        sd = lo.StructureData(self.pdb)
        fitter = lo.DensityFitter(sd, self.dmap)
        before = sd.get_coordinates().copy()
        center = before.mean(axis=0)
        fitter._coords = before
        fitter._center = center
        for params in (np.zeros(6), np.array([0.2, -0.1, 0.3, 1.0, -2.0, 0.5])):
            fitter._eval(params, center)
        self.assertTrue(np.array_equal(sd.get_coordinates(), before),
                        "_eval 不得改变结构坐标")

    def test_eval_records_params_and_fit_replays_them(self):
        """`_eval` 只记参数；`fit()` 结尾按同一约定重放，得到同一坐标。"""
        params = np.array([0.1, 0.2, -0.15, 1.5, -0.7, 2.0])

        # 参照：直接对原始坐标应用该变换（与 fit() 结尾同一算法）
        sd_ref = lo.StructureData(self.pdb)
        coords0 = sd_ref.get_coordinates()
        center = coords0.mean(axis=0)
        R = Rotation.from_euler("xyz", params[:3]).as_matrix()
        expected = np.dot(coords0 - center, R.T) + center + params[3:6]

        # 被测：_eval 记录 params，然后用 fit() 的重放逻辑落地
        sd = lo.StructureData(self.pdb)
        fitter = lo.DensityFitter(sd, self.dmap)
        fitter._coords = coords0
        fitter._center = center
        fitter._eval(params, center)
        self.assertIsNotNone(fitter.best_params)
        self.assertTrue(np.allclose(fitter.best_params, params))

        bp = fitter.best_params
        sd.apply_transformation(
            Rotation.from_euler("xyz", bp[:3]).as_matrix(), bp[3:6])
        self.assertTrue(np.allclose(sd.get_coordinates(), expected, atol=1e-5),
                        "fit() 的重放必须与直接变换等价")

    def test_structure_data_copy_removed(self):
        """P4 连带删除的死代码。"""
        self.assertFalse(hasattr(lo.StructureData, "copy"))


# ===========================================================================
# P2 · 步长策略（表达式澄清，逐值相同）
# ===========================================================================
class StepSizePolicyTest(unittest.TestCase):
    """步长只按 ×1.2 增长或 ×0.5 减半；增长无上界（既有行为，非设计主张）。"""

    def test_growth_has_no_cap(self):
        step = 1.25
        seq = []
        for _ in range(6):
            step = min(step * 1.2, step * 2)
            seq.append(step)
        # 澄清后的表达式与之逐值相同
        step2 = 1.25
        for i, expected in enumerate(seq):
            step2 *= 1.2
            self.assertAlmostEqual(step2, expected, places=12,
                                   msg="第 %d 次增长：%r != %r" % (i, step2, expected))

    def test_growth_is_monotonic_and_unbounded(self):
        step = 1.25
        last = step
        for _ in range(50):
            step *= 1.2
            self.assertGreater(step, last)
            last = step
        self.assertGreater(step, 100.0, "增长确实无上界")


# ===========================================================================
# P4 · 删除 ScipyFitter 后的行为契约
# ===========================================================================
class NoScipyFallbackTest(unittest.TestCase):
    """局部优化只能改善；全部轨迹更差时必须保留原始位姿。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

    def test_scipy_fitter_is_gone(self):
        self.assertFalse(hasattr(lo, "ScipyFitter"))
        for name in ("minimize", "add_gaussian_to_grid", "add_sphere_mask",
                     "atomic_number_dict", "VDW_RADII"):
            self.assertFalse(hasattr(lo, name),
                             "%s 应随 ScipyFitter 一并删除" % name)

    def test_original_pose_is_kept_when_every_track_is_worse(self):
        """构造"6 条轨迹全部更差、精修也没改善"的场景。

        做法：把密度团放在离结构极远的位置，且让 contour 阈值把唯一的高密度区
        完全滤掉 —— 这样所有轨迹的 CC 都低于起点，最终必须原样返回输入结构。
        """
        shape = (24, 24, 24)
        path = os.path.join(self.tmp, "map.mrc")
        data = np.zeros(shape, dtype=np.float32)
        data[2:6, 2:6, 2:6] = 5.0            # 一个小团，远离结构
        with mrcfile.new(path, overwrite=True) as mrc:
            mrc.set_data(data)
            mrc.voxel_size = (2.0, 2.0, 2.0)
            mrc.update_header_from_data()
            mrc.update_header_stats()

        coords = np.array([[60.0, 60.0, 60.0], [64.0, 60.0, 60.0],
                           [68.0, 60.0, 60.0]], dtype=np.float32)
        pdb = _write_atoms(os.path.join(self.tmp, "far.pdb"), coords)
        out = os.path.join(self.tmp, "out.pdb")

        # contour 定在团之上：阈值化后图内密度全 0 → 任何轨迹都好不了
        ok, path_out, cc = lo.local_optimize(
            pdb, path, out, resolution=5.0, contour=6.0, max_iterations=20)

        self.assertTrue(ok)
        self.assertTrue(os.path.exists(path_out))
        written = lo.StructureData(path_out).get_coordinates()
        # 算法契约：坐标不变（copy2 的逐字节相同是次要的）
        self.assertTrue(np.allclose(written, coords, atol=1e-4),
                        "所有轨迹更差时，最终坐标必须与输入一致")
        self.assertLessEqual(cc, 1e-9)


if __name__ == "__main__":
    unittest.main()
