"""Local rigid-body optimization for CC_mask maximization.

Algorithm: coarse density gradient (6 copies) -> CC check -> scipy fallback -> fine -> check.
Uses core/ modules. No duplicates. Can be imported or run as CLI.
"""
import os, sys, copy, shutil, logging, argparse, warnings
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from scipy.ndimage import map_coordinates

from protassem.core.scoring import calculate_cc_mask, read_mrc_full, _pearson
from protassem.core.constants import atomic_number_dict, VDW_RADII
from protassem.core.numba_kernels import add_gaussian_to_grid, add_sphere_mask
from protassem.runtime.execution import ExecutionContext

log = logging.getLogger(__name__)


# ===========================================================================
# Euler 旋转矩阵的解析导数（与 scipy 的 from_euler("xyz", ...) 约定一致）
# ---------------------------------------------------------------------------
# 实测确认（scipy 1.10.1）：from_euler("xyz",[a,b,c]).as_matrix()
#                       == Rz(c) @ Ry(b) @ Rx(a)   （maxdiff 2.2e-16）
# 因此 dR/dθ 可按乘积法则直接给出，与矩阵中心差分残差 ~1e-10（差分噪声量级）。
# ===========================================================================
def _rot_x(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _drot_x(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[0.0, 0.0, 0.0], [0.0, -s, -c], [0.0, c, -s]])


def _drot_y(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[-s, 0.0, c], [0.0, 0.0, 0.0], [-c, 0.0, -s]])


def _drot_z(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[-s, -c, 0.0], [c, -s, 0.0], [0.0, 0.0, 0.0]])


def rotation_derivatives(theta):
    """返回 (dR/dα, dR/dβ, dR/dγ)，对应 Rotation.from_euler("xyz", theta)。"""
    a, b, c = float(theta[0]), float(theta[1]), float(theta[2])
    Rz, Ry, Rx = _rot_z(c), _rot_y(b), _rot_x(a)
    return (Rz @ Ry @ _drot_x(a),
            Rz @ _drot_y(b) @ Rx,
            _drot_z(c) @ Ry @ Rx)


class StructureData:
    """Full atom data PDB handler for read/transform/write."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.atoms = []
        with open(filepath) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")) and len(line) >= 54:
                    self.atoms.append({
                        "record": line[0:6].strip(),
                        "serial": int(line[6:11].strip()) if line[6:11].strip() else 1,
                        "name": line[12:16].strip(),
                        "altLoc": line[16:17],
                        "resName": line[17:20].strip(),
                        "chainID": line[21:22],
                        "resSeq": int(line[22:26].strip()) if line[22:26].strip() else 1,
                        "iCode": line[26:27],
                        "x": float(line[30:38]),
                        "y": float(line[38:46]),
                        "z": float(line[46:54]),
                        "occupancy": float(line[54:60].strip()) if len(line) > 54 and line[54:60].strip() else 1.0,
                        "tempFactor": float(line[60:66].strip()) if len(line) > 60 and line[60:66].strip() else 0.0,
                        "element": line[76:78].strip() if len(line) > 76 else "",
                    })

    def get_coordinates(self):
        return np.array([[a["x"], a["y"], a["z"]] for a in self.atoms], dtype=np.float32)

    def set_coordinates(self, coords):
        for i, a in enumerate(self.atoms):
            a["x"], a["y"], a["z"] = float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2])

    def apply_transformation(self, R, t):
        c = self.get_coordinates()
        center = np.mean(c, axis=0)
        self.set_coordinates(np.dot(c - center, R.T) + center + t)

    def copy(self):
        s = StructureData.__new__(StructureData)
        s.filepath = self.filepath
        s.atoms = copy.deepcopy(self.atoms)
        return s

    def write_pdb(self, path):
        with open(path, "w") as f:
            for a in self.atoms:
                nm = a["name"]
                nf = (" " + nm + "   ")[:4] if len(nm) < 4 else nm[:4]
                el = a.get("element") or nm[0]
                f.write("ATOM  %5d %4s%s%-3s %s%4d%s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s\n" % (
                    a["serial"] % 100000, nf, a["altLoc"], a["resName"],
                    a["chainID"][:1], a["resSeq"] % 10000, a["iCode"],
                    a["x"], a["y"], a["z"], a["occupancy"], a["tempFactor"], el))
            f.write("END\n")


class DensityMap:
    """MRC density with trilinear interpolation."""

    def __init__(self, mrc_file, contour=None):
        self.filename = mrc_file
        self.data, self.voxel_size, self.origin, self.shape = read_mrc_full(mrc_file)
        self.data = self.data.astype(np.float32)
        if contour:
            self.data[self.data < contour] = 0.0
        # 梯度场惰性计算（每个实例一次）；192^3 约 85 MB，400^3 约 768 MB
        self._grad = None

    def get_density_at_position(self, pos):
        vc = (pos - self.origin) / self.voxel_size
        vc = vc[:, [2, 1, 0]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return map_coordinates(self.data, vc.T, order=1,
                                   mode="nearest", cval=0.0).astype(np.float32)

    def _gradient_fields(self):
        """密度梯度场 (∂ρ/∂x, ∂ρ/∂y, ∂ρ/∂z)，**每个实例惰性计算一次**。

        data 的轴序是 (nz, ny, nx)，所以按轴给 spacing：
        轴 0/1/2 分别对应 z/y/x 方向，据此得到 per-Å 的偏导。
        """
        if self._grad is None:
            gz, gy, gx = np.gradient(self.data, self.voxel_size[2],
                                     self.voxel_size[1], self.voxel_size[0])
            self._grad = (gx, gy, gz)
        return self._grad

    def gradient_at_positions(self, pos):
        """在原子位置插值出密度梯度，返回 (N, 3) 的 (gx, gy, gz)。

        逐分量调用 map_coordinates：scipy 1.10.1 的 map_coordinates 不支持
        "向量值一次插值"（传 (3, nz, ny, nx) + 4 行坐标的通道技巧可用但实测慢 2.1×）。
        """
        gx, gy, gz = self._gradient_fields()
        vc = ((pos - self.origin) / self.voxel_size)[:, [2, 1, 0]]
        out = np.empty((len(pos), 3), dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for k, field in enumerate((gx, gy, gz)):
                out[:, k] = map_coordinates(field, vc.T, order=1,
                                            mode="nearest", cval=0.0)
        return out


class DensityFitter:
    """Steepest ascent density gradient optimizer with adaptive step and early stop."""

    def __init__(self, structure, dmap):
        self.structure = structure
        self.mrc = dmap
        self.best_score = -np.inf
        # 最佳状态只存 6 个位姿参数；fit() 结束时对原始坐标重放一次变换。
        # 旧实现每次改进都 copy.deepcopy(atoms) 两遍，实测占 fit 总耗时约 76%。
        self.best_params = None
        self.no_improve = 0
        self.patience = 400
        # fit() 期间结构只读，坐标与旋转中心只取一次（旧实现每轮各取一次）
        self._coords = None
        self._center = None

    def fit(self, max_iter=2000, step_size=1.25):
        self.best_score = -np.inf
        self.best_params = None
        self.no_improve = 0
        params = np.zeros(6)
        self._coords = self.structure.get_coordinates()
        self._center = np.mean(self._coords, axis=0)
        center = self._center
        vs = float(np.mean(self.mrc.voxel_size))
        rot_s, trans_s = 0.1, vs
        min_step = 0.01
        seg = 4
        step = 0
        self._eval(params, center)

        while step < max_iter and step_size > min_step:
            seg_start = params.copy()
            for _ in range(seg):
                if step >= max_iter or self.no_improve >= self.patience:
                    break
                if step % 2 == 0:
                    g = self._trans_grad(params, center)
                    n = np.linalg.norm(g)
                    if n > 1e-8:
                        params[3:6] += (g / n) * step_size * trans_s
                else:
                    g = self._rot_grad(params, center)
                    n = np.linalg.norm(g)
                    if n > 1e-8:
                        params[0:3] += (g / n) * step_size * rot_s
                params[0:3] = np.clip(params[0:3], -np.pi, np.pi)
                params[3:6] = np.clip(params[3:6], -200, 200)
                step += 1
                self._eval(params, center)

            if self.no_improve >= self.patience:
                break
            motion = np.linalg.norm(params - seg_start)
            expected = seg * step_size * max(trans_s, rot_s)
            if motion < 0.25 * expected:
                step_size *= 0.5
            else:
                # 有进展则加速。这里原本写 min(step_size * 1.2, step_size * 2)，
                # 但 1.2 < 2 恒成立，等价于直接乘 1.2 —— 保留既有行为（无上界），
                # 不是经过验证的算法设计。
                step_size *= 1.2

        if self.best_params is not None:
            bp = self.best_params
            self.structure.apply_transformation(
                Rotation.from_euler("xyz", bp[:3]).as_matrix(), bp[3:6])
        return self.best_score

    def _pose(self, params, center):
        """位姿参数 -> (原子坐标, R)。

        fit() 期间结构只读，坐标与旋转中心缓存在实例上；独立调用时惰性取。
        """
        if self._coords is None:
            self._coords = self.structure.get_coordinates()
            self._center = np.mean(self._coords, axis=0)
        R = Rotation.from_euler("xyz", params[:3]).as_matrix()
        return np.dot(self._coords - center, R.T) + center + params[3:6], R

    def _eval(self, params, center):
        tr, _R = self._pose(params, center)
        score = float(np.mean(self.mrc.get_density_at_position(tr)))
        if score > self.best_score:
            self.best_score = score
            self.no_improve = 0
            self.best_params = params.copy()
        else:
            self.no_improve += 1

    def _trans_grad(self, params, center):
        """∂φ/∂t = mean_i ∇ρ(r_i)：在原子位置插值密度梯度后取均值。

        旧实现用全原子中心有限差分（每个方向 ±eps，共 6 次插值）估计同一量；
        改后 1 次插值即得三个分量。
        """
        tr, _R = self._pose(params, center)
        return self.mrc.gradient_at_positions(tr).mean(axis=0)

    def _rot_grad(self, params, center):
        """解析 Euler 梯度：∂φ/∂θ_j = mean_i ∇ρ(r_i) · [(∂R/∂θ_j)(x_i − c)]。

        力臂用原始坐标 q = x − c（平移 t 不进入旋转梯度）。**不用 torque**：
        torque 是 axis-angle 意义下的旋转方向，与 Euler 参数空间不对应，
        实测在复合姿态下方向余弦可低至 −0.05。
        """
        tr, _R = self._pose(params, center)
        g = self.mrc.gradient_at_positions(tr)              # (N, 3)
        q = self._coords - center                           # (N, 3) 原始力臂
        out = np.empty(3, dtype=np.float64)
        for j, dR in enumerate(rotation_derivatives(params[:3])):
            out[j] = np.mean(np.sum(g * (q @ dR.T), axis=1))
        return out


class ScipyFitter:
    """L-BFGS-B CC_mask optimizer (fallback when density method drops CC)."""

    def __init__(self, structure, dmap, resolution):
        self.structure = structure
        self.dmap = dmap
        self.best_cc = -np.inf
        self.best_coords = None
        sf = 1.0 / (np.pi * np.sqrt(2.0))
        self.sigma_vox = np.array([resolution * sf / dmap.voxel_size[i] for i in range(3)])
        self.norm = np.power(2 * np.pi, -1.5) * np.power(resolution * sf, -3)

    def fit(self, max_iter=1500, num_copies=4):
        orig = copy.deepcopy(self.structure.atoms)
        best_cc, best_atoms = -np.inf, None
        for _ in range(num_copies):
            self.structure.atoms = copy.deepcopy(orig)
            self.best_cc = -np.inf
            self.best_coords = None
            init = np.random.normal(0, 0.02, 6)
            init[:3] *= 0.1
            minimize(self._obj, init, method="L-BFGS-B",
                     bounds=[(-np.pi / 4, np.pi / 4)] * 3 + [(-20, 20)] * 3,
                     options={"maxiter": max_iter, "ftol": 5e-3,
                              "gtol": 5e-2, "eps": 5e-3, "disp": False})
            if self.best_cc > best_cc:
                best_cc = self.best_cc
                best_atoms = self.best_coords.copy() if self.best_coords is not None else None
        self.structure.atoms = copy.deepcopy(orig)
        if best_atoms is not None:
            self.structure.set_coordinates(best_atoms)
        return best_cc

    def _obj(self, params):
        coords = self.structure.get_coordinates()
        center = np.mean(coords, axis=0)
        R = Rotation.from_euler("xyz", params[:3]).as_matrix()
        tc = np.dot(coords - center, R.T) + center + params[3:6]
        cc = self._cc(tc)
        if cc > self.best_cc:
            self.best_cc = cc
            self.best_coords = tc.copy()
        return -cc

    def _cc(self, coords):
        origin, vs, shape = self.dmap.origin, self.dmap.voxel_size, self.dmap.shape
        sim = np.zeros(shape, dtype=np.float32)
        elems = [a.get("element", "C") or "C" for a in self.structure.atoms]
        for c, e in zip(coords, elems):
            add_gaussian_to_grid(sim, np.array(c, dtype=np.float64),
                                 atomic_number_dict.get(e, 1.0),
                                 origin, vs, self.sigma_vox, 5.0)
        sim *= self.norm
        mask = np.zeros(shape, dtype=np.bool_)
        nz, ny, nx = shape
        for c, e in zip(coords, elems):
            r = VDW_RADII.get(e, 1.70) + 1.1
            rsq = r * r
            ic = (c[0] - origin[0]) / vs[0]
            jc = (c[1] - origin[1]) / vs[1]
            kc = (c[2] - origin[2]) / vs[2]
            rv = [r / vs[d] for d in range(3)]
            i0, i1 = max(0, int(ic - rv[0])), min(nx, int(ic + rv[0]) + 1)
            j0, j1 = max(0, int(jc - rv[1])), min(ny, int(jc + rv[1]) + 1)
            k0, k1 = max(0, int(kc - rv[2])), min(nz, int(kc + rv[2]) + 1)
            if i0 < i1 and j0 < j1 and k0 < k1:
                add_sphere_mask(mask, ic, jc, kc, vs[0], vs[1], vs[2],
                                rsq, i0, i1, j0, j1, k0, k1)
        n = np.count_nonzero(mask)
        if n == 0:
            return 0.0
        return float(_pearson(self.dmap.data[mask].astype(np.float64),
                              sim[mask].astype(np.float64)))


def _density_copy_worker(arg):
    """Run one density-gradient copy in a worker process; returns cc/density/pdb."""
    (structure_file, density_mrc, contour, resolution,
     step_size, max_iter, out_pdb) = arg
    try:
        s = StructureData(structure_file)
        dmap = DensityMap(density_mrc, contour if contour else None)
        ds = DensityFitter(s, dmap).fit(max_iter=max_iter, step_size=step_size)
        s.write_pdb(out_pdb)
        cc = calculate_cc_mask(density_mrc, out_pdb, resolution, contour)
        return {"cc": cc, "ds": ds, "pdb": out_pdb, "step": step_size}
    except Exception as e:
        log.warning("density copy (step=%.1f) failed: %s", step_size, e)
        return {"cc": -1.0, "ds": -1.0, "pdb": None, "step": step_size}


def local_optimize(structure_file, density_mrc, output_file,
                   resolution, contour=0.0, max_iterations=2000,
                   initial_step_size=1.25, initial_cc=None,
                   context=None):
    """Local optimization.

    1. ALWAYS run multi-copy density gradient (parallel via `context` when it has >1 worker)
    2. If best copy CC dropped vs initial -> also run scipy CC-objective opt
    3. Select highest-CC candidate (original + copies + scipy)
    4. Fine density optimization (fewer steps) on the selected best
    5. Revert if fine drops

    initial_cc: CC already computed during PARENet fitting (reuse, no recompute).
                If None it is computed here.
    Returns (success, output_path, final_cc).
    """
    try:
        if initial_cc is None:
            initial_cc = calculate_cc_mask(density_mrc, structure_file, resolution, contour)
        log.info("local_optimize start: cc=%.4f", initial_cc)

        # ---- Step 1: always multi-copy density gradient (parallel) ----
        step_sizes = [initial_step_size, 3.0, 3.5, 4.5, 5.5, 6.0]
        args = [(structure_file, density_mrc, contour, resolution, ss,
                 max_iterations, output_file + ".cp%d.pdb" % i)
                for i, ss in enumerate(step_sizes)]

        if context is not None:
            results = context.map(_density_copy_worker, args)
        else:
            results = [_density_copy_worker(a) for a in args]

        results = [r for r in results if r["pdb"] and os.path.exists(r["pdb"])]
        if not results:
            log.warning("all density copies failed, keeping input structure")
            shutil.copy2(structure_file, output_file)
            return True, output_file, initial_cc
        for r in results:
            log.info("  copy step=%.1f density=%.4f cc=%.4f", r["step"], r["ds"], r["cc"])

        # Keep the unmodified pose as a first-class candidate.  Local
        # optimisation is allowed to improve a structure, never to replace it
        # with a lower-CC pose merely because that pose was the best of the
        # optimisation attempts.
        candidates = [{"cc": initial_cc, "pdb": structure_file,
                       "source": "original"}]
        candidates.extend({"cc": r["cc"], "pdb": r["pdb"],
                           "source": "density"} for r in results)
        best_density_cc = max(r["cc"] for r in results)

        # ---- Step 2: density CC dropped -> scipy CC-objective opt (extra candidate) ----
        if best_density_cc < initial_cc:
            log.info("density best CC %.4f < initial %.4f -> scipy CC opt",
                     best_density_cc, initial_cc)
            s2 = StructureData(structure_file)
            dmap2 = DensityMap(density_mrc, contour if contour else None)
            scipy_cc = ScipyFitter(s2, dmap2, resolution).fit()
            scipy_pdb = output_file + ".scipy.pdb"
            s2.write_pdb(scipy_pdb)
            candidates.append({"cc": scipy_cc, "pdb": scipy_pdb})
            log.info("  scipy cc=%.4f", scipy_cc)

        # ---- Step 3: select highest-CC candidate ----
        best = max(candidates, key=lambda c: c["cc"])

        # ---- Step 4: fine density optimization (fewer steps) on best ----
        bs = StructureData(best["pdb"])
        dmap3 = DensityMap(density_mrc, contour if contour else None)
        DensityFitter(bs, dmap3).fit(max_iter=250, step_size=0.5)
        fine_pdb = output_file + ".fine.pdb"
        bs.write_pdb(fine_pdb)
        fine_cc = calculate_cc_mask(density_mrc, fine_pdb, resolution, contour)
        log.info("fine: %.4f -> %.4f", best["cc"], fine_cc)

        # ---- Step 5: revert if fine drops ----
        if fine_cc >= best["cc"]:
            final_src, final_cc = fine_pdb, fine_cc
        else:
            final_src, final_cc = best["pdb"], best["cc"]
        shutil.copy2(final_src, output_file)

        # cleanup temp files
        for r in results:
            if r["pdb"] and os.path.exists(r["pdb"]):
                os.remove(r["pdb"])
        for extra in (output_file + ".scipy.pdb", fine_pdb):
            if os.path.exists(extra):
                os.remove(extra)

        log.info("local_optimize done: %.4f -> %.4f (%+.4f)",
                 initial_cc, final_cc, final_cc - initial_cc)
        return True, output_file, final_cc

    except Exception as e:
        log.error("local_optimize failed: %s", e)
        return False, None, 0.0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Local CC_mask optimization")
    p.add_argument("structure_file")
    p.add_argument("mrc_file")
    p.add_argument("output_file")
    p.add_argument("--resolution", type=float, required=True)
    p.add_argument("--contour", type=float, default=0.0)
    p.add_argument("--max_iterations", type=int, default=2000)
    p.add_argument("--initial_step_size", type=float, default=1.25)
    p.add_argument("--num_processes", type=int, default=1,
                   help="并行副本数（1 = 串行；>1 走 ExecutionContext 共享池）")
    p.add_argument("--pool_start_method", default=None,
                   help="进程池启动方式：默认 None = 系统默认（Linux 为 fork）")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    # P3 之后并行度改由 ExecutionContext 表达：此处是本模块的独立入口，自己建上下文并释放。
    # worker 函数 `_density_copy_worker` 是模块级可序列化对象，满足共享池的契约。
    context = ExecutionContext(pool_workers=a.num_processes,
                               start_method=a.pool_start_method)
    try:
        ok, _, _ = local_optimize(a.structure_file, a.mrc_file, a.output_file,
                                  a.resolution, a.contour,
                                  max_iterations=a.max_iterations,
                                  initial_step_size=a.initial_step_size,
                                  context=context)
    finally:
        context.close()
    sys.exit(0 if ok else 1)
