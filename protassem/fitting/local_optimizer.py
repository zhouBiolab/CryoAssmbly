"""Local rigid-body optimization for CC_mask maximization.

Algorithm: coarse density gradient (6 copies) -> CC check -> scipy fallback -> fine -> check.
Uses core/ modules. No duplicates. Can be imported or run as CLI.
"""
import os, sys, copy, shutil, logging, argparse, warnings
import numpy as np

from protassem.runtime.pool import timed_pool
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from scipy.ndimage import map_coordinates

from protassem.core.scoring import calculate_cc_mask, read_mrc_full, _pearson
from protassem.core.constants import atomic_number_dict, VDW_RADII
from protassem.core.numba_kernels import add_gaussian_to_grid, add_sphere_mask

log = logging.getLogger(__name__)


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

    def get_density_at_position(self, pos):
        vc = (pos - self.origin) / self.voxel_size
        vc = vc[:, [2, 1, 0]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return map_coordinates(self.data, vc.T, order=1,
                                   mode="nearest", cval=0.0).astype(np.float32)


class DensityFitter:
    """Steepest ascent density gradient optimizer with adaptive step and early stop."""

    def __init__(self, structure, dmap):
        self.structure = structure
        self.mrc = dmap
        self.best_score = -np.inf
        self.best_state = None
        self.no_improve = 0
        self.patience = 400

    def fit(self, max_iter=2000, step_size=1.25):
        self.best_score = -np.inf
        self.best_state = None
        self.no_improve = 0
        params = np.zeros(6)
        coords = self.structure.get_coordinates()
        center = np.mean(coords, axis=0)
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
                step_size = min(step_size * 1.2, step_size * 2)

        if self.best_state is not None:
            self.structure.atoms = copy.deepcopy(self.best_state)
        return self.best_score

    def _eval(self, params, center):
        R = Rotation.from_euler("xyz", params[:3]).as_matrix()
        coords = self.structure.get_coordinates()
        tr = np.dot(coords - center, R.T) + center + params[3:6]
        score = float(np.mean(self.mrc.get_density_at_position(tr)))
        if score > self.best_score:
            self.best_score = score
            self.no_improve = 0
            orig = copy.deepcopy(self.structure.atoms)
            self.structure.apply_transformation(R, params[3:6])
            self.best_state = copy.deepcopy(self.structure.atoms)
            self.structure.atoms = orig
        else:
            self.no_improve += 1

    def _trans_grad(self, params, center):
        R = Rotation.from_euler("xyz", params[:3]).as_matrix()
        coords = self.structure.get_coordinates()
        tr = np.dot(coords - center, R.T) + center + params[3:6]
        eps = 0.001
        g = np.zeros(3)
        for i in range(3):
            p, m = tr.copy(), tr.copy()
            p[:, i] += eps
            m[:, i] -= eps
            g[i] = np.mean(self.mrc.get_density_at_position(p) -
                           self.mrc.get_density_at_position(m)) / (2 * eps)
        return g

    def _rot_grad(self, params, center):
        eps = 0.01
        g = np.zeros(3)
        coords = self.structure.get_coordinates()
        for i in range(3):
            p = params[:3].copy()
            p[i] += eps
            R = Rotation.from_euler("xyz", p).as_matrix()
            dp = np.mean(self.mrc.get_density_at_position(
                np.dot(coords - center, R.T) + center + params[3:6]))
            p[i] -= 2 * eps
            R = Rotation.from_euler("xyz", p).as_matrix()
            dm = np.mean(self.mrc.get_density_at_position(
                np.dot(coords - center, R.T) + center + params[3:6]))
            g[i] = (dp - dm) / (2 * eps)
        return g


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
                   initial_step_size=1.25, num_processes=1, initial_cc=None,
                   metrics=None):
    """Local optimization.

    1. ALWAYS run multi-copy density gradient (parallel if num_processes > 1)
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

        if num_processes and num_processes > 1:
            with timed_pool(metrics, min(num_processes, len(args)),
                            "local_optimize_copies") as pool:
                results = pool.map(_density_copy_worker, args)
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
    p.add_argument("--num_processes", type=int, default=1)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ok, _, _ = local_optimize(a.structure_file, a.mrc_file, a.output_file,
                              a.resolution, a.contour,
                              a.max_iterations, a.initial_step_size,
                              a.num_processes)
    sys.exit(0 if ok else 1)
