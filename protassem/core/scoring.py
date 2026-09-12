"""CC_mask calculation — direct computation, no subprocess."""

import numpy as np
import mrcfile
from scipy.ndimage import distance_transform_edt, gaussian_filter
from numba import njit
from protassem.core.io import read_structure
from protassem.core.constants import atomic_number_dict, VDW_RADII
from protassem.core.numba_kernels import add_gaussian_to_grid, add_sphere_mask


def read_mrc_full(filename):
    """Read MRC with axis reordering (handles mapc/mapr/maps)."""
    with mrcfile.open(filename, permissive=True) as mrc:
        data = mrc.data.copy()
        vx = float(mrc.voxel_size.x) or 1.0
        vy = float(mrc.voxel_size.y) or 1.0
        vz = float(mrc.voxel_size.z) or 1.0
        voxel_size = np.array([vx, vy, vz], dtype=np.float32)

        origin = np.array([
            float(mrc.header.origin.x),
            float(mrc.header.origin.y),
            float(mrc.header.origin.z)
        ], dtype=np.float32)

        nstart = np.array([
            int(mrc.header.nxstart),
            int(mrc.header.nystart),
            int(mrc.header.nzstart)
        ], dtype=np.float32)

        mapcrs = np.array([
            int(mrc.header.mapc),
            int(mrc.header.mapr),
            int(mrc.header.maps)
        ], dtype=int)

        sort = np.array([0, 1, 2], dtype=int)
        for i in range(3):
            sort[mapcrs[i] - 1] = i
        nstart = np.array([nstart[i] for i in sort], dtype=np.float32)
        data = np.transpose(data, axes=2 - sort[::-1])

        if np.all(origin == 0):
            origin = origin + nstart * voxel_size

    return data, voxel_size, origin, data.shape


@njit(fastmath=True)
def _pearson(x, y):
    n = len(x)
    if n == 0:
        return 0.0
    sx, sy = 0.0, 0.0
    for i in range(n):
        sx += x[i]
        sy += y[i]
    mx, my = sx / n, sy / n
    num, vx, vy = 0.0, 0.0, 0.0
    for i in range(n):
        dx = x[i] - mx
        dy = y[i] - my
        num += dx * dy
        vx += dx * dx
        vy += dy * dy
    denom = np.sqrt(vx * vy)
    if denom == 0.0:
        return 0.0
    return num / denom


def _make_sim_map(origin, voxel_size, box_size, coords, elements, resolution):
    sigma_factor = 1.0 / (np.pi * np.sqrt(2.0))
    sigma_real = resolution * sigma_factor
    sigma_vox = np.array([sigma_real / voxel_size[i] for i in range(3)])

    grid = np.zeros(box_size, dtype=np.float32)

    for coord, elem in zip(coords, elements):
        w = atomic_number_dict.get(elem, 1.0)
        atom_coord = np.array(coord, dtype=np.float64)
        add_gaussian_to_grid(grid, atom_coord, w, origin, voxel_size,
                             sigma_vox, 5.0)

    norm = np.power(2 * np.pi, -1.5) * np.power(sigma_real, -3)
    grid *= norm
    return grid


def _make_phenix_mask(origin, voxel_size, box_size, coords, elements,
                      solvent_radius=1.1, mask_radius=2.0, resolution=None):
    mask = np.zeros(box_size, dtype=np.bool_)
    nz, ny, nx = mask.shape

    for coord, elem in zip(coords, elements):
        r_vdw = VDW_RADII.get(elem, 1.70)
        radius = r_vdw + solvent_radius
        radius_sq = radius * radius
        ic = (coord[0] - origin[0]) / voxel_size[0]
        jc = (coord[1] - origin[1]) / voxel_size[1]
        kc = (coord[2] - origin[2]) / voxel_size[2]
        rv = np.array([radius / voxel_size[i] for i in range(3)])
        i0 = max(0, int(np.floor(ic - rv[0])))
        i1 = min(nx, int(np.ceil(ic + rv[0])) + 1)
        j0 = max(0, int(np.floor(jc - rv[1])))
        j1 = min(ny, int(np.ceil(jc + rv[1])) + 1)
        k0 = max(0, int(np.floor(kc - rv[2])))
        k1 = min(nz, int(np.ceil(kc + rv[2])) + 1)
        if i0 >= i1 or j0 >= j1 or k0 >= k1:
            continue
        add_sphere_mask(mask, ic, jc, kc,
                        voxel_size[0], voxel_size[1], voxel_size[2],
                        radius_sq, i0, i1, j0, j1, k0, k1)

    avg_vs = np.mean(voxel_size)
    dist = distance_transform_edt(~mask) * avg_vs
    expanded = (dist <= mask_radius) | mask

    sigma = (resolution / 4.0) if resolution else 1.0
    sigma_vox = np.array([sigma / voxel_size[i] for i in range(3)])
    soft = gaussian_filter(expanded.astype(np.float32), sigma=sigma_vox,
                           mode="constant", cval=0.0)
    if soft.max() > 0:
        soft /= soft.max()
    return soft > 0.5


def calculate_cc_mask(density_mrc, structure_file, resolution, contour):
    """Compute CC_mask between experimental map and structure."""
    exp_map, voxel_size, origin, dims = read_mrc_full(density_mrc)
    exp_map = np.where(exp_map > contour, exp_map, -1.0)

    coords, elements = read_structure(str(structure_file))
    elem_list = list(elements)

    sim_map = _make_sim_map(origin, voxel_size, dims, coords, elem_list, resolution)
    mask = _make_phenix_mask(origin, voxel_size, dims, coords, elem_list,
                            resolution=resolution)

    sel = mask
    if np.count_nonzero(sel) == 0:
        return 0.0
    exp_vals = exp_map[sel].astype(np.float64)
    sim_vals = sim_map[sel].astype(np.float64)
    return float(_pearson(exp_vals, sim_vals))
