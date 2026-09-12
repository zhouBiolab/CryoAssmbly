"""Point cloud filtering and density map masking — direct computation."""

import os

import numpy as np

from protassem.core.io import read_structure, read_mrc, write_mrc
from protassem.core.points_txt import read_point_cloud, write_filtered
from protassem.core.scoring import read_mrc_full
from protassem.core.constants import VDW_RADII
from protassem.core.numba_kernels import add_sphere_mask


def mask_fitted_region(target_txt, mask_pdb, density_mrc,
                       output_txt, output_mrc, solvent_radius=1.1):
    """Filter points near mask_pdb atoms and zero density in that region.

    Returns True on success.
    """
    cloud = read_point_cloud(target_txt)
    if len(cloud.points) == 0:
        return False

    coords, elements = read_structure(str(mask_pdb))
    elem_list = list(elements)

    data, voxel_size, origin, dims = read_mrc_full(density_mrc)

    mask = _build_atom_mask(origin, voxel_size, dims, coords, elem_list,
                            solvent_radius)

    kept = _filter_points(cloud.points, mask, origin, voxel_size)
    directory = os.path.dirname(output_txt)
    if directory:
        os.makedirs(directory, exist_ok=True)
    write_filtered(output_txt, cloud, kept)

    data[mask] = 0.0
    write_mrc(data, origin, voxel_size, str(output_mrc))
    return True


def _build_atom_mask(origin, voxel_size, box_size, coords, elements,
                     solvent_radius):
    mask = np.zeros(box_size, dtype=np.bool_)
    nz, ny, nx = mask.shape
    for coord, elem in zip(coords, elements):
        r = VDW_RADII.get(elem, 1.70) + solvent_radius
        rsq = r * r
        ic = (coord[0] - origin[0]) / voxel_size[0]
        jc = (coord[1] - origin[1]) / voxel_size[1]
        kc = (coord[2] - origin[2]) / voxel_size[2]
        rv = [r / voxel_size[d] for d in range(3)]
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
                        rsq, i0, i1, j0, j1, k0, k1)
    return mask


def _filter_points(points, mask, origin, voxel_size):
    nz, ny, nx = mask.shape
    kept = []
    for i, pt in enumerate(points):
        vi = int(np.round((pt[0] - origin[0]) / voxel_size[0]))
        vj = int(np.round((pt[1] - origin[1]) / voxel_size[1]))
        vk = int(np.round((pt[2] - origin[2]) / voxel_size[2]))
        if 0 <= vi < nx and 0 <= vj < ny and 0 <= vk < nz:
            if mask[vk, vj, vi]:
                continue
        kept.append(i)
    return kept
