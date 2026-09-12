"""Point cloud filtering and density map masking — direct computation."""

import numpy as np
from protassem.core.io import read_structure, read_mrc, write_mrc
from protassem.core.scoring import read_mrc_full
from protassem.core.constants import VDW_RADII
from protassem.core.numba_kernels import add_sphere_mask


def mask_fitted_region(target_txt, mask_pdb, density_mrc,
                       output_txt, output_mrc, solvent_radius=1.1):
    """Filter points near mask_pdb atoms and zero density in that region.

    Returns True on success.
    """
    points, original_lines, line_mapping = _read_points_with_lines(target_txt)
    if len(points) == 0:
        return False

    coords, elements = read_structure(str(mask_pdb))
    elem_list = list(elements)

    data, voxel_size, origin, dims = read_mrc_full(density_mrc)

    mask = _build_atom_mask(origin, voxel_size, dims, coords, elem_list,
                            solvent_radius)

    kept = _filter_points(points, mask, origin, voxel_size)
    _save_filtered_txt(original_lines, kept, line_mapping, output_txt)

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


def _read_points_with_lines(file_path):
    points, line_mapping = [], []
    with open(file_path, "r") as f:
        lines = f.readlines()
    if len(lines) < 6:
        return np.array([]), lines, []
    sample = float(lines[0].strip())
    ox, oy, oz = (float(v) for v in lines[3].strip().split())
    for i in range(5, len(lines)):
        line = lines[i].strip()
        if i % 2 == 1 and line:
            parts = line.split()
            if len(parts) >= 4:
                _, x, y, z = parts[:4]
                points.append([float(x) * sample + ox,
                               float(y) * sample + oy,
                               float(z) * sample + oz])
                line_mapping.append((i, i + 1))
    return np.array(points), lines, line_mapping


def _save_filtered_txt(original_lines, kept_indices, line_mapping, output_path):
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for i in range(min(5, len(original_lines))):
            f.write(original_lines[i])
        for idx in kept_indices:
            if idx < len(line_mapping):
                ci, vi = line_mapping[idx]
                if ci < len(original_lines) and vi < len(original_lines):
                    f.write(original_lines[ci])
                    f.write(original_lines[vi])
