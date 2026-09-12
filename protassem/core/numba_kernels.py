import numpy as np
from numba import njit


@njit(fastmath=True, nogil=True)
def add_gaussian_to_grid(grid_data, atom_coord, weight, origin, voxel_size,
                         sigma_voxels, cutoff_range=5.0):
    i_center = (atom_coord[0] - origin[0]) / voxel_size[0]
    j_center = (atom_coord[1] - origin[1]) / voxel_size[1]
    k_center = (atom_coord[2] - origin[2]) / voxel_size[2]

    i_range = int(np.ceil(cutoff_range * sigma_voxels[0]))
    j_range = int(np.ceil(cutoff_range * sigma_voxels[1]))
    k_range = int(np.ceil(cutoff_range * sigma_voxels[2]))

    nz, ny, nx = grid_data.shape

    for k in range(max(0, int(k_center) - k_range),
                   min(nz, int(k_center) + k_range + 1)):
        for j in range(max(0, int(j_center) - j_range),
                       min(ny, int(j_center) + j_range + 1)):
            for i in range(max(0, int(i_center) - i_range),
                           min(nx, int(i_center) + i_range + 1)):
                di = (i - i_center) / sigma_voxels[0]
                dj = (j - j_center) / sigma_voxels[1]
                dk = (k - k_center) / sigma_voxels[2]
                r2 = di * di + dj * dj + dk * dk
                if r2 <= cutoff_range * cutoff_range:
                    grid_data[k, j, i] += weight * np.exp(-0.5 * r2)


@njit(fastmath=True, parallel=False)
def add_sphere_mask(mask, i_center, j_center, k_center,
                    vx, vy, vz, radius_sq,
                    i0, i1, j0, j1, k0, k1):
    for k in range(k0, k1):
        dz = (k - k_center) * vz
        dz2 = dz * dz
        for j in range(j0, j1):
            dy = (j - j_center) * vy
            dy2 = dy * dy
            for i in range(i0, i1):
                dx = (i - i_center) * vx
                if dx * dx + dy2 + dz2 <= radius_sq:
                    mask[k, j, i] = True
