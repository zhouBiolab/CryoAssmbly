import math
import numpy as np
from protassem.core.io import read_structure, write_mrc
from protassem.core.constants import atomic_number_dict, atom_mass_dict
from protassem.core.numba_kernels import add_gaussian_to_grid


def _compute_grid_params(atoms, voxel_size, resolution):
    """Compute grid dimensions and origin with padding."""
    pad = 3.0 * resolution
    min_coords = atoms.min(axis=0)
    max_coords = atoms.max(axis=0)

    dims = tuple(
        math.ceil((max_coords[i] - min_coords[i] + 2 * pad) / voxel_size[i])
        for i in (2, 1, 0)  # z, y, x
    )
    origin = tuple(min_coords[i] - pad for i in range(3))
    return dims, origin


def _build_density(origin, voxel_size, box_size, atoms, atom_types,
                   resolution, sigma_factor, cutoff_range, use_atomic_number):
    """Build 3D Gaussian density map from atom positions."""
    grid = np.zeros(box_size, dtype=np.float32)
    sigma_real = resolution * sigma_factor
    sigma_voxels = np.array([sigma_real / v for v in voxel_size])

    weight_dict = atomic_number_dict if use_atomic_number else atom_mass_dict
    origin_arr = np.array(origin)
    voxel_arr = np.array(voxel_size)

    for i, coord in enumerate(atoms):
        w = weight_dict.get(atom_types[i], 1.0)
        add_gaussian_to_grid(grid, coord, w, origin_arr, voxel_arr,
                             sigma_voxels, cutoff_range)

    norm = np.power(2 * np.pi, -1.5) * np.power(sigma_real, -3)
    grid *= norm
    return grid


def pdb2vol(input_file, resolution, output_mrc=None, ref_map=None,
            sigma_factor=1.0 / (np.pi * np.sqrt(2)),
            cutoff_range=5.0, use_atomic_number=True,
            backbone_only=False, contour=None, bin_mask=False,
            return_data=False):
    """Convert a PDB/CIF file to a volumetric MRC density map.

    Args:
        input_file: path to .pdb or .cif
        resolution: map resolution in angstroms
        output_mrc: output .mrc path (None to skip writing)
        ref_map: reference MRC to match grid (optional)
        sigma_factor: Gaussian sigma = resolution * sigma_factor
        cutoff_range: Gaussian cutoff in sigma units
        use_atomic_number: use atomic number (True) or mass as weight
        backbone_only: only backbone atoms
        contour: threshold for zeroing low values
        bin_mask: binarize output
        return_data: return the density array

    Returns:
        density ndarray if return_data=True, else None
    """
    atoms, types = read_structure(input_file, backbone_only=backbone_only)

    if ref_map:
        import mrcfile
        with mrcfile.open(ref_map, permissive=True) as mrc:
            voxel_size = np.array([mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z])
            dims = mrc.data.shape
            origin = (float(mrc.header.origin.x),
                      float(mrc.header.origin.y),
                      float(mrc.header.origin.z))
    else:
        grid_spacing = resolution / 3.0
        voxel_size = np.array([grid_spacing] * 3)
        dims, origin = _compute_grid_params(atoms, voxel_size, resolution)

    density = _build_density(origin, voxel_size, dims, atoms, types,
                             resolution, sigma_factor, cutoff_range,
                             use_atomic_number)

    if contour:
        density = np.where(density > contour, density, 0)
    if bin_mask:
        density = np.where(density > 0, 1.0, 0.0).astype(np.float32)

    if output_mrc is not None:
        write_mrc(density, np.array(origin), voxel_size, output_mrc)

    if return_data:
        return density
