import importlib

ext_module = importlib.import_module('pareconv.ext')

def grid_subsample_with_density(points, densities, lengths, voxel_size):
    """Grid subsampling including density averaging.

    Args:
        points (Tensor): (N, 3)
        densities (Tensor): (N,)
        lengths (Tensor): (B,)
        voxel_size (float): voxel size.

    Returns:
        s_points (Tensor): (M, 3)
        s_densities (Tensor): (M,)
        s_lengths (Tensor): (B,)
    """
    s_points, s_densities, s_lengths = ext_module.grid_subsampling(
        points, densities, lengths, voxel_size
    )
    return s_points, s_densities, s_lengths