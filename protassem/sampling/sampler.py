import os
import numpy as np
from protassem.core.io import load_sample_points, calculate_contour


SAMPLE_BINARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Sample")


def sample_density_map(mrc_path, contour=None, voxel_size=2.0, output_dir=None):
    if contour is None:
        contour = calculate_contour(mrc_path)
        print(f"Auto contour (3*sigma): {contour:.6f}")

    map_name = os.path.basename(mrc_path)
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(mrc_path))
    os.makedirs(output_dir, exist_ok=True)

    sample_file = os.path.join(output_dir, f"{map_name[:-4]}_{voxel_size:.2f}.txt")

    if not os.path.exists(SAMPLE_BINARY):
        raise FileNotFoundError(f"Sample binary not found: {SAMPLE_BINARY}")

    cmd = f"{SAMPLE_BINARY} -a {mrc_path} -t {contour:.4f} -s {voxel_size:.2f} > {sample_file}"
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"Sample binary failed with code {ret}")

    points, normals = load_sample_points(sample_file)
    print(f"Sampled {len(points)} points from {map_name}")
    return points, normals, sample_file
