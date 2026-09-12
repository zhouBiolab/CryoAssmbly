"""Density-map sampling: call the bundled Sample (VoxEM) binary, then read its TXT.

The binary prints the point cloud to stdout; we redirect stdout into the TXT file
so the caller keeps a copy of exactly what was sampled.
"""

import os
import subprocess

from protassem.core.io import calculate_contour, load_sample_points

SAMPLE_BINARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Sample")


def sample_density_map(mrc_path, contour=None, voxel_size=2.0, output_dir=None):
    """Sample a density map into a point cloud.

    Args:
        mrc_path: input .mrc density map
        contour: density contour level; None means 3*sigma of the map
        voxel_size: sampling step in angstroms (also the coordinate scale of the TXT)
        output_dir: directory for the TXT (default: the directory of mrc_path)

    Returns:
        (points, normals, txt_path): two (N, 3) float arrays and the TXT path.

    Raises:
        FileNotFoundError: the bundled Sample binary is missing.
        RuntimeError: the binary failed, or produced no output.
    """
    if contour is None:
        contour = calculate_contour(mrc_path)
        print("Auto contour (3*sigma): %.6f" % contour)

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(mrc_path))
    os.makedirs(output_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(mrc_path))[0]
    sample_file = os.path.join(output_dir, "%s_%.2f.txt" % (stem, voxel_size))

    if not os.path.isfile(SAMPLE_BINARY):
        raise FileNotFoundError("Sample binary not found: %s" % SAMPLE_BINARY)

    command = [SAMPLE_BINARY,
               "-a", str(mrc_path),
               "-t", "%.4f" % contour,
               "-s", "%.2f" % voxel_size]
    with open(sample_file, "w", encoding="utf-8") as output:
        result = subprocess.run(command, stdout=output, stderr=subprocess.PIPE,
                                text=True, check=False)

    if result.returncode != 0:
        raise RuntimeError("Sample binary failed with code %d: %s"
                           % (result.returncode, result.stderr.strip()[-2000:]))
    if os.path.getsize(sample_file) == 0:
        raise RuntimeError("Sample binary produced no output: %s" % sample_file)

    points, normals = load_sample_points(sample_file)
    print("Sampled %d points from %s" % (len(points), os.path.basename(mrc_path)))
    return points, normals, sample_file
