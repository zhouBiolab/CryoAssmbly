import os
import glob
from pathlib import Path

import numpy as np
import mrcfile
from Bio.PDB import PDBParser, MMCIFParser
from numba.typed import List as NumbaList

from protassem.core.points_txt import read_point_cloud


def read_structure(file_path, backbone_only=False):
    """Read atom coordinates and element types from PDB or CIF."""
    ext = file_path.rsplit(".", 1)[-1].lower()
    if ext == "pdb":
        parser = PDBParser(QUIET=True)
    elif ext == "cif":
        parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError(f"Unsupported format: {ext}")

    structure = parser.get_structure("s", file_path)
    first_model = list(structure.get_models())[0]

    coords = []
    elements = NumbaList()

    if backbone_only:
        for chain in first_model:
            for residue in chain:
                if "CA" in residue:
                    coords.append(residue["CA"].get_coord())
                    elements.append(residue["CA"].element)
                    for name in ("C", "N", "O"):
                        if name in residue:
                            coords.append(residue[name].get_coord())
                            elements.append(residue[name].element)
    else:
        for atom in first_model.get_atoms():
            coords.append(atom.get_coord())
            elements.append(atom.element)

    if not coords:
        raise ValueError(f"No atoms found in {file_path}")
    return np.array(coords), elements


def read_mrc(file_path):
    """Read MRC file. Returns (data, voxel_size, origin)."""
    with mrcfile.open(file_path, permissive=True) as mrc:
        data = mrc.data.copy()
        voxel_size = np.array([mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z])
        origin = np.array([mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z])
    return data, voxel_size, origin


def write_mrc(data, origin, voxel_size, file_path):
    """Write ndarray to MRC file."""
    with mrcfile.new(file_path, overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32))
        mrc.update_header_from_data()
        mrc.voxel_size = tuple(voxel_size)
        mrc.header.origin.x = origin[0]
        mrc.header.origin.y = origin[1]
        mrc.header.origin.z = origin[2]
        mrc.update_header_stats()
        mrc.flush()


def load_sample_points(file_path, with_density=False):
    """读取点云 TXT（薄包装，保留历史签名）。

    Returns:
        (points, normals) 或 (points, normals, densities)，均为 float64 数组。
        格式契约与解析规则见 protassem.core.points_txt。
    """
    cloud = read_point_cloud(file_path)
    if with_density:
        return cloud.points, cloud.vectors, cloud.densities
    return cloud.points, cloud.vectors


def save_points_as_pdb(coords, file_path, chain_id="A"):
    """Save 3D coordinates as a PDB file with CA atoms."""
    with open(file_path, "w") as f:
        f.write("MODEL\n")
        for i, c in enumerate(coords, 1):
            f.write("ATOM%7d  %3s %3s%2s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"
                    % (i, "CA ", "ALA", " " + chain_id, 1,
                       c[0], c[1], c[2], 1.0, 1.0))
        f.write("ENDMDL\n")


def calculate_contour(mrc_file_path, sigma_multiplier=3):
    """Calculate contour as sigma_multiplier * std of the density map."""
    data, _, _ = read_mrc(mrc_file_path)
    return sigma_multiplier * float(np.std(data))


def find_files(directory, *extensions):
    """Find files matching given extensions in a directory."""
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, f"*{ext}")))
    files.sort()
    return files


def read_param_file(path):
    """Read a single numeric value from a text file.

    Raises:
        FileNotFoundError: path does not exist.
        ValueError: content is not a single number; the message names the file
            and shows the offending content.
    """
    text = Path(path).read_text().strip()
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError("%s: expected a single number, got %r" % (path, text)) from exc
