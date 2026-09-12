import os
import glob
import numpy as np
import mrcfile
from Bio.PDB import PDBParser, MMCIFParser
from numba.typed import List as NumbaList


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
    """Load sampled point cloud from txt file."""
    points, normals, densities = [], [], []
    with open(file_path, "r") as f:
        lines = f.readlines()
        sample = float(lines[0].strip())
        ox, oy, oz = (float(v) for v in lines[3].strip().split())
        for i in range(5, len(lines)):
            line = lines[i].strip()
            if not line:
                continue
            if i % 2 == 1:
                parts = line.split()
                if len(parts) >= 4:
                    _, x, y, z = parts[:4]
                    points.append([float(x) * sample + ox,
                                   float(y) * sample + oy,
                                   float(z) * sample + oz])
            else:
                parts = line.split()
                if len(parts) >= 4:
                    vx, vy, vz, d = parts[:4]
                    normals.append([float(vx), float(vy), float(vz)])
                    densities.append([float(d)])

    if with_density:
        return np.array(points), np.array(normals), np.array(densities)
    return np.array(points), np.array(normals)


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
    """Read a single numeric value from a text file."""
    with open(path) as f:
        return float(f.read().strip())
