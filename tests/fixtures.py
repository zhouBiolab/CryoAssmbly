"""Synthetic fixtures for boundary tests.

Every helper writes a tiny file into a directory the caller owns (normally a
tempfile.TemporaryDirectory). No GPU, no real data, no new dependency.
"""

import os
import stat

import numpy as np
import mrcfile

# Sample TXT layout: 5 header lines, then alternating (index x y z) / (vx vy vz d).
SAMPLE_TXT_HEADER = [
    "2.000000",
    "8 8 8",
    "0.000000 0.000000 0.000000",
    "1.000000 2.000000 3.000000",
    "0.000000 0.000000 0.000000",
]

SAMPLE_TXT_PAIRS = [
    ("1 0 0 0", "1.000000 0.000000 0.000000 5.000000"),
    ("2 1 0 0", "0.000000 1.000000 0.000000 6.000000"),
]


def sample_txt_text(pairs=None, header=None):
    """Return the text of a sample TXT (5 header lines + data line pairs)."""
    if header is None:
        header = SAMPLE_TXT_HEADER
    if pairs is None:
        pairs = SAMPLE_TXT_PAIRS
    lines = list(header)
    for coord_line, vector_line in pairs:
        lines.append(coord_line)
        lines.append(vector_line)
    return "\n".join(lines) + "\n"


def make_sample_txt(path, pairs=None, header=None):
    """Write a sample TXT and return its path."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(sample_txt_text(pairs=pairs, header=header))
    return path


def make_fake_sample_bin(path, exit_code=0, output=None, stderr_text=""):
    """Write an executable standing in for the Sample binary.

    The binary prints the point cloud to stdout, which is what the sampler
    redirects into the TXT file. output=None prints the default sample TXT,
    output="" prints nothing.
    """
    if output is None:
        output = sample_txt_text()
    lines = ["#!/bin/sh"]
    if output:
        lines.append("cat <<'SAMPLE_EOF'")
        lines.append(output.rstrip("\n"))
        lines.append("SAMPLE_EOF")
    if stderr_text:
        lines.append("echo %s >&2" % _sh_quote(stderr_text))
    lines.append("exit %d" % exit_code)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
    return path


def make_mrc(path, shape=(8, 8, 8), voxel_size=2.0, origin=(1.0, 2.0, 3.0),
             value=5.0):
    """Write a small MRC map (filled inner block) and return its path."""
    data = np.zeros(shape, dtype=np.float32)
    data[2:-2, 2:-2, 2:-2] = value
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = (voxel_size, voxel_size, voxel_size)
        mrc.header.origin.x = origin[0]
        mrc.header.origin.y = origin[1]
        mrc.header.origin.z = origin[2]
        mrc.update_header_from_data()
        mrc.update_header_stats()
    return path


def _sh_quote(text):
    return "'" + text.replace("'", "'\\''") + "'"
