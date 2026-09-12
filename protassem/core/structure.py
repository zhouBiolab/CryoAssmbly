"""Structure operations: format conversion, gyration radius, sequence alignment."""

import os
import re
import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, MMCIFIO, Superimposer
from Bio import pairwise2

AA_MAP = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def cif_to_pdb(cif_file, output_pdb, chain_id="A"):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("s", str(cif_file))
    if chain_id is not None:
        for model in structure:
            for chain in model:
                chain.id = chain_id
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_pdb))


def pdb_to_cif(pdb_file, output_cif, chain_id=None):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", str(pdb_file))
    if chain_id is not None:
        for model in structure:
            for chain in model:
                chain.id = chain_id
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(str(output_cif))


def chain_id_pool():
    """Yield chain IDs in order: A-Z, a-z, then two-letter (uppercase first)."""
    import string
    base = list(string.ascii_uppercase) + list(string.ascii_lowercase)
    for c in base:
        yield c
    for a in base:
        for b in base:
            yield a + b


def write_structure_with_chain_map(input_file, chain_map, output_file):
    """Rename chains of input_file per chain_map (old_id -> new_id), save out.

    Output writer chosen by output extension (.cif -> MMCIFIO, else PDBIO).
    Chains not in chain_map keep their id. Safe against rename collisions
    (detach all, rename, re-add). Multi-char chain IDs only survive in CIF.
    """
    in_ext = os.path.splitext(str(input_file))[1].lower()
    parser = MMCIFParser(QUIET=True) if in_ext == ".cif" else PDBParser(QUIET=True)
    structure = parser.get_structure("s", str(input_file))
    model = list(structure)[0]
    chains = list(model)
    for ch in chains:
        model.detach_child(ch.id)
    for ch in chains:
        ch.id = chain_map.get(ch.id, ch.id)
    for ch in chains:
        model.add(ch)
    out_ext = os.path.splitext(str(output_file))[1].lower()
    io = MMCIFIO() if out_ext == ".cif" else PDBIO()
    io.set_structure(structure)
    io.save(str(output_file))


def cif_to_pdb_placeholders(cif_file, output_pdb):
    """CIF -> PDB using single-char placeholder chain IDs (fitting ignores
    chain IDs; PDB cannot hold multi-char IDs).

    Returns {placeholder_id: real_id} so the real (possibly multi-char) IDs
    can be restored when writing the final CIF. Reuses
    write_structure_with_chain_map as the rename primitive.
    """
    import string
    pool = (list(string.ascii_uppercase) + list(string.ascii_lowercase)
            + list(string.digits))
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("s", str(cif_file))
    real_ids = [ch.id for ch in list(structure)[0]]
    real2ph, ph2real = {}, {}
    for i, rid in enumerate(real_ids):
        ph = pool[i]
        real2ph[rid] = ph
        ph2real[ph] = rid
    write_structure_with_chain_map(str(cif_file), real2ph, str(output_pdb))
    return ph2real


def calculate_gyration_radius(points):
    if len(points) == 0:
        return 0.0
    centroid = np.mean(points, axis=0)
    return float(np.sqrt(np.mean(np.sum((points - centroid) ** 2, axis=1))))


def extract_chain_id(filename):
    basename = Path(filename).stem
    match = re.match(r"chain_([^_]+)_", basename)
    return match.group(1) if match else basename


def align_by_resid(ref_file, mob_file, output_file):
    """Align mob to ref by matching residue numbers (domain is subset of chain)."""
    ref_struct = _get_parser(ref_file).get_structure("ref", str(ref_file))
    mob_struct = _get_parser(mob_file).get_structure("mob", str(mob_file))

    ref_ca = {}
    for model in ref_struct:
        for chain in model:
            for res in chain:
                if res.get_resname() in AA_MAP and "CA" in res:
                    ref_ca[res.get_id()[1]] = res["CA"]

    atoms_ref, atoms_mob = [], []
    for model in mob_struct:
        for chain in model:
            for res in chain:
                resid = res.get_id()[1]
                if res.get_resname() in AA_MAP and "CA" in res and resid in ref_ca:
                    atoms_ref.append(ref_ca[resid])
                    atoms_mob.append(res["CA"])

    if len(atoms_ref) < 3:
        return False

    sup = Superimposer()
    sup.set_atoms(atoms_ref, atoms_mob)
    sup.apply(mob_struct.get_atoms())

    out_io = PDBIO() if str(output_file).endswith(".pdb") else MMCIFIO()
    out_io.set_structure(mob_struct)
    out_io.save(str(output_file))
    return True


def align_by_sequence(ref_file, mob_file, output_file):
    """Align mob to ref using global sequence alignment on CA atoms."""
    ref_struct = _get_parser(ref_file).get_structure("ref", str(ref_file))
    mob_struct = _get_parser(mob_file).get_structure("mob", str(mob_file))

    seq1, ca1 = _get_sequence_and_ca(ref_struct)
    seq2, ca2 = _get_sequence_and_ca(mob_struct)
    if not seq1 or not seq2:
        return False

    alignments = pairwise2.align.globalms(seq1, seq2, 2, -1, -0.5, -0.5,
                                          one_alignment_only=True)
    if not alignments:
        return False

    atoms1, atoms2 = _get_aligned_atoms(alignments[0], ca1, ca2)
    if len(atoms1) < 3:
        return False

    sup = Superimposer()
    sup.set_atoms(atoms1, atoms2)
    sup.apply(mob_struct.get_atoms())

    out_io = PDBIO() if str(output_file).endswith(".pdb") else MMCIFIO()
    out_io.set_structure(mob_struct)
    out_io.save(str(output_file))
    return True


def _get_parser(filepath):
    ext = os.path.splitext(str(filepath))[1].lower()
    if ext == ".cif":
        return MMCIFParser(QUIET=True)
    return PDBParser(QUIET=True)


def _get_sequence_and_ca(structure):
    seq, ca_info = "", {}
    pos = 0
    for model in structure:
        for chain in model:
            for res in chain:
                if res.get_resname() in AA_MAP and "CA" in res:
                    seq += AA_MAP[res.get_resname()]
                    ca_info[pos] = res["CA"]
                    pos += 1
    return seq, ca_info


def _get_aligned_atoms(alignment, ca1, ca2):
    aligned_seq1, aligned_seq2 = alignment[0], alignment[1]
    atoms1, atoms2 = [], []
    p1, p2 = 0, 0
    for i in range(len(aligned_seq1)):
        c1, c2 = aligned_seq1[i], aligned_seq2[i]
        if c1 != "-" and c2 != "-" and c1 == c2:
            if p1 in ca1 and p2 in ca2:
                atoms1.append(ca1[p1])
                atoms2.append(ca2[p2])
        if c1 != "-":
            p1 += 1
        if c2 != "-":
            p2 += 1
    return atoms1, atoms2


def read_chain_ids(filepath):
    """Read chain IDs from a structure file. Returns list of chain ID strings."""
    parser = _get_parser(filepath)
    structure = parser.get_structure("s", str(filepath))
    model = next(structure.get_models())
    return [ch.id for ch in model]


def split_structure_to_chains(filepath, output_dir):
    """Split a multi-chain structure into individual single-chain PDB files.

    Returns list of (chain_id, output_pdb_path) tuples.
    """
    from Bio.PDB import Structure as StructModule, Model as ModelModule
    parser = _get_parser(filepath)
    structure = parser.get_structure("s", str(filepath))
    model = next(structure.get_models())
    results = []
    stem = Path(filepath).stem
    for chain in model:
        cid = chain.id
        out_pdb = os.path.join(output_dir, f"chain_{cid}_{stem}.pdb")
        s = StructModule.Structure("single")
        m = ModelModule.Model(0)
        ch_copy = chain.copy()
        ch_copy.id = cid
        m.add(ch_copy)
        s.add(m)
        io = PDBIO()
        io.set_structure(s)
        io.save(str(out_pdb))
        results.append((cid, out_pdb))
    return results
