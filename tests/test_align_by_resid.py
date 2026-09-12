"""align_by_resid 的链号匹配测试。

覆盖三种空间关系：两边同空间（单链/多链同编号）、需要 mob->ref 链号映射的
占位空间（真 B -> 占位 A -> 恢复 B），以及匹配不足时的拒绝。
"""

import os
import tempfile
import unittest

import numpy as np
from Bio.PDB import PDBParser, Superimposer

from protassem.core import structure as structure_module
from protassem.core.structure import (align_by_resid, cif_to_pdb_placeholders,
                                      split_structure_to_chains)
from tests import fixtures


def _ca_coords(path, chain_id):
    parsed = PDBParser(QUIET=True).get_structure("s", path)
    model = next(parsed.get_models())
    return np.array([residue["CA"].get_coord() for residue in model[chain_id]
                     if "CA" in residue])


def _legacy_matched_pairs(ref_file, mob_file):
    """S4a 之前 align_by_resid 的匹配逻辑：只用残基号建键（仅用于取证）。"""
    ref_struct = PDBParser(QUIET=True).get_structure("ref", ref_file)
    mob_struct = PDBParser(QUIET=True).get_structure("mob", mob_file)
    ref_ca = {}
    for model in ref_struct:
        for chain in model:
            for residue in chain:
                if residue.get_resname() in structure_module.AA_MAP and "CA" in residue:
                    ref_ca[residue.get_id()[1]] = residue["CA"]
    pairs = []
    for model in mob_struct:
        for chain in model:
            for residue in chain:
                resid = residue.get_id()[1]
                if (residue.get_resname() in structure_module.AA_MAP
                        and "CA" in residue and resid in ref_ca):
                    pairs.append((ref_ca[resid], residue["CA"]))
    return pairs


def _fitted_rmsd(pairs):
    sup = Superimposer()
    sup.set_atoms([a for a, _ in pairs], [b for _, b in pairs])
    return sup.rms


def _two_model_pdb(path):
    """手写一个含两个 model 的 PDB：model 2 整体沿 x 平移 100 Å。"""
    lines = []
    for model, shift in ((1, 0.0), (2, 100.0)):
        lines.append("MODEL     %d" % model)
        for i in range(1, 11):
            lines.append("ATOM  %5d  CA  ALA A%4d    %8.3f%8.3f%8.3f  1.00  0.00           C"
                         % (i, i, i * 3.8 + shift, 0.0, 0.0))
        lines.append("ENDMDL")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


class AlignByResidTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

    def path(self, name):
        return os.path.join(self.tmp, name)

    def test_single_chain_is_aligned(self):
        ref = fixtures.make_chain_structure(self.path("ref.pdb"), [("A", (0.0, 0.0, 0.0))])
        mob = fixtures.make_chain_structure(self.path("mob.pdb"), [("A", (5.0, 2.0, -1.0))])
        out = self.path("out.pdb")
        self.assertTrue(align_by_resid(ref, mob, out))
        np.testing.assert_allclose(_ca_coords(out, "A"), _ca_coords(ref, "A"), atol=1e-3)

    def test_multi_chain_same_numbering_new_matches_per_chain(self):
        chains = [("A", (0.0, 0.0, 0.0)), ("B", (0.0, 20.0, 0.0))]
        ref = fixtures.make_chain_structure(self.path("ref.pdb"), chains)
        mob = fixtures.make_chain_structure(self.path("mob.pdb"),
                                            [("A", (5.0, 0.0, 0.0)), ("B", (5.0, 20.0, 0.0))])
        out = self.path("out.pdb")
        self.assertTrue(align_by_resid(ref, mob, out))
        np.testing.assert_allclose(_ca_coords(out, "A"), _ca_coords(ref, "A"), atol=1e-3)
        np.testing.assert_allclose(_ca_coords(out, "B"), _ca_coords(ref, "B"), atol=1e-3)
        # 取证：旧逻辑把两条链的同号残基混在一个字典里，叠合 RMSD 很大
        legacy_rmsd = _fitted_rmsd(_legacy_matched_pairs(ref, mob))
        self.assertGreater(legacy_rmsd, 5.0)

    def test_placeholder_chain_map_is_required(self):
        cif = fixtures.make_chain_structure(self.path("real.cif"),
                                            [("Q", (0.0, 0.0, 0.0)), ("R", (0.0, 30.0, 0.0))])
        placeholder = self.path("placeholder.pdb")
        chain_map = cif_to_pdb_placeholders(cif, placeholder)
        self.assertEqual(chain_map, {"A": "Q", "B": "R"})
        split_dir = self.path("split")
        os.makedirs(split_dir)
        per_chain = dict(split_structure_to_chains(cif, split_dir))

        ref = per_chain["Q"]                      # 真链号 Q
        out = self.path("out.pdb")
        self.assertFalse(align_by_resid(ref, placeholder, out))
        self.assertFalse(os.path.exists(out))     # 无映射：无交集，不写文件

        self.assertTrue(align_by_resid(ref, placeholder, out, mob_chain_map={"A": "Q"}))
        # 映射只用于构造匹配键：输出仍是 mob 自己的链号（A），几何与 ref 的 Q 一致
        np.testing.assert_allclose(_ca_coords(out, "A"), _ca_coords(ref, "Q"), atol=1e-3)

    def test_too_few_matches_returns_false_without_writing(self):
        ref = fixtures.make_chain_structure(self.path("ref.pdb"), [("A", (0.0, 0.0, 0.0))])
        mob = fixtures.make_chain_structure(self.path("mob.pdb"), [("C", (0.0, 0.0, 0.0))])
        out = self.path("out.pdb")
        self.assertFalse(align_by_resid(ref, mob, out))
        self.assertFalse(os.path.exists(out))

    def test_only_first_model_is_used(self):
        path = _two_model_pdb(self.path("two_models.pdb"))
        parsed = PDBParser(QUIET=True).get_structure("s", path)
        atoms = structure_module._ca_by_residue(parsed)
        self.assertEqual(len(atoms), 10)
        np.testing.assert_allclose(atoms[("A", 1, " ")].get_coord(),
                                   [3.8, 0.0, 0.0], atol=1e-3)


if __name__ == "__main__":
    unittest.main()
