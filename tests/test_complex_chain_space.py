"""复合物链号空间回归测试（审计 P1-1 / P1-2）。

三个空间必须分清：
  - 真实链号（最终输出）：如 B/C；
  - 占位链号（内部位姿/PDB、`source_chain_id`）：如 A/B；
  - 组件 ID：如 `Q+R`（`chain_id`，中间产物用）。

约定：**占位 -> 真实的映射只在最终输出处做一次。**
本文件用"真实链号与占位链号有交集"（B/C vs A/B）的构造覆盖审计里的触发条件 ——
Q/R 案例无法暴露该问题，因为 Q/R 不是 chain_map 的键。
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from Bio.PDB import MMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionException

from protassem.assembly.domain_assembler import (_handle_single_domain,
                                                  merge_domains)
from protassem.core.structure import (cif_to_pdb_placeholders,
                                      write_structure_with_chain_map)
from tests import fixtures


def _chain_ids(path):
    structure = MMCIFParser(QUIET=True).get_structure("s", str(path))
    return sorted(chain.id for chain in list(structure)[0])


class _Orch:
    """domain_assembler 需要的最小 orchestrator 替身。"""

    def __init__(self, root, chain_map, chain_id="Q+R"):
        self.work_dir = Path(root) / "work"
        self.final_dir = Path(root) / "final"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        (self.final_dir / "domain_chains").mkdir(parents=True, exist_ok=True)
        self.chain_records = [{"chain_id": chain_id, "chain_map": chain_map}]
        self.accepted_chains = []
        self._domain_chain_order = 0
        self.complex_min_cc = 0.35
        self.original_density_mrc = "unused.mrc"
        self.resolution = 6.0
        self.contour = 0.04

    def chain_record(self, chain_id):
        for record in self.chain_records:
            if record["chain_id"] == chain_id:
                return record
        return None


class ComplexChainSpaceTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def path(self, name):
        return str(self.root / name)

    def _placeholder_map(self):
        """真实 B/C -> 占位 A/B（与真实链号有交集）。"""
        real_cif = fixtures.make_chain_structure(
            self.path("real.cif"), [("B", (0.0, 0.0, 0.0)), ("C", (30.0, 0.0, 0.0))],
            residues=6)
        chain_map = cif_to_pdb_placeholders(real_cif, self.path("placeholder.pdb"))
        self.assertEqual(chain_map, {"A": "B", "B": "C"})
        return chain_map

    def test_merge_domains_keeps_placeholder_space(self):
        """逐域合并的结果必须留在内部（占位）空间，供拟合流程继续使用。"""
        chain_map = self._placeholder_map()
        dom_a = fixtures.make_chain_structure(self.path("dom_a.cif"), [("A", (0.0, 0.0, 0.0))],
                                              residues=6)
        dom_b = fixtures.make_chain_structure(self.path("dom_b.cif"), [("B", (30.0, 0.0, 0.0))],
                                              residues=6)
        fitted = [{"domain_num": 1, "fitted_cif": dom_a, "source_chain_id": "A"},
                  {"domain_num": 2, "fitted_cif": dom_b, "source_chain_id": "B"}]
        ranges = {1: [(1, 6)], 2: [(1, 6)]}
        orch = _Orch(self.root, chain_map)

        merged = merge_domains(orch, "Q+R", fitted, ranges, is_complex=True)
        self.assertIsNotNone(merged)
        self.assertEqual(_chain_ids(merged), ["A", "B"])

        # 最终输出：只映射一次 → 真实链号 B/C
        final = self.path("final.cif")
        write_structure_with_chain_map(merged, chain_map, final)
        self.assertEqual(_chain_ids(final), ["B", "C"])

    def test_second_mapping_is_the_audited_failure(self):
        """取证：对已是真实链号的文件再映射一次 → 撞名（审计报告的 PDBConstructionException）。"""
        chain_map = self._placeholder_map()
        dom_a = fixtures.make_chain_structure(self.path("d_a.cif"), [("A", (0.0, 0.0, 0.0))],
                                              residues=6)
        dom_b = fixtures.make_chain_structure(self.path("d_b.cif"), [("B", (30.0, 0.0, 0.0))],
                                              residues=6)
        fitted = [{"domain_num": 1, "fitted_cif": dom_a, "source_chain_id": "A"},
                  {"domain_num": 2, "fitted_cif": dom_b, "source_chain_id": "B"}]
        orch = _Orch(self.root, chain_map)
        merged = merge_domains(orch, "Q+R", fitted, {1: [(1, 6)], 2: [(1, 6)]},
                               is_complex=True)
        real_space = self.path("real_space.cif")
        write_structure_with_chain_map(merged, chain_map, real_space)   # 第一次（正确）
        with self.assertRaises(PDBConstructionException):
            write_structure_with_chain_map(real_space, chain_map,
                                           self.path("twice.cif"))       # 第二次（旧缺陷）

    def test_multichar_real_chain_ids_survive_the_placeholder_round_trip(self):
        """多字符真实链号（AA/BB）：占位 A/B → 合并保持占位 → 最终恢复成 AA/BB。

        多字符链号只能落在 CIF 里（PDB 写不下），这条用例覆盖"恢复后再写 improved PDB"的兼容风险。
        """
        real_cif = fixtures.make_chain_structure(
            self.path("multi.cif"), [("AA", (0.0, 0.0, 0.0)), ("BB", (30.0, 0.0, 0.0))],
            residues=6)
        chain_map = cif_to_pdb_placeholders(real_cif, self.path("multi_ph.pdb"))
        self.assertEqual(sorted(chain_map.values()), ["AA", "BB"])
        self.assertEqual(sorted(chain_map.keys()), ["A", "B"])

        dom_a = fixtures.make_chain_structure(self.path("m_dom_a.cif"), [("A", (0.0, 0.0, 0.0))],
                                              residues=6)
        dom_b = fixtures.make_chain_structure(self.path("m_dom_b.cif"), [("B", (30.0, 0.0, 0.0))],
                                              residues=6)
        fitted = [{"domain_num": 1, "fitted_cif": dom_a, "source_chain_id": "A"},
                  {"domain_num": 2, "fitted_cif": dom_b, "source_chain_id": "B"}]
        orch = _Orch(self.root, chain_map)

        merged = merge_domains(orch, "Q+R", fitted, {1: [(1, 6)], 2: [(1, 6)]},
                               is_complex=True)
        self.assertEqual(_chain_ids(merged), ["A", "B"])       # 内部空间

        final = self.path("multi_final.cif")
        write_structure_with_chain_map(merged, chain_map, final)
        self.assertEqual(_chain_ids(final), ["AA", "BB"])      # 最终空间（CIF 能承载多字符）

    def test_improved_pose_survives_accept_chain_and_keeps_real_ids(self):
        """串起真实链路：逐域改善产物（内部空间）→ `_accept_chain()` → 最终 CIF。

        只替代昂贵的评分/优化：改善步骤用"把合并结果写成 PDB"代替（与
        `chain_fitter._improve_by_domain_reassembly` 的产物形态一致）。
        """
        from Bio.PDB import PDBIO
        from protassem.assembly.orchestrator import AssemblyOrchestrator

        chain_map = self._placeholder_map()                 # 真实 B/C，占位 A/B
        dom_a = fixtures.make_chain_structure(self.path("i_dom_a.cif"), [("A", (0.0, 0.0, 0.0))],
                                              residues=6)
        dom_b = fixtures.make_chain_structure(self.path("i_dom_b.cif"), [("B", (30.0, 0.0, 0.0))],
                                              residues=6)
        fitted = [{"domain_num": 1, "fitted_cif": dom_a, "source_chain_id": "A"},
                  {"domain_num": 2, "fitted_cif": dom_b, "source_chain_id": "B"}]
        orch = _Orch(self.root, chain_map)
        merged = merge_domains(orch, "Q+R", fitted, {1: [(1, 6)], 2: [(1, 6)]},
                               is_complex=True)

        # 改善步骤：合并结果（内部空间）→ improved PDB
        improved = self.path("improved_Q+R.pdb")
        structure = MMCIFParser(QUIET=True).get_structure("a", merged)
        io = PDBIO()
        io.set_structure(structure)
        io.save(improved)
        self.assertEqual(_chain_ids(improved if improved.endswith(".cif") else merged),
                         ["A", "B"])

        # 接受步骤：真实 `_accept_chain`（唯一恢复点）
        stub = mock.MagicMock()
        stub._chain_order = 0
        stub.final_dir = Path(self.root) / "final"
        (stub.final_dir / "chains").mkdir(parents=True, exist_ok=True)
        stub.accepted_chains = []
        stub.accepted_fitted_pdbs = []
        record = {"chain_id": "Q+R", "is_complex": True, "chain_map": chain_map,
                  "status": "pending"}
        AssemblyOrchestrator._accept_chain(stub, record, improved, 0.4376)

        final = record["fitted_cif"]
        self.assertTrue(os.path.exists(final))
        self.assertEqual(_chain_ids(final), ["B", "C"])      # 恰好映射一次
        self.assertEqual(stub.accepted_chains[0]["cc_mask"], 0.4376)

    def test_single_domain_complex_restores_real_chain_id(self):
        """只接受一个域时，最终 domain_chain 文件必须是真实链号（不是组件 ID、不是占位）。"""
        chain_map = self._placeholder_map()
        fitted_pdb = fixtures.make_chain_structure(
            self.path("fitted_domain.pdb"), [("A", (0.0, 0.0, 0.0))], residues=6)
        component_cif = fixtures.make_chain_structure(
            self.path("domain_component.cif"), [("Q+R", (0.0, 0.0, 0.0))], residues=6)
        domain_rec = {"domain_num": 1, "source_chain_id": "A",
                      "fitted_pdb": fitted_pdb, "fitted_cif": component_cif,
                      "cc_mask": 0.5}
        orch = _Orch(self.root, chain_map)

        with mock.patch("protassem.assembly.domain_assembler.calculate_cc_mask",
                        return_value=0.5):
            _handle_single_domain(orch, "Q+R", orch.chain_records[0], domain_rec)

        self.assertEqual(len(orch.accepted_chains), 1)
        final = orch.accepted_chains[0]["fitted_cif"]
        self.assertTrue(os.path.exists(final))
        self.assertEqual(_chain_ids(final), ["B"])
        # 过滤版与全量同源（cc 达标）→ 指向同一个最终文件
        self.assertEqual(orch.accepted_chains[0]["fitted_cif_filtered"], final)

    def test_single_domain_without_chain_map_keeps_chain_id(self):
        """非复合物（无 chain_map）：保持原链号，不做任何映射。"""
        fitted_pdb = fixtures.make_chain_structure(
            self.path("plain_domain.pdb"), [("A", (0.0, 0.0, 0.0))], residues=6)
        domain_rec = {"domain_num": 1, "source_chain_id": "A",
                      "fitted_pdb": fitted_pdb, "fitted_cif": fitted_pdb,
                      "cc_mask": 0.1}
        orch = _Orch(self.root, {}, chain_id="A")
        with mock.patch("protassem.assembly.domain_assembler.calculate_cc_mask",
                        return_value=0.1):
            _handle_single_domain(orch, "A", orch.chain_records[0], domain_rec)
        final = orch.accepted_chains[0]["fitted_cif"]
        self.assertEqual(_chain_ids(final), ["A"])
        # cc 未达标 → 不进过滤复合物
        self.assertIsNone(orch.accepted_chains[0]["fitted_cif_filtered"])


if __name__ == "__main__":
    unittest.main()
