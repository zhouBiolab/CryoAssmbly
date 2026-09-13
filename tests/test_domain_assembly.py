"""域链组装测试：复合物多链域合并必须按源链分开，并把占位链号恢复为真链号。

复现 probe3 的真实故障：`assemble_domain_chains` 未传 `is_complex` 时，两条链的域
（残基编号各自从 1 开始）被并进同一条链 → 残基 ID 重复 → Biopython 报错 →
`Chain Q+R: domain merge failed; chain dropped` → 空复合物。
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from protassem.assembly import domain_assembler
from protassem.core.structure import pdb_to_cif
from tests import fixtures

COMPLEX_ID = "Q+R"
CHAIN_MAP = {"A": "Q", "B": "R"}


class _StubOrchestrator:
    """最小 orchestrator 状态：只提供域链组装真正读取的字段。"""

    def __init__(self, work_dir, final_dir):
        self.work_dir = work_dir
        self.final_dir = final_dir
        self.needs_domain_assembly = []
        self.chain_records = []
        self.domain_records = {}
        self.domain_adjacency = {}
        self.accepted_chains = []
        self._domain_chain_order = 0
        self.complex_min_cc = 0.10
        self.original_density_mrc = "unused.mrc"
        self.resolution = 5.6
        self.contour = 0.04

    def chain_record(self, chain_id):
        for record in self.chain_records:
            if record["chain_id"] == chain_id:
                return record
        return None


class ComplexDomainMergeTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name
        self.work = Path(self.tmp) / "work"
        self.final = Path(self.tmp) / "final_results"
        self.work.mkdir()
        (self.final / "domain_chains").mkdir(parents=True)

        self.orch = _StubOrchestrator(self.work, self.final)
        self.orch.chain_records.append({
            "chain_id": COMPLEX_ID, "is_complex": True, "chain_map": CHAIN_MAP,
            "pdb_file": "unused.pdb", "txt_file": "unused.txt"})

        # 与 _accept_domain 写出的一致：单链、链号 = 复合物 ID、残基按域切片编号
        self.fitted = [
            self._domain(1, "A", 1, 5, 0.45),
            self._domain(2, "A", 6, 10, 0.45),
            self._domain(4, "B", 1, 5, 0.40),      # 与 domain 1 编号重叠 -> 旧代码在此炸
            self._domain(5, "B", 6, 10, 0.40),
        ]
        self.orch.domain_records[COMPLEX_ID] = list(self.fitted)
        self.orch.domain_adjacency[COMPLEX_ID] = (
            {1: [(1, 5)], 2: [(6, 10)], 4: [(1, 5)], 5: [(6, 10)]}, {})

    def _domain(self, dnum, source_cid, start, end, cc):
        length = end - start + 1
        pdb = fixtures.make_chain_structure(
            os.path.join(self.tmp, "src_%d.pdb" % dnum), [(source_cid, (0.0, 0.0, 0.0))],
            residues=length, residue_start=start)
        cif = os.path.join(self.tmp, "domain_%d.cif" % dnum)
        pdb_to_cif(pdb, cif, chain_id=COMPLEX_ID)
        return {"domain_num": dnum, "status": "accepted", "fitted_cif": cif,
                "source_chain_id": source_cid, "cc_mask": cc}

    def _chain_ids(self, cif_path):
        from Bio.PDB import MMCIFParser
        structure = MMCIFParser(QUIET=True).get_structure("s", cif_path)
        return sorted(chain.id for chain in next(structure.get_models()))

    def test_merge_without_is_complex_reproduces_the_failure(self):
        """旧行为取证：不按链分开 -> 残基 ID 重复 -> 返回 None（整链丢弃）。"""
        result = domain_assembler.merge_domains(
            self.orch, COMPLEX_ID, self.fitted,
            self.orch.domain_adjacency[COMPLEX_ID][0], is_complex=False)
        self.assertIsNone(result)

    def test_merge_with_is_complex_keeps_one_chain_per_source_and_real_ids(self):
        out = domain_assembler.merge_domains(
            self.orch, COMPLEX_ID, self.fitted,
            self.orch.domain_adjacency[COMPLEX_ID][0], is_complex=True)
        self.assertIsNotNone(out)
        self.assertEqual(self._chain_ids(out), ["Q", "R"])   # 占位 A/B -> 真 Q/R

        from Bio.PDB import MMCIFParser
        structure = MMCIFParser(QUIET=True).get_structure("s", out)
        model = next(structure.get_models())
        self.assertEqual(sum(1 for _ in model["Q"]), 10)
        self.assertEqual(sum(1 for _ in model["R"]), 10)

    def test_real_chain_ids_are_not_mapped_twice(self):
        """已经是真链号的 source_chain_id 必须原样通过。"""
        fitted = [self._domain(1, "Q", 1, 5, 0.45), self._domain(4, "R", 1, 5, 0.40)]
        out = domain_assembler.merge_domains(
            self.orch, COMPLEX_ID, fitted, {1: [(1, 5)], 4: [(1, 5)]},
            is_complex=True)
        self.assertIsNotNone(out)
        self.assertEqual(self._chain_ids(out), ["Q", "R"])

    def test_assemble_domain_chains_produces_domain_chain_with_real_ids(self):
        self.orch.needs_domain_assembly.append(COMPLEX_ID)
        with mock.patch.object(domain_assembler, "calculate_cc_mask",
                               return_value=0.42):
            domain_assembler.assemble_domain_chains(self.orch)

        self.assertEqual(len(self.orch.accepted_chains), 1)
        record = self.orch.accepted_chains[0]
        self.assertEqual(record["type"], "domain_chain")
        self.assertEqual(sorted(record["domains"]), [1, 2, 4, 5])
        self.assertTrue(os.path.exists(record["fitted_cif"]))
        self.assertEqual(self._chain_ids(record["fitted_cif"]), ["Q", "R"])


if __name__ == "__main__":
    unittest.main()
