"""O4 空结果输出契约测试：全空 / 仅过滤版为空 / 正常非空 / 旧输出残留。

用最小 orchestrator 状态直接驱动真实的 build_complex / clear_stale_outputs /
_output_status / refine_step.maybe_refine，不需要 GPU 与完整流水线。
"""

import os
import tempfile
import unittest
from pathlib import Path

from protassem.assembly import complex_builder, refine_step
from protassem.assembly.orchestrator import _output_status
from tests import fixtures


class _StubOrchestrator:
    """只提供 refine 门控与状态统计读取的字段。"""

    def __init__(self, final_dir):
        self.final_dir = final_dir
        self.work_dir = final_dir / "work"
        self.do_refine = True
        self.domain_records = {}
        self.accepted_domain_pdbs = []
        self.accepted_chains = []
        self.excluded_domains = []

    def _is_similar_to_any(self, *_args):
        return False


class ComplexOutputContractTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name) / "final_results"
        self.out.mkdir(parents=True)
        self.component = fixtures.make_chain_structure(
            os.path.join(self._tmp.name, "component.cif"), [("Q", (0.0, 0.0, 0.0))],
            residues=5)

    def _build(self, records, cif_key, out_name):
        return complex_builder.build_complex(records, str(self.out),
                                             cif_key=cif_key, out_name=out_name)

    def test_all_empty_writes_nothing_and_returns_none(self):
        all_cif, _ = self._build([], "fitted_cif", "assembled_complex_all.cif")
        filtered_cif, _ = self._build([], "fitted_cif_filtered", "assembled_complex.cif")
        self.assertIsNone(all_cif)
        self.assertIsNone(filtered_cif)
        self.assertFalse((self.out / "assembled_complex_all.cif").exists())
        self.assertFalse((self.out / "assembled_complex.cif").exists())

    def test_filtered_empty_keeps_full_and_records_both_statuses(self):
        record = {"type": "domain_chain", "chain_id": "Q",
                  "fitted_cif": self.component, "fitted_cif_filtered": None,
                  "domains": [1, 2]}
        all_cif, _ = self._build([record], "fitted_cif", "assembled_complex_all.cif")
        filtered_cif, _ = self._build([record], "fitted_cif_filtered",
                                      "assembled_complex.cif")
        self.assertIsNotNone(all_cif)
        self.assertIsNone(filtered_cif)                      # 过滤版为空 → 不写出
        self.assertTrue((self.out / "assembled_complex_all.cif").exists())
        self.assertFalse((self.out / "assembled_complex.cif").exists())

        orch = _StubOrchestrator(self.out)
        orch.accepted_chains = [record]
        orch.domain_records = {"Q": [{"status": "accepted"}, {"status": "rejected"}]}
        status = _output_status(orch)
        self.assertEqual(status["accepted_components"], 1)
        self.assertEqual(status["accepted_domains"], 1)
        self.assertEqual(status["merged_domain_chains"], 1)
        self.assertEqual(status["filtered_out_domain_chains"], 1)

    def test_normal_non_empty_writes_both_with_expected_chains(self):
        record = {"type": "chain", "chain_id": "Q",
                  "fitted_cif": self.component, "fitted_cif_filtered": self.component}
        all_cif, _ = self._build([record], "fitted_cif", "assembled_complex_all.cif")
        filtered_cif, _ = self._build([record], "fitted_cif_filtered",
                                      "assembled_complex.cif")
        self.assertIsNotNone(all_cif)
        self.assertIsNotNone(filtered_cif)
        from Bio.PDB import MMCIFParser
        for path in (all_cif, filtered_cif):
            structure = MMCIFParser(QUIET=True).get_structure("s", path)
            self.assertEqual([c.id for c in next(structure.get_models())], ["Q"])

    def test_stale_outputs_from_previous_run_are_removed(self):
        stale = ["assembled_complex.cif", "assembled_complex_all.cif",
                 "refined_complex.cif", "homo_chain_refined_complex.cif"]
        for name in stale:
            (self.out / name).write_text("stale from previous run\n", encoding="utf-8")
        removed = complex_builder.clear_stale_outputs(str(self.out))
        self.assertEqual(sorted(removed), sorted(stale))
        for name in stale:
            self.assertFalse((self.out / name).exists())

        # 空结果重跑后不应复活任何旧文件
        self._build([], "fitted_cif", "assembled_complex_all.cif")
        self.assertFalse((self.out / "assembled_complex_all.cif").exists())

    def test_stale_target_is_removed_when_current_run_is_empty(self):
        target = self.out / "assembled_complex.cif"
        target.write_text("stale from previous run\n", encoding="utf-8")
        result, _ = self._build([], "fitted_cif_filtered", "assembled_complex.cif")
        self.assertIsNone(result)
        self.assertFalse(target.exists())

    def test_refine_is_skipped_when_no_assembled_complex_exists(self):
        orch = _StubOrchestrator(self.out)
        orch.domain_records = {"Q": [{"status": "accepted"}]}
        orch.accepted_domain_pdbs = ["something.pdb"]
        self.assertIsNone(refine_step.maybe_refine(orch))     # 不抛异常、明确跳过


if __name__ == "__main__":
    unittest.main()
