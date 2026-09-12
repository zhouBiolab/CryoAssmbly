"""链号池、占位容量、跳过输入与失败策略测试（无 GPU）。"""

import itertools
import os
import tempfile
import unittest
from unittest import mock

from protassem.assembly.orchestrator import (AssemblyOrchestrator,
                                             _pre_screen_cc_worker)
from protassem.core.structure import (cif_to_pdb_placeholders, logical_chain_ids,
                                      pdb_placeholder_ids)
from protassem.fitting import pipeline as fitting_pipeline
from tests import fixtures


class ChainIdPoolTest(unittest.TestCase):

    def test_logical_pool_order(self):
        pool = list(itertools.islice(logical_chain_ids(), 54))
        self.assertEqual(pool[0], "A")
        self.assertEqual(pool[25], "Z")
        self.assertEqual(pool[26], "a")
        self.assertEqual(pool[51], "z")
        self.assertEqual(pool[52], "AA")
        self.assertEqual(pool[53], "AB")

    def test_placeholder_pool_has_sixty_two_single_characters(self):
        pool = list(pdb_placeholder_ids())
        self.assertEqual(len(pool), 62)
        self.assertEqual(pool[:3], ["A", "B", "C"])
        self.assertEqual(pool[52:], ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
        self.assertTrue(all(len(c) == 1 for c in pool))


class CifPlaceholderTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

    def _cif_with_chains(self, name, count):
        ids = list(itertools.islice(logical_chain_ids(), count))
        cif = fixtures.make_chain_structure(
            os.path.join(self.tmp, name),
            [(cid, (0.0, i * 12.0, 0.0)) for i, cid in enumerate(ids)],
            residues=2)
        return cif, ids

    def test_full_capacity_maps_and_includes_digits(self):
        cif, ids = self._cif_with_chains("full.cif", 62)
        chain_map = cif_to_pdb_placeholders(cif, os.path.join(self.tmp, "full.pdb"))
        self.assertEqual(len(chain_map), 62)
        self.assertEqual(chain_map["A"], ids[0])
        self.assertEqual(chain_map["0"], ids[52])   # 数字占位仍在使用
        self.assertEqual(chain_map["9"], ids[61])

    def test_over_capacity_raises_instead_of_truncating(self):
        cif, _ = self._cif_with_chains("over.cif", 63)
        with self.assertRaises(ValueError) as ctx:
            cif_to_pdb_placeholders(cif, os.path.join(self.tmp, "over.pdb"))
        self.assertIn("capacity is 62", str(ctx.exception))


class PrepareChainsTest(unittest.TestCase):
    """跳过的输入要被记录；一个可用链都没有时直接失败。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

    def _orchestrator(self, source_dir):
        return AssemblyOrchestrator(
            target_txt=os.path.join(source_dir, "target.txt"),
            source_dir=source_dir,
            density_mrc=os.path.join(source_dir, "map.mrc"),
            resolution=3.0, contour=0.02,
            output_dir=os.path.join(source_dir, "out"),
            num_processes=1)

    def _write_usable_chain(self, source_dir, chain_id="A"):
        fixtures.make_chain_structure(os.path.join(source_dir, "chain_%s.pdb" % chain_id),
                                      [(chain_id, (0.0, 0.0, 0.0))])
        fixtures.make_sample_txt(os.path.join(source_dir, "chain_%s.txt" % chain_id))

    def test_unreadable_structure_is_recorded_not_dropped(self):
        source = os.path.join(self.tmp, "src")
        os.makedirs(source)
        self._write_usable_chain(source)
        with open(os.path.join(source, "broken.pdb"), "w", encoding="utf-8") as handle:
            handle.write("not a pdb at all\n")

        orch = self._orchestrator(source)
        orch._prepare_chains()

        self.assertEqual(orch.input_count, 2)
        self.assertEqual(len(orch.chain_records), 1)
        self.assertEqual(len(orch.skipped_inputs), 1)
        self.assertIn("broken.pdb", orch.skipped_inputs[0][0])

    def test_no_usable_chain_raises(self):
        source = os.path.join(self.tmp, "empty_src")
        os.makedirs(source)
        with open(os.path.join(source, "broken.pdb"), "w", encoding="utf-8") as handle:
            handle.write("not a pdb at all\n")

        orch = self._orchestrator(source)
        with self.assertRaises(RuntimeError) as ctx:
            orch._prepare_chains()
        self.assertIn("no usable chains", str(ctx.exception))


class WorkerFailureTest(unittest.TestCase):
    """worker 失败必须带文件名报错，不能退化成 0 分/None。"""

    def test_cc_worker_raises_with_file_name(self):
        with mock.patch.object(fitting_pipeline, "calculate_cc_mask",
                               side_effect=ValueError("boom")):
            with self.assertRaises(RuntimeError) as ctx:
                fitting_pipeline._cc_worker(("pred_x_0.500000.pdb", "map.mrc", 3.0, 0.02))
        self.assertIn("pred_x_0.500000.pdb", str(ctx.exception))
        self.assertIn("boom", str(ctx.exception))

    def test_pre_screen_worker_raises_with_file_name(self):
        import protassem.assembly.orchestrator as orchestrator_module
        with mock.patch.object(orchestrator_module, "calculate_cc_mask",
                               side_effect=ValueError("boom")):
            with self.assertRaises(RuntimeError) as ctx:
                _pre_screen_cc_worker(("map.mrc", "chain_A.pdb", 3.0, 0.02))
        self.assertIn("chain_A.pdb", str(ctx.exception))


class DigitPlaceholderDownstreamTest(unittest.TestCase):
    """V2 实测：数字占位链号在读取/评分/USalign 上可用（DomainParser 未覆盖）。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

    def test_read_structure_scoring_and_usalign(self):
        from protassem.core.io import read_structure
        from protassem.core.scoring import calculate_cc_mask
        from protassem.core.similarity import calculate_tm_score

        digit_zero = fixtures.make_chain_structure(
            os.path.join(self.tmp, "digit0.pdb"), [("0", (0.0, 0.0, 0.0))])
        digit_one = fixtures.make_chain_structure(
            os.path.join(self.tmp, "digit1.pdb"), [("1", (5.0, 0.0, 0.0))])
        density = fixtures.make_mrc(os.path.join(self.tmp, "map.mrc"))

        coords, _ = read_structure(digit_zero)
        self.assertEqual(len(coords), 10)
        self.assertIsInstance(calculate_cc_mask(density, digit_zero, 3.0, 0.02), float)
        self.assertGreater(calculate_tm_score(digit_zero, digit_one), 0.0)


if __name__ == "__main__":
    unittest.main()
