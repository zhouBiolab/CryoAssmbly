"""run_pipeline 入口校验测试：无效输入必须在建立输出目录之前失败。"""

import os
import tempfile
import unittest

from protassem import pipeline
from tests import fixtures


def _write(path, text="placeholder\n"):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


class ValidateInputsTest(unittest.TestCase):
    """边界函数本身：每种非法输入给出确定的异常类型与消息关键字。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name
        self.mrc = fixtures.make_mrc(os.path.join(self.tmp, "map.mrc"))
        self.struct = _write(os.path.join(self.tmp, "chain_A.pdb"))

    def validate(self, **overrides):
        kwargs = {"density_mrc": self.mrc,
                  "structure_files": [self.struct],
                  "resolution": 3.0,
                  "contour": 0.02,
                  "voxel_size": 2.0}
        kwargs.update(overrides)
        return pipeline._validate_inputs(**kwargs)

    def test_valid_inputs_pass(self):
        self.assertIsNone(self.validate())

    def test_missing_density_map(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            self.validate(density_mrc=os.path.join(self.tmp, "nope.mrc"))
        self.assertIn("density map not found", str(ctx.exception))

    def test_density_map_must_be_mrc(self):
        other = _write(os.path.join(self.tmp, "map.txt"))
        with self.assertRaises(ValueError) as ctx:
            self.validate(density_mrc=other)
        self.assertIn("must be a .mrc", str(ctx.exception))

    def test_empty_structure_list(self):
        with self.assertRaises(ValueError) as ctx:
            self.validate(structure_files=[])
        self.assertIn("no structure files", str(ctx.exception))

    def test_missing_structure_file_is_named(self):
        missing = os.path.join(self.tmp, "chain_B.pdb")
        with self.assertRaises(FileNotFoundError) as ctx:
            self.validate(structure_files=[self.struct, missing])
        self.assertIn("chain_B.pdb", str(ctx.exception))

    def test_resolution_must_be_positive_and_finite(self):
        for value in (0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.validate(resolution=value)

    def test_contour_none_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self.validate(contour=None)
        self.assertIn("contour", str(ctx.exception))

    def test_contour_must_be_finite(self):
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.validate(contour=value)

    def test_voxel_size_must_be_positive(self):
        for value in (0, -2.0):
            with self.assertRaises(ValueError):
                self.validate(voxel_size=value)


class RunPipelineBoundaryTest(unittest.TestCase):
    """校验必须发生在 os.makedirs/setup_logging 之前。"""

    def test_invalid_input_does_not_create_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out")
            with self.assertRaises(FileNotFoundError):
                pipeline.run_pipeline(
                    density_mrc=os.path.join(tmp, "missing.mrc"),
                    structure_files=[os.path.join(tmp, "chain_A.pdb")],
                    resolution=3.0, contour=0.02, output_dir=out)
            self.assertFalse(os.path.exists(out))


if __name__ == "__main__":
    unittest.main()
