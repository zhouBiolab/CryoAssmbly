"""动态密度缓存版本测试（审计 P2-7）。

`_mask_region` 每轮都会覆盖 `current_density.mrc`，而评分缓存按「路径 + size + mtime_ns」
建键 —— 覆盖同名文件不是版本契约，而且共享池 worker 内的缓存主进程清不掉。
修复后每轮写 `current_density_mNN.mrc`（路径即版本）。
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from protassem.assembly import orchestrator as orchestrator_module


class _Stub:
    """`_mask_region` 需要的最小状态。"""

    def __init__(self, root):
        self.work_dir = Path(root) / "work"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.current_target_txt = str(self.work_dir / "current_target.txt")
        self.current_density_mrc = str(self.work_dir / "current_density.mrc")
        self.original_target_txt = str(Path(root) / "orig_target.txt")
        self.original_density_mrc = str(Path(root) / "orig_density.mrc")
        self._mask_iter = 0
        for path in (self.current_target_txt, self.current_density_mrc,
                     self.original_target_txt, self.original_density_mrc):
            with open(path, "w") as handle:
                handle.write("HEADER\n" * 6)

    def _ensure_work_files(self):
        return orchestrator_module.AssemblyOrchestrator._ensure_work_files(self)


def _fake_mask(source_txt, fitted_pdb, density_mrc, out_txt, out_mrc):
    """模拟掩膜：按输入密度内容派生输出（内容不同 → 代表版本不同）。"""
    with open(out_txt, "w") as handle:
        handle.write("HEADER\n" * 4)
    with open(density_mrc) as handle:
        payload = handle.read() + "masked\n"
    with open(out_mrc, "w") as handle:
        handle.write(payload)
    return True


class DynamicDensityVersionTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name
        self.stub = _Stub(self.root)
        with open(self.stub.current_density_mrc, "w") as handle:
            handle.write("density-v0\n")

    def _mask(self, fitted="fitted.pdb"):
        with mock.patch.object(orchestrator_module, "mask_fitted_region",
                               side_effect=_fake_mask):
            orchestrator_module.AssemblyOrchestrator._mask_region(self.stub, fitted)

    def test_each_mask_round_writes_a_new_density_path(self):
        first_path = self.stub.current_density_mrc
        self._mask()
        after_first = self.stub.current_density_mrc
        self.assertNotEqual(after_first, first_path)
        self.assertIn("current_density_m01.mrc", after_first)
        self.assertTrue(os.path.exists(first_path), "旧版本文件不能被覆盖掉")

        self._mask()
        after_second = self.stub.current_density_mrc
        self.assertIn("current_density_m02.mrc", after_second)
        self.assertNotEqual(after_second, after_first)
        self.assertTrue(os.path.exists(after_first))

        # 内容随轮次变化：评分缓存按路径建键，因此不会命中去年的密度
        contents = [open(path).read() for path in (first_path, after_first, after_second)]
        self.assertEqual(len(set(contents)), 3)

    def test_ensure_work_files_restores_the_current_version(self):
        self._mask()
        current = self.stub.current_density_mrc
        os.remove(current)
        self.stub._ensure_work_files()
        self.assertTrue(os.path.exists(current))


if __name__ == "__main__":
    unittest.main()
