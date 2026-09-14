"""掩码路径的部分失败测试（第二轮审计#1）。

`use_mask=True` 时，同一个 mask 内可能有多次评估（config × voxel/fps）。旧 `_publish` 先看
成功文件再看错误，于是"voxel 成功 + fps 抛异常"会被发布成 ok —— 日志里有异常、`Failed` 却是 0、
台账还报 ok，三处互相矛盾。这里直接驱动 `run_inference(use_mask=True, ...)`（模型/点云预处理/
单对推理为 stub，尾部流水线关闭），覆盖全成功 / 部分失败 / 全部失败三种情况。
"""

import json
import os
import tempfile
import unittest
from unittest import mock

from protassem.fitting import demo_mask
from protassem.fitting.candidate_ledger import LEDGER_NAME


def _read_ledger(path):
    records = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


class MaskedPartialFailureTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name
        self.output_dir = os.path.join(self.dir, "out")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "temp"), exist_ok=True)
        self.mask_file = os.path.join(self.output_dir, "temp", "mask_0001.txt")
        with open(self.mask_file, "w", encoding="utf-8") as handle:
            handle.write("1.0 2.0 3.0\n")

    def _run(self, failing_sampling):
        """failing_sampling: 会抛异常的采样名集合（{"voxel"} / {"fps"} / 两者 / 空）。"""

        def fake_process_single_pair(src_data, tgt_data, source, target, chain_pdb,
                                     model_net, cfg, cid, sm, output_dir,
                                     use_mask=False, **kwargs):
            if sm in failing_sampling:
                raise RuntimeError("模拟评估失败 (%s)" % sm)
            path = os.path.join(output_dir, "pred_%d_%s.pdb" % (cid, sm))
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("HEADER\nEND\n")
            return {"config_id": cid, "sampling_method": sm,
                    "pred_pdb_path": path, "overlap": 0.3,
                    "target_file": self.mask_file}

        with mock.patch.object(demo_mask, "_get_model",
                               return_value=(object(), None)), \
                mock.patch.object(demo_mask, "preprocess_point_cloud_data",
                                  return_value={}), \
                mock.patch.object(demo_mask, "generate_masks_once",
                                  return_value=(None, None)), \
                mock.patch.object(demo_mask, "find_mask_files",
                                  return_value=[self.mask_file]), \
                mock.patch.object(demo_mask, "process_single_pair",
                                  side_effect=fake_process_single_pair):
            demo_mask.run_inference(
                target="t.txt", source="s.txt", chain_pdb=None,
                output_dir=self.output_dir, weights=None, use_mask=True,
                configs="1", seed=1, tail_pipeline_enabled=False)
        return _read_ledger(os.path.join(self.output_dir, LEDGER_NAME))

    def _states(self, records):
        return [r["state"] for r in records if r["kind"] == "candidate"]

    def _end(self, records):
        return [r for r in records if r["kind"] == "end"][0]

    def test_all_evaluations_succeed(self):
        records = self._run(set())
        self.assertEqual(self._states(records), ["ok"])
        self.assertEqual(self._end(records)["status"], "ok")

    def test_partial_failure_is_error_not_ok(self):
        """一个采样成功、另一个抛异常 → 该候选必须是 error（不能被成功掩盖）。"""
        records = self._run({"fps"})
        self.assertEqual(self._states(records), ["error"])
        self.assertIn("模拟评估失败", records[0]["error"])
        end = self._end(records)
        self.assertEqual(end["status"], "error")
        self.assertIn("1 个候选执行失败", end["error"])

    def test_all_evaluations_fail(self):
        records = self._run({"voxel", "fps"})
        self.assertEqual(self._states(records), ["error"])
        self.assertEqual(self._end(records)["status"], "error")

    def test_failures_are_counted_in_the_run_totals(self):
        """异常要进入统一统计：台账里有 error 记录，请求级状态也必须是 error。"""
        records = self._run({"fps"})
        failures = [r for r in records
                    if r["kind"] == "candidate" and r["state"] == "error"]
        self.assertEqual(len(failures), 1)
        self.assertEqual(self._end(records)["status"], "error")


if __name__ == "__main__":
    unittest.main()
