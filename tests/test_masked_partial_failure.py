"""掩码路径的评估失败测试。

`use_mask=True` 时，同一个 mask 内可能有多次评估（config × voxel/fps）。契约：

  - 同一 mask 内**有成功候选** → 按原规则选最优并发布 `ok`；失败记录保留在结果集里参与统计；
  - 同一 mask 内**没有成功候选**且有执行错误 → 发布 `error` 且 `reason=MASK_ERROR_REASON`
    （客户端会跳过它，请求级结束状态不受影响）；
  - 每次异常在掩码结果集与全局结果集**各记一次**（不重复追加）。

这里直接驱动 `run_inference(use_mask=True, ...)`（模型/点云预处理/单对推理为 stub，
尾部流水线关闭）。
"""

import json
import os
import tempfile
import unittest
from unittest import mock

from protassem.fitting import demo_mask
from protassem.fitting.candidate_ledger import LEDGER_NAME, MASK_ERROR_REASON


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
        """failing_sampling: 会抛异常的采样名集合（{"voxel"} / {"fps"} / 两者 / 空）。

        返回 `(台账记录, 服务端日志行)`。
        """

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
            with self.assertLogs("protassem.fitting.demo_mask", level="INFO") as captured:
                demo_mask.run_inference(
                    target="t.txt", source="s.txt", chain_pdb=None,
                    output_dir=self.output_dir, weights=None, use_mask=True,
                    configs="1", seed=1, tail_pipeline_enabled=False)
        records = _read_ledger(os.path.join(self.output_dir, LEDGER_NAME))
        return records, captured.output

    def _states(self, records):
        return [r["state"] for r in records if r["kind"] == "candidate"]

    def _end(self, records):
        return [r for r in records if r["kind"] == "end"][0]

    def _totals(self, log_lines):
        for line in log_lines:
            if "Total:" in line:
                return line.split("Total:", 1)[1].strip()
        raise AssertionError("日志里没有 Total 行：%r" % (log_lines,))

    def test_all_evaluations_succeed(self):
        records, log_lines = self._run(set())
        self.assertEqual(self._states(records), ["ok"])
        self.assertEqual(self._end(records)["status"], "ok")
        self.assertEqual(self._totals(log_lines), "2, Success: 2, Failed: 0")

    def test_partial_failure_still_publishes_the_successful_candidate(self):
        """一个采样成功、另一个抛异常 → 用成功候选发布 ok，失败记录保留在统计里。"""
        records, log_lines = self._run({"fps"})
        self.assertEqual(self._states(records), ["ok"])
        self.assertEqual(self._end(records)["status"], "ok")
        candidate = [r for r in records if r["kind"] == "candidate"][0]
        self.assertNotIn("error", candidate)
        # 失败没有被丢掉：仍进入统一统计（且只记一次）
        self.assertEqual(self._totals(log_lines), "2, Success: 1, Failed: 1")

    def test_all_evaluations_fail_publishes_mask_error_reason(self):
        """全部评估失败 → error + reason，客户端据此跳过；请求级仍是 ok。"""
        records, log_lines = self._run({"voxel", "fps"})
        self.assertEqual(self._states(records), ["error"])
        candidate = [r for r in records if r["kind"] == "candidate"][0]
        self.assertEqual(candidate["reason"], MASK_ERROR_REASON)
        self.assertIn("模拟评估失败", candidate["error"])
        self.assertEqual(self._end(records)["status"], "ok")
        self.assertEqual(self._totals(log_lines), "2, Success: 0, Failed: 2")

    def test_each_exception_is_recorded_once_per_result_set(self):
        """同一异常在掩码结果集与全局结果集各记一次，不重复追加。

        旧实现在异常分支里对 `mask_results` 追加两次，同一个失败会被计成两个：
        日志 `Total: 3, Success: 1, Failed: 2`（正确值是 `Total: 2, Success: 1, Failed: 1`）。
        """
        _records, log_lines = self._run({"fps"})
        self.assertEqual(self._totals(log_lines), "2, Success: 1, Failed: 1")


if __name__ == "__main__":
    unittest.main()
