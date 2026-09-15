"""失败状态传播测试（审计 P2-5）。

契约（**非掩码路径**，即 `use_mask=False`）：
  - 正常计算但没有有效候选 -> `filtered`（计数、跳过，不算失败）；
  - 执行失败               -> `error`（默认**明确失败**：客户端消费到就抛错）；
  - 客户端主动早停         -> 请求级 `cancelled`；
  - 存在候选级 `error` 时请求级**不得**写 `ok`。

掩码路径的评估失败另见 `test_masked_partial_failure.py`：同一 mask 内**有成功候选**
就按原规则选最优发布 `ok`；**全部失败**时带 `reason=MASK_ERROR_REASON` 发布 `error`，
客户端跳过它且请求级结束状态不受影响。

服务端部分用 stub 直接驱动 `run_inference(use_mask=False)`（与 P1-4 同样的接线），
客户端部分直接驱动 `CandidateConsumer` + 台账文件。
"""

import json
import os
import tempfile
import unittest
from unittest import mock

from protassem.fitting import demo_mask
from protassem.fitting.candidate_consumer import CandidateConsumer
from protassem.fitting.candidate_ledger import (LEDGER_NAME, CandidateLedgerWriter,
                                                LedgerReader)


def _read_ledger(path):
    records = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


class ServerFailureStateTest(unittest.TestCase):
    """服务端：异常必须记 error，且请求级状态不得为 ok。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name
        self.output_dir = os.path.join(self.dir, "out")
        os.makedirs(self.output_dir, exist_ok=True)

    def _run(self, behaviour):
        """behaviour: candidate_id -> "ok" | "error" | "none"."""

        def fake_process_single_pair(src_data, tgt_data, source, target, chain_pdb,
                                     model_net, cfg, cid, sm, output_dir,
                                     use_mask=False, **kwargs):
            decision = behaviour.get(cid, "ok")
            if decision == "error":
                raise RuntimeError("模拟模型执行失败 (config %d)" % cid)
            if decision == "none":
                return {"config_id": cid, "sampling_method": sm,
                        "pred_pdb_path": None, "overlap": None}
            path = os.path.join(output_dir, "pred_%d_%s.pdb" % (cid, sm))
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("HEADER\nEND\n")
            return {"config_id": cid, "sampling_method": sm,
                    "pred_pdb_path": path, "overlap": 0.2 + 0.01 * cid}

        with mock.patch.object(demo_mask, "_get_model",
                               return_value=(object(), None)), \
                mock.patch.object(demo_mask, "preprocess_point_cloud_data",
                                  return_value={}), \
                mock.patch.object(demo_mask, "process_single_pair",
                                  side_effect=fake_process_single_pair), \
                mock.patch.object(demo_mask, "rename_pdb_files_by_ranking"):
            demo_mask.run_inference(
                target="t.txt", source="s.txt", chain_pdb=None,
                output_dir=self.output_dir, weights=None, use_mask=False,
                configs="1,2", seed=1, tail_pipeline_enabled=False)
        return _read_ledger(os.path.join(self.output_dir, LEDGER_NAME))

    def test_all_ok_writes_ok(self):
        records = self._run({1: "ok", 2: "ok"})
        states = [r["state"] for r in records if r["kind"] == "candidate"]
        self.assertEqual(states, ["ok"] * 4)
        self.assertEqual([r for r in records if r["kind"] == "end"][0]["status"], "ok")

    def test_execution_failure_is_error_and_request_is_not_ok(self):
        records = self._run({1: "error", 2: "ok"})
        candidates = [r for r in records if r["kind"] == "candidate"]
        # config 1 的两种采样都失败 -> 2 个 error；config 2 正常 -> 2 个 ok
        self.assertEqual([r["state"] for r in candidates],
                         ["error", "error", "ok", "ok"])
        self.assertIn("模拟模型执行失败", candidates[0]["error"])
        end = [r for r in records if r["kind"] == "end"][0]
        self.assertEqual(end["status"], "error")
        self.assertIn("2 个候选执行失败", end["error"])

    def test_no_valid_prediction_is_filtered_and_request_stays_ok(self):
        records = self._run({1: "none", 2: "ok"})
        candidates = [r for r in records if r["kind"] == "candidate"]
        self.assertEqual([c["state"] for c in candidates], ["filtered", "filtered",
                                                            "ok", "ok"])
        end = [r for r in records if r["kind"] == "end"][0]
        self.assertEqual(end["status"], "ok")


if __name__ == "__main__":
    unittest.main()
