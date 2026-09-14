"""无掩码路径的候选发布顺序（审计 P1-4）。

契约：台账里的候选名必须是**最终名**（改名之后不再变化），且只有在尾部任务完成、
文件真的写出之后才能发布 `ok`。旧实现边算边发布：改名后台账指向失效文件；
tail 开启时还可能把尚未写出的候选记成 `filtered`。

测试直接驱动 `demo_mask.run_inference(use_mask=False)`：模型、点云预处理与单对推理都被
替换成 stub，但**尾部流水线是真实的**（stub 通过 `tail_pipeline.submit` 延迟写文件），
因此能真实复现"发布早于写出/改名"的时序问题。
"""

import json
import os
import tempfile
import unittest
from unittest import mock

from protassem.fitting import demo_mask
from protassem.fitting.candidate_ledger import LEDGER_NAME, LedgerReader


def _write_pdb(path, overlap):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("HEADER overlap=%f\nEND\n" % overlap)


class MasklessPublishOrderTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name
        self.output_dir = os.path.join(self.dir, "out")
        os.makedirs(self.output_dir, exist_ok=True)

    def _run(self, tail_enabled):
        """跑一次无掩码推理；返回 (台账记录, 磁盘文件名集合)。"""
        written = {}

        def fake_process_single_pair(src_data, tgt_data, source, target, chain_pdb,
                                     model_net, cfg, cid, sm, output_dir,
                                     use_mask=False, **kwargs):
            raw = os.path.join(output_dir, "pred_raw_%d_%s.pdb" % (cid, sm))
            overlap = 0.2 + 0.01 * cid
            result = {"config_id": cid, "sampling_method": sm,
                      "pred_pdb_path": raw, "overlap": overlap}

            def write():
                _write_pdb(raw, overlap)
                result["pred_pdb_path"] = raw

            tail = kwargs.get("tail_pipeline")
            if tail_enabled and tail is not None:
                tail.submit(write)          # 真实尾部：写盘被推迟
            else:
                write()
            return result

        def fake_rename(results):
            """模拟 rename_pdb_files_by_ranking：改名并回写 pred_pdb_path。"""
            for index, result in enumerate(results, 1):
                old = result["pred_pdb_path"]
                new = os.path.join(os.path.dirname(old),
                                   "pred_rank%02d_%.6f.pdb" % (index, result["overlap"]))
                os.rename(old, new)
                result["pred_pdb_path"] = new

        with mock.patch.object(demo_mask, "_get_model",
                               return_value=(object(), None)), \
                mock.patch.object(demo_mask, "preprocess_point_cloud_data",
                                  return_value={}), \
                mock.patch.object(demo_mask, "process_single_pair",
                                  side_effect=fake_process_single_pair), \
                mock.patch.object(demo_mask, "rename_pdb_files_by_ranking",
                                  side_effect=fake_rename):
            demo_mask.run_inference(
                target="target.txt", source="source.txt", chain_pdb=None,
                output_dir=self.output_dir, weights=None, use_mask=False,
                configs="1,2", seed=1, tail_pipeline_enabled=tail_enabled)

        reader = LedgerReader(os.path.join(self.output_dir, LEDGER_NAME),
                              request_id="server-%d" % os.getpid())
        records = []
        for line in open(os.path.join(self.output_dir, LEDGER_NAME),
                         encoding="utf-8"):
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records, reader

    def _assert_contract(self, records, tail_enabled):
        candidates = [r for r in records if r["kind"] == "candidate"]
        ends = [r for r in records if r["kind"] == "end"]
        self.assertEqual(len(candidates), 4, "configs 1,2 x voxel,fps = 4 个候选")
        self.assertEqual([r["id"] for r in candidates], [0, 1, 2, 3])   # 生成顺序、连续
        self.assertEqual({r["state"] for r in candidates}, {"ok"})
        self.assertEqual(len(ends), 1)
        self.assertEqual(ends[0]["status"], "ok")
        for record in candidates:
            path = os.path.join(self.output_dir, record["name"])
            self.assertTrue(os.path.exists(path),
                            "台账名字必须指向真实文件（tail=%s）" % tail_enabled)
            self.assertIn("pred_rank", record["name"],
                          "台账名字必须是**改名后**的最终名")

    def test_publish_after_rename_without_tail(self):
        records, _reader = self._run(tail_enabled=False)
        self._assert_contract(records, False)

    def test_publish_after_rename_with_tail(self):
        records, _reader = self._run(tail_enabled=True)
        self._assert_contract(records, True)


if __name__ == "__main__":
    unittest.main()
