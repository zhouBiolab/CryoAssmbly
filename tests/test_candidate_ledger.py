"""候选台账协议测试（O6）：完整行读取、请求身份、状态与结束记录。"""

import json
import os
import tempfile
import unittest

from protassem.fitting.candidate_ledger import (LEDGER_NAME, CandidateLedgerWriter,
                                                LedgerReader)


class LedgerWriterTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name
        self.path = os.path.join(self.dir, LEDGER_NAME)

    def test_round_trip(self):
        writer = CandidateLedgerWriter(self.dir, "req-1")
        writer.publish(0, "ok", name="/tmp/pred_a_0.250000.pdb", overlap=0.25,
                       source={"mask": "mask_0.txt", "config": 3, "sampling": "voxel"})
        writer.publish(1, "filtered", reason="no valid prediction")
        writer.publish(2, "error", error="CUDA out of memory")
        writer.finish("ok")

        reader = LedgerReader(self.path, request_id="req-1")
        records = reader.poll()
        self.assertEqual(len(records), 4)
        self.assertEqual(sorted(reader.candidates), [0, 1, 2])
        self.assertEqual(reader.candidates[0]["name"], "pred_a_0.250000.pdb")
        self.assertEqual([r["id"] for r in reader.skipped], [1, 2])
        self.assertEqual(reader.end["status"], "ok")
        self.assertEqual(reader.expected_count(), 3)

    def test_request_id_is_required_and_enforced(self):
        with self.assertRaises(ValueError):
            CandidateLedgerWriter(self.dir, "")
        CandidateLedgerWriter(self.dir, "req-1").publish(0, "ok", name="x.pdb")
        with self.assertRaises(RuntimeError):
            LedgerReader(self.path, request_id="req-2").poll()

    def test_partial_line_is_buffered_until_complete(self):
        writer = CandidateLedgerWriter(self.dir, "req-1")
        writer.publish(0, "ok", name="a.pdb")
        reader = LedgerReader(self.path, request_id="req-1")
        self.assertEqual(len(reader.poll()), 1)
        # 手工追加半行：读到的必须是 0 条，且半行留在缓冲里
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write('{"v": 1, "request_id": "req-1", "kind": "candidate", "id": 1')
        self.assertEqual(reader.poll(), [])
        self.assertTrue(reader.pending_partial())
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(', "state": "ok", "name": "b.pdb"}\n')
        records = reader.poll()
        self.assertEqual([r["id"] for r in records], [1])
        self.assertEqual(reader.pending_partial(), "")

    def test_protocol_violations_raise(self):
        writer = CandidateLedgerWriter(self.dir, "req-1")
        with self.assertRaises(ValueError):
            writer.publish(0, "unknown")
        with self.assertRaises(ValueError):
            writer.finish("unknown")
        with self.assertRaises(ValueError):
            writer.finish("error")           # error 必须带原因
        writer.publish(0, "ok", name="a.pdb")
        writer.finish("cancelled")
        reader = LedgerReader(self.path, request_id="req-1")
        reader.poll()
        self.assertEqual(reader.end["status"], "cancelled")

    def test_reader_rejects_duplicate_or_bad_records(self):
        # id 重复：读侧必须报错（写侧不阻止，防止掩盖协议问题）
        with open(self.path, "w", encoding="utf-8") as handle:
            for _ in range(2):
                handle.write(json.dumps({"v": 1, "request_id": "req-1",
                                         "kind": "candidate", "id": 0,
                                         "state": "ok", "name": "a.pdb"}) + "\n")
        with self.assertRaises(RuntimeError):
            LedgerReader(self.path, request_id="req-1").poll()
        # 协议版本不符 / 未知状态 / 未知记录类型
        for bad in ({"v": 99, "request_id": "req-1", "kind": "candidate", "id": 0,
                     "state": "ok"},
                    {"v": 1, "request_id": "req-1", "kind": "candidate", "id": 0,
                     "state": "weird"},
                    {"v": 1, "request_id": "req-1", "kind": "nope"}):
            with open(self.path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(bad) + "\n")
            with self.assertRaises(RuntimeError):
                LedgerReader(self.path, request_id="req-1").poll()

    def test_finish_is_idempotent_and_error_carries_text(self):
        writer = CandidateLedgerWriter(self.dir, "req-1")
        writer.publish(0, "ok", name="a.pdb")
        self.assertIsNotNone(writer.finish("ok"))
        self.assertIsNone(writer.finish("ok"))
        reader = LedgerReader(self.path, request_id="req-1")
        reader.poll()
        self.assertEqual(len([r for r in reader.candidates.values()]), 1)
        self.assertEqual(reader.end["status"], "ok")

        writer2 = CandidateLedgerWriter(self.dir, "req-2")
        os.remove(self.path)                      # 新请求写新台账（客户端会先清旧文件）
        writer2.finish("error", error="boom")
        reader2 = LedgerReader(self.path, request_id="req-2")
        reader2.poll()
        self.assertEqual(reader2.end["status"], "error")
        self.assertEqual(reader2.end["error"], "boom")


if __name__ == "__main__":
    unittest.main()
