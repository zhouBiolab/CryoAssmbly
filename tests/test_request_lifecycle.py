"""请求生命周期测试（审计 P1-3）：服务意外退出必须被识别，而不是无限等待。

用**真实的假子进程**（`bash -c "sleep 30"`）模拟 PARENet 服务，不用手工把
`request_finished` 置 True —— 后者只能证明消费者的分支，证明不了"句柄能看出服务已死"。
"""

import os
import subprocess
import sys
import tempfile
import unittest

from protassem.fitting.candidate_consumer import CandidateConsumer
from protassem.fitting.candidate_ledger import (LEDGER_NAME, CandidateLedgerWriter,
                                                LedgerReader)
from protassem.fitting.parenet_client import ParenetRequest


def _spawn_fake_server():
    """一个不会自己结束的假服务（父进程句柄可 terminate/poll）。"""
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


class RequestHandleTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name
        self.servers = []

    def tearDown(self):
        for server in self.servers:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=5)
                except Exception:
                    server.kill()

    def _server(self):
        server = _spawn_fake_server()
        self.servers.append(server)
        return server

    def test_running_request_polls_none(self):
        handle = ParenetRequest(self.dir, "req-1", server=self._server())
        self.assertIsNone(handle.poll())
        self.assertTrue(handle.server_alive())
        self.assertEqual(handle.describe_failure(), "")

    def test_dead_server_is_reported_as_finished(self):
        server = self._server()
        handle = ParenetRequest(self.dir, "req-1", server=server)
        server.terminate()
        server.wait(timeout=10)
        self.assertFalse(handle.server_alive())
        self.assertIsNotNone(handle.poll())                     # 不再判"在跑"
        self.assertIn("服务进程已退出", handle.describe_failure())
        self.assertIn("returncode", handle.describe_failure())

    def test_done_marker_wins_over_dead_server(self):
        server = self._server()
        handle = ParenetRequest(self.dir, "req-1", server=server)
        with open(handle.done_file, "w") as fh:
            fh.write("done\n")
        server.terminate()
        server.wait(timeout=10)
        self.assertEqual(handle.poll(), 0)                      # 请求确实完成了
        self.assertEqual(handle.describe_failure(), "")

    def test_handle_without_server_keeps_old_behaviour(self):
        handle = ParenetRequest(self.dir, "req-1")
        self.assertIsNone(handle.poll())

    def test_old_poll_logic_cannot_see_a_dead_service(self):
        """行为对照（不是接口不兼容）：旧 `poll` 只认 done 文件，服务死亡时永远返回 None。

        旧逻辑在这里原样重写一遍，跑在**同一个假服务**上，与其新句柄对比：
        旧逻辑 → None（客户端会一直等）；新句柄 → 非 None（明确结束）。
        """
        from protassem.fitting.parenet_client import DONE_MARKER

        server = self._server()
        handle = ParenetRequest(self.dir, "req-1", server=server)

        def old_poll(output_dir):
            done_file = os.path.join(output_dir, DONE_MARKER)
            return 0 if os.path.exists(done_file) else None

        server.terminate()
        server.wait(timeout=10)
        self.assertIsNone(old_poll(self.dir), "旧逻辑在服务死亡后仍判'在跑'")
        self.assertIsNotNone(handle.poll(), "新句柄必须把服务死亡识别为结束")
        self.assertIn("服务进程已退出", handle.describe_failure())


class ConsumerDeadServerTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name
        self.ledger_path = os.path.join(self.dir, LEDGER_NAME)
        self.server = _spawn_fake_server()
        self.addCleanup(self._stop_server)

    def _stop_server(self):
        if self.server.poll() is None:
            self.server.terminate()
            try:
                self.server.wait(timeout=5)
            except Exception:
                self.server.kill()

    def test_consumer_fails_fast_when_service_dies(self):
        """服务死亡 + 没有 end 记录 → 明确报错，而不是等满轮询。"""
        writer = CandidateLedgerWriter(self.dir, "req-1")
        writer.publish(0, "ok", name="pred_0.pdb")
        handle = ParenetRequest(self.dir, "req-1", server=self.server)
        reader = LedgerReader(self.ledger_path, request_id="req-1")

        sleeps = []

        def sleep(seconds):
            sleeps.append(seconds)
            if len(sleeps) == 1:               # 第一轮睡眠期间服务被杀
                self.server.terminate()
                self.server.wait(timeout=10)

        consumer = CandidateConsumer(
            reader, 10, lambda records: ([], False),
            request_finished=lambda: handle.poll() is not None,
            describe_request=lambda: handle.describe_failure(),
            sleep=sleep, poll_interval=0.01)
        with self.assertRaises(RuntimeError) as caught:
            consumer.run()
        message = str(caught.exception)
        self.assertIn("end", message)
        self.assertIn("服务进程已退出", message)
        self.assertLessEqual(len(sleeps), 4)   # 快速失败：不靠轮询耗尽

    def test_consumer_succeeds_when_end_arrives(self):
        writer = CandidateLedgerWriter(self.dir, "req-1")
        writer.publish(0, "ok", name="pred_0.pdb")
        writer.finish("ok")
        reader = LedgerReader(self.ledger_path, request_id="req-1")
        handle = ParenetRequest(self.dir, "req-1", server=self.server)
        consumer = CandidateConsumer(
            reader, 1, lambda records: ([{"cc_mask": 0.5}], False),
            request_finished=lambda: handle.poll() is not None,
            describe_request=lambda: handle.describe_failure(),
            sleep=lambda seconds: None, poll_interval=0.01)
        outcome = consumer.run()
        self.assertEqual(outcome["consumed"], 1)
        self.assertFalse(outcome["early_stop"])


if __name__ == "__main__":
    unittest.main()
