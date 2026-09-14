"""候选消费顺序测试（O6）：固定 ID 区间批次与文件出现时机无关。

用"假生产者 + 虚拟时钟"驱动真 `CandidateConsumer` 与真 `LedgerReader`：
发布时机（快/慢/随机/结束早于最后一批）、worker 数（1/2/10，模拟完成顺序打乱、
结果仍按输入顺序归并）、状态（ok/filtered/error）、结束态（ok/error/cancelled）、
分数并列 —— 这些场景必须给出**同一候选 id 序列与同一早停决策**。
"""

import json
import os
import random
import tempfile
import unittest

from protassem.fitting.candidate_consumer import CandidateConsumer
from protassem.fitting.candidate_ledger import (LEDGER_NAME, CandidateLedgerWriter,
                                                LedgerReader)

POLL_S = 2.5
THRESHOLD = 0.4200


def _candidate_record(candidate_id, overlap, state="ok"):
    record = {"v": 1, "request_id": "req", "kind": "candidate",
              "id": candidate_id, "state": state}
    if state == "ok":
        record["name"] = "pred_%02d_%.6f.pdb" % (candidate_id, overlap)
        record["overlap"] = overlap
    elif state == "error":
        record["error"] = "worker crashed"
    else:
        record["reason"] = "no valid prediction"
    return record


class FakeProducer:
    """按虚拟时间发布台账记录；`request_finished_at` 模拟请求句柄结束。"""

    def __init__(self, path, schedule, end_at, request_finished_at, end_status="ok"):
        self.path = path
        self.schedule = list(schedule)
        self.end_at = end_at
        self.request_finished_at = request_finished_at
        self.end_status = end_status
        self.count = 0
        self.cancelled = False
        self._ended = False

    def advance(self, now):
        while self.schedule and self.schedule[0][0] <= now:
            _, record = self.schedule.pop(0)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            self.count += 1
        if not self._ended and now >= self.end_at:
            record = {"v": 1, "request_id": "req", "kind": "end",
                      "status": self.end_status, "count": self.count}
            if self.end_status == "error":
                record["error"] = "server failed"
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            self._ended = True

    def cancel(self):
        self.cancelled = True


def evaluate_like_pool(records, workers, threshold=THRESHOLD):
    """模拟一次批内评估：完成顺序与 worker 数有关，结果按输入顺序归并。

    返回 `(results, hit)`；`results` 顺序 == 输入顺序（与 `ExecutionContext.map` 一致）。
    候选的 CC 直接取 overlap（数值本身不重要，重要的是**顺序与并列**）。
    """
    usable = [r for r in records if r["state"] == "ok"]
    completion_order = list(range(len(usable)))
    random.Random(workers).shuffle(completion_order)     # worker 越多，完成顺序越乱
    scores = {}
    for index in completion_order:
        scores[index] = usable[index]["overlap"]
    results = [{"pdb_file": usable[i]["name"], "cc_mask": scores[i],
                "candidate_id": usable[i]["id"]} for i in range(len(usable))]
    ordered = sorted(results, key=lambda r: (-r["cc_mask"], r["candidate_id"]))
    optimized = []
    hit = False
    for result in ordered:
        optimized.append(result["candidate_id"])
        if result["cc_mask"] >= threshold:
            hit = True
            break
    return results, hit, optimized


def drive(schedule, end_at, request_finished_at, batch_size=10, workers=1,
          end_status="ok", poll_interval=POLL_S):
    """用虚拟时钟驱动消费者，返回 `(outcome, optimized_ids, producer)`。"""
    tmp = tempfile.TemporaryDirectory()
    path = os.path.join(tmp.name, LEDGER_NAME)
    writer = CandidateLedgerWriter(tmp.name, "req")   # 仅用于校验写侧协议一致
    del writer
    producer = FakeProducer(path, schedule, end_at, request_finished_at,
                            end_status=end_status)
    reader = LedgerReader(path, request_id="req")
    optimized = []
    clock = [0.0]

    def on_batch(records):
        results, hit, ids = evaluate_like_pool(records, workers)
        optimized.extend(ids)
        return results, hit

    def sleep(seconds):
        clock[0] += seconds
        producer.advance(clock[0])

    consumer = CandidateConsumer(
        reader, batch_size, on_batch,
        request_finished=lambda: clock[0] >= producer.request_finished_at,
        on_cancel=producer.cancel, sleep=sleep, poll_interval=poll_interval)
    producer.advance(0.0)
    outcome = consumer.run()
    tmp.cleanup()
    return outcome, optimized, producer


def _schedule(count=24, overlaps=None, mode="fast", publish_end_at=0.0,
              end_at=None, request_finished_at=None, states=None):
    """构造 (publish_time, record) 列表。"""
    overlaps = overlaps or [0.30 + 0.002 * i for i in range(count)]
    states = states or {}
    rng = random.Random(7)
    schedule = []
    for cid in range(count):
        if mode == "fast":
            moment = 0.0
        elif mode == "slow":
            moment = cid * POLL_S
        else:
            moment = publish_end_at * rng.random()
        schedule.append((moment, _candidate_record(cid, overlaps[cid],
                                                   states.get(cid, "ok"))))
    schedule.sort(key=lambda item: item[0])
    last = max(item[0] for item in schedule)
    end_at = last + 0.1 if end_at is None else end_at
    request_finished_at = (end_at + 0.1 if request_finished_at is None
                           else request_finished_at)
    return schedule, end_at, request_finished_at


class CandidateOrderingTest(unittest.TestCase):

    def test_timing_does_not_change_batches_or_choice(self):
        """快发 / 慢发 / 随机延迟：批次划分与优化顺序必须一致。"""
        outcomes = []
        for mode in ("fast", "slow", "random"):
            schedule, end_at, finished = _schedule(mode=mode)
            outcome, optimized, producer = drive(schedule, end_at, finished)
            outcomes.append((mode, outcome, optimized, producer))
        reference = outcomes[0][2]
        for mode, outcome, optimized, _producer in outcomes:
            self.assertEqual([(b["start"], b["stop"]) for b in outcome["batches"]],
                             [(0, 10), (10, 20), (20, 24)], "mode=%s" % mode)
            self.assertEqual(optimized, reference, "mode=%s" % mode)
            self.assertEqual(outcome["consumed"], 24, "mode=%s" % mode)

    def test_worker_count_does_not_change_choice(self):
        schedule, end_at, finished = _schedule(mode="random")
        reference = None
        for workers in (1, 2, 10):
            outcome, optimized, _ = drive(schedule, end_at, finished, workers=workers)
            if reference is None:
                reference = optimized
            self.assertEqual(optimized, reference, "workers=%d" % workers)
            self.assertEqual(outcome["consumed"], 24)

    def test_batches_are_fixed_even_after_end_arrived(self):
        """服务端早已结束（记录与 end 都在 t=0）：仍必须按固定批次依次消费。"""
        schedule, end_at, finished = _schedule(mode="fast", end_at=0.0,
                                               request_finished_at=0.0)
        outcome, _optimized, _producer = drive(schedule, end_at, finished)
        self.assertEqual([(b["start"], b["stop"]) for b in outcome["batches"]],
                         [(0, 10), (10, 20), (20, 24)])

    def test_early_stop_uses_first_hit_in_id_order(self):
        """分数并列时用 id 作次序；早停发生在**首个达标候选**，不要求整批优化完。"""
        overlaps = [0.10] * 10 + [0.44, 0.44, 0.44] + [0.10] * 11
        schedule, end_at, finished = _schedule(overlaps=overlaps, mode="slow")
        outcome, optimized, producer = drive(schedule, end_at, finished)
        self.assertTrue(outcome["early_stop"])
        # 第一批（id 0..9）分数相同且未达标 → 按 id 次序全优化
        self.assertEqual(optimized[:10], list(range(10)))
        # 第二批里首个达标的是并列中最小的 id，且达标即停（不优化同批其余候选）
        self.assertEqual([cid for cid in optimized if cid >= 10], [10])
        self.assertTrue(producer.cancelled)           # 早停后取消请求

    def test_filtered_states_do_not_block_batches(self):
        """`filtered`（正常计算但无有效候选）只计数跳过，不阻塞批次、不改变顺序。"""
        states = {3: "filtered", 21: "filtered"}
        schedule, end_at, finished = _schedule(mode="slow", states=states)
        outcome, optimized, _ = drive(schedule, end_at, finished)
        self.assertEqual([(b["start"], b["stop"]) for b in outcome["batches"]],
                         [(0, 10), (10, 20), (20, 24)])
        self.assertEqual([b["skipped"] for b in outcome["batches"]], [1, 0, 1])
        self.assertNotIn(3, optimized)
        self.assertNotIn(21, optimized)
        self.assertEqual(len(outcome["skipped"]), 2)

    def test_error_candidate_cancels_the_request(self):
        """P2-5 + 审计#2：候选执行失败必须抛错，并且抛错前**取消并结束该请求**。"""
        states = {7: "error"}
        schedule, end_at, finished = _schedule(mode="slow", states=states)
        tmp = tempfile.TemporaryDirectory()
        path = os.path.join(tmp.name, LEDGER_NAME)
        producer = FakeProducer(path, schedule, end_at, finished)
        reader = LedgerReader(path, request_id="req")
        clock = [0.0]
        finished_calls = []

        def sleep(seconds):
            clock[0] += seconds
            producer.advance(clock[0])

        def request_finished():
            finished_calls.append(clock[0])
            return clock[0] >= producer.request_finished_at

        consumer = CandidateConsumer(
            reader, 10, lambda records: ([], False),
            request_finished=request_finished, on_cancel=producer.cancel,
            sleep=sleep, poll_interval=POLL_S)
        producer.advance(0.0)
        with self.assertRaises(RuntimeError) as caught:
            consumer.run()
        tmp.cleanup()
        self.assertIn("候选 7 执行失败", str(caught.exception))
        self.assertTrue(producer.cancelled, "抛错前必须取消请求")
        self.assertGreater(len(finished_calls), 0,
                           "抛错前必须确认请求结束（_await_request_end）")

    def test_missing_end_raises(self):
        """句柄结束但没有任何 end 记录 → 不完整请求，必须抛错（不得静默用部分结果）。"""
        schedule = [(0.0, _candidate_record(cid, 0.30)) for cid in range(4)]
        with self.assertRaises(RuntimeError):
            drive(schedule, end_at=10 ** 6, request_finished_at=0.0)

    def test_end_error_raises(self):
        schedule, end_at, finished = _schedule(mode="slow")
        with self.assertRaises(RuntimeError):
            drive(schedule, end_at, finished, end_status="error")

    def test_cancelled_end_is_normal(self):
        schedule, end_at, finished = _schedule(mode="fast", end_at=0.0,
                                               request_finished_at=0.0)
        outcome, _optimized, _ = drive(schedule, end_at, finished,
                                       end_status="cancelled")
        self.assertEqual(outcome["consumed"], 24)
        self.assertFalse(outcome["early_stop"])


if __name__ == "__main__":
    unittest.main()
