"""候选台账（O6）：请求级、追加写、按稳定整数 ID 标识候选。

协议（`ENGINEERING_HARDENING_PLAN.md` 附录 D.1）：

- 每个请求一份 `<request_out>/candidates.jsonl`；每行一条**完整** JSON 记录（以换行结束）；
- candidate 记录：`{"v","request_id","kind":"candidate","id","state","name","overlap","source","error","reason"}`；
- end 记录：`{"v","request_id","kind":"end","status","count","error"}`；
- 服务端发布顺序：候选文件**完整写出且不再改名/删除** → 追加记录 → `flush + fsync`；
- `state`：`ok`（有效候选）/ `filtered`（正常计算但无有效候选）/ `error`（执行失败）；
- `error` 的 `reason`：只有 `MASK_ERROR_REASON` 需要客户端特殊处理 —— 掩码级评估失败可跳过
  （保留 id 与批次位置），其余执行失败一律抛错；
- `end.status`：`ok` / `error`（必须带 `error`）/ `cancelled`（客户端主动早停，属正常控制流）；
- 客户端只消费**完整行**：读到文件末尾的半行时等待后续数据，不判损坏。

本模块只负责协议本身（写与增量读），批次语义在 `candidate_consumer.py`。
"""

import json
import os

LEDGER_VERSION = 1
LEDGER_NAME = "candidates.jsonl"

CANDIDATE_STATES = ("ok", "filtered", "error")
END_STATUSES = ("ok", "error", "cancelled")

# 掩码级评估失败的 reason。语义：
#   服务端：同一个 mask 内没有成功候选时发布 `error` 并带此 reason；若同一 mask 内有成功候选，
#           则按原规则选最优发布 `ok`，失败只留在结果集里参与统计。
#   客户端：带此 reason 的 `error` 候选跳过（不送入 CC 与局部优化），保留原 id 与批次位置；
#           它也不使请求级结束状态变成 `error`。
MASK_ERROR_REASON = "mask_evaluations_failed"


class CandidateLedgerWriter:
    """服务端侧：按生成顺序发布候选与请求结束。"""

    def __init__(self, output_dir, request_id):
        if not request_id:
            raise ValueError("request_id 不能为空：台账必须能判定归属")
        self.path = os.path.join(output_dir, LEDGER_NAME)
        self.request_id = str(request_id)
        self.count = 0
        self.finished = False

    def publish(self, candidate_id, state, name=None, overlap=None,
                source=None, error=None, reason=None):
        """发布一个候选的**终态**（调用方保证文件已完整写出且不会再改名/删除）。"""
        if state not in CANDIDATE_STATES:
            raise ValueError("未知候选状态: %r" % (state,))
        record = {"v": LEDGER_VERSION, "request_id": self.request_id,
                  "kind": "candidate", "id": int(candidate_id), "state": state}
        if name is not None:
            record["name"] = os.path.basename(str(name))
        if overlap is not None:
            record["overlap"] = float(overlap)
        if source:
            record["source"] = dict(source)
        if error is not None:
            record["error"] = str(error)
        if reason is not None:
            record["reason"] = str(reason)
        self._append(record)
        self.count = max(self.count, int(candidate_id) + 1)
        return record

    def finish(self, status, error=None):
        """写请求结束记录（唯一权威结束标记）；`error` 状态必须带原因。"""
        if status not in END_STATUSES:
            raise ValueError("未知结束状态: %r" % (status,))
        if status == "error" and not error:
            raise ValueError("status=error 必须带 error 文本")
        if self.finished:
            return None
        record = {"v": LEDGER_VERSION, "request_id": self.request_id,
                  "kind": "end", "status": status, "count": self.count}
        if error is not None:
            record["error"] = str(error)
        self._append(record)
        self.finished = True
        return record

    def _append(self, record):
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


class LedgerReader:
    """客户端侧：增量读取台账，只消费完整行。"""

    def __init__(self, path, request_id=None):
        self.path = path
        self.request_id = str(request_id) if request_id is not None else None
        self.candidates = {}      # id -> record（按到达顺序无关；id 是唯一键）
        self.skipped = []         # state != ok 的记录（filtered/error）
        self.end = None
        self._offset = 0
        self._partial = b""      # **字节**缓冲（不能提前解码：见 poll()）

    def poll(self):
        """读取自上次以来的完整记录；末尾半行留在缓冲里等下次。"""
        if not os.path.exists(self.path):
            return []
        with open(self.path, "rb") as handle:
            handle.seek(self._offset)
            chunk = handle.read()
            self._offset += len(chunk)
        if not chunk:
            return []
        # 先按**字节**切出完整行，再解码：多字节（中文原因/路径）可能正好被读取边界截断，
        # 先 decode 整个 chunk 会在半行缓冲之前就抛 UnicodeDecodeError。
        lines = (self._partial + chunk).split(b"\n")
        self._partial = lines.pop()          # 末尾：可能是半行，也可能是空串
        records = []
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            record = self._parse(line.decode("utf-8"))
            self._apply(record)
            records.append(record)
        return records

    def pending_partial(self):
        """当前缓冲的半行（诊断用；正常情况下为空）。"""
        return self._partial.decode("utf-8", "replace")

    def ready_ids(self, start, stop):
        """[start, stop) 区间内**已到达终态**的候选 id 列表。"""
        return [cid for cid in range(start, stop) if cid in self.candidates]

    def expected_count(self):
        """请求已结束时返回候选总数（含 filtered/error），否则 None。"""
        return self.end["count"] if self.end else None

    def _parse(self, line):
        try:
            record = json.loads(line)
        except ValueError as exc:
            raise RuntimeError("台账记录不是合法 JSON（%s）：%s" % (self.path, exc))
        if record.get("v") != LEDGER_VERSION:
            raise RuntimeError("台账协议版本不匹配：%r" % (record.get("v"),))
        if self.request_id is not None and record.get("request_id") != self.request_id:
            raise RuntimeError("台账归属不匹配（期望 %s，实际 %s）：拒绝读取旧台账"
                               % (self.request_id, record.get("request_id")))
        return record

    def _apply(self, record):
        kind = record.get("kind")
        if kind == "candidate":
            cid = int(record["id"])
            if record["state"] not in CANDIDATE_STATES:
                raise RuntimeError("未知候选状态: %r" % (record["state"],))
            if cid in self.candidates:
                raise RuntimeError("候选 id 重复发布: %d" % cid)
            self.candidates[cid] = record
            if record["state"] != "ok":
                self.skipped.append(record)
        elif kind == "end":
            if self.end is not None:
                raise RuntimeError("请求结束记录重复")
            self.end = record
        else:
            raise RuntimeError("未知台账记录类型: %r" % (kind,))
