"""候选消费（O6）：批次成员由**固定 ID 区间**决定，与文件出现时机无关。

规则（`ENGINEERING_HARDENING_PLAN.md` 附录 D.1.2）：

1. 批次 = `[0, batch_size)`、`[batch_size, 2·batch_size)`…（`batch_size` 取 `--batch-size`）；
   **不得**用"本次轮询发现的文件集合"当批次；
2. 批内保持原有候选排序与逐个优化策略，第一个达标候选即触发早停（不要求整批优化完）；
3. 即使 `end` 已到、全部候选已生成，仍按固定批次依次消费（不得改成"剩余一起 final_select"）；
4. `final_select` 只在"全部批次消费完 且 未早停"时执行（由调用方判断，本模块只如实返回）；
5. 未见 `end` 而请求已结束 → 抛错；`end.status="error"` → 抛错；`cancelled` → 正常返回；
6. 早停后：确认请求已结束后才返回。
"""

import time

from protassem.fitting.candidate_ledger import MASK_ERROR_REASON

POLL_INTERVAL_S = 2.5
END_GRACE_POLLS = 2


class CandidateConsumer:
    """按固定 ID 区间消费台账；`on_batch(records)` 返回 `(results, hit_threshold)`。"""

    def __init__(self, reader, batch_size, on_batch, request_finished,
                 on_cancel=None, sleep=time.sleep, poll_interval=POLL_INTERVAL_S,
                 describe_request=None):
        self.reader = reader
        self.batch_size = max(1, int(batch_size))
        self.on_batch = on_batch
        self.request_finished = request_finished
        self.on_cancel = on_cancel or (lambda: None)
        self.sleep = sleep
        self.poll_interval = poll_interval
        # P1-3：可选的诊断来源（例如"服务进程已退出（returncode=…）"），让报错能区分
        # "服务死亡"与"服务端漏写 end"。
        self.describe_request = describe_request or (lambda: "")
        # 已消费范围内被跳过的掩码级评估失败数量（见 _check_states）。
        self.skipped_mask_errors = 0

    def _check_states(self, records):
        """掩码级评估失败跳过，其余执行失败抛错。

        `reason=MASK_ERROR_REASON` 表示该候选所属的 mask 内所有评估都失败：它保留原有的
        id 与批次位置（批次区间不变），只是不参与 CC 评估与局部优化；服务端不会因此把
        请求级结束状态写成 `error`。其余执行失败（服务/台账/非掩码路径）仍然抛错，
        不能让"少了一些候选"悄悄变成交付结果。
        """
        for record in records:
            if record["state"] != "error":
                continue
            if record.get("reason") == MASK_ERROR_REASON:
                self.skipped_mask_errors += 1
                continue
            raise RuntimeError("候选 %d 执行失败（state=error）：%s"
                               % (record["id"], record.get("error") or "未提供原因"))

    def _missing_end_error(self):
        detail = ""
        try:
            detail = str(self.describe_request() or "")
        except Exception:
            detail = ""
        message = "请求已结束但没有台账 end 记录（不完整请求）"
        return RuntimeError("%s：%s" % (message, detail) if detail else message)

    def run(self):
        """消费全部候选；返回批次摘要（调用方据此决定是否 final_select）。

        异常（候选失败、缺 end、服务死亡…）原样传播，不做额外的取消或等待；
        只有**正常早停**会取消请求并确认其结束。
        """
        batches = []
        results = []
        next_id = 0
        early_stop = False
        missing_end_polls = 0

        while True:
            self.reader.poll()
            end = self.reader.end
            total = end["count"] if end else None
            progressed = False

            # 1) 消费所有"区间成员已齐"的批次（含 end 到达后的末尾不足额批）
            while True:
                window = list(range(next_id, next_id + self.batch_size))
                if total is not None:
                    window = [cid for cid in window if cid < total]
                if not window:
                    break
                if len(self.reader.ready_ids(window[0], window[-1] + 1)) < len(window):
                    break
                records = [self.reader.candidates[cid] for cid in window]
                self._check_states(records)     # 掩码级评估失败在这里跳过
                batch_results, hit = self.on_batch(records)
                results.extend(batch_results)
                batches.append({"start": window[0], "stop": window[-1] + 1,
                                "skipped": sum(1 for r in records
                                               if r["state"] != "ok")})
                next_id = window[-1] + 1
                progressed = True
                if hit:
                    early_stop = True
                    break

            # 2) 退出与错误判定
            if early_stop:
                self.on_cancel()
                self._await_request_end()
                break
            if end is not None:
                if end["status"] == "error":
                    raise RuntimeError("请求失败（台账 end.status=error）：%s"
                                       % end.get("error"))
                if next_id >= total:
                    break
                if not progressed:
                    raise RuntimeError("台账不完整：流已结束但 [%d, %d) 缺少候选记录"
                                       % (next_id, next_id + self.batch_size))
                continue
            if self.request_finished():
                missing_end_polls += 1
                if missing_end_polls > END_GRACE_POLLS:
                    raise self._missing_end_error()
            else:
                missing_end_polls = 0
            self.sleep(self.poll_interval)

        return {"batches": batches, "results": results, "early_stop": early_stop,
                "skipped": list(self.reader.skipped), "consumed": next_id,
                "skipped_mask_errors": self.skipped_mask_errors}

    def _await_request_end(self):
        """客户端主动早停后：确认请求真的结束了，才允许复用/改写目录或输入。"""
        while True:
            self.reader.poll()
            if self.reader.end is not None:
                if self.reader.end["status"] == "error":
                    raise RuntimeError("请求失败（台账 end.status=error）：%s"
                                       % self.reader.end.get("error"))
                if self.request_finished():
                    return
            elif self.request_finished():
                # 句柄已结束但 end 记录缺失：再给一个轮询周期（写盘与句柄不同步的窗口）
                self.sleep(self.poll_interval)
                self.reader.poll()
                if self.reader.end is None:
                    raise self._missing_end_error()
                continue
            self.sleep(self.poll_interval)
