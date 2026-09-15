# 候选处理精简 —— 验收报告

日期：2026-09-15　分支：`refactor/mask-candidate-policy`（起点 `de51657`）
代码提交：`1fee657`　文档提交：`a2a9792`　工作副本：`/xiangyux/claude_c_work/demo_reg_policy`

**状态：本轮功能收口。** 验证范围、已接受取舍与交付差异见第五节 —— 三类均为明确划定的边界，**不作为待办项**；本轮不追加功能、不追加长跑、不做顺带重构或防御性逻辑。

---

## 一、改了什么

| 步 | 位置 | 改动 |
|---|---|---|
| ① | `protassem/fitting/demo_mask.py` `_publish` | **成功优先**：`name` 存在即发布 `ok`。同一 mask 内个别评估抛异常不再掩盖已经得到的成功结果 |
| ② | 同上 + 掩码分支调用点 | 同一 mask 内**没有成功候选**且有执行错误 → 发布 `error`，`reason=MASK_ERROR_REASON`（`"mask_evaluations_failed"`），保留 `error` 原因与 `source`（mask/config/sampling） |
| ③ | 同上（结束状态） | 掩码级失败走独立计数，**不进入 `ledger_errors`** → 请求级结束状态不受影响：正常结束 `ok`、早停 `cancelled`、请求级故障仍 `error` |
| ④ | `protassem/fitting/candidate_consumer.py` `_check_states` | 只跳过带该 reason 的 `error` 候选；保留原 `id` 与批次区间（`window`/`next_id` 不变），不送入 CC 评估与局部优化；其余执行失败仍然抛错 |
| ⑤ | 同上 `run()` | 撤回异常路径的取消/等待包装，`_run()` 合回 `run()`；**只保留正常早停**的 `on_cancel()` + `_await_request_end()` |
| ⑥ | `protassem/fitting/demo_mask.py` 异常分支 + 客户端 | 异常在掩码结果集与全局结果集**各记一次**（去掉重复追加）；新增 `skipped_mask_errors`，客户端只统计**已消费范围内**被跳过的掩码级失败，由 `fitting/pipeline.py` 记入日志 |

**协议常量**：`MASK_ERROR_REASON = "mask_evaluations_failed"` 定义在
`protassem/fitting/candidate_ledger.py`，写（服务端）读（客户端）两侧共用；该模块的协议说明
同步补充了 `error` 的 `reason` 语义。

**未改动**（按任务卡约束）：模型、精度、采样、评分公式、阈值、O6 顺序、缓存容量、
显存释放策略；`_cc_worker()` 未动；`error_policy` 保持移除状态；
`protassem/pipeline.py` 的评分缓存统计修复保留（`_record_score_cache` 定义于 77 行、调用在 361 行）。

**未新增**：策略开关、重试、抽象层。`_publish` 只多一个布尔参数区分掩码级与候选级。

**其余约定**：全部 mask 不可用时走已有无拟合结果路径（不造结构、不造分数）；CC 低分与负数仍按原逻辑；
服务死亡、台账损坏、非掩码模式错误保持原有报错。

## 二、单测与静态检查

| 项 | 结果 |
|---|---|
| `python -m unittest discover -s tests -t .` | **261 项通过**（`de51657` 为 259 项，净增 2） |
| `python -m compileall -q protassem tests main.py tools` | 通过 |
| `git diff --check` | 无空白问题 |
| 工作树 | `git status --porcelain -uno` 为空 |

**测试同步**

| 文件 | 改动 |
|---|---|
| `tests/test_masked_partial_failure.py` | 按新契约重写：全成功 → `ok`；**部分失败 → 用成功候选发布 `ok`**；全失败 → `error` + `reason`；并新增**去重证据**断言（`Total: 2, Success: 1, Failed: 1`；旧实现为 `Total: 3, Success: 1, Failed: 2`） |
| `tests/test_candidate_ordering.py` | 异常路径用例改为"抛错但**不再取消**"；新增 `test_mask_evaluation_failure_is_skipped`（ID 与批次区间不变、不进优化）与 `test_skipped_mask_errors_count_only_consumed_range`（早停后不计入） |
| `tests/test_failure_states.py` | 契约说明注明只针对**非掩码路径**；用例未改（该路径行为本就不变） |

## 三、真实 case 验证

代码树冻结在 `1fee657`，用 `demo_reg_cases/policy_case_validation_20260915.py` 串行执行，
输出根目录 `demo_reg_cases/out_policy_20260915_2004`，配置 `{"blas_threads": 1, "seed": 7351}`。

**脚本判定：两个 case 全部 `passed=True`，脚本退出码 0。**

| case | 墙钟 | exit | 设备显存峰值 | 设备采样 | 主机 PSS 峰值 | 主机 RSS 峰值 |
|---|---|---|---|---|---|---|
| `test/1` | **1184.1 s** | 0 | **8547 MiB** | 219 点 | 10051 MiB | 15627 MiB |
| `test/2` | **2267.8 s** | 0 | **6839 MiB** | 418 点 | 32177 MiB | 33692 MiB |

两次运行的设备占用均由 2 MiB 起、结束时仍为 6837 MiB（进程退出后释放），无残留进程。

### 3.1 产物 md5

| case | 产物 | 实测 | 期望 | 判定 |
|---|---|---|---|---|
| `test/1` | `assembled_complex.cif` | `76638d0f97d4ee5775110c66b15e73c6` | 同 | ✅ |
| `test/1` | `assembled_complex_all.cif` | `76638d0f97d4ee5775110c66b15e73c6` | 同 | ✅ |
| `test/1` | `refined_complex.cif` | `bd281f40e25352455ba34d41e6bbc9f7` | 同 | ✅ |
| `test/2` | `assembled_complex.cif` | `6d2d87405bfc433ac0ac6dfbc81d3ed0` | 同 | ✅ |
| `test/2` | `assembled_complex_all.cif` | `6d2d87405bfc433ac0ac6dfbc81d3ed0` | 同 | ✅ |
| `test/2` | `refined_complex.cif` | `80a17ee5b6ba46ff38508dafcb25466b` | 同 | ✅ |

### 3.2 接受决策

`test/1`：

```
Chain A rejected (cc=0.4192 < 0.450)
Chain B accepted (cc=0.4235 >= 0.420)
Chain A: assembled from 2 domains (cc=0.4430)
```

`test/2`：

```
Chain A rejected (cc=0.2337 < 0.450)
Chain C rejected (cc=0.3898 < 0.450)
Chain B rejected (cc=0.2387 < 0.420)
Chain A: assembled from 2 domains (cc=0.5610)
Chain C: assembled from 2 domains (cc=0.5474)
Chain B: assembled from 2 domains (cc=0.5316)
```

`test/2` 的三个组装 cc 值（0.5610 / 0.5474 / 0.5316）与历史审计记录一致。

### 3.3 候选台账

| case | 请求数 | 每请求候选数 | id 连续 | 状态 | reason | end |
|---|---|---|---|---|---|---|
| `test/1` | 3 | 24 / 28 / 115 | 全部 ✔ | 全 `ok` | 无 | 全 `ok` |
| `test/2` | 8 | 44 / 61 / 35 / 28 / 19 / 18 / 29 / 53 | 全部 ✔ | 全 `ok` | 无 | 3×`ok` + 5×`cancelled` |

`test/2` 的 5 个 `end=cancelled` 是**客户端主动早停**（正常控制流），与历史运行一致。
**两次运行均未出现任何 `filtered` 或 `error` 候选**，即本次未触发新策略的分支。

### 3.4 候选消费顺序

| case | 已消费候选数 | 消费序列 sha1 | 与历史基线 |
|---|---|---|---|
| `test/1` | **58** | **`de02d03105ebff9f`** | 候选数、已消费数、sha1 **三项全部一致** |
| `test/2` | 41 | `0a1f221a73994698` | 无历史 sha1 基线可比（`test/2` 无冻结基线） |

sha1 算法沿用历史脚本 `/tmp/o6_summary.py`：
`sha1("\n".join("%d:%s" % (序号, 文件名)))` 取前 16 位，序列取自 pipeline 日志的
`Candidate #N: local optimization on <file>` 行。

### 3.5 错误日志与残留进程

| 项 | `test/1` | `test/2` |
|---|---|---|
| 运行日志内 `ERROR`/`Traceback`/`CRITICAL` | **0 条** | **0 条** |
| `main.py` / `demo_mask.py` 残留进程 | **无** | **无** |

## 四、结论（就本轮改动而言）

1. **默认路径行为完全不变**：两次运行的三个 CIF md5、接受决策、候选台账（数量/id 连续性/状态/end）、
   `test/1` 的候选消费序列 sha1 全部与改动前一致。
2. **新增分支未在本轮真实运行中触发**：两次运行的所有候选都是 `ok`，没有出现掩码级评估失败。
   新策略（部分失败发布 `ok` / 全失败标 `reason` / 客户端跳过）由第二节的定向单测覆盖，
   真实数据的触发条件见第五节 5.1（验证范围）。
3. 未承诺加速，本轮也未观测到需要归因的耗时异常。

## 五、验证范围、已接受取舍与交付差异

本轮**功能收口**。以下三类是明确划定的边界，**不作为待办项**；本轮不追加功能、不追加长跑。

### 5.1 验证范围

| # | 边界 | 说明 |
|---|---|---|
| 1 | 真实数据上的**掩码级评估失败路径不在本轮验证范围内** | 本轮两次真实运行的候选全部为 `ok`，未触发新分支；其正确性由定向单测覆盖（`test_masked_partial_failure.py` 的部分失败 / 全失败 / 去重三类用例，`test_candidate_ordering.py` 的掩码跳过与"只计已消费范围"用例）。要真实触发需构造必然失败的输入，属**后续独立验证**，不在本轮 |

### 5.2 已接受取舍

| # | 取舍 | 理由与影响 |
|---|---|---|
| 3 | 异常路径**不再取消 / 等待请求** | 任务卡 ⑤ 明确要求撤回该包装以简化嵌套异常处理。影响面：服务死亡、台账损坏等场景下客户端抛错后不再主动收口请求。这是本轮**有意选择**，不是缺陷 |
| 4 | `skipped_mask_errors` **只写日志，不新增 metrics 事件** | 任务卡 ⑥ 只要求"记录已消费范围内的数量"，日志已满足。新增 metrics 字段会改变 `performance.jsonl` 的字段协议，本轮不做 |

### 5.3 交付差异

| # | 差异 | 说明 |
|---|---|---|
| 2 | `test/2` **此前没有消费序列 sha1 基线** | 与该 case 的历史差异**仅是基线缺失，不是行为差异**。本轮补记 **`0a1f221a73994698`**（41 个已消费候选）作为后续可比基线；`test/1` 的基线 `de02d03105ebff9f` 已逐位一致 |

## 六、交付物

| 项 | 内容 |
|---|---|
| 代码提交 | `1fee657`（`protassem/fitting/` 4 个文件 + 3 个测试文件） |
| 验证脚本 | `demo_reg_cases/policy_case_validation_20260915.py`（用户提供，未修改） |
| 额外核对脚本 | `demo_reg_cases/policy_extra_checks.py`（本轮新增，覆盖决策/台账/消费序列/错误/残留） |
| 验证输出 | `demo_reg_cases/out_policy_20260915_2004/`（两个 case 的日志、采样、manifest、结果 JSON） |
| 本报告 | `tests/reports/2026-09-15_candidate_policy_acceptance.md` |
