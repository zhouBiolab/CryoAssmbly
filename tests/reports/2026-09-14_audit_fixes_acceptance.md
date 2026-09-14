# 审计发现的修复与验收（2026-09-14）

分支：`feat/old-card-closeout`　第一轮审计起点：`decc52f`　第一轮修复终点：`8f1d2ce`；第二轮复核起点 `aee3e78`
范围：**只做功能修复**——不新增性能优化、不动 `tail_pipeline` 默认值。

**证据分级（第二轮复核要求补充）**：每条修复分开标注三类证据 ——
**(a) 旧接口不兼容**（旧代码没有新参数，只能说明接口变了）、
**(b) 旧行为缺陷复现**（旧代码在同一条用例上给出错误结果）、
**(c) 新行为验收**（修复后的行为断言）。

## 一、第一轮 7 项（各自一次提交）

| 审计项 | 提交 | 修复要点 | 定向测试 | 证据分级 |
|---|---|---|---|---|
| **P1-1** 复合物逐域改善后重复恢复链号 | `f9e3ef1` | 固化三个链号空间（真实 / 占位 / 组件 ID）与"**占位→真实只在最终输出映射一次**"；`merge_domains(is_complex)` 保持占位链号；新增 `_restore_chain_ids()` 作为域链落盘唯一恢复点；`_accept_chain` 仍是链级唯一恢复点 | `tests/test_complex_chain_space.py`（真实 `B/C` + 占位 `A/B`） | **(b)** 旧代码：合并结果 `['B','C'] != ['A','B']`；二次映射抛 `PDBConstructionException: C defined twice`。**(c)** 新行为：合并=`{A,B}`、落盘后=`{B,C}`，且不抛异常 |
| **P1-2** 复合物只接受一个域时链号没恢复 | `f9e3ef1` | `_handle_single_domain` 从内部空间 `fitted_pdb` 出发，按该域 `source_chain_id` 做**单条**映射；顺带修掉"把 `.pdb` 内容写进 `.cif` 名字" | 同上 + `assemble_domain_chains` 级单域用例 | **(b)** 旧代码：最终链号 `['Q+R'] != ['B']`。**(c)** 新行为：最终 `domain_chains/*.cif` 链号 = 归属真链号 |
| **P1-3** 服务意外退出→客户端永久等待 | `bd92f45` | 句柄持有服务进程；`poll()` 区分"请求完成 / 服务已退出 / 仍在运行"；新增 `describe_failure()`；消费者把诊断带进报错并**快速失败** | `tests/test_request_lifecycle.py`（**真实假子进程**中途 `terminate()`） | **(a)** 旧代码：4 项 `TypeError`（旧接口不接受服务进程）—— 只说明接口变化。**(b)** 新增 `test_old_poll_logic_cannot_see_a_dead_service`：把旧 `poll` 逻辑原样重写、跑在同一个假服务上 → 返回 `None`（永远"在跑"），而新句柄返回非 None。**(c)** 新行为：死亡后 `describe_failure()` 含 returncode、消费者 ≤4 次轮询即抛错 |
| **P1-4** 无掩码路径发布后才改名 | `9b5947f` | 循环内只登记，`_drain_tail("rename")` + `rename_pdb_files_by_ranking()` 之后再按生成顺序一次性发布 | `tests/test_ledger_publish_order.py` | **(b)** 旧代码：无 tail 时台账名字指向失效文件；有 tail 时状态里出现 `filtered`。**(c)** 新行为：两种设置下 id 连续、全 `ok`、名字指向磁盘真实文件 |
| **P2-5** 失败被静默跳过、请求仍报成功 | `4b5ad7e` | 服务端掩码分支异常写入结果集；请求级状态优先级明确；客户端消费到 `error` 立即抛错 | `tests/test_failure_states.py` + `test_candidate_ordering.py` | **(b)** 旧服务端：`Failed: 0` 而日志有异常；旧客户端：把 `error` 当 `filtered` 跳过。**(c)** 新行为：见第二节 #1 |
| **P2-6** 台账半行 UTF-8 截断 | `9b6c3c6` | `LedgerReader` 按**字节**缓冲，切完整行后再解码 | `test_candidate_ledger.py` 逐字节追加含中文/`ü` 的记录 | **(b)** 旧代码：`UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe4 …`。**(c)** 新行为：无异常、内容与写入一致 |
| **P2-7** 动态密度没有版本契约 | `8f1d2ce` | 每轮掩膜写 `current_density_mNN.mrc`（**路径即版本**），主进程与共享池 worker 内缓存都自然失效 | `tests/test_density_version.py` | **(b)** 旧代码：两轮后路径相同。**(c)** 新行为：路径不同、旧版本仍在、内容互不相同 |

## 一bis、第二轮复核的 4 项（提交见下）

| 复核项 | 提交 | 修复要点 | 定向测试 | 证据分级 |
|---|---|---|---|---|
| **#1** 同一 mask 内部分评估失败仍报成功 | `28d4440` | `_publish` 改为**错误优先**；掩码分支异常写入 `mask_results` 与 `all_results`，使日志 / `Failed` / 台账口径一致 | `tests/test_masked_partial_failure.py`（新，4 项） | **(b)** 旧代码：`['ok'] != ['error']`、计数 `0 != 1`。**(c)** 新行为：部分失败→`error` + `end.status="error"`；全成功→`ok` |
| **#2** 客户端抛错未收口请求 | `28d4440` | `run()` 包装 `_run()`：异常时先 `on_cancel()` 再 `_await_request_end()`，清理异常不得覆盖原始错误 | `test_candidate_ordering.py::test_error_candidate_cancels_the_request` | **(b)** 旧 consumer：该用例失败（未取消）。**(c)** 新行为：抛错前 `cancelled=True` 且检查过请求结束 |
| **#3** `error_policy="skip"` 名不副实 | `28d4440` | **移除该参数**（不新增宽松策略）：候选级 `error` 一律抛错 | `test_failure_states.py::test_error_policy_is_not_a_supported_mode` | **(c)** 新行为：传该参数 `TypeError`；报告删掉"需要宽松就用 skip"的措辞（原文不准确） |
| **#4** 评分缓存统计读旧字段 | `60ea283` | 统计写入抽成 `_record_score_cache()` / `_record_tm_cache()`，占用/峰值取**顶层共享字段** | `tests/test_score_cache_metric.py`（新，4 项） | **(b)** 旧表达式 `snapshot["density"]["bytes"]` → `KeyError: 'bytes'`，两次审计运行日志均有 `Score cache snapshot failed: 'bytes'`。**(c)** 新行为：事件写出且字段齐全 |

单测：228 → **259 项全绿**（第二轮拆分测试文件后复核：`test_masked_partial_failure.py` 的 4 项从 `test_failure_states.py` 迁出，避免同一组用例被计两次）。文档同步：`PROJECT_ARCHITECTURE.md`、`ENGINEERING_HARDENING_PLAN.md`（v4.12 第 106–115 条、v4.13 第 116–119 条）。

## 二、验证 ①：默认 `test/1` 与冻结基线逐位一致

`python main.py test/1/EMD-8436.mrc test/1 5.6 0.04 <out> --log --runtime-config {"blas_threads":1}`，HEAD `8f1d2ce`。

| 项 | 结果 |
|---|---|
| 墙钟 / `pipeline_total` | 1240 s / 1230.21 s（在 O6 时代的波动带 1211–1327 s 内） |
| `assembled_complex.cif` / `_all.cif` | `76638d0f97d4ee5775110c66b15e73c6` = **基线** |
| `refined_complex.cif` | `bd281f40e25352455ba34d41e6bbc9f7` = **基线** |
| 决策 | Chain A 拒 0.4192、Chain B 受 0.4235、Chain A 2 域 0.4430（与基线逐条一致） |
| 候选台账 | 3 请求 24/28/115，id 连续、全 `ok`、`end=ok` |
| 消费序列 | 58 个候选，sha1 `de02d03105ebff9f`（与修复前一致） |
| 服务端失败计数 | `Total … Failed: 0`（全部请求） |

→ **7 项修复没有改变默认路径的产物、决策与消费顺序**（P1-4/P2-5 动的是非默认与失败路径，默认路径逐位不变）。

## 三、验证 ②：`test/2` 额外真实数据（无基线，只作真实运行验证）

输入：`EMD-29607.mrc`（245 MB）+ `chain_A_1.pdb`/`chain_B_2.pdb`/`chain_C_3.pdb`（**三条独立单链 PDB**，内部链号分别 B/C/A），`resolution.txt` = 5.50、`contour_level.txt` = 0.011。

| 项 | 结果 |
|---|---|
| 墙钟 / `pipeline_total` | 2342 s（39 分钟）/ 2332.73 s；exit 0 |
| 整链拟合 | A 拒 0.2337、C 拒 0.3898、B 拒 0.2387 → 三条链全部转入域路径 |
| 域链组装 | A: 2 域 cc=0.5610、C: 2 域 cc=0.5474、B: 2 域 cc=0.5316 |
| 产物 | `assembled_complex.cif` = `6d2d87405bfc433ac0ac6dfbc81d3ed0`、`refined_complex.cif` = `80a17ee5b6ba46ff38508dafcb25466b` |
| 候选台账 | 8 请求 296 个候选，id 连续、**全 `ok`**；3 × `end=ok`（整链）+ **5 × `end=cancelled`**（域拟合主动早停，新契约下正常控制流） |
| 服务端失败计数 | `Failed: 0`（全部 11 个请求，两次运行合计） |

**边界声明（重要）**：`test/2` 是**三条独立的单链 PDB 输入**，不经过"多链 CIF → PDB 占位链号 → 恢复真实链号"这条路径，因此**不能**证明 P1-1/P1-2 已修好；它只是"另一个真实数据集的整体流程可用性"验证。P1-1/P1-2 的证据是第二节的合成定向测试（真实 `B/C` + 占位 `A/B`、单域复合物）。`test/2` 没有冻结基线，故也不称"前后等价"。

## 四、残留范围与开放项

1. **真实复合物端到端未跑**：P1-1/P1-2 的证据是第二节的合成定向测试（真实 `B/C` + 占位 `A/B`、多字符 `AA/BB`、单域复合物、以及"逐域改善 → `_accept_chain()` → 最终 CIF"串联用例 `test_improved_pose_survives_accept_chain_and_keeps_real_ids`）；真实数据仍需另建只含 `6lu9.cif` + `EMD-0979.mrc` 的干净目录（res 8.8 / contour 0.316）。
2. **没有宽松模式**：候选级 `error` 一律抛错（`error_policy` 已移除）。本次 11 个请求 `Failed=0` 未触发该路径；若某类输入出现单点评估失败，整次请求会中止 —— 这是定稿契约的行为，需要容错时必须单独设计"部分成功"语义，而不是靠一个已被移除的开关。
3. **多字符真实链号**：由 `test_multichar_real_chain_ids_survive_the_placeholder_round_trip` 覆盖（合成 CIF），真实数据未出现。
4. `test/2` 墙钟 2342 s：地图比 `test/1` 大约 10 倍，**不作性能结论**。
5. 第二轮复核后**尚未重跑真实全流程**（本轮只改失败路径与统计写入，按你"先修完再考虑长跑"的要求，test/1 复测见下）。

## 四bis、第二轮修复后的复测（`7517f11`）

本轮只改失败传播与指标写入，未改数值路径；仍按验收纪律跑了默认 `test/1`（13:51–14:12）：

| 项 | 结果 |
|---|---|
| 墙钟 / `pipeline_total` | 1258 s / 1248.91 s（在 1211–1330 s 波动带内） |
| 三个 CIF md5 | `76638d0f…` / `76638d0f…` / `bd281f40…` = **`baseline_after_o6.md5`** |
| 决策 | A 拒 0.4192、B 受 0.4235、A 两域 0.4430（逐条一致） |
| 台账 / 消费序列 | 3 请求 24/28/115、58 个候选、sha1 `de02d03105ebff9f`（一致） |
| 服务端失败计数 | 三个请求均 `Failed: 0` |
| **`score_cache` 指标事件** | **本次真实写出**（修复前被 KeyError 吞掉）：`score_cache_mb=128, entries=52, bytes=110.5 MB, peak_bytes=127.8 MiB, evictions=28, density_hits=72, density_misses=4, structure_hits=0, structure_misses=76`；日志中**没有** `Score cache snapshot failed` |
| `tm_cache` 指标事件 | 写出：`mode=auto, hits=4, misses=0, writes=0, rows=12` |
| 未归因时间 | 2.02 s |

→ 第二轮 4 项修复**没有改变默认路径产物**，且审计#4 的症状在真实运行里已消失。

## 五、复现路径（`/xiangyux/claude_c_work/demo_reg_cases/`，不进仓库）

| 产物 | 说明 |
|---|---|
| `audit_verify_result.txt` / `run_audit_verify.sh` | 两次验证运行的原始输出与脚本 |
| `out_audit_test1` / `out_audit_test2` | 运行产物（含每个请求的 `candidates.jsonl`、`metrics/`） |
| `rt_audit.json` | 运行配置（`{"blas_threads": 1}`） |
| `tests/reports/2026-09-14_audit_fixes_acceptance.md` | 本报告 |
