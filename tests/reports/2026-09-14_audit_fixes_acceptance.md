# 审计 7 项发现的修复与验收（2026-09-14）

分支：`feat/old-card-closeout`　审计起点：`decc52f`　修复终点：`8f1d2ce`（12:00 定时器触发后执行）
范围：**只做功能修复**——不新增性能优化、不动 `tail_pipeline` 默认值。

## 一、修复清单（各自一次提交）

| 审计项 | 提交 | 修复要点 | 定向测试 | "旧代码会失败"的取证 |
|---|---|---|---|---|
| **P1-1** 复合物逐域改善后重复恢复链号 | `f9e3ef1` | 模块开头固化三个链号空间（真实 / 占位 / 组件 ID）与"**占位→真实只在最终输出映射一次**"；`merge_domains(is_complex)` 保持占位链号；新增 `_restore_chain_ids()` 作为域链落盘唯一恢复点；`_accept_chain` 仍是链级唯一恢复点 | `tests/test_complex_chain_space.py`（真实 `B/C` + 占位 `A/B`，含多字符链号） | 旧代码：合并结果 `['B','C'] != ['A','B']`；二次映射抛 `PDBConstructionException: C defined twice`（探针先复现：真实 CIF `B/C` → 占位 `A/B` → `chain_map {A:B, B:C}`） |
| **P1-2** 复合物只接受一个域时链号没恢复 | `f9e3ef1` | `_handle_single_domain` 从内部空间的 `fitted_pdb` 出发，按该域 `source_chain_id` 做**单条**映射（显式指定，不做 `get(id,id)` 猜测；不可能误伤其他链）；顺带修掉"把 `.pdb` 内容写进 `.cif` 名字" | 同上 + `assemble_domain_chains` 级单域用例 | 旧代码：最终链号 `['Q+R'] != ['B']` |
| **P1-3** 服务意外退出→客户端永久等待 | `bd92f45` | 句柄持有服务进程；`poll()` 区分"请求完成 / 服务已退出 / 仍在运行"；新增 `describe_failure()`；消费者把诊断带进报错并**快速失败** | `tests/test_request_lifecycle.py`（**真实假子进程**中途 `terminate()`） | 旧代码：4 项 `TypeError`（`ParenetRequest` 根本不接受服务进程） |
| **P1-4** 无掩码路径发布后才改名 | `9b5947f` | 循环内只登记，`_drain_tail("rename")` + `rename_pdb_files_by_ranking()` 之后再按生成顺序一次性发布（名字为最终名） | `tests/test_ledger_publish_order.py`（驱动真实 `run_inference(use_mask=False)`，尾部流水线真实） | 旧代码：无 tail 时台账名字指向失效文件；有 tail 时状态里出现 `filtered` |
| **P2-5** 失败被静默跳过、请求仍报成功 | `4b5ad7e` | 服务端掩码分支异常写入结果集；请求级状态优先级明确（请求异常→`error`／主动早停→`cancelled`／候选失败→`error`／否则 `ok`）；客户端 `error_policy` **默认 `"fail"`**，`"skip"` 为显式选择 | `tests/test_failure_states.py`（服务端 3 + 客户端 5）+ `test_candidate_ordering.py` 新增默认策略用例 | 旧代码：客户端 4 项 `TypeError`（无 `error_policy`）；旧 `demo_mask` 服务端用例失败 |
| **P2-6** 台账半行 UTF-8 截断 | `9b6c3c6` | `LedgerReader` 按**字节**缓冲，`split(b"\n")` 切完整行后再解码 | `test_candidate_ledger.py` 逐字节追加含中文/`ü` 的记录 | 旧代码：`UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe4 …` |
| **P2-7** 动态密度没有版本契约 | `8f1d2ce` | 每轮掩膜写 `current_density_mNN.mrc`（**路径即版本**），主进程与共享池 worker 内缓存都自然失效；`_ensure_work_files` 同步版本名 | `tests/test_density_version.py` | 旧代码：两轮后路径相同（`current_density.mrc`），测试失败 |

单测：228 → **253 项全绿**。文档同步：`PROJECT_ARCHITECTURE.md` 五行、`ENGINEERING_HARDENING_PLAN.md` v4.12 第 106–113 条。

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

1. **真实复合物端到端未跑**：P1-1/P1-2 目前只有合成定向测试。若要真实数据背书，需要另建只含 `6lu9.cif` + `EMD-0979.mrc` 的干净目录（`/xiangyux/test_data/fiting_lg/6lu9`，res 8.8 / contour 0.316）——等用户点头再安排。
2. **P2-5 的默认"明确失败"**：本次 11 个请求 `Failed=0`，未触发；但某些输入下若单个候选失败，整次请求会中止（这是定稿契约要求的）。如需更宽松，把 `error_policy` 显式设为 `skip` 或在该调用点处理。
3. **多字符真实链号**：只有合成测试覆盖，真实数据未出现。
4. `test/2` 墙钟 2342 s：地图比 `test/1` 大约 10 倍，**不作性能结论**。

## 五、复现路径（`/xiangyux/claude_c_work/demo_reg_cases/`，不进仓库）

| 产物 | 说明 |
|---|---|
| `audit_verify_result.txt` / `run_audit_verify.sh` | 两次验证运行的原始输出与脚本 |
| `out_audit_test1` / `out_audit_test2` | 运行产物（含每个请求的 `candidates.jsonl`、`metrics/`） |
| `rt_audit.json` | 运行配置（`{"blas_threads": 1}`） |
| `tests/reports/2026-09-14_audit_fixes_acceptance.md` | 本报告 |
