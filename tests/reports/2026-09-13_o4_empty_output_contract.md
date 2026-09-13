# O4 空结果输出契约：实现与验证

日期：2026-09-13　代码：`feat/engineering-hardening`（O4 提交见分支，父提交 `a295872`）

## 契约（按用户决定）

1. 没有最终组件时**不写出** `assembled_complex*.cif`（不再写"合法空 CIF"）。
2. 运行摘要**始终生成**，记录输入数、接受域数、合并链数、过滤数量与最终状态。
3. 结构路径返回 `None`，不返回不存在或无效文件的路径。
4. 完整版有组件、过滤版为空时**保留完整版**，两个状态分别记录。
5. 同一输出目录重跑时清理上一轮的程序管理产物（精确文件名白名单）。
6. Step4/Step5 明确判断前置产物，不把"无结果"当有效结构继续处理。

## 实现

| 位置 | 变更 |
|---|---|
| `complex_builder.build_complex` | `added == 0` 时删除同名旧文件、不写出、返回 `(None, remap_log)`；docstring 写明返回契约 |
| `complex_builder.clear_stale_outputs` | 新函数：重跑前清理 `assembled_complex.cif` / `assembled_complex_all.cif` / `refined_complex.cif` / `homo_chain_refined_complex.cif` |
| `orchestrator.run()` | 构建完成版与过滤版后分别判定；过滤版为空时告警但保留完整版；`homo`/最终返回回退链改为 `homo_cif or refined_cif or complex_cif or all_cif`；无任何产物时告警 |
| `orchestrator._output_status` | 新函数：`accepted_components` / `accepted_domains` / `merged_domain_chains` / `filtered_out_domain_chains` |
| 运行摘要 | 新增 `assembled_complex_all` / `assembled_complex_filtered` / `final_status`（`assembled` 或 `no_result`）与上述统计 |
| `refine_step._skip_reason` | 新增门控：没有任何组装产物时返回 `no assembled complex`，Step4 明确跳过 |
| `refine_step.maybe_refine` | 备份日志改为指向实际存在的产物（过滤版优先、否则完整版），不再假定文件存在 |

（`homo_chain_step.maybe_homo_refine` 本就对 `None` / 不存在路径跳过，无需改动。）

## 测试（6 项，无 GPU）

`tests/test_complex_output_contract.py`：

| 测试 | 覆盖 |
|---|---|
| `test_all_empty_writes_nothing_and_returns_none` | 全空：不写出、返回 `None` |
| `test_filtered_empty_keeps_full_and_records_both_statuses` | 仅过滤版为空：保留完整版；状态统计 1/1/1/1 |
| `test_normal_non_empty_writes_both_with_expected_chains` | 正常非空：两份都写出、链号正确 |
| `test_stale_outputs_from_previous_run_are_removed` | 旧输出残留：白名单清理 + 空结果不复活 |
| `test_stale_target_is_removed_when_current_run_is_empty` | 空结果时删除同名旧文件 |
| `test_refine_is_skipped_when_no_assembled_complex_exists` | Step4 在无组装产物时明确跳过 |

## 真实集成验证

### 验证 1：链被整链接受（11:53，187 s）

预置三份旧产物后重跑，日志确认 `Removed stale output(s) from previous run: assembled_complex.cif,
assembled_complex_all.cif, refined_complex.cif`；摘要出现新状态字段
（`assembled_complex_all: True` / `assembled_complex_filtered: True` / `final_status: assembled` /
`accepted_components: 1`）。
（该场景下过滤版与完整版同为整链记录：按 `build_complex` 既有约定"整链/复合物记录不做域级过滤"，
两个文件都用同一 `fitted_cif`，因此过滤版非空。）

### 验证 2：域链被接受但所有域低于 `complex-min-cc`（12:01，410 s）

这是"仅过滤版为空"的真实场景（probe3 配置 + `--complex-min-cc 0.99`）：

```text
=== O4 final check finished at 2026-09-13 12:01:57 ===
elapsed_seconds=410
--- 关键行 ---
2026-09-13 12:01:48,349 INFO [protassem.assembly.complex_builder] Removed stale output(s) from previous run: assembled_complex.cif, assembled_complex_all.cif, refined_complex.cif
2026-09-13 12:01:51,895 INFO [protassem.assembly.domain_assembler] Chain Q+R: assembled from 5 domains (cc=0.4490)
2026-09-13 12:01:52,877 INFO [protassem.assembly.complex_builder] No component for assembled_complex.cif; file not written
2026-09-13 12:01:52,878 WARNING [protassem.assembly.orchestrator] No component passed the complex_min_cc filter; assembled_complex.cif not written (assembled_complex_all.cif kept)
2026-09-13 12:01:52,884 INFO [protassem.refine] Step 4 (refine) disabled (--no-refine)
2026-09-13 12:01:52,901 INFO [protassem.assembly.orchestrator] Assembly complete. Accepted: 1 components
2026-09-13 12:01:52,902 INFO [protassem.assembly.orchestrator] Complex: /xiangyux/claude_c_work/demo_reg_cases/out_o4_final/assembly/final_results/assembled_complex_all.cif
--- final_results 内容（应只有 all 版，没有 filtered 版）---
assembled_complex_all.cif
assembly_summary.txt
chains
domain_chains
--- summary ---
Assembly Summary Report
======================================================================

Date: 2026-09-13 12:01:52
chain_threshold: 0.45
complex_threshold: 0.35
domain_initial_threshold: 0.3
domain_min_cc: 0.25
complex_min_cc: 0.99
similarity_threshold: 0.85
resolution: 5.6
contour: 0.04
input_files: 1
processed_chains: 1
skipped_inputs: 0
skipped_detail: 
assembled_complex_all: True
assembled_complex_filtered: False
final_status: assembled
accepted_components: 1
accepted_domains: 5
merged_domain_chains: 1
filtered_out_domain_chains: 1

Total accepted: 1
  As chain: 0
  As complex: 0
  As domain chain: 1

1. [01] Q+R (domain_chain)
--- 链号 ---
assembled_complex_all.cif            ['Q', 'R']
domain_chains/domain_chain_Q+R_01.cif ['Q', 'R']
done```

结论：旧产物被清理；域链正常组装（R1 修复生效，链号 `Q`,`R`）；**过滤版为空时没有写出**
`assembled_complex.cif`，完整版 `assembled_complex_all.cif` 保留；摘要始终生成并分别记录两个状态；
最终返回与日志回退到完整版；链号均为真链号。
