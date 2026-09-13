# R1：复合物域链合并丢链修复报告

日期：2026-09-13　代码：`feat/engineering-hardening` @ `a338dfb`（R1 提交见分支）

## 现象（来自 S4b probe3）

放宽阈值让域达标后，日志出现：

```text
[Round 1] 本轮接受 3 域 ... / [Round 2] 接受 1 域 / [Round 3] 接受 1 域
Unified assembly done: 3 轮, accepted=5 组件
ERROR [protassem.assembly.domain_assembler] merge_domains error: (' ', 1, ' ') defined twice
INFO  [protassem.assembly.domain_assembler] Chain Q+R: domain merge failed; chain dropped
Complex CIF created: .../assembled_complex_all.cif (0 chains)
```

即：**域是合法接受的，但在域链合并阶段整链被丢弃**，最终写出 0 链、Bio.PDB 无法解析的复合物 CIF。

## 根因（已核实）

1. 复合物经 `--complex-domain-opt` 拆成两条链再各自切域，**每条链的域 CIF 残基编号各自从 1 开始**：
   - chain A（占位）：`domain_1` = 1–238、`domain_2` = 239–564
   - chain B（占位）：`domain_4` = 1–238、`domain_5` = 239–304、`domain_6` = 305–564
2. `domain_assembler.assemble_domain_chains()` 调用 `merge_domains(orch, cid, fitted, ranges)`
   **没有传 `is_complex`** → 走非复合物分支，把 5 个域全部并进同一条 `Chain("Q+R")`
   → 两条链的 1–238 号残基重复 → Biopython `Residue defined twice` → 整链丢弃。
3. 另外，`merge_domains` 的复合物分支用 `source_chain_id`（占位链号 A/B）作为输出链号，
   即使合并成功也会把占位链号写进最终 CIF（这正是 S4b 关心的问题）。

## 修复

- `assemble_domain_chains()`：读取 `chain_rec["is_complex"]` 并传给 `merge_domains`
  与 `_filtered_domain_cif`（过滤版同样需要按链分开）。
- `merge_domains()` 复合物分支：新增 `chain_map_of(orch, chain_id)`，用原始 `chain_records` 的
  `chain_map` 把占位链号映射回真链号（`A→Q`、`B→R`）；`chain_map` 的键只含占位链号，
  **已是真链号的输入原样通过，不会二次映射**。
- 未改动 `chain_id_for_cif`（S4b 的约定）；`chain_fitter` 的逐域微调路径本来已传 `is_complex`。

## 定向测试（4 项，无 GPU）

`tests/test_domain_assembly.py`，用最小 orchestrator 状态 + 合成域 CIF 直接驱动真实的
`merge_domains` / `assemble_domain_chains`：

| 测试 | 断言 |
|---|---|
| `test_merge_without_is_complex_reproduces_the_failure` | 不传 `is_complex` → 返回 `None`；测试输出复现原错误 `(' ', 1, ' ') defined twice` |
| `test_merge_with_is_complex_keeps_one_chain_per_source_and_real_ids` | 传 `is_complex=True` → 链号 `Q`,`R`，各 10 残基 |
| `test_real_chain_ids_are_not_mapped_twice` | `source_chain_id` 已是真链号时原样通过 |
| `test_assemble_domain_chains_produces_domain_chain_with_real_ids` | 端到端组装产出 `domain_chain` 记录，其 CIF 链号为 `Q`,`R` |

## 真实运行验证（同 probe3 配置，421 s）

| 项 | 修复前 | 修复后 |
|---|---|---|
| 域链组装 | `merge_domains error ... chain dropped` | `Chain Q+R: assembled from 5 domains (cc=0.4490)` |
| `final_results/domain_chains/domain_chain_Q+R_01.cif` | 不存在 | 链号 **`Q`,`R`** |
| `assembled_complex.cif` / `assembled_complex_all.cif` | 0 链、不可解析 | 链号 **`Q`,`R`** |
| 摘要 | `Total accepted: 0` | `Total accepted: 1` / `As domain chain: 1` |

## 与 O5 的关系

- `refine_step._backfill_chains_as_domains()` 只处理 `type == "chain"`，复合物记录是
  `type="complex"` → **该分支对复合物不可达**（S4b 的残留担忧不成立于这条路径）。
- 真正消费复合物域产物的位置是 `assemble_domain_chains` → `merge_domains(is_complex=True)`
  → `_save_domain_chain` → `build_complex`；本次已在该路径上验证链号恢复为真链号。
