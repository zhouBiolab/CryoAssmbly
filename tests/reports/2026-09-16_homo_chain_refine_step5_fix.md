# 修复：Step 5 同源链精修静默失效（`local_optimize` 调用点未随签名更新）

日期：2026-09-16
基线：tag `release/20260915-211518`（annotated tag 对象 `5b801a6…` → commit `3b3fbbd`）
修复分支：`fix/homo-chain-refine-local-optimize`（隔离工作树 `/xiangyux/claude_c_work/demo_reg_homofix`）
**tag 指向未改动**；只在本分支修改，未合入其他分支。

## 1. 缺陷确认（运行时实测，非读代码推断）

```
local_optimize 参数： [structure_file, density_mrc, output_file, resolution, contour,
                      max_iterations, initial_step_size, initial_cc, metrics, context]
是否接受 num_processes： False
调用形态 bind： TypeError -> got an unexpected keyword argument 'num_processes'
```

### 回归来源

`773acf7 feat(runtime): reuse a single run-level CPU pool (P3)`（2026-09-13）改动：

```diff
-                   initial_step_size=1.25, num_processes=1, initial_cc=None,
+                   metrics=None, context=None):
```

该提交 **stat 中没有 `homo_chain_refine.py`** —— 只改了签名，漏改调用点。
「旧版正常」也成立：初始提交 `6d301e1` 签名里**有** `num_processes=1`，调用点当时可绑定。

### 为什么长期未被发现

1. P3 验收报告第 22 行**自己声明**：「**仍保留自建池（未纳入 P3，已记录）**：`homo_chain_refine.py` 5 处（Step5，需 `--homo-chain-refine` 才跑）」；
2. `tests/` 下原本**没有任何** homo_chain 覆盖；本次新增测试前，全仓扫不出这类问题的检查也不存在；
3. 该路径需显式 `--homo-chain-refine` 才走，历次真实验收都没带该开关；
4. 失败被 `except Exception` 吞掉（两处只写 stderr，`_fill_worker` 连 stderr 都不写）。

## 2. 修复清单（共 4 个文件、7 个调用点）

| # | 位置 | 缺陷表现 | 修法 |
|---|---|---|---|
| A1 | `protassem/assembly/homo_chain_refine.py:204` `_cand_worker` | TypeError，仅 stderr | `num_processes=1` → `context=None` |
| A2 | 同上 `:223` `_anchor_cand_worker` | 同上 | 同上 |
| A3 | 同上 `:283` `_fill_worker` | TypeError，**完全静默** | 同上 |
| B | `protassem/fitting/local_optimizer.py:405`（`__main__`） | `a.num_processes` 当**位置**参数传入 → 错位进 `initial_cc`（不报错） | 改正为关键字传参、删除错位实参；`--num_processes` 保留语义并接回 `ExecutionContext`；新增 `--pool_start_method` |
| C1 | `geo_sym_refine.py:166` `_cand_worker` | **与新测试扫描一并发现**，同类缺陷 | `context=None` |
| C2 | `geo_sym_refine.py:226` `_fill_worker` | 同上 | 同上 |

**A/C 处为何仍是"不并行"**：这三个 worker 运行在 `Pool(...)` 内，`runtime/execution.py` 与 `runtime/pool.py` 的约定是「**worker 内不得再建池**」；`num_processes=1` 的原意就是禁用内层并行。改为**显式 `context=None`** 而非直接删参数，把该语义留在代码里，防止以后有人顺手接上共享池。

**B 处为何恢复并行**：该 `__main__` 是独立入口，不嵌套任何池；`_density_copy_worker` 是模块级可序列化函数，满足共享池契约。

## 3. 修复前后的实测对照（决定性问题）

`python protassem/fitting/local_optimizer.py <tiny>/chain_a.pdb <tiny>/tiny.mrc <out> --resolution 6.0 --contour 0.04 --num_processes 10`

| | exit | 产物 bytes | sha256 前16位 | 含义 |
|---|---:|---:|---|---|
| 输入 `chain_a.pdb` | — | 1058 | `b011210f7bd0b4e2` | 未优化原结构 |
| **修复前** | **0（假成功）** | **1058** | **`b011210f7bd0b4e2`** | **原样返回：优化完全没有发生** |
| 修复后 `--num_processes 1` | 0 | 952 | `66161f532a167ba7` | 真优化 |
| 修复后 `--num_processes 10` | 0 | 952 | `66161f532a167ba7` | **与串行逐位一致**（10 进程并行可用） |

⚠️ **注意修复前的 exit code 是 0**：它不报错，而是把输入结构**原样复制**成输出（`local_optimize` 内部捕获异常后走「所有副本失败 → keep input structure」分支，并 `return True`）。日志里 `copy step=4.5 density=0.0000` 之类的行仍然真实（副本计算发生在异常之前），所以**日志看起来完全正常**。这是该缺陷能长期潜伏的关键原因，也说明「静默失效」比字面描述更严重。

库层并行度对照（`ExecutionContext`，同一 tiny 输入）：`pool_workers=1` 2.69 s → `pool_workers=4` 0.13 s（≈20 倍）；CLI 端到端因含解释器与 numba 启动开销，tiny 规模下两者接近（7.52 s vs 8.05 s），并行收益需在真实规模上体现。

## 4. 新增回归测试

`tests/test_local_optimize_call_sites.py`（8 项，全部通过）：

| 测试 | 拦什么 |
|---|---|
| `test_local_optimize_rejects_removed_num_processes` | 锁定基线事实：`num_processes` 已不是参数 |
| `test_all_project_sources_parse` | 自研代码必须全部可解析——否则扫描会**静默漏掉整个文件** |
| `test_no_call_site_has_unbindable_arguments` | **全仓** `ast` 扫描 → 逐调用点 `inspect.signature().bind()`，不允许抛 `TypeError` |
| `test_homo_chain_refine_still_calls_local_optimize` | 契约：该文件仍须携带 ≥3 个调用点且都显式给出 `context`（防止整段被删后测试退化成空跑） |
| `test_cli_serial/parallel_exits_zero_and_writes_output` | 独立入口 1 与 10 进程都必须 exit=0 且产出文件存在 |
| `test_serial_branch_runs_in_process` / `test_parallel_map_with_module_level_function` | `ExecutionContext` 串行/并行分支行为与顺序归并语义 |

**测试自身修掉的两个坑**（记录以免重犯）：
- `homo_chain_refine.py` 带 **UTF-8 BOM**，Python 3.8 的 `ast.parse` 直接读会 `SyntaxError` → 该文件被静默跳过，测试假通过。改为 `utf-8-sig` 读取，并补 `test_all_project_sources_parse` 兜底。
- CLI 子进程需显式 `PYTHONPATH=仓库根`（与生产调用方式一致）。

**该测试第一次运行就抓出了 C1/C2 两处我此前未发现的缺陷**——这正是"全仓扫描"相对"只改报告里那三行"的价值。

## 5. 验证结果

| 命令 | 结果 |
|---|---|
| `python -m unittest discover -s tests -t .` | **Ran 269 tests — OK**（修复前 261 + 新增 8） |
| `python -m compileall -q protassem tests main.py tools` | COMPILE OK |
| `git diff --check` | 无空白问题 |
| 调用点全扫 | 9 个调用点全部可绑定，不可绑定 = **0** |
| 诊断入口 1 / 10 进程 | 均 exit=0，产物 sha256 相同 |

**改动面**：`geo_sym_refine.py`(2)、`protassem/assembly/homo_chain_refine.py`(3)、`protassem/fitting/local_optimizer.py`(1 处错位 + 并行接回 + `__main__` 选项)；共 3 文件 23 插入 11 删除。

**未做的事**：不改 `num_processes` 在其他模块的合法用法；不把 `homo_chain_refine.py` 的 5 处自建池并入 P3 共享池（属行为变更，超出本次范围）；不动其他分支（`refactor/mask-candidate-policy`、`codex/source-cache-result-level` 仍带此缺陷）。

## 6. 复现

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate point
cd /xiangyux/claude_c_work/demo_reg_homofix
export PYTHONPATH=$PWD

# 1) 调用点扫描（应全部可绑定）
python -m unittest tests.test_local_optimize_call_sites -v

# 2) 独立入口 1 / 10 进程（产物应逐位一致）
T=/xiangyux/claude_c_work/demo_reg_cases/tiny_case
python protassem/fitting/local_optimizer.py $T/chain_a.pdb $T/tiny.mrc /tmp/p1.pdb \
    --resolution $(cat $T/resolution.txt) --contour $(cat $T/contour_level.txt) --num_processes 1
python protassem/fitting/local_optimizer.py $T/chain_a.pdb $T/tiny.mrc /tmp/p10.pdb \
    --resolution $(cat $T/resolution.txt) --contour $(cat $T/contour_level.txt) --num_processes 10
sha256sum /tmp/p1.pdb /tmp/p10.pdb

# 3) 全量回归
python -m unittest discover -s tests -t .
```
