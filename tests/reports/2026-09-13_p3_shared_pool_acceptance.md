# P3 复用 CPU 池：实现与验收（实测报告）

日期：2026-09-13　代码：`feat/engineering-hardening`，**P3 提交 `773acf7`，父提交 `6b6f641`**（P2-fine）。
**基线口径**：本报告的比较基线是**老卡原始冻结基线**（T01 之前）`2497fefc2e866ad65db628fb75f1ec0a`（`assembled_complex.cif`/`_all.cif`）
与 `dbc937efc071cd9d1d2611f3ad84743f`（`refined_complex.cif`）；T01 之后基线已重冻结为 `76638d0f…`/`bd281f40…`，本报告不引用后者。
**口径说明（v4.11）**：报告中的"未归因 1.13 s"是**计时覆盖完善**（`final_select` 267.6 s 补埋点），**不是加速**；
池复用只覆盖**已接入的主拟合路径**，未接入位置见第一节"仍保留自建池"。
case：`test/1`（两条单链 PDB + `EMD-8436.mrc`，res 5.6，contour 0.04），全部运行均为 1 BLAS 线程。

## 一、实现（保持候选策略与数值计算不变）

| 变更 | 位置 |
|---|---|
| 新增运行级执行上下文：惰性共享池、`map()` 按输入顺序归并、`close()` 幂等、`workers<=1` 时串行不建池 | `protassem/runtime/execution.py`（新） |
| 池原语 `open_pool()` / `close_pool()`（只计创建与 close+join），`timed_pool` 复用之 | `protassem/runtime/pool.py` |
| `run_pipeline` 建立上下文并在 `finally` 释放；`run_assembly` 独立调用时自建自释 | `pipeline.py`、`assembly/orchestrator.py` |
| `context` 显式穿过 `run_fitting → _fit_chain/_fit_domain → _fit_single → _monitor_and_evaluate → _batch_cc/_optimize_candidate/_select_final_result/_optimize_top_n` | `fitting/pipeline.py` |
| `local_optimize`、`prefill_tm_cache`、预筛 CC、chain_fitter 的局部优化全部改用 `context.map` | `fitting/local_optimizer.py`、`core/similarity.py`、`assembly/orchestrator.py`、`assembly/chain_fitter.py` |
| 删除临时建池：`_batch_cc` 的每批 `mp.Pool`、`local_optimize` 的每批池、TM 预填池、预筛池（共 4 类） | 同上 |
| 新增 `pool_start_method` 运行配置（默认 fork；`spawn` 用于单独测启动成本） | `protassem/runtime/config.py` |

**仍保留自建池（未纳入 P3，已记录）**：`homo_chain_refine.py` 5 处（Step5，需 `--homo-chain-refine` 才跑）、
vendored `assembly/refine/chain_enumerator.py:606`（Step4 枚举器）、legacy `sampling/extract_points/VoxEM.py:299`。

## 二、验收结果

### 池创建次数（同一 case，四组运行）

| 运行 | 代码 | worker | pool_start | pool_close | 墙钟(s) |
|---|---|---|---|---|---|
| P2-fine | P3 前 | 10 | **79** | 79 | 1136.2 |
| P3-A10 | P3（接线前，4 池） | 10 | 4 | 4 | 1074.2 |
| **P3-A2** | **P3（接线后）** | **10** | **1** | **1** | **1021.2** |
| P3-B1 | P3 | 1 | **0**（串行不建池） | 0 | 1470.1 |

### 结果等价性（`assembled_complex.cif` / `refined_complex.cif` md5）

| 运行 | 配置 | md5 | 与冻结基线 |
|---|---|---|---|
| 冻结基线 | 10 worker | `2497fefc…` / `dbc937efc071…` | — |
| P3-A2 | 10 worker | `2497fefc…` / `dbc937efc071…` | **一致** ✓ |
| P3-B1 | 1 worker | `cb56a863…` / `1bfe6a48…` | **不一致** |
| **冻结基线代码** | **1 worker** | **`cb56a863…` / `1bfe6a48…`** | 与 P3-B1 **完全相同** |

**判定**：P3 未改变结果——同一配置下"冻结基线代码"与"P3 代码"产出完全一致。
1 worker 与 10 worker 的结果差异是**既有配置敏感性**（监测循环按批次大小与 2.5 s 轮询节奏挑选候选并触发早停），
不是 P3 引入；已在方案中登记为新开放项 O6，供后续决定是否固定候选消费顺序。

### 各阶段耗时（客户端，秒）

| 阶段 | P2-fine（79 池） | P3-A2（1 池） | 变化 |
|---|---|---|---|
| pipeline_total | 1136.2 | 1021.2 | −115.0（含运行间波动） |
| local_optimize | 201.0 | 160.4 | −40.6 |
| final_select | 267.6 | 211.2 | −56.4 |
| cc_batch | 38.9 | 33.7 | −5.2 |
| gpu_wait | 547.5 | 542.5 | −5.0 |
| 未归因 | 1.13 | 1.09 | 口径保持 |

**注意**：这是两次不同时刻的运行（相隔约 1 小时），服务端负载不同；不得把 −115 s 全部归因于池复用。
可归因的硬证据是"池创建 79 → 1 次、创建总耗时 4.34 s → 0.08 s"以及"同配置结果逐字节一致"。

### 串行（1 worker）对照

`cc_batch` 38.9 → 175.1 s、`final_select` 267.6 → 524.3 s、墙钟 1136 → 1470 s（+29%）：
说明并行 CC/局部优化确实在起作用（`--num-processes 1` 仅用于对照，不建议生产使用）。

### 建池成本（独立基准 `tools/pool_bench.py`，10 worker × 3 次）

| 启动方式 | 建池耗时 (s) | 均值 |
|---|---|---|
| fork（默认） | 0.030 / 0.023 / 0.021 | ≈25 ms |
| **spawn** | 0.036 / 0.065 / 0.056 | ≈52 ms |

spawn 建池约为 fork 的 2 倍；**该项单独测量，未与池复用收益混合**。
（注：spawn 无法从 stdin 脚本运行——它会重新导入 `__main__`；基准与运行都用真实脚本文件。）

## 三、残留与后续

- 每次运行结束都检查残留进程：`(无残留)` ✓（含 `demo_mask.py --server`）。
- 测试：`python -m unittest discover -s tests` → 92 项通过；`compileall` 与 `git diff --check` 干净。
- 后续（按测量，不提前上 GPU batching）：最高杠杆仍是服务端 CPU 后处理（postprocess ≈460–485 s）与
  写盘（2338 次 ≈85–92 s）；P4（CC 缓存）需先统计**全部**评分调用点；P5 暂缓。
