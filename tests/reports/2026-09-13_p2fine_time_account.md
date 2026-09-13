# P2 细化：一次 fit_request 的时间账（实测报告）

日期：2026-09-13　代码：`feat/runtime-metrics`（父提交 `5b38c70`）
case：`test/1`，1 线程默认配置，墙钟 **1136 s**（rc=0，三个最终 CIF 与冻结基线 md5 一致，无残留进程）。

## 一、客户端：fit_request 的细分（合计 1062.3 s）

| 阶段 | 次数 | 秒 | 说明 |
|---|---|---|---|
| gpu_wait | 3 | 547.5 | 轮询等待预测（纯 sleep 累计） |
| **final_select** | 3 | **267.6** | 此前**未埋点**：最终候选选择（含其内部 CC 评分） |
| local_optimize | 24 | 201.0 | 局部优化整体 |
| cc_batch | 17 | 38.9 | 批次 CC |
| cc_verify | 3 | 5.3 | 最终 CC 复核 |
| candidate_scan | 3 | 0.8 | 候选文件扫描 |
| save_result / analyze_sources | 3 / 2 | 0.14 | 结果保存、源文件分析 |
| **未归因** | — | **1.13** | 口径齐全后基本为 0 |

`first_pred` 延迟（相对监测循环开始，非累加）：22.5 s / 7.5 s / 7.5 s。

**结论：此前约 260 s 的"未归因"就是 `final_select`（267.6 s），不是未覆盖的计算。**
指标口径本身没有重复计数：`gpu_wait` 只累计 sleep，`local_optimize`/`cc_batch` 各自包住对应调用。

## 二、服务端（PARENet 常驻进程，3 个请求共 2338 个 pair）

| 阶段 | 次数 | 秒 |
|---|---|---|
| server_postprocess | 2338 | **485.3** |
| server_forward | 2338 | 375.0 |
| server_write_pred | 2338 | 92.3 |
| server_masks | 3 | 4.8 |
| server_mask_preprocess | 3 | 1.4 |
| server_to_gpu | 2338 | 1.2 |
| server_preprocess | 3 | 0.3 |
| server_request_total（每请求合计） | 3 | 738.5 |
| server_queue_wait（请求间隔的空闲） | 3 | 289.2 |

**结论：服务端不是"GPU 计算占主导"，而是 CPU 后处理（485 s）> 模型 forward（375 s）> 预测写盘（92 s）。**
且 `queue_wait + request_total ≈ 1027.7 s`，与客户端 `fit_request`（1062.3 s）吻合——父子计时区间重叠，
本报告只分列展示，**不把两边的秒数相加**。

## 三、进程池生命周期

| 事件 | 次数 | 秒 | 分布 |
|---|---|---|---|
| pool_start（仅创建） | 79 | 4.34 | local_optimize_copies 59、batch_cc 17、tm_prefill 2、prescreen_cc 1 |
| pool_close（仅关闭+回收） | — | — | **口径修正后需重跑才有效值** |

**口径说明（自查发现）**：首版 `timed_pool` 把 `pool_close` 记成了"创建→join 结束"，
等于池的存活时长（与上层阶段重叠、读数 367.75 s），属于我的指标定义错误；
已改为只计 `close()+join()` 的耗时，下一次运行才有效。**不要用旧读数 367.75 s 推断关闭成本。**

**结论：池创建成本仅 4.34 s（0.4% 墙钟，≈55 ms/池）。** P3（复用池）的直接收益上限因此很小；
它的价值主要在"减少 79 次进程启停的抖动 + 让 P4 的每 worker 缓存在多批任务间保持有效"。

## 四、下一步优先级（依本轮测量修订）

| 新测量发现 | 结论与下一步 |
|---|---|
| 服务端 CPU 后处理（485 s）> forward（375 s），另有写盘 92 s；每请求 779 个 pair | **最高杠杆**：减少/复用 pair 级 CPU 工作（后处理与写盘）与预处理重叠，而不是先上 GPU batching |
| 客户端 gpu_wait 547 s（52%）+ 服务端 queue_wait 289 s | 等待与空闲占比大：改善候选发布/消费节奏（发布即评估、减少无效 pair） |
| final_select 267 s（其中含 CC 评分），cc_batch 仅 38.9 s | **P4 的适用范围要重估**：必须先统计**全部** `calculate_cc_mask` 调用（batch / final_select / 复核 / 预筛），不能只看 cc_batch |
| local_optimize 201 s，其中建池仅 4.34 s | P3 仍可做（低风险、收口临时建池），但收益有限；之后应评估**重复计算与数组复用**，而非继续加进程 |
| tm_prefill 0.65 s | P5 在本 case 无收益，继续暂缓 |

## 五、两处措辞收紧（按要求）

- 线程效果：改为"**本次配对运行**观察到 1680.8 s → 1023.7 s（约 1.64 倍），三个 CIF 与基线一致"；
  不表述为普遍稳定收益；后续性能实验应交替运行基线与新版并记录设备负载。
- P4 上限："约 4%"只适用于已计入 `cc_batch` 的那部分；`final_select` 等处的评分尚未统计，
  若计入则 P4 影响范围更大。

## 六、原始输出

```text
=== P2 细化测量 rc=0 elapsed_seconds=1144 ===
--- 生效线程 ---
2026-09-13 13:07:38,765 INFO [protassem.pipeline] Effective threads: {'env': {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1'}, 'blas': [{'internal_api': 'openblas', 'num_threads': 1}, {'internal_api': 'openblas', 'num_threads': 1}, {'internal_api': 'openmp', 'num_threads': 1}], 'torch_intra': 1}

== 客户端阶段（全部） ==
  pipeline_total           count=1    seconds=  1136.15
  assembly_total           count=1    seconds=  1129.97
  assembly_rounds          count=1    seconds=  1108.46
  fit_request              count=3    seconds=  1062.26
  gpu_wait                 count=3    seconds=   547.50
  pool_close               count=79   seconds=   367.75
  final_select             count=3    seconds=   267.59
  local_optimize           count=24   seconds=   200.97
  cc_batch                 count=17   seconds=    38.87
  first_pred               count=3    seconds=    37.58
  domain_split             count=1    seconds=     5.50
  cc_verify                count=3    seconds=     5.25
  sampling                 count=1    seconds=     4.40
  pool_start               count=79   seconds=     4.34
  refine_step4             count=1    seconds=     3.52
  domain_assembly          count=1    seconds=     2.73
  prescreen_chains         count=1    seconds=     2.10
  mask                     count=3    seconds=     2.03
  build_complex            count=1    seconds=     1.77
  voxelization             count=1    seconds=     1.58
  candidate_scan           count=3    seconds=     0.81
  tm_prefill               count=1    seconds=     0.65
  standardize              count=1    seconds=     0.15
  save_result              count=3    seconds=     0.12
  request_submit           count=3    seconds=     0.04
  analyze_sources          count=2    seconds=     0.02
  reorder_chains           count=1    seconds=     0.00
  refine_step5             count=1    seconds=     0.00

== 客户端：fit_request 的细分 ==
  gpu_wait                 count=3    seconds=   547.50
  final_select             count=3    seconds=   267.59
  local_optimize           count=24   seconds=   200.97
  cc_batch                 count=17   seconds=    38.87
  cc_verify                count=3    seconds=     5.25
  candidate_scan           count=3    seconds=     0.81
  save_result              count=3    seconds=     0.12
  analyze_sources          count=2    seconds=     0.02

first_pred 延迟（相对监测循环开始，非累加）: 22.5 s, 7.5 s, 7.5 s

== 服务端阶段（3 个 server_timing.jsonl） ==
  server_request_total     count=3    seconds=   738.52
  server_postprocess       count=2338 seconds=   485.26
  server_forward           count=2338 seconds=   374.99
  server_queue_wait        count=3    seconds=   289.16
  server_write_pred        count=2338 seconds=    92.25
  server_masks             count=3    seconds=     4.76
  server_mask_preprocess   count=3    seconds=     1.36
  server_to_gpu            count=2338 seconds=     1.15
  server_preprocess        count=3    seconds=     0.29

== 进程池生命周期（包含在上层阶段内，不重复计入） ==
  pool_close               count=79   seconds=   367.75
  pool_start               count=79   seconds=     4.34

== 时间账 ==
  fit_request 合计          1062.26 s
  已识别子阶段合计          1061.13 s
  未归因                       1.13 s
  pipeline_total（墙钟）    1136.15 s
  池创建次数              79

--- 产物等价性（对冻结基线）---
2497fefc2e866ad65db628fb75f1ec0a  assembled_complex.cif
2497fefc2e866ad65db628fb75f1ec0a  assembled_complex_all.cif
dbc937efc071cd9d1d2611f3ad84743f  refined_complex.cif
2497fefc2e866ad65db628fb75f1ec0a  assembled_complex.cif
2497fefc2e866ad65db628fb75f1ec0a  assembled_complex_all.cif
dbc937efc071cd9d1d2611f3ad84743f  refined_complex.cif
2f184d9d4b9f91074a7e58b430449a38  assembly_summary.txt

--- 残留进程检查 ---
(无残留)
done```
