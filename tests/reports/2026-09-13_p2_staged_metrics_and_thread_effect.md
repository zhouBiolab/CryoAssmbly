# P2 分阶段计时与线程影响（实测报告）

日期：2026-09-13　代码：`feat/runtime-metrics`（P2 提交见分支，父提交 `dfad8ef`）
case：`test/1`（两条单链 PDB + `EMD-8436.mrc`，res 5.6，contour 0.04），命令两侧相同，
仅线程配置不同（A：`--runtime-config {"blas_threads":128}`；B：默认 `blas_threads=1`）。

## 分阶段拆解（同一 instrumented 代码）

| 阶段 | A 128 线程 (s) | B 1 线程 (s) | 说明 |
|---|---|---|---|
| pipeline_total（墙钟） | 1680.8 | 1023.7 | 总耗时 |
| assembly_total | 1674.3 | 1017.1 | Step3+4+5 |
| assembly_rounds | 1650.6 | 991.7 | 全局轮次主循环 |
| fit_request | 1603.8 | 953.3 | 3 次拟合请求整体（含下列子阶段） |
| **gpu_wait** | **882.5** | **470.0** | 等 PARENet 预测（轮询 sleep 累计，不与其它阶段重叠） |
| **local_optimize** | **357.5** | **185.8** | 23 次局部优化 |
| cc_batch | 81.0 | 36.7 | 17 批 CC_mask |
| domain_split | 6.2 | 5.7 | DomainParser 切域 |
| sampling | 4.6 | 4.7 | 采样（实验图 + 各模板） |
| refine_step4 | 3.8 | 4.7 | Step4 精修 |
| prescreen / domain_assembly / mask / build_complex / voxelization / tm_prefill / standardize / reorder / refine_step5 | ≤2.8 各 | ≤2.8 各 | 全部可忽略 |

说明：并行阶段的 seconds 是**累计值**，不能相加当作墙钟；`gpu_wait` 只累计轮询等待时间，
与 CC/局部优化不重叠（那些在 `fit_request` 内另行计时）。

## 结论 1：P1 线程默认是实测有效的

背靠背同条件对比：**1023.7 s vs 1680.8 s（约 1.6×）**，且 gpu_wait 从 882.5 s 降到 470.0 s
（每个 worker 不再各占 128 个 BLAS 线程、把整机 CPU 抢满）。三个最终 CIF 的 md5 与冻结基线
完全一致，说明这是纯配置收益，不是语义变化。

## 结论 2：耗时分布决定 P3–P5 的优先级（依测量，而非预设）

- **最大头是 GPU 等待（46–52%）**：`fit_request` 953–1604 s 中约一半在等预测；
  扣除 gpu_wait / local_optimize / cc_batch 后，`fit_request` 内仍有约 260 s 未归因
  （PARENet 服务启动、候选登记、`_verify_cc`、`_select_final_result` 等），下一步应细化这部分的埋点。
- `local_optimize` 占 18–21%，其中包含每次调用的进程池启动开销（23 次调用）→ **P3 是收益最明确的一项**。
- `cc_batch` 只占 3.6–4.8% → **P4（CC 缓存）上限约 4%**。
- `tm_prefill` 仅 0.7 s → **P5（SQLite TM 缓存）在本 case 上几乎无收益**，应推迟或仅在重复运行场景做。

## 注意事项（不夸大）

- A 组（1680.8 s）与 S7 阶段实测的 956 s 名义线程配置相同却相差更大，说明**外部负载/GPU 争用**
  对墙钟影响显著；因此只把背靠背的 A vs B 作为线程影响的证据。
- 本次测量只覆盖 `test/1`；更大 case 未测。

## 原始输出

```text
=== P2 测量开始 2026-09-13 12:06:26 ===
### run a128: rc=0 elapsed_seconds=1690
2026-09-13 12:06:31,987 INFO [protassem.pipeline] Runtime config: blas_threads=128 seed=7351
2026-09-13 12:06:33,297 INFO [protassem.pipeline] Effective threads: {'env': {'OMP_NUM_THREADS': '128', 'MKL_NUM_THREADS': '128', 'OPENBLAS_NUM_THREADS': '128', 'NUMEXPR_NUM_THREADS': '128', 'VECLIB_MAXIMUM_THREADS': '128'}, 'blas': [{'internal_api': 'openblas', 'num_threads': 128}, {'internal_api': 'openblas', 'num_threads': 64}, {'internal_api': 'openmp', 'num_threads': 112}], 'torch_intra': 112}
2026-09-13 12:34:34,122 INFO [protassem.pipeline] Performance summary: /xiangyux/claude_c_work/demo_reg_cases/out_p2_a128/metrics/performance_summary.json
--- 分阶段（metrics/performance_summary.json）---
  total_wall_s = 1680.8
  pipeline_total     count=1    seconds=  1680.8
  assembly_total     count=1    seconds=  1674.3
  assembly_rounds    count=1    seconds=  1650.6
  fit_request        count=3    seconds=  1603.8
  gpu_wait           count=3    seconds=   882.5
  local_optimize     count=23   seconds=   357.5
  cc_batch           count=17   seconds=    81.0
  domain_split       count=1    seconds=     6.2
  sampling           count=1    seconds=     4.6
  refine_step4       count=1    seconds=     3.8
  prescreen_chains   count=1    seconds=     2.8
  domain_assembly    count=1    seconds=     2.7
  mask               count=3    seconds=     2.0
  build_complex      count=1    seconds=     1.8
  voxelization       count=1    seconds=     1.7
  tm_prefill         count=1    seconds=     0.8
  standardize        count=1    seconds=     0.2
  reorder_chains     count=1    seconds=     0.0
  refine_step5       count=1    seconds=     0.0
--- 产物等价性（对冻结基线 md5）---
2497fefc2e866ad65db628fb75f1ec0a  assembled_complex.cif
2497fefc2e866ad65db628fb75f1ec0a  assembled_complex_all.cif
dbc937efc071cd9d1d2611f3ad84743f  refined_complex.cif
a3c5e1b29cf7c0b066168e54775212bf  assembly_summary.txt

### run b1: rc=0 elapsed_seconds=1033
2026-09-13 12:34:41,873 INFO [protassem.pipeline] Runtime config: blas_threads=1 seed=7351
2026-09-13 12:34:43,369 INFO [protassem.pipeline] Effective threads: {'env': {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1'}, 'blas': [{'internal_api': 'openblas', 'num_threads': 1}, {'internal_api': 'openblas', 'num_threads': 1}, {'internal_api': 'openmp', 'num_threads': 1}], 'torch_intra': 1}
2026-09-13 12:51:47,122 INFO [protassem.pipeline] Performance summary: /xiangyux/claude_c_work/demo_reg_cases/out_p2_b1/metrics/performance_summary.json
--- 分阶段（metrics/performance_summary.json）---
  total_wall_s = 1023.7
  pipeline_total     count=1    seconds=  1023.7
  assembly_total     count=1    seconds=  1017.1
  assembly_rounds    count=1    seconds=   991.7
  fit_request        count=3    seconds=   953.3
  gpu_wait           count=3    seconds=   470.0
  local_optimize     count=23   seconds=   185.8
  cc_batch           count=17   seconds=    36.7
  domain_split       count=1    seconds=     5.7
  refine_step4       count=1    seconds=     4.7
  sampling           count=1    seconds=     4.7
  prescreen_chains   count=1    seconds=     2.8
  domain_assembly    count=1    seconds=     2.8
  mask               count=3    seconds=     1.9
  build_complex      count=1    seconds=     1.9
  voxelization       count=1    seconds=     1.7
  tm_prefill         count=1    seconds=     0.7
  standardize        count=1    seconds=     0.2
  reorder_chains     count=1    seconds=     0.0
  refine_step5       count=1    seconds=     0.0
--- 产物等价性（对冻结基线 md5）---
2497fefc2e866ad65db628fb75f1ec0a  assembled_complex.cif
2497fefc2e866ad65db628fb75f1ec0a  assembled_complex_all.cif
dbc937efc071cd9d1d2611f3ad84743f  refined_complex.cif
12bbf4534fe88a3ae1770210d2afa473  assembly_summary.txt

=== 冻结基线 md5（参照）===
2497fefc2e866ad65db628fb75f1ec0a  assembled_complex.cif
2497fefc2e866ad65db628fb75f1ec0a  assembled_complex_all.cif
dbc937efc071cd9d1d2611f3ad84743f  refined_complex.cif
2f184d9d4b9f91074a7e58b430449a38  assembly_summary.txt
=== P2 测量结束 2026-09-13 12:51:49 ===
done```

（本文里的百分号均为字面量。）
