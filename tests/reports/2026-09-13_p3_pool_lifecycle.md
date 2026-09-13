# P3 池生命周期验证（老卡收口第 2 步）

日期：2026-09-13　代码：`feat/old-card-closeout`（本步提交见 `git log`，父提交 `af02707`）
用途：补 P3 报告里缺的"池生命周期"证据，并**核查 fork 默认的适用范围**（不因原方案写过 spawn 就机械切换）。

## 一、方法与命令

| 手段 | 命令 | 覆盖 |
|---|---|---|
| 生命周期探针（新） | `python tools/pool_probe.py --workers 10 --repeat 2 [--start-method spawn] --json …` | 构造 / 全部 worker 就绪 / 就绪后首任务 / 暖任务 / 回收 / 异常回收 |
| 真实 `main.py` 父进程留证 | `python main.py <mrc> <dir> <res> <contour> <out> --log --num-processes 10 --runtime-config rt.json` | 建池瞬间父进程线程数与 CUDA 状态 |
| 单测 | `python -m unittest tests.test_runtime_execution -v` | 顺序归并 / 只建一次池 / 串行回退 / 幂等 / 异常释放 / **异常回收** / 父进程状态入账 |

探针细节：`N` 个任务（`chunksize=1`，一 worker 一个）各写一个 pid 标记后忙等，直到出现 `N` 个**不同 pid**
→ "全部 worker 就绪"；随后测一次单任务（就绪后首任务）与 5 次取中位（暖任务）；回收用 `close()+join()`；
异常路径让 worker 抛异常，再用 `close()+join()` 并核对 `/proc` 与 `multiprocessing.active_children()`。

## 二、探针结果（10 worker，各 2 轮）

| 启动方式 | 构造 (s) | 全部就绪批 (s) | 就绪后首任务 (s) | 暖任务 (s) | 回收 (s) | 不同 worker pid | 异常回收后存活 |
|---|---|---|---|---|---|---|---|
| 默认（Linux fork） | 0.030 / 0.027 | 0.009 / 0.008 | 0.001 / 0.000 | 0.001 / 0.001 | 0.007 / 0.008 | 10 / 10 | 0 |
| `spawn` | 0.048 / 0.046 | 0.082 / 0.086 | 0.001 / 0.001 | 0.002 / 0.002 | 0.018 / 0.018 | 10 / 10 | 0 |

- **构造**：fork ≈27–30 ms，spawn ≈46–48 ms；**但真正的差距在"全部 worker 就绪"**：fork ≈8–9 ms vs spawn ≈82–86 ms（约 10×）。
  这与 P3 旧记录（fork≈25 ms / spawn≈52 ms，只测了构造）方向一致，且把口径补全到"就绪"。
- 就绪后单任务与暖任务在两种方式下都 ≈0.001–0.002 s（进程池稳态开销可忽略）。
- 回收 ≈7–8 ms（fork）/ ≈18 ms（spawn）；异常路径下 `close()+join()` 后**存活子进程 0**（两种方式）。

## 三、fork 适用范围（真实运行留证）

- `runtime/pool.py::open_pool` 现在把**建池瞬间的父进程状态**写进 `pool_start` 记录（`threads`、`cuda_initialized`）；
  `tests/test_runtime_execution.py::test_pool_start_records_parent_state` 保证它不会被悄悄删掉。
- 真实 `main.py` 运行（10 worker，`--runtime-config {"blas_threads":1}`，合成小 case）实测：
  `{"pool":"shared","workers":10,"start_method":"default","threads":1,"cuda_initialized":false,"elapsed_s":0.068}`。
- 结论【已核实】：建池发生在**父进程**（`run_pipeline`/`run_fitting`），此时父进程**线程数 = 1、CUDA 未初始化**，
  且池内 worker 只做 CPU 计算（CC 评分、局部优化、TM 预填）——**fork 的已知风险（复制已初始化 CUDA/多线程状态）在此不成立**，
  因此**维持系统默认（fork）**；`spawn` 保留为可配置项（`pool_start_method`）与对照基线，不作为默认。
- 若将来把池的创建点挪到"已初始化 CUDA 的进程"里，或让 worker 触碰 CUDA，则必须重新评估启动方式——
  `pool_start` 的这两个字段会让这种漂移在真实运行里立刻可见，而不是靠约定。

## 四、限制与未验证

1. 本次只验证**已接入的主拟合路径**的池；Step4/Step5（`assembly/homo_chain_refine.py` 5 处 `pool.map`）、
   `assembly/refine/chain_enumerator.py`、`sampling/extract_points/VoxEM.py` 仍各自建池，**未纳入**（本轮只文档化）。
2. 探针测的是"空载"生命周期；真实运行里池与 GPU 服务、磁盘 I/O 并行，绝对耗时会有差异——
   本步只回答"建池/就绪/回收的量级与资源是否干净"。
3. 合成小 case 的真实运行在进入 PARENet 请求后不收敛（合成点云无意义），被 `timeout 300` 终止；
   终止后 `main.py` 与 PARENet 服务端均无残留进程（`ps` 计数 0）。它的用途仅是取"建池瞬间的父进程状态"。

## 五、复现路径（`/xiangyux/claude_c_work/demo_reg_cases/`，不进仓库）

| 产物 | 说明 |
|---|---|
| `pool_probe_fork.json` / `pool_probe_spawn.json` | 探针原始结果（各 2 轮） |
| `out_tiny/metrics/performance.jsonl` | 真实 `main.py` 父进程的 `pool_start` 记录（含线程数/CUDA 状态） |
| `tiny_case/`、`rt_tiny.json` | 合成小 case 与运行配置（由 `tests/fixtures.py` 生成） |
