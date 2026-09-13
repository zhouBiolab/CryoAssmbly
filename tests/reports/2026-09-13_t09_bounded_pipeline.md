# T09：有界 CPU 尾部流水线（实测报告）

日期：2026-09-13　代码：`feat/runtime-metrics`，工作区基线 `d91b2a2`（T08）+ 本卡改动
（T09 代码 diff sha1 `e0f3fbec7630f059`）
输入：`tests/cases/registration_manifest.json` 与 `test/1`（端到端）
GPU：A800 **MIG 7g.80gb 切片**。

**本卡结论**：把"模型之后的 CPU 尾部（后处理 + 写盘）"从深度 1 的单 worker 上执行，
与下一次配准的 GPU 工作重叠；单测覆盖顺序/深度/开关等价/异常传播/资源释放（全量 **184 项通过**）；
真实运行 A/B（10 worker）墙钟 **1092 → 994 s（−9.0%）** 且三个 CIF md5 与决策完全一致；
微基准（单客户端）反而 **+9.5%** —— 收益依赖"多 worker 把 GPU 压满"，故**默认仍为关闭**（见第四、五、七节）。

## 一、为什么是"尾部"

真实运行（本卡第五节 off 轮，2338 次配准）的服务端耗时构成：

| 环节 | 累计 | 每次配准 | 性质 |
|---|---|---|---|
| `server_postprocess`（变换 + overlap） | 127.8 s | 55 ms | CPU（numpy） |
| `server_write_pred`（PDB 写出） | 93.3 s | 40 ms | CPU + 磁盘 |
| 模型（forward / encode+register） | 331.6 s | 142 ms | GPU（含 host 启动） |
| 几何 + 缓存 + 其他/客户端等待 | ~530 s | ~227 ms | CPU + 10 worker 排队 |

CPU 尾部合计 ≈ **221 s / 1092 s ≈ 20%** 墙钟，而且它**串行发生在每次配准的模型之后**——
这就是本卡要重叠的部分（任务卡："分离准备/传输/编码/输出；按预期 CPU 准备与 GPU 执行重叠"）。

## 二、实现

| 内容 | 位置 |
|---|---|
| `TailPipeline`：单 worker 线程、**深度 1**（`queue.Queue(maxsize=1)`，worker 忙时 `submit()` 等待 → 内存最多多一个任务）、FIFO 顺序 = 提交顺序 = 消费顺序、`wait()` 等待全部完成、`close()` 幂等（先 `join` 队列再投哨兵 + 线程 `join(30s)`，超时报错）、任务异常由 worker 记录并在 `wait()`/`close()` **抛出**（不吞掉）、`enabled=False` 时 `submit()` 就地执行（**同一份调用代码**）、上下文管理器保证异常退出也关闭 | `runtime/tail_pipeline.py`（新） |
| `process_single_pair`：把"后处理 + 写盘 + 结果回填"抽成 `finish_pair()` 闭包；只保留必要的 CPU 结果（`pred_R`/`pred_t` numpy、两侧点数），GPU 张量随 `output_dict` 释放；`tail_pipeline=None` 时就地执行 | `fitting/demo_mask.py` |
| `run_inference`：建立流水线（每次请求一个，`finally` 关闭）；在**每个消费点前** `wait()`：掩码内选优、排序、排名改名、运行摘要；记录 `server_tail_wait`（主线程等待时长 = 重叠不足的部分）与 `server_tail_stats`（submitted/completed/waited） | 同上 |
| 配置：`RuntimeConfig.tail_pipeline`（默认 **False** = 原路径）+ `--tail-pipeline` / `--no-tail-pipeline`（服务端/CLI/benchmark）+ `configure_tail_pipeline()` + `run_pipeline` 透传 | `runtime/config.py`、`fitting/parenet_client.py`、`pipeline.py`、`tools/benchmark_registration.py` |
| 测试：`tests/test_tail_pipeline.py`（8 项） | `tests/` |

**没有并行任何随机过程**（任务卡偏差处理）：尾部只有确定性的 numpy 后处理与写盘；
几何/编码/配准仍在主线程按原顺序执行，随机消费顺序不变。

## 三、验收对照（任务卡 T09）

| 验收项 | 证据 | 结论 |
|---|---|---|
| 开关结果/顺序一致 | 单测：`enabled=False` 与 `True` 的结果序列一致；微基准四轮 **72/72 预测聚合 md5 相同**；端到端 A/B 三个 CIF md5 与决策逐条相同（第四、五节） | 通过 |
| 顺序不被完成顺序取代 | 单测 `test_enabled_preserves_submission_order`（20 个任务按提交顺序完成）；`run_inference` 在每个消费点 `wait()`，消费顺序仍是循环顺序 | 通过 |
| 取消/错误/正常退出均释放资源 | 单测：`close()` 幂等 + 线程已退出（`threading.enumerate()` 检查）、任务异常在 `wait()`/`close()` 抛出、上下文管理器异常退出也不残留线程；真实运行结束无残留进程 | 通过 |
| 内存最多增加一个预取任务 | 单测 `test_depth_is_bounded_to_one_pending_task`（并发峰值 = 1）；`queue(maxsize=1)` 结构保证 | 通过 |
| 早停取消 | `run_inference` 的 `_stopped()` 循环退出后走 `finally: tail_pipeline.close()`（先等已提交任务完成再关线程），已提交的写盘不会被丢弃 | 通过（单测覆盖关闭路径） |
| 传输等待明显后才开 pinned memory | 本卡不做 pinned memory / 异步拷贝（任务卡要求"传输等待明显后才开"）；G2H 拷贝只有 4×4 位姿 | 未启用 |

## 四、微基准 A/B（`--tail-pipeline` off/on，四轮交叉）

命令：`tools/benchmark_registration.py run --manifest tests/cases/registration_manifest.json
--out-dir … --repeat 2 --geometry-cache-mb 0 [--no-tail-pipeline]`，顺序 off→on→off→on。

| 轮次 | 目标 0 | 目标 1 | 目标 2 | 合计 |
|---|---|---|---|---|
| off1 | 10.84 / 6.88 | 4.85 / 4.48 | 4.53 / 4.30 | **35.88 s** |
| off2 | 9.17 / 6.86 | 5.07 / 5.52 | 5.50 / 4.86 | **36.98 s** |
| on1 | 12.19 / 7.19 | 4.63 / 5.25 | 6.00 / 4.83 | **40.09 s** |
| on2 | 12.07 / 6.24 | 4.81 / 5.48 | 5.89 / 5.21 | **39.70 s** |

（每格 = 该目标两次重复；每轮 3 目标 × 12 预测 = 36 次配准。）

- 结果：四轮的 72 个 `pred_*.pdb` **聚合 md5 完全相同**（`f6c942de99f40bccdc9b9430df09e511`），
  best overlap 全部 0.3268 / 0.0527 / 0.058 → **开关等价**成立。
- 性能：off 均值 36.43 s、on 均值 39.90 s → **on 反而慢 9.5%**（组内重复差仅 0.4–1.1 s，
  差异集中在目标 0 的首轮冷启动：9.17/10.84 → 12.07/12.19）。

**这条负面结果必须保留**：微基准是"单客户端、请求串行、GPU 不被压满"的场景。此时
（a）尾部只与**同一请求**内下一次配准的 GPU 工作重叠，收益有限；
（b）流水线线程与主线程在同一进程里争 CPU/GIL，把主线程的 host 侧准备拖慢
（真实运行实测 `server_forward` +85 s、后处理 +38 s、写盘 +11 s 即为同一机制的放大）。
所以"尾巴重叠"的收益**只在多 worker 把 GPU 压成瓶颈时才出现**（第五节：10 worker → −9.0%）。
本卡不据此宣称加速，也不把它设为默认（第七节第 4 条）。

## 五、真实运行 A/B（`test/1`，默认 10 worker + 1 BLAS 线程）

| 运行 | 配置 | 墙钟 | `pipeline_total` | 三个 CIF md5 | 关键决策（A/B/域） |
|---|---|---|---|---|---|
| off | `{"blas_threads": 1, "tail_pipeline": false}` | **1092 s** | 1082.95 s | `76638d0f…` / `bd281f40…` / `0356ae49…` | A rejected 0.4192、B accepted 0.4235、Chain A 2 domains 0.4430 |
| on | `{"blas_threads": 1, "tail_pipeline": true}` | **994 s** | 985.19 s | 同上（逐位相同） | 同上（逐条相同） |

**差 −98 s（−9.0%）**，产物与决策完全一致 —— 本卡的验收判据（等价）通过。

时间账（`server_timing.jsonl` 全量聚合，2338 次配准）：

| 阶段 | off | on | 说明 |
|---|---|---|---|
| `server_forward` | 331.60 s | 416.76 s | +85 s：尾部线程与主线程**争 CPU**，把 forward 的 host 侧拖长 |
| `server_postprocess` | 127.76 s | 166.21 s | 同上（同一块 CPU 上多了一个常驻线程） |
| `server_write_pred` | 93.25 s | 104.38 s | 同上 |
| `server_tail_wait` | — | **17.13 s**（167 次，均值 103 ms，最大 276 ms） | 主线程在消费点真正等待尾部的时长 |
| `server_tail_stats` | — | submitted 2338 / completed 2338（3 个请求） | 无丢任务、无残留 |

尾部 CPU 工作 on 轮合计 270.6 s，主线程只等了 17.1 s → **约 94% 的尾部被重叠**。
即使 forward+后处理+写盘因争抢合计变慢 ≈ +135 s，净墙钟仍 −98 s：收益来自"GPU 不再等
CPU 尾部排队"，代价是"CPU 上多一个线程分时"。单轮 A/B 的偏差风险见第七节第 3 条。

## 六、偏差记录

首次微基准 `--tail-pipeline` 运行**全部失败**：`_record_timing(output_dir, "server_tail_wait",
waited, stage=stage)` 的字段名 `stage` 与 `_record_timing(output_dir, stage, elapsed_s, **fields)`
的形参重名 → `TypeError: _record_timing() got multiple values for argument 'stage'`。
已改字段名为 `where=` 并重跑（与 T07 的 `inference_mode=None` 同属"埋点/负载契约"类缺陷，
都由"开启新开关后的实跑"暴露，单测未覆盖埋点字段名）。

## 七、限制与后续卡指向

1. 尾部重叠的**上界**是"每次配准的 CPU 尾部耗时"（实测 ≈100 ms/次）；实际收益取决于
   主线程等待比例（`server_tail_wait` 会给出主线程真正等待的时长）。
2. 只重叠了 CPU 尾部；几何构建（含 GPU kNN）与编码仍在主线程串行 —— 若要进一步重叠，
   需要把"下一次的几何/编码"也放进 worker（任务卡 T09 的"准备"部分），但那会与 GPU 竞争
   且牵涉随机过程隔离，本卡按"偏差处理：随机过程未隔离等价前不并行它"未做。
3. 客户端监测循环按 `pred_*.pdb` 出现的批次评估候选（批次 10 / 轮询 2.5 s）；尾部流水线只
   改变文件出现的**微小时机**（≤ 一个尾部 ≈100 ms），候选集合不变，但离散批次边界理论上
   可能移动（与 O6"worker 数敏感性"同类）→ 端到端 md5 对比是本卡的判据。
4. **默认保持关闭**（`RuntimeConfig.tail_pipeline=False`，原路径）：端到端 −9.0% 只测了一对
   （机器上的历史同配置波动带 1054–1097 s，on 轮 994 s 落在带外，但仍是单样本），
   而同一份代码在单客户端微基准里慢 9.5% —— 收益依赖"多 worker 把 GPU 压满"。
   提升默认属 T10（完整回归与交付）的判据，届时用固定 manifest + 复测的端到端 A/B 决定；
   交付物是已验证、可回退的 `tail_pipeline: true` 配置。
5. T10 复测要一并看的：`server_tail_wait / (postprocess + write_pred)` 的重叠比例（本轮 94%）、
   `server_forward` 的争抢膨胀（本轮 +25%）以及 O6（worker 数敏感性）对本结论的影响。

## 八、复现路径（`/xiangyux/claude_c_work/demo_reg_cases/`，不进仓库）

| 产物 | 说明 |
|---|---|
| `t09_ab_result.txt` / `run_t09_ab.sh` | 微基准 + 端到端 A/B 的原始输出与脚本 |
| `t09_bench_result.txt` / `run_t09_bench.sh` / `t09b_off1|on1|off2|on2` | 微基准四轮 A/B 的原始输出与产物（每轮 3 目标 × 12 预测） |
| `out_t09_off` / `out_t09_on` / `rt_t09_off.json` / `rt_t09_on.json` | 端到端 A/B 产物与运行配置 |
| T10 的复测产物 | 见 `tests/reports/2026-09-13_t10_full_regression.md` 第六节 |

## 九、补遗（T10 复测修正，2026-09-13 21:20）

> 本节由 T10（完整回归与交付）追加，**修正第五节与第七节第 4 条的判定依据**，原文保留不改。

1. 第五节的"on 轮三个 CIF md5 与冻结基线完全一致"是**单次观测**，**不可复现**：
   T10 又跑了 2 次同配置 on（`out_t10_on`、`out_t10_on2`），两次都给出**另一个产物**
   （`assembled_complex.cif` `51009d69…`、`refined_complex.cif` `1ca5f6f9…`，CC 0.4189/0.4230/0.4431），
   彼此逐位一致；而 off 路径在此期间的第 2 次运行仍复现冻结基线（`76638d0f…`）。
2. 根因不在尾部流水线的计算：链 A 的 24 个 `registration/pred_*.pdb` 在 off/on 两次运行里
   **逐位相同**（模型层等价成立）。差异出现在**客户端候选消费顺序**——候选按"文件出现时机"
   成批评估（`fitting/pipeline.py`：`sorted(glob('pred_*.pdb'))` 取差集、`>= batch_size(10)` 评估一批、
   每轮 `sleep(2.5)`）：`pred_chain_A_ed_points_0.301508.pdb` 在 off 是候选 `#8`、在 on 是 `#11`。
   尾部流水线把写盘挪到后台线程，正好移动了这个批次边界。
3. 速度结论不变且更稳：on 三次墙钟 994 / 1006 / 993 s，off 两次 1092 / 1083 s（−8.4% … −9.1%）。
4. 因此 `tail_pipeline` 维持**默认关闭**；T10 报告的第三节给出完整证据链。
5. 第四节"微基准 on 慢 9.5%"同样**不成立为结论**：T10 交叉复测里 on 反而快 3.2%，而组内两轮
   之差可达 2.5 s → 微基准只能判等价与显存，判收益要用真实运行。
