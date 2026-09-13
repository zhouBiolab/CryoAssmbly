# T03：推理内存与中间回传（实测报告）

日期：2026-09-13　代码：`feat/runtime-metrics`，工作区基线 `00a97e8` + 本卡改动
（T03 diff sha1 `67fedf877098ca81`）
输入：`tests/cases/registration_manifest.json`（sha256 `39c163c4adf48345…`，1 源 7335 点 +
3 个掩码 8006/3565/1672 点，每目标 6 配置 × 2 采样 = 12 次配准）
GPU：A800 **MIG 7g.80gb 切片**；显存数字是切片内进程级。

## 一、本卡做了什么（任务卡 T03 逐项）

| 要求 | 实现 | 位置 |
|---|---|---|
| 保留 `eval` | 未改动：`_get_model()` 仍 `_MODEL.eval()` | `fitting/demo_mask.py` |
| 以 `no_grad` 覆盖推理 | `@torch.no_grad()` 装饰 `process_single_pair`（**唯一**调用模型的入口），覆盖 collate→上卡→前向→后处理全路径 | `fitting/demo_mask.py:157` |
| 仅回传最终变换与必要字段 | 新增 `INFERENCE_OUTPUT_FIELDS = ("estimated_transform", "ref_points", "src_points")` 与 `select_output_fields()`（`None` = 旧的全量行为，训练/诊断路径不受影响）；白名单字段缺失直接 `KeyError`，不静默兜底 | `fitting/parenet/model.py:20-38, 508`、`fitting/demo_mask.py:254` |
| 删除完整输入/输出字典的递归 `release_cuda` | 删除两行递归释放与 `release_cuda` 导入（实测 0.52–0.75 s / 36 对，见第四节） | `fitting/demo_mask.py` |
| 结束局部引用 | `data_dict`/`output_dict` 是 `process_single_pair` 的局部变量，返回即结束引用；`result` 只留标量与路径（审计：无日志、列表持有 GPU 张量） | 同上 |
| 不逐候选 `empty_cache` | 删除正常路径与异常路径两处 `torch.cuda.empty_cache()`（实测 0.94–1.43 s / 36 对） | 同上 |
| 审计 GT correspondence 消费者 | GT 对应只被 `if self.training:` 的 `coarse_target` 消费 → 推理不再计算（实测 0.15–0.20 s / 36 对）；`get_node_correspondences` 无随机、无副作用、不写全局状态 | `fitting/parenet/model.py:206-227` |
| 审计 attention scores 消费者 | `self.transformer(...)` 返回的 `scores_list` 在 `forward` 内**从未被读取**，也从未进入 `output_dict`；它产生于 `pareconv` 的 `GeometricTransformer` 内部，属 T06（单侧编码拆分）范围 → 本轮记录不改 | `fitting/parenet/model.py:246` |
| 不改 dtype | 未改任何 dtype | — |

新增埋点与工具：

- `model_node_partition`（CUDA event，前向内节点分区阶段）；
- `server_mem_after`（每对结束时的 `allocated`/`reserved`，用于"连续运行是否持续增长"）；
- `tools/compare_registration_runs.py`（新）：两套微基准产物的 A/B（一致性 / 墙钟 / 阶段 / 显存），T10 可复用；
- `tools/probe_forward_identity.py`（新）：固定输入前向的**逐字段按位取证**（落盘 `.npz` + 字段哈希），用于 `atol=1e-6/rtol=1e-5` 与位姿等价性检查；
- `tools/summarize_timing.py` 增加服务端显存小节（真实运行也能看 allocated 轨迹）；
- `tests/test_inference_contract.py`（新，10 项）：白名单契约、唯一模型调用点、`no_grad` 装饰、`release_cuda`/`empty_cache` 不得回归。

## 二、A/B 方法与可控性

- **基线树**：`git worktree` 检出 `00a97e8`（T03 之前）到 `/xiangyux/claude_c_work/demo_reg_prev`，
  仅加**探针埋点**（不进仓库）：`probe_release_cuda`、`probe_empty_cache`、`model_gt_corr`、
  `model_node_partition`、`server_mem_after`。补丁脚本与 diff 存档：
  `demo_reg_cases/t03_probe_patch.py`、`demo_reg_cases/t03_base_probe.diff`。
- **候选树**：工作区 `demo_reg`（`00a97e8` + T03 改动）。
- 两棵树同一 manifest（sha256 相同）、同一 `pareconv`（`/xiangyux/PARENet-main`，非仓库副本）、
  同一权重文件；顺序执行，各自 `--repeat 2`（冷 = 第 1 次重放，暖 = 第 2 次，共 72 次配准）。
- **两轮交叉**：第 1 轮先 base 后 new（`t03_bench_*`），第 2 轮先 new 后 base（`t03_bench2_*`），
  用于排除顺序/时段偏差。

## 三、结果一致性（验收硬条件）

### 3.1 微基准两轮：72 / 72 预测逐字节一致

| 轮次 | 预测文件数 | `pred_*.pdb` 内容 sha256 一致 | 文件名集合（含 `_topNN` 排名） | 逐对 overlap 列表 |
|---|---|---|---|---|
| 第 1 轮 | 72 / 72 | **72 / 72** | 完全一致 | 完全一致 |
| 第 2 轮 | 72 / 72 | **72 / 72** | 完全一致 | 完全一致 |

文件名集合一致即候选排名与分数一致，文件内容一致即位姿一致 → **变换、评分、候选身份均未改变**。

### 3.2 前向逐字段取证：41 / 41 共有字段**按位**一致

固定 pair（target order 0 + 定标 `out_p3_a2/assembly/work/mask_1/filtered.txt`，sha256 `d31d7f2d…`）
在两棵树上各跑一次前向，把 `output_dict` 全部字段落盘（`demo_reg_cases/t03_fwd_base|new`）：

- `estimated_transform`、`hypotheses`(2200,4,4)、`corr_scores`(2200)、`ref/src_corr_points`、
  `matching_scores`(256,64,64)、`ref/src_feats_c|f|_re`、`ref/src_node_corr_*`、`ref/src_points*`、
  `input__transform`/`input__scale`/两侧质心 —— **全部 `tobytes()` 相等**（最大偏差 0）。
- 字段集合差异仅 2 项，且是**本卡刻意跳过**的训练专用输出：
  `out__gt_node_corr_indices`、`out__gt_node_corr_overlaps`（只存在于 base 侧）。

这说明 `no_grad` 与"仅训练分支算 GT 对应"没有改变任何数值路径——连候选假设张量都逐位相同。

## 四、耗时（每轮冷/暖各 36 对累计，秒）

### 4.1 墙钟（`t00_report.json`）

| 轮次 | base 合计 | new 合计 | Δ |
|---|---|---|---|
| 第 1 轮（base→new） | 58.64 | 46.52 | **−12.13（−20.7%）** |
| 第 2 轮（new→base） | 54.97 | 47.25 | **−7.72（−14.0%）** |

逐 target 明细（含模型加载的 target 0 run 1 绝对值不可跨运行比较）：

| 轮次 | target/repeat | base | new | Δ% |
|---|---|---|---|---|
| 1 | 0 / 1 | 19.93 | 17.17 | −13.9% |
| 1 | 0 / 2 | 9.44 | 8.15 | −13.7% |
| 1 | 1 / 1 | 8.47 | 5.33 | −37.1% |
| 1 | 2 / 1 | 7.14 | 4.98 | −30.2% |
| 2 | 0 / 1 | 18.42 | 18.34 | −0.4%（加载主导） |
| 2 | 0 / 2 | 8.14 | 7.58 | −6.8% |
| 2 | 1 / 1 | 7.76 | 5.58 | −28.1% |
| 2 | 2 / 1 | 6.71 | 5.16 | −23.1% |

### 4.2 阶段归因（冷，36 对）

| 阶段 | 第 1 轮 base→new | 第 2 轮 base→new |
|---|---|---|
| `server_forward`（主机 wall，含下列 CUDA 阶段） | −3.82 | −2.20 |
| `model_backbone`（不再建计算图） | −1.28 | −0.83 |
| `model_lgr` | −1.15 | −0.63 |
| `model_transformer` | −0.31 | −0.10 |
| `probe_empty_cache`（**已删除**的工作） | −1.43 | −0.94 |
| `probe_release_cuda`（**已删除**的工作） | −0.75 | −0.52 |
| `model_gt_corr`（**已删除**的工作） | −0.20 | −0.15 |
| `server_write_pred` | −0.64 | +2.20（I/O 噪声） |
| `server_collate` | +0.31 | +0.30（8 ms/对，见第七节） |
| `server_postprocess` | −0.17 | +0.10 |

**归因**：两轮方向一致，量级差 ±50%（与 T00 记录的运行间波动一致，不承诺固定加速倍数）。
可直接归因的两组是：显式删除的工作（`empty_cache` + `release_cuda` + GT 对应，
两轮分别 2.38 s / 1.61 s，36 对）、以及对 backbone/transformer/point_matching 关闭梯度后
模型内 CUDA 阶段的下降（两轮分别 2.74 s / 1.56 s）。

## 五、显存（微基准 + 真实运行）

### 5.1 微基准（MIG 切片内进程级，两轮数字相同）

| 指标 | base | new | Δ |
|---|---|---|---|
| 前向返回瞬间 `allocated` | 3746.9 MiB | 23.5 MiB | **−3723.4** |
| 进程 `max_allocated` 高水位 | 4849.1 MiB | 1177.1 MiB | **−3672.0（−75.7%）** |
| 进程 `max_reserved` 高水位 | 4952.0 MiB | 2558.0 MiB | −2394.0 |
| 每对结束 `reserved`（示例 target_2） | 32.0 MiB | 1476.0 MiB | +1444.0 |

- 连续 72 次配准的 `allocated` **不增长**：首 / 最小 / 最大 / 末 = 14.5 / 11.5 / 23.5 / 11.5 MiB
  （末/首 = 0.79；末 6 次恒为 11.5）。
- `reserved` 变大是删除 `empty_cache` 的**预期结果**：缓存分配器保留高水位块复用，
  不是泄漏（同一时刻真实占用 `allocated` ≈ 12 MiB）。
- 旧代码"前向返回瞬间 allocated 3.7 GiB"正是计算图 + 全量 `output_dict` 驻留的直接证据。

### 5.2 真实运行（`test/1`，2338 次配准，同一常驻服务进程）

| 指标 | 值 |
|---|---|
| `server_mem_after` 记录数 | 2338 |
| `allocated` 首/最小/最大/末 | 12.2 / 8.3 / 18.8 / **8.4** MiB |
| `reserved` 首/最小/最大/末 | 1944.0 / 1282.0 / 2998.0 / 1504.0 MiB |
| 进程高水位 `max_allocated` | 1592.4 MiB |
| 进程高水位 `max_reserved` | 2998.0 MiB |

小结：真实流程里也没有随请求数增长的驻留（末值低于首值），高水位 1.59 GiB 与微基准同量级。

## 六、端到端回归（`test/1`，默认 10 worker + 1 BLAS 线程）

命令与 `out_t01_fixed` 完全一致：
`python main.py test/1/EMD-8436.mrc test/1 5.6 0.04 <out> --log --runtime-config rt_blas1.json`
（结果存档 `demo_reg_cases/t03_result.txt`，输出目录 `out_t03`，墙钟 1055.6 s）

| 产物 | T03 md5 | 冻结基线 md5 | 判定 |
|---|---|---|---|
| `assembled_complex.cif` | `76638d0f…` | `76638d0f…` | **一致** |
| `assembled_complex_all.cif` | `76638d0f…` | `76638d0f…` | **一致** |
| `refined_complex.cif` | `bd281f40…` | `bd281f40…` | **一致** |
| `assembly_summary.txt` | `3a3f4cd4…` | `0356ae49…` | 内容逐行比较后**仅 Date 与 performance_summary 路径不同**（运行时间戳与输出目录） |

其它验收项：

- 池创建 1 次 / 关闭 1 次；无残留进程；`fit_request` 未归因 **1.13 s**（P2-fine 口径保持）；
- 关键决策与基线一致：`Chain A rejected (cc=0.4192 < 0.450)`、`Chain B accepted (cc=0.4235 >= 0.420)`、
  `Chain A: assembled from 2 domains (cc=0.4430)`；
- 服务端阶段（同为 2338 次配准）：`server_forward` 369.26 → 322.92 s、`server_write_pred` 87.33 → 82.45 s，
  但 `server_postprocess` 477.24 → 139.15 s（见第七节的**不作归因**说明）。

## 七、限制与**不作结论**的部分

1. **端到端跨运行差值不作归因**：`out_t01_fixed`（15:12–15:29）与 `out_t03`（15:49–16:07）相隔约 2.4 小时，
   其中 `server_postprocess`（T03 完全没碰的纯 numpy 阶段）从 0.204 s/次降到 0.060 s/次，
   整个分布（p50/p95）同步平移约 3.4×，且比 T03 机制能解释的幅度大得多 → 判定为**跨运行系统条件差异**
   （同时段共享主机负载等），按任务卡"不把冷暖/时段差异当算子加速"的要求，端到端只作
   **结果一致性与无回归**证据，量化收益只用同小时内的可控微基准 A/B。
2. `max_allocated` 是**进程累计**高水位，不是单对瞬时峰值；绝对值随 MIG 切片与输入规模变化。
3. `server_collate` 在两轮里都慢约 0.30 s / 36 对（8 ms/对，占单次配准 <1%）：未定位到机制，
   不掩盖任何结果差异，登记为观察项。
4. 微基准的两轮墙钟差（−20.7% / −14.0%）差异本身也来自运行间波动；可复核的稳定事实是
   "同一轮内每个测量点方向一致"与"逐位一致的结果"。
5. 本卡没有动候选顺序、随机消费、早停规则、尺度定义与 dtype；`no_grad` 与跳过 GT 对应都不消耗随机数。
6. 任务卡"第一轮不做"清单未触碰（无 FP16/BF16、无真多样本 batching、无自定义 CUDA）。

## 八、验收对照（任务卡 T03）

| 验收项 | 证据 | 结论 |
|---|---|---|
| 变换/评分/候选身份一致 | 微基准两轮 72/72 文件哈希 + overlap 列表；前向取证 41/41 字段按位一致 | 通过 |
| 多掩码连续运行 `allocated` 不持续增长 | 微基准 72 次（末/首 = 0.79）；真实运行 2338 次（12.2 → 8.4 MiB，高水位 1.59 GiB） | 通过 |
| 峰值可能下降 | `max_allocated` 4.85 GiB → 1.17 GiB（−75.7%） | 通过（超预期） |
| 不靠 `empty_cache` 掩盖泄漏 | 正常/异常路径两处 `empty_cache` 与递归 `release_cuda` 均已删除，并有测试防回归 | 通过 |
| 端到端结果不变 | 三个 CIF md5 与冻结基线一致；`assembly_summary.txt` 差异已逐行确认为时间戳/路径 | 通过 |
| 测试 | 110 项（含本卡新增 10 项）全过；`compileall` 通过 | 通过 |

## 九、对后续卡的指向
1. T07 的 256 MiB GPU 编码缓存预算：原占前向峰值 5%，现约占 **22%**，预算更宽裕；
2. T08（假设评分分块）仍应以 `model_lgr` 实测热点为准（两轮里它仍是模型内最大头 5.17–6.79 s / 36 对）；
3. T04（单侧几何拆分）不受本卡影响；`no_grad` 与字段白名单是后续卡做等价性检查时可复用的基线工具
   （`tools/probe_forward_identity.py` + `tools/compare_registration_runs.py`）；
4. 真实流程若需再降显存，应针对 `max_allocated` 1.59 GiB 的构成（T02/T08 范围），而不是继续清缓存。

## 十、复现路径（全部在 `/xiangyux/claude_c_work/demo_reg_cases/`，不进仓库）

| 产物 | 说明 |
|---|---|
| `t03_base_probe.diff` / `t03_probe_patch.py` | 基线树（`00a97e8`）的探针埋点补丁与幂等打补丁脚本 |
| `t03_ab_run.sh` / `t03_ab2_run.sh` / `t03_e2e_run.sh` | 第 1 轮 A/B、第 2 轮 A/B + 前向取证、`test/1` 端到端 |
| `t03_bench_base|new` / `t03_bench2_base|new` | 两轮微基准产物（各 72 个预测 + `server_timing.jsonl`） |
| `t03_compare.md` / `t03_compare2.md` | `tools/compare_registration_runs.py` 的两轮对比输出 |
| `t03_fwd_base|new` | 前向逐字段取证（`forward_outputs.npz` + 字段哈希 JSON） |
| `t03_compare_forward.py` | 前向取证的对比脚本（判定：共有字段全部逐位一致） |
| `t03_result.txt` / `out_t03` | 端到端运行日志、时间账、显存轨迹与产物 |
| `demo_reg_prev/` | 基线工作树（`git worktree`，含探针补丁；用完可 `git worktree remove`） |
