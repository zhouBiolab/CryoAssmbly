# P4：有界评分缓存（老卡收口第 5 步）

日期：2026-09-13/14　代码：`feat/old-card-closeout`（实现提交见 `git log`，本报告 = 第 5 步验收）
对照基线：`tests/cases/baseline_after_o6.md5`（`76638d0f…` / `bd281f40…`）。

**结论**：P4 通过——**缓存开关前后 `test/1` 三个 CIF md5 与 O6 基线逐位一致**，候选轨迹也完全一致
（28 次 `local_optimize` / 18 次 `cc_batch`）；读取与结构解析 12 → 1 次（命中率 91.7%），
占用 28.3 MB / 128 MiB；单次评分耗时 −4.5%。**本 case 端到端墙钟无可测收益**（评分只占 ~2.7%，
且并行在 10 个 worker 里），P4 的价值是"去重 + 有界计费 + 失效契约"。

## 一、改了什么（接口见 `ENGINEERING_HARDENING_PLAN.md` 附录 D.2）

| 内容 | 位置 |
|---|---|
| `DensityMapContext`（密度数组 + voxel/origin/shape/contour + 指纹；**保持原 dtype 与阈值顺序**）、`score_coords(ctx, coords, elements, resolution)`、`calculate_cc_mask()` 退化为薄包装（签名不变，新增可选 `density_version`） | `core/scoring.py` |
| 进程内**按字节计费**的双层缓存（密度上下文 + 结构坐标共享 `score_cache_mb`，默认 128 MiB，0 = 关闭）；键 = 版本 + 路径 + size + mtime_ns + contour(+显式版本)；`invalidate_density()` 显式失效；worker 缓存配置变化即重建/清空 | `core/scoring.py`、`runtime/byte_cache.py`（新，`ByteLruCache` 从 `fitting/feature_cache.py` 迁出并保留 re-export） |
| 配置与生效：`RuntimeConfig.score_cache_mb`（负数报错）+ `apply_score_cache()`（同时写 `PROTASSEM_SCORE_CACHE_MB`，fork/spawn worker 都继承）；`run_pipeline` 应用并记录 | `runtime/config.py`、`pipeline.py` |
| 探针：缓存关/开的读取次数、命中率、占用与单次耗时 | `tools/score_cache_probe.py`（新） |

**不做**（按附录 D.2）：低分辨率替代、bbox 硬剪枝、CC 公式调整、目标函数统一、无界缓存。

## 二、等价性与契约（单测 11 项，`tests/test_scoring_cache.py`）

| 验收项 | 结论 |
|---|---|
| 缓存开/关 CC 一致 | ✔ 逐位相同（`configure_score_cache(0)` 对照） |
| 非零 origin / 非立方图 / **轴映射**（mapc/mapr/maps）/ 各向异性 voxel | ✔ 全部按 header 重排后一致；置换时形状随置换变化（`sorted(shape)` 相同） |
| 同路径密度更新 | ✔ size/mtime_ns 变化 → 未命中，结果等于重新计算 |
| 显式密度版本 | ✔ 不同 `density_version` → 不同 key（两个条目并存） |
| 显式失效 | ✔ `invalidate_density()` 清空条目 |
| 预算 0 / 极小 | ✔ 0 = 关闭且不存；1 MiB 装不下大图 → 不缓存但结果正确，占用 ≤ 预算 |
| 预算变化 | ✔ 重建并清空旧条目 |
| 结构坐标更新 | ✔ 未命中，结果等于重新计算 |
| 缓存只读 | ✔ 代码内不原地修改缓存数组；需要修改时用独立数组 |
| `fitting/feature_cache.py` 兼容 | ✔ 仍导出 `ByteLruCache`（与 `runtime/byte_cache` 同一对象） |

全量测试：**212 项通过**。

## 三、探针实测（真实 `test/1` 输入，12 次调用，空闲机器）

| 组 | 单次耗时 | 首次调用 | 首次之后中位 | 密度读取 | 结构读取 | 命中率 | 占用/预算 |
|---|---|---|---|---|---|---|---|
| 缓存关 | 1071.5 ms | 2.84 s | 893.1 ms | 12 | 12 | 0% | 0 / — |
| 缓存开 | **992.8 ms（−7.3%）** | 2.06 s | 803.3 ms | **1** | **1** | **91.7%** | **29.05 MB / 128 MiB**（2 条目：1 密度 + 1 结构） |

（首版探针 1052.7 → 1004.9 ms，−4.5%；两次都在 −4.5%…−7.3% 之间，取本次为报告值。）

- 结果一致（`same_result: true`）。
- **共享预算**：一个 LRU、两类键前缀（`d:`/`s:`）—— 第 7 步真实运行曾暴露"两条独立 LRU 各按上限计费"
  （密度 113 MB + 结构 44 MB > 128 MiB），已改为共享并加单测 `test_density_and_structure_share_one_budget`。
- **−7.3% 单次耗时**：省掉的是"读图 + 阈值化"与"结构解析"；评分本身的大头是
  `_make_phenix_mask`（全图 `distance_transform_edt`）与 `_make_sim_map`，
  本步按附录 D.2 **不动**它们（bbox 剪枝属后续实验）。
- 结论：P4 是**读取/解析去重 + 有界计费**，不是评分算法加速；收益随图更大、评分调用更密而增大。

## 四、真实运行（`test/1`，10 worker，1 BLAS 线程，`out_p4`）

| 项 | 结果 |
|---|---|
| 墙钟 | **1327 s** |
| 三个 CIF md5 | `76638d0f…` / `76638d0f…` / `bd281f40…` → **与 `baseline_after_o6.md5` 一致** |
| 决策 | Chain A 拒 0.4192、Chain B 受 0.4235、Chain A 2 域 0.4430（与基线一致） |
| O6 台账 | 3 个请求 24 / 28 / 115 个候选，id 连续、全 `ok`、`end=ok` |
| `Runtime config` | `… score_cache_mb=128`；`Score cache: {...capacity 134217728…}`（启动时快照） |
| 未归因时间 | 1.93 s（`candidate_stream` 改为"区间重叠、不重复计入"后修正） |
| **父进程侧缓存统计**（第 9 次运行实测，`out_warm2`） | 密度 **hits 72 / misses 4（命中率 94.7%）**、结构 hits 0 / misses 76；合计 **113 MB + 44 MB** → 当时**超预算**，已修为共享预算（见下） |

**共享预算修正（第 7 步发现并修复）**：`out_warm2` 的统计显示密度 113 MB 与结构 44 MB 各自按
128 MiB 上限计费，合计 157 MB > 128 MiB，与附录 D.2"两者共享该预算"不符。已把 `_ScoreCache` 改为
**一个 LRU + 两类键前缀**（`d:`/`s:`），命中/未命中按类别计数、占用与峰值按共享预算计；新增单测
`test_density_and_structure_share_one_budget`（断言两类 `capacity_bytes` 相同、`bytes`/`peak_bytes`
不超过 `capacity_bytes`），并修正探针输出。复测（本报告第三节）：**29.05 MB ≤ 128 MiB** ✓。
该修正只改计费口径，不触碰任何数值路径。

阶段对比（同轨迹、同候选数）：

| 阶段 | o6_a | o6_b | **p4** |
|---|---|---|---|
| `pipeline_total` | 1239.6 | 1250.8 | 1318.1 |
| `gpu_wait` | 652.5 | 667.5 | 665.0 |
| `local_optimize`（28 次） | 208.9 | 204.7 | **238.7** |
| `final_select`（3 次） | 271.4 | 268.5 | **297.4** |
| `cc_batch`（18 次） | 33.4 | 33.4 | 36.2 |

**解释与不确定项（已由第 6 步回答）**：三者的候选轨迹完全相同（28/18 次、决策与 md5 一致），因此差异不能归因于搜索路径；
`local_optimize` 走的是它自己的 `DensityMap`（**不经过** P4 的评分缓存），所以 +30 s 也不像是 P4 的开销。
**第 6 步（P5）的运行给出同轨迹第三个样本：1211 s**（比 P4 那次快 116 s）→ 同一轨迹下的运行间波动达到
1211–1327 s（**±5%**），因此 **P4 的 1327 s 属机器波动，P4 没有可测的墙钟影响**（既不加速也不减速）。

**墙钟不作加速结论**：评分阶段本身只占 `pipeline_total` 的 2.7%（36 s / 1318 s）且并行在 10 个 worker 中，
P4 省的读取/解析也在其中；按附录 D.2 的口径，本步只承诺"减少重复读取与解析"，收益以测量为准。

## 五、复现路径（`/xiangyux/claude_c_work/demo_reg_cases/`，不进仓库）

| 产物 | 说明 |
|---|---|
| `p4_result.txt` / `/tmp/run_closeout_check.sh` | 真实运行与核对输出 |
| `out_p4/` | 运行产物（含 `metrics/`、候选台账） |
| `p4_probe.json` | 探针原始 JSON |
| `rt_p4.json` | 运行配置（`{"blas_threads": 1}`，评分缓存取默认 128 MiB） |
