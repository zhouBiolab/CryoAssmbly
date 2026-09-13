# T08：位姿假设评分分块（实测报告）

日期：2026-09-13　代码：`feat/runtime-metrics`，工作区基线 `2da6645`（T07）+ 本卡改动
（T08 diff sha1 `48a655d26328f3ef`）
输入：`tests/cases/registration_manifest.json`（1 源 7335 点 + 3 个掩码 8006/3565/1672 点）
GPU：A800 **MIG 7g.80gb 切片**；运行配置 = 生产默认（joint + 框架默认精度，TF32 打开）。

**本卡结论**：热点定位在 LGR/HypothesisProposer 的"整批假设评分"；**chunk=64 与 chunk=0 结果逐位一致
（72/72 预测哈希）、峰值显存 −15.6%、墙钟 −3…−7%**；**chunk=1 会改变结果**（24/72）且在 TF32 下
慢 63%。按任务卡"原路径继续默认"，本卡**不改变默认**，交付 `--hypothesis-chunk 64` 这一
已验证可回退配置。

## 一、热点定位（不改共享 pareconv）

`pareconv/modules/registration/combineRegisraition.py` 有两处"整批假设评分"：

```python
aligned   = apply_transform(src_corr_points.unsqueeze(0), transforms)   # (P, N, 3)
residuals = torch.linalg.norm(ref_corr_points.unsqueeze(0) - aligned, 2) # (P, N)
masks     = residuals < acceptance_radius                                # (P, N)
counts    = masks.sum(dim=1)                                             # (P,)
```

- `LocalGlobalRegistration.local_to_global_registration`（补丁数 P × 验证点数 N）
- `HypothesisProposer.feature_based_hypothesis_proposer`（假设数 P=2200 × 验证点数 N）

P=2200、N≈2200 时 `(P, N, 3)` float32 ≈ 58 MB、`(P, N)` 掩码与其整型中间量再占数十 MB；
实测这一段的进程峰值 `max_allocated` 为 1194 MiB。

任务卡要求"不直接修改共享 `/xiangyux/PARENet-main`"，而该项目的扩展 `pareconv.ext`（编译产物）
只存在于共享目录（仓库副本没有 `.so`），把副本切成导入源会缺扩展。因此本卡**不动 pareconv**，
改用**子类覆盖单个方法**（方法体与上游逐行一致，唯一差别是分块调用）：

| 文件 | 内容 |
|---|---|
| `protassem/fitting/hypothesis_scoring.py`（新） | `select_best_hypothesis(ref, src, transforms, acceptance_radius, chunk_size)`：按假设分块统计内点、返回 `(best_index, best_mask)`；`chunk_size<=0` 或 `>= P` 走原整批路径；argmax 用严格大于比较保持"首个最大值"的 tie-break |
| `protassem/fitting/chunked_registration.py`（新） | `ChunkedLocalGlobalRegistration` / `ChunkedHypothesisProposer`（各只覆盖一个方法）+ `build_registration(hypothesis_chunk=0, **kwargs)` 工厂（kwargs 与上游 `combineRegisraition.__init__` 同名，避免参数漂移） |
| `parenet/model.py` | `PARE_Net(cfg, hypothesis_chunk=0)` 持有分块大小并据此构造；`create_model(cfg, hypothesis_chunk=0)` |
| 配置/透传 | `RuntimeConfig.hypothesis_chunk`（默认 0）+ `--hypothesis-chunk`（服务端与 benchmark）+ `configure_hypothesis_chunk()`；在**构造模型之前**应用 |
| 测试 | `tests/test_hypothesis_chunking.py`（5 项） |

**为什么用显式参数而不是改 cfg**：`parenet/config.py::make_cfg()` 返回**模块级单例**，改
`cfg.fine_matching.*` 会污染同进程内的其他模型（实测两个模型拿到同一份被改过的 cfg）；
因此分块大小作为构造参数显式传入。

## 二、验收对照（任务卡 T08）

| 验收项 | 证据 | 结论 |
|---|---|---|
| chunk=0/64/1 结果对照 | 生产配置（TF32 打开）72 次配准：**chunk=64 与 chunk=0 的预测哈希 72/72 一致**；**chunk=1 只有 24/72 一致**，且 target 0/2 的 overlap 与文件名都变（0.058→0.055 等） | 64 ✓ / 1 ✗ |
| 分块逻辑等价 | GPU 单测（TF32 关闭，合成输入）：chunk=1/64/0 的 `estimated_transform`/`hypotheses`/`corr_scores`/候选索引**逐位一致**；CPU 单测：chunk=1/2/3/7/29/30/64 与整批的 `best_index` 与掩码逐位一致；tie-break 取首个下标 | 通过 |
| 峰值显存可测 | `server_mem_peak`：`max_allocated` **1194 → 1008 MiB（−15.6%）**、`max_reserved` **2564 → 1758 MiB（−31%）** | 通过 |
| 时间需实测 | 墙钟（每轮 72 次配准，两轮）：chunk=0 `40.65 / 39.05 s`、chunk=64 `37.64 / 37.78 s`（**−7.4% / −3.3%**）、chunk=1 `66.40 / 64.18 s`（+63%）；`model_lgr`（CUDA event）12.27 → 12.76 → 39.50 s | 见结论 |
| 不改随机/不减少假设 | 假设数、评分、全局顺序与 tie-break 均未改；只把"假设"这一维分块 | 通过 |
| 原路径继续默认 | `RuntimeConfig.hypothesis_chunk` 默认 0；`build_registration(0, ...)` 返回上游类实例（单测断言） | 通过 |

## 三、测量明细（生产配置，72 次配准/轮，两轮交替）

| chunk | 与 chunk=0 的哈希一致 | 墙钟 a/b | `max_allocated` | `max_reserved` | `model_lgr` |
|---|---|---|---|---|---|
| 0（原路径，默认） | —（自身两轮 72/72） | 40.65 / 39.05 s | 1194 MiB | 2564 MiB | 12.27 s |
| **64** | **72 / 72** | **37.64 / 37.78 s** | **1008 MiB** | **1758 MiB** | 12.76 s（+4%） |
| 1 | 24 / 72 ✗ | 66.40 / 64.18 s | 1008 MiB | 1756 MiB | 39.50 s（×3.2） |

- **chunk=64 的墙钟收益（−3…−7%）大于 `model_lgr` 阶段的变化（+4%）**：分块后分配器足迹明显变小
  （reserved −31%），host 侧/分配开销下降，收益体现在阶段计时之外——与 T05/T07 观察到的
  "分配器 churn 影响墙钟"一致。
- **chunk=1 为什么不等价**：它把每次 `apply_transform` 的批次降到 1，**TF32 下 cuBLAS 会选择
  不同的内核**（1 元素批次），数值随之改动（T06 已证明 TF32 的数值依赖张量形状）；
  关闭 TF32 时 chunk=1 与整批逐位一致（单测）。因此"分块大小"必须逐位验证，
  不能假设"块越小越安全"。

## 四、结论与默认决策

1. **chunk=64 是安全且有收益的配置**：结果与旧路径逐位一致、峰值显存降 15.6%、
   微基准墙钟降 3–7%；推荐作为可选配置（`--hypothesis-chunk 64`，或运行配置
   `{"hypothesis_chunk": 64}`）。
2. **默认仍是 0（原路径）**：任务卡明确"原路径继续默认"，且 T07 的教训是
   "微基准收益在真实运行里会被非配准阶段稀释"，是否提升默认由 T10 用固定 manifest 复测后决定。
3. **chunk=1 不推荐也不默认**：结果会变（TF32 内核选择），且慢 63%。
4. 只分块了"独立的假设维度"：假设数、评分公式、全局顺序、tie-break 全部保留；
   `chunk<=0` 时完全走上游实现（子类不被使用）。

## 五、限制与后续卡指向

1. 本卡的包装是对上游两个方法的最小复制（方法体逐行一致）。上游若改动这两个方法，
   必须同步 `chunked_registration.py` 并重跑 `tests/test_hypothesis_chunking.py`
   （该测试直接比较分块与整批的输出，能发现语义漂移）。
2. 共享 pareconv 未被修改；仓库副本 `pareconv_src` 仍未作为导入源（缺编译扩展 `ext*.so`），
   这一点在 T00 已记录，本卡再次确认（`import pareconv` → `/xiangyux/PARENet-main`）。
3. 分块收益与输入规模相关：验证点数 N 小时整批张量本来就小，收益递减；N 大时收益更明显。
4. 峰值显存数字来自进程级 `max_allocated`；单次 LGR 的瞬时峰值未单独隔离测量（chunk 只影响
   这一段，两次运行的差值即该段的贡献）。

## 六、复现路径（`/xiangyux/claude_c_work/demo_reg_cases/`，不进仓库）

| 产物 | 说明 |
|---|---|
| `t08_ab_result.txt` / `run_t08_ab.sh` | 六轮（chunk=0/64/1 × 两轮）微基准原始输出与脚本 |
| `t08_bench_c0a|c64a|c1a|c0b|c64b|c1b` | 六轮产物（各 72 个预测 + 阶段计时 + 显存峰值） |
| `tools/benchmark_registration.py --hypothesis-chunk N` | 复跑入口（默认 0 = 原路径） |
