# T06：单侧编码与双侧配准（实测报告）

日期：2026-09-13　代码：`feat/runtime-metrics`，工作区基线 `13c1d9b`（T05）+ 本卡改动
输入：`tests/cases/registration_manifest.json`（1 源 7335 点 + 3 个掩码 8006/3565/1672 点）
GPU：A800 **MIG 7g.80gb 切片**。

**本卡结论（三句话）**：
1. 单侧编码 + 双侧配准已实现并与旧 `forward` **逐层等价**（真实输入位姿 max|Δ|=3.81e-06、
   候选索引/掩码/原始点 10/10 逐位一致），旧 `forward` 原样保留为兼容对照；
2. 该拆分**只有在关闭 TF32 时才成立**：TF32 让数值依赖张量形状（同一拆分在 TF32 下位姿差 0.127）；
   为此新增 `inference_mode` 与显式 TF32 策略；
3. **默认路径保持 joint + 框架默认精度**（实测与 T05 基线 72/72 预测哈希逐位一致），
   拆分路径作为已验证的可选路径保留，是否切换默认留给 T07 用真实运行实测净收益决定。

## 一、本卡做了什么

| 变更 | 位置 |
|---|---|
| `EncodedCloud`（单侧可缓存编码：多尺度点、细/粗层描述子与等变特征、分数、节点分区、`node_knn_points`、scale、几何指纹）+ `PARE_Net.encode_cloud(geometry, scale)`（backbone + 节点分区）+ `PARE_Net.register_pair(target_encoded, source_encoded)`（cross-attention → 粗匹配 → 点匹配 → LGR） | `fitting/parenet/model.py` |
| `forward()` **原样保留**为兼容对照；`register_pair` 在 `training=True` 时直接报错（训练入口仍是 `forward`），权重键名未改 | 同上 |
| 编码适配：`backbone_input()`（单侧几何 → backbone 输入，还原 pareconv 的 upsampling 紧凑约定）、`node_partition()`（纯函数；`attach_node_partition` 复用它） | `fitting/cloud_encoding.py` |
| **推理路径显式化**：`RuntimeConfig.inference_mode`（`joint` 默认 / `split`）+ `allow_tf32`（None = 跟随模式）+ `effective_allow_tf32()`（`split` + TF32 → 构造配置即报错，不给"看起来正常"的错误结果）；服务端在加载模型时应用策略，客户端经 `configure_inference_mode`/`configure_allow_tf32` 转交；CLI/benchmark 均有 `--inference-mode` 与 `--allow-tf32/--no-allow-tf32` | `runtime/config.py`、`fitting/demo_mask.py`、`fitting/parenet_client.py`、`pipeline.py`、`tools/benchmark_registration.py` |
| 埋点：`server_encode_tgt` / `server_encode_src` / `server_register`（拆分路径）；`server_forward` 仍属 joint 路径；`server_pair_info` 增记 `inference_mode` | `fitting/demo_mask.py` |
| 工具：`tools/check_encoding_split.py`（逐层对照，真实 + 合成）、`tools/compare_pred_pdbs.py`（PDB/坐标容差对比） | `tools/` |
| 测试：`tests/test_encoding_split.py`（10 项：CPU 契约 + GPU 等价性防回归） | `tests/` |

## 二、关键发现：TF32 让"数值依赖张量形状"

| 精度模式 | 真实输入上"联合 forward vs 单侧拆分" | 说明 |
|---|---|---|
| TF32 打开（框架默认：`cudnn.allow_tf32=True`、`matmul.allow_tf32=True`） | 位姿 **max\|Δ\|=0.127**；特征相对差最高 **3.4%** | 拆分改变结果 |
| TF32 关闭 | 位姿 **max\|Δ\|=3.81e-06**；特征 ≤1.3e-05（float32 kernel 求和顺序） | 可接受（第三节） |

- 同一路径连续两次前向**逐位一致**（max\|Δ\|=0）→ 差异来自 shape 相关的 kernel 选择，不是随机性。
- 关闭 TF32 **没有可测减速**（联合前向 0.193 s → 0.188 s，属噪声）；本模型瓶颈在 gather/scatter 与小批量算子。
- **因此把 `split` 与 TF32 绑定**：`inference_mode="split"` 要求 `allow_tf32=false`，非法组合在
  构造 `RuntimeConfig` 时即报错；`joint`（默认）跟随框架默认，保持既有数值。

## 三、验收：逐层与旧 `forward` 比较（TF32 关闭）

`tools/check_encoding_split.py`，真实输入（8006/7335 → 227/127）：

| 项 | 结果 |
|---|---|
| 位姿 `estimated_transform` | max\|Δ\| = **3.81e-06**（判据 1e-3）→ 通过 |
| 候选索引/掩码/原始点（10 个字段） | **10/10 逐位一致** → 通过 |
| `corr_scores` / `matching_scores` | 7.15e-07 / 7.30e-07 |
| 编码层字段（每侧 8 个） | 逐位一致 5/8，其余 ≤1.22e-05（`atol=1e-6` 逐元素判定 6/8 通过） |
| 输出字段（36 个共有） | 逐位一致 20 个；`atol=1e-6/rtol=1e-5` 内 28 个 |

**口径说明**：任务卡的 `atol=1e-6` 针对**几何/索引**级诊断（实测这部分全部逐位一致）；
float32 下同一数学的两次不同 shape 计算不可能逐位相同，残差 ≤1.3e-05（相对 ≤3e-06），
来源是卷积/矩阵乘的求和顺序。最终判定按任务卡另一条口径：R/t（3.81e-06 ≪ 1e-3）✓、
候选身份（逐位一致）✓、接受结果（见第五、六节）。合成输入 3000×2000 / 2000×3000 同样通过；
极小点云（500×400）下粗匹配排序会因 1e-6 级差异翻转（数据规模问题，已登记为限制）。

## 四、默认路径不变（回归证据）

| 对比 | 结果 |
|---|---|
| 默认路径（joint + 框架默认精度，本卡代码） vs T05 基线（`t05_bench_off`） | **72/72 预测哈希、文件名集合、overlap 列表逐位一致** |
| 端到端默认路径 | 与 T05 冻结基线同一代码路径 → 三个 CIF md5 不变（`76638d0f…` / `76638d0f…` / `bd281f40…`） |

即：本卡重构**没有改变生产默认行为**，冻结基线继续有效（`baseline_after_t01_fix.md5` 不需要重做）。

## 五、拆分路径的性能（诚实口径）

微基准（72 次配准，冷/暖各一遍；同一脚本内连续两轮）：

| 阶段（36 次累计，冷/暖） | joint + TF32（默认） | split + TF32 off | Δ |
|---|---|---|---|
| `server_forward` vs `encode_tgt`+`encode_src`+`register` | 10.05 / 9.78 s | 1.83+1.61+7.21 = **10.65 / 10.90 s** | +6% / +11% |
| `model_backbone`（两侧） | 1.19 / 1.17 s | **2.33 / 2.25 s** | ≈ ×2 |
| `model_transformer` | 0.28 / 0.30 s | 0.48 / 0.50 s | +0.20 s |
| `model_lgr` | 6.66 / 6.54 s | 5.88 / 5.78 s | −0.78 s |
| 全部重放墙钟 | 47.66 s | 52.03 s | **+9.2%** |

- 原因：**单侧编码拆掉了 backbone 的小批量效率**。单独测量：`backbone(joint)` 0.0337 s vs
  两次单侧 0.0382 s（+13%），而 `encode_tgt` 整体 0.0244 s、`encode_src` 0.0189 s。
- **T07 的账**：拆分路径下源编码 ≈1.61 s / 36 次 = **45 ms/次**；真实运行 2338 次配准若命中缓存，
  可省 ≈105 s（约 1071 s 的 10%），但拆分同时引入 `backbone` ×2 与 transformer 的额外开销
  （≈0.6–1.1 s / 36 次 ≈ 40–70 s / 2338 次）→ **净收益需按真实运行实测**，不能只按阶段外推。
  这正是 T07 的验收内容；若实测为负，按任务卡"依据实测选择默认项、保留关闭选项"保持 `joint`。

## 六、端到端对照（`test/1`，TF32 关闭下 joint vs split）

| 运行 | 墙钟 | `assembled_complex.cif` | `refined_complex.cif` | 关键决策 |
|---|---|---|---|---|
| joint（参考） | 1054 s | `14d0cf0d…` | `850f5722…` | Chain A rejected (0.4188)、Chain B accepted (0.4230)、2 domains (0.4430) |
| split（候选） | 1181 s | `f07db3ba…` | `36ed8123…` | Chain A rejected (0.4188)、Chain B accepted (0.4228)、2 domains |

- 两次运行的 **组件数（9192 原子）、原子顺序与标签列完全一致**；最终 `refined_complex.cif`
  坐标最大偏差 **0.151 Å**（拆分引入的 ≤4e-6 位姿差被流水线里的离散优化步骤放大）。
- 作为对照：TF32 打开 vs 关闭（同为 joint 路径）的坐标最大偏差 **135.6 Å**（换了一个同样得分的
  摆放）→ TF32 的影响比拆分本身大三个数量级。
- 结论：拆分路径**层级等价、结果形状一致**，但流水线会把它放大到 0.15 Å 量级；
  因此**若** T07 决定把它作为默认，需要同时冻结新基线并接受该量级漂移；本卡默认不动。

## 七、限制与后续卡指向

1. **TF32 是隐藏的"形状依赖"来源**，本卡首次量化（位姿 0.127 / 特征 3.4%）。任何改变张量形状的
   优化（本卡的拆分、T08 的分块、未来的 batching）都必须先固定精度策略，否则等价性无从谈起。
2. 拆分路径的收益取决于"源编码复用次数 vs backbone 小批量损失"；T07 必须在真实掩码序列上实测
   （命中率、缓存搬运、阶段耗时），并按实测决定默认项。
3. `forward()` 与拆分路径是**两份实现**（任务卡要求保留对照），`tools/check_encoding_split.py`
   是同步契约；改动任一侧都必须重跑该工具。
4. 极小点云（<500 点）下粗匹配排序对 1e-6 级差异敏感；真实输入与 ≥3000 点合成输入逐位稳定。
5. `pred_*.pdb`/CIF 的 md5 相等不再是 T06 的判据（`%.3f` 舍入 + 流水线放大），改用坐标容差
   （`tools/compare_pred_pdbs.py`）；同代码重复运行仍然要求逐位一致（实测 0 Å）。

## 八、复现路径（`/xiangyux/claude_c_work/demo_reg_cases/`，不进仓库）

| 产物 | 说明 |
|---|---|
| `t06_check_real.txt` | 真实输入逐层对照（`tools/check_encoding_split.py`） |
| `t06_bench_joint|split|joint2|split2` | 配准级两轮交叉 A/B（TF32 off） |
| `t06_bench_default` / `t06_bench_split_final` / `t06_final_measure.txt` | 默认路径回归 + 拆分路径阶段计时 + 性能探针 |
| `t06_pdb_compare.md` | PDB 坐标容差对比（0.001 Å） |
| `06_probe_numeric.py` / `06_probe_tf32.py` / `06_probe_backbone.py` / `06_probe_perf.py` | TF32、backbone、性能定位探针 |
| `out_t06_joint` / `out_t06_split` / `t06_e2e_result.txt` | 端到端对照（墙钟、md5、决策、缓存与时间账） |
