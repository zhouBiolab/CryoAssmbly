# T04：单侧几何拆分（实测报告）

日期：2026-09-13　代码：`feat/runtime-metrics`，工作区基线 `832fa34`（T03）+ 本卡改动
（T04 diff sha1 `6db8d8e5546acdde`）
输入：`tests/cases/registration_manifest.json`（1 源 7335 点 + 3 个掩码 8006/3565/1672 点）
GPU：A800 **MIG 7g.80gb 切片**。

## 一、本卡做了什么

新增 `protassem/fitting/cloud_encoding.py`（单侧几何记录 + 构建 + 联合适配）：

| 内容 | 说明 |
|---|---|
| `CloudGeometry` | 单侧多尺度几何：`points`（每阶段 (N_i,D)）、`lengths`、`features`、`neighbors`、`subsampling`、`upsampling`、`centroid`、生效的 `voxel_sizes`/`sampling_method`，可选 `node_partition`；**所有索引都是本侧局部索引** |
| `NodePartition` | 节点分区（`masks`/`knn_indices`/`knn_masks`），由 `attach_node_partition()` 生成 |
| `build_stage_points()` | 单侧多尺度点，等价于 pareconv `precompute_subsample` 的单侧实现（下采样是 CPU 实现，随后搬设备） |
| `build_neighbors()` | 单侧邻居/上下采样索引（pointops `knnquery_heap`，CUDA） |
| `join_geometries()` | 适配层：拼成旧联合入口的 `data_dict`，**跨侧只给 src 侧加偏移**，偏移按"行/值所在阶段"分别取 |
| `offset_indices()` | 哨兵规则：`pointops` 邻居不足时用 0 填充尾部槽位，只有 `source_count >= k` 才无哨兵；有哨兵时保持 0、不加偏移 |
| `CloudGeometry.fingerprint()` | 几何指纹（实际点集与顺序 + 特征 + 生效采样配置 + `GEOMETRY_VERSION`），供 T05 缓存 key |

`demo_mask.process_single_pair` 改为：两侧各建几何 → 各补邻居 → `join_geometries()` 拼装 →
送模型。同时删掉三个不再使用的 pareconv 导入（`registration_collate_fn_stack_mode`、
`precompute_neibors`、`to_cuda`）与一个既有的未使用导入（`apply_transform`）。

**键发现（顺带解决一个隐性成本）**：旧入口 `registration_collate_fn_stack_mode` 内部有
`torch.cuda.empty_cache()`（pareconv 代码，不能改），每个请求都会触发一次显存同步+回收；
单侧路径不再调用它（T03 删掉的是我们自己的两处）。

**配置口径**：pareconv 的 `precompute_subsample` 从**模块级全局变量**读体素配置，而本仓库
没有任何地方设置它们 → 实际始终是 `config_id=0` + `voxel`（T02 已实测"6 个标签输入完全相同"）。
本卡把生效值用 `effective_sampling_config()` 显式读出、作为参数传入几何构建，并在
`server_pair_info` 记录 `effective_sampling` 与 `voxel_sizes`（标签 `configs=all`/`fps` 与实际
生效配置的对照从此可见）。生产行为不变：仍是 config 0 + voxel。

## 二、验收：独立构建 vs 联合构建（逐阶段逐张量**按位**比较）

工具：`tools/check_geometry_split.py`（新，本卡交付）。比较范围：`features`、`transform`
设备/dtype、每阶段 `points`/`lengths`/`neighbors`/`subsampling`/`upsampling`、两侧节点分区
（`masks`/`knn_indices`/`knn_masks`）。

| 用例 | 阶段点数（ref / src） | 结果 |
|---|---|---|
| 真实输入（manifest target 0） | (8006,7335) → (2851,2076) → (805,511) → (227,127) | **全部一致**（含分区） |
| 合成 300 × 120 | (300,120) → (293,120) → (254,114) → (110,80) | **全部一致** |
| 合成 2048 × 96 | (2048,96) → (1793,96) → (870,92) → (125,71) | **全部一致** |
| 合成 96 × 2048 | (96,2048) → (96,1787) → (93,872) → (65,125) | **全部一致** |

- 覆盖"小点云（96 点）"与"两侧点数差异大（2048 vs 96）"两类边界；对比期间临时把体素配置写进
  pareconv 的全局槽位（工具内 `try/finally` 恢复），证明单侧路径可以用**任意**显式配置，
  而旧入口只能通过改全局变量。
- 四组用例的"哨兵风险（阶段点数 < 邻居数）"均为**无** → 本次比较不涉及 0 填充槽位；
  哨兵规则由单元测试单独覆盖（见第四节）。

## 三、结果与耗时（微基准 A/B：`t03_bench_new`（T03 代码） vs `t04_bench`）

| 项 | 结果 |
|---|---|
| 预测文件数 | 72 / 72 |
| `pred_*.pdb` 内容 sha256 一致 | **72 / 72** |
| 文件名集合（排名+overlap）与逐对 overlap 列表 | 完全一致 |

阶段累计（冷 = 第 1 次重放，36 对；秒）：

| 阶段 | T03 冷 | T04 冷 | Δ冷 | T03 暖 | T04 暖 | Δ暖 |
|---|---|---|---|---|---|---|
| `server_collate`（单侧多尺度点） | 0.49 | 0.14 | −0.36 | 0.49 | 0.15 | −0.34 |
| `server_neighbors`（单侧 k-NN） | 0.86 | 0.07 | −0.79 | 0.87 | 0.07 | −0.81 |
| `server_join`（适配层拼装，**新增**） | — | 0.55 | +0.55 | — | 0.53 | +0.53 |
| `server_to_gpu`（**取消**：搬设备并入单侧构建） | 0.04 | — | −0.04 | 0.03 | — | −0.03 |
| `server_forward`（主机 wall） | 8.39 | 10.85 | +2.46 | 10.12 | 10.07 | −0.05 |
| `model_lgr` | 5.17 | 6.75 | +1.58 | 6.79 | 6.42 | −0.37 |
| `model_backbone` | 1.36 | 1.13 | −0.23 | 1.39 | 1.12 | −0.27 |
| 全部重放墙钟 | 46.52 | 45.72 | −0.80（−1.7%） | — | — | — |

**结论**：几何阶段（collate+neighbors）自身从 1.35 s 降到 0.21 s，但适配层 `server_join`
（把两侧 `torch.cat` 成联合布局 + 加偏移）花掉 0.55 s，**净变化在噪声量级**（总墙钟 −1.7%，
逐 target 有正有负）。这与任务卡"本卡不承诺加速"一致：T04 的收益是**结构性的**
（源几何成为独立、可指纹化、可缓存的单元），性能收益要等 T05/T07。

注：`server_neighbors` 的绝对变小部分来自**异步边界变化**（k-NN 是异步 kernel，旧路径在
`radius_search` 内部还多做两次 `.contiguous()` 切片拷贝与 `torch.cat`），因此这一行不作
算子级加速结论；可复核的硬事实是"结果逐位一致"与"几何阶段总耗时未增加"。

## 四、单元测试（新增 12 项，共 122 项）

`tests/test_cloud_geometry.py`：

- 多尺度点数单调不增、首阶段与输入逐位相同、确定性（两次构建指纹相同）；
- 输入契约：非二维/空点云/点数与特征行数不一致/列数 < 3/空体素配置/未知采样方式 → `GeometryError`；
  `build_stage_points` 拒绝 CUDA 输入（下采样是 CPU 实现）、`build_neighbors` 拒绝 CPU 输入、邻居数
  长度与阶段数不一致 → `GeometryError`；
- **哨兵规则**：`source_count >= k` → 整体加偏移；`source_count < k` → 尾部槽位保持 0；
- **跨侧偏移的层次**：neighbors 按本阶段点数、subsampling 行=下一阶段点数/值=本阶段点数、
  upsampling 行=本阶段点数/值=下一阶段点数，逐项断言展开后的张量；
- 指纹：改动一个点或改采样配置 → 指纹变化；`sha256` 十六进制格式。

## 五、端到端回归（`test/1`，默认 10 worker + 1 BLAS 线程）

命令与 `out_t03`/`out_t01_fixed` 完全一致（`--runtime-config rt_blas1.json`）；
产物 `out_t04`，日志 `demo_reg_cases/t04_result.txt`，墙钟 1097.4 s。

| 产物 | T04 md5 | 冻结基线 md5 | 判定 |
|---|---|---|---|
| `assembled_complex.cif` | `76638d0f…` | `76638d0f…` | **一致** |
| `assembled_complex_all.cif` | `76638d0f…` | `76638d0f…` | **一致** |
| `refined_complex.cif` | `bd281f40…` | `bd281f40…` | **一致** |
| `assembly_summary.txt` | `bdaa197c…` | `0356ae49…` | 与 `out_t03` 逐行比较后**仅 Date 与 performance_summary 路径不同** |

- 关键决策与前一版逐字一致：`Chain A rejected (cc=0.4192 < 0.450)`、
  `Chain B accepted (cc=0.4235 >= 0.420)`、`Chain A: assembled from 2 domains (cc=0.4430)`；
  连候选日志数值都相同（如 `Candidate #2 … CC 0.2533 -> 0.4174`）。
- 池创建 1 次 / 关闭 1 次；无残留进程；`fit_request` 未归因 **1.33 s**。
- 服务端 2338 次配准：`server_forward` 327.90 s、`server_postprocess` 144.35 s、
  `server_write_pred` 102.02 s（T03 对应 322.92 / 139.15 / 82.45）——**跨运行差值不作归因**
  （T03 报告已证明这些阶段存在远超代码差异的时段波动）。
- 显存（同一常驻进程 2338 次）：`allocated` 首/末 = 17.9 / 10.3 MiB（**不增长**，高水位
  `max_allocated` 1594.5 MiB，与 T03 的 1592.4 MiB 同量级）；`reserved` 高水位
  6288 MiB（T03 为 2998 MiB）——见第六节第 6 条。
- 观察项：`local_optimize` 次数 23（T03 为 24），即监测循环少触发一次局部优化；最终产物
  逐字节相同，属既有"批次/轮询时机敏感性"（开放项 O6）范畴，非本卡引入。

## 六、限制与后续卡指向

1. **模型仍在 `forward` 内自己算节点分区**：本卡把单侧分区算出来并证明它与联合切片的结果逐位
   一致（验收要求），但**尚未接线**给模型。接线（去掉重复计算）属 T06"单侧编码与双侧配准"，
   避免在本卡给 `forward` 增加第二条代码路径。
2. `cloud_encoding` 不读取、也不写入 pareconv 的模块级全局变量；`demo_mask` 只把生效值读出来
   传参并记录。要真正支持多配置，应把配置显式化到运行配置（目前无人设置全局变量，属既有行为）。
3. `join_geometries` 的偏移按"行所在阶段/值所在阶段"分别取，已由单元测试与真实数据双向覆盖；
   新增阶段或改变 pareconv 的 upsampling 列表约定（元素 j ↔ stage j+1）时必须同步。
4. 哨兵（0 填充槽位）在当前数据与基准输入中**不出现**（所有阶段点数 ≥ 邻居数）；一旦出现
   （极小点云/极大邻居数），联合入口会把哨兵一起偏移、单侧路径不会——这是**有意修正**，
   单元测试固定该行为。
5. `fps` 分支沿用 pareconv 的 `farthest_point_sampling_gpu`（内部 pytorch3d
   `random_start_point=True`），因此**随机消费与旧路径完全一致**；当前生效配置是 `voxel`，
   该分支不会被执行（改为 fps 属于配置变更，不是本卡范围）。
6. 微基准 A/B 的两侧不是同一时刻的运行（相隔约 40 分钟），阶段差值只作方向性参考；
   T10 会用固定 manifest 重跑并做完整对比。
7. **新观察项（`reserved` 升高）**：真实运行里 `reserved` 高水位 6288 MiB（T03 为 2998 MiB），
   而同一 manifest 的微基准 `max_allocated` 仅 +1.4%（1177 → 1194 MiB）、`max_reserved` +6 MiB。
   原因是本卡在拼装后仍**同时持有**单侧几何张量与联合张量（拼装是拷贝），分配器的缓存块集合更碎；
   真实输入更大所以被放大。**真实占用（`allocated`）稳定在 ~10–30 MiB 且不增长**，不构成泄漏或
   容量风险；几何对象的生存期归 T05（几何缓存）设计，届时一并决定何时释放单侧张量。

## 七、复现路径（`/xiangyux/claude_c_work/demo_reg_cases/`，不进仓库）

| 产物 | 说明 |
|---|---|
| `t04_check_manifest.txt` | 真实输入的独立 vs 联合逐阶段按位比较（`tools/check_geometry_split.py --manifest`） |
| `t04_check_synthetic.txt` | 三组合成输入（300×120 / 2048×96 / 96×2048）比较输出 |
| `t04_bench` / `t04_bench2` | T04 微基准（改动前后各一次，用于证明注释/校验重构不改结果） |
| `t04_compare.md` | T03 → T04 的微基准对比（一致性/墙钟/阶段/显存） |
| `t04_bench_vs_bench2.md` | 改动前后微基准对比（72/72 哈希一致） |
| `out_t04` / `t04_result.txt` / `t04_e2e_run.sh` | 端到端产物、日志与时间账、运行脚本 |
| `04_probe_split.py`（前缀 `t04_probe_split.py`） | 立项前的可行性探针（单侧 grid_subsample/kNN 与联合切片按位一致） |
