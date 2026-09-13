# T05：有界 CPU 几何缓存（实测报告）

日期：2026-09-13　代码：`feat/runtime-metrics`，工作区基线 `44c8d7c`（T04）+ 本卡改动
（T05 diff sha1 `969fe38f3bb36c83`）
输入：`tests/cases/registration_manifest.json`（1 源 7335 点 + 3 个掩码 8006/3565/1672 点）
GPU：A800 **MIG 7g.80gb 切片**。

## 一、本卡做了什么

| 变更 | 位置 |
|---|---|
| 新增按字节计费的有界 LRU：`ByteLruCache`（容量以字节计、`capacity_bytes<=0` 即关闭、单条超容量时正常返回但不缓存、`CacheStats` 记 hits/misses/evictions/rejected/bytes/peak/hit_rate），明确"值只读、不做深拷贝、不加锁（服务端单进程顺序处理）" | `protassem/fitting/feature_cache.py`（新） |
| 几何缓存层：`GeometryCache`（存 **CPU** 张量、按字节计费、容量 0 关闭）、`geometry_cache_key()`（构建前可算的 key）、`geometry_bytes()`、`geometry_to()`（返回新对象，不改缓存条目）、`acquire_geometry()`（缓存开关**同一份构建代码**） | `protassem/fitting/cloud_encoding.py` |
| key 覆盖：`GEOMETRY_VERSION` + 点集**内容与顺序**（含 dtype/shape）+ 特征 + 生效体素尺寸 + 最后一层采样方式 + **每阶段邻居数** + 质心（仅元信息，一并入 key 避免跨质心误命中） | 同上 |
| 只缓存确定性采样（`DETERMINISTIC_SAMPLING = ("voxel",)`）；`fps` 走 pytorch3d `random_start_point`（依赖全局 RNG），按任务卡偏差处理**保留采样、不缓存** | 同上 |
| 缓存由 **PARENet 常驻服务进程拥有**（`_server_loop` 建一次、跨请求复用），`geometry_cache` 沿 `run_inference → process_single_pair` 显式传递；单跑 CLI 也建自己的缓存 | `protassem/fitting/demo_mask.py` |
| 新增 `--geometry-cache-mb`（默认 512，0 = 关闭）；运行配置新增 `geometry_cache_mb`，由 `run_pipeline` 显式转交给服务端（`parenet_client.configure_geometry_cache`，服务已启动时不偷偷重启，只告警） | `demo_mask.py`、`runtime/config.py`、`parenet_client.py`、`pipeline.py` |
| 埋点：每次配准记 `server_cache_hit`（含 `tgt_hit`/`src_hit`/`cacheable`）与 `server_cache_store`；每个请求记一次 `server_cache_stats`（累计） | `demo_mask.py` |
| 工具：微基准新增 `--geometry-cache-mb`（并把缓存统计写进 `t00_report.json`）；`tools/summarize_timing.py` 新增几何缓存小节（逐次命中率 + 累计统计） | `tools/benchmark_registration.py`、`tools/summarize_timing.py` |

**开关等价是结构性的**：`acquire_geometry(cache=None, ...)` 与开启缓存走**同一份**构建代码，
不存在"两条路径"；`cache=None`、容量 0、非确定性采样三种情况都退化为"只构建"。

## 二、验收对照（任务卡 T05）

| 验收项 | 证据 | 结论 |
|---|---|---|
| 开关等价 | 单元测试 `test_cache_off_matches_cache_on`；微基准四轮（off→on→on→off）**各 72/72 预测哈希一致**、文件名集合与 overlap 列表一致 | 通过 |
| 内容变化失效 | 单测：改动一个点坐标/一个特征值 → key 变 | 通过 |
| 点序变化失效 | 单测：点集逆序 → key 变 | 通过 |
| 采样变化失效 | 单测：`fps`、体素尺寸、邻居数、质心任一改变 → key 变；`fps` 另外被排除在缓存之外（`cacheable=False`，命中数不增加） | 通过 |
| 容量不超预算 | 单测：容量 0/负值/超容量/替换键/连续插入 50 条，逐次断言 `bytes <= capacity`、`peak_bytes <= capacity`；真实运行 `entries` 与 `bytes` 见第五节 | 通过 |
| 命中跳过源几何重建 | 微基准命中率 **140/144 = 97.2%**（4 次未命中 = 1 次源 + 3 个掩码）；几何阶段耗时 0.44 s → 0.12 s（每 72 次配准） | 通过 |

## 三、微基准 A/B（`--geometry-cache-mb 0` vs `512`，冷/暖各一次重复，四轮、顺序交叉）

| 轮次 | 顺序 | 关闭墙钟 | 开启墙钟 | 预测哈希一致 |
|---|---|---|---|---|
| 1 | off → on | 42.20 s | 41.60 s | 72 / 72 |
| 2 | on → off | 41.60 s | 42.03 s | 72 / 72 |
| 3 | off → on | 42.03 s | 44.65 s | 72 / 72 |
| 4 | on → off | 44.65 s | 42.20 s | 72 / 72 |

- **墙钟差值在运行间波动范围内**（T00 已记录同输入两次重放可差约 29%；本组 41.6–44.7 s）：
  本卡**不主张墙钟加速**。
- 可归因的硬证据在**阶段层**（每 72 次配准，冷/暖两列）：

| 阶段 | 关闭 冷/暖 | 开启 冷/暖 | 说明 |
|---|---|---|---|
| `server_collate`（多尺度下采样） | 0.39 / 0.39 | 0.00 / 0.00 | 命中后完全跳过 |
| `server_neighbors`（k-NN） | 0.05 / 0.05 | 0.01 / 0.00 | 命中后完全跳过 |
| `server_cache_hit`（搬回设备） | — | 0.08 / 0.09 | 新增：命中的代价 |
| `server_cache_store`（写 CPU 副本） | — | 0.04 / 0.00 | 新增：仅 4 次未命中时发生 |
| 几何相关合计 | **0.44** | **0.13** | 每次配准约省 4.3 ms（两侧合计） |

- 缓存占用（微基准，开启轮）：`entries=4`、`bytes=12.1 MiB`、`peak=12.1 MiB`、`evictions=0`、
  `rejected_too_large=0`、`hit_rate=0.972`——远低于 512 MiB 预算。

## 四、单元测试（新增 24 项，全量 146 项）

- `tests/test_feature_cache.py`（11 项）：容量 0 关闭、负容量/非法 `size_of` 报错、put/get/字节计费、
  LRU 淘汰最近最少使用、超容量条目不缓存、替换键释放旧体积、连续插入不超预算、`clear`/`snapshot`、
  `None` 键报错。
- `tests/test_cloud_geometry.py` 扩展（13 项）：key 的内容/点序/采样/邻居数/质心失效与稳定性、
  `geometry_bytes` 与张量字节之和一致、`geometry_to` 不改原对象、手工几何的缓存往返、
  容量 0 关闭；CUDA 用例（`torch.cuda.is_available()` 守卫）：未命中→命中后**逐位一致**、
  缓存开关结果一致、内容变化重新未命中、关闭缓存永不命中、`fps` 被旁路。

## 五、端到端回归（`test/1`，默认 10 worker + 1 BLAS 线程 + 几何缓存 512 MiB）

命令与 T03/T04 一致；产物 `out_t05`，日志 `demo_reg_cases/t05_result.txt`，墙钟 1070.8 s；
服务端启动行确认容量生效：`PARENet server ready (pid=…, geometry_cache_mb=512)`。

| 产物 | T05 md5 | 冻结基线 md5 | 判定 |
|---|---|---|---|
| `assembled_complex.cif` | `76638d0f…` | `76638d0f…` | **一致** |
| `assembled_complex_all.cif` | `76638d0f…` | `76638d0f…` | **一致** |
| `refined_complex.cif` | `bd281f40…` | `bd281f40…` | **一致** |
| `assembly_summary.txt` | `2e241b3e…` | `0356ae49…` | 与 `out_t04` 逐行比较后**仅 Date 与 performance_summary 路径不同** |

- 关键决策逐字一致：`Chain A rejected (cc=0.4192 < 0.450)`、`Chain B accepted (cc=0.4235 >= 0.420)`、
  `Chain A: assembled from 2 domains (cc=0.4430)`；池创建 1 次、无残留进程、未归因 **1.21 s**。
- **缓存实测（2338 次配准，同一常驻进程，跨 3 个请求复用）**：

| 指标 | 值 |
|---|---|
| 源几何命中 | **2335 / 2338 = 99.9%**（3 次未命中 = 3 个不同源文件） |
| 目标几何命中 | **2171 / 2338 = 92.9%**（未命中 = 每个新掩码的首次请求） |
| 累计 | hits 4506 / misses 170 → hit_rate **0.9636** |
| 占用 | entries 170、bytes = peak = **188.94 MiB / 512 MiB**、**evictions 0**、rejected 0 |
| 命中搬设备 | 4.84 s（4506 次，≈1.07 ms/次） |
| 未命中构建（collate+neighbors） | 0.53 s（170 次，≈3.1 ms/次，两侧合计） |
| 写缓存（D2H + 记账） | 0.91 s（170 次，≈5.4 ms/次） |

- **收益核算（诚实口径）**：无缓存时几何成本 ≈ 2338 × 3.1 ms ≈ **7.3 s**；开启后
  4.84 + 0.53 + 0.91 = **6.28 s** → 净省约 **1.0 s（占 1071 s 的 0.1%）**，在运行间噪声内。
  原因：T04 之后"重建一侧几何"只要约 1.6 ms，而命中路径要把 int64 索引张量搬回 GPU（≈1.07 ms），
  两者同量级。**T05 不作加速承诺**，其价值是结构性与前置性（见第六节）。
- 显存：`allocated` 首/末 17.9 / 10.3 MiB（不增长），高水位 `max_allocated` 1594.5 MiB
  （与 T03/T04 同量级）；`reserved` 高水位 6288 MiB（与 T04 相同）。

## 六、限制与后续卡指向

1. **本卡的直接收益很小**（实测约 1 s / 1071 s）：T04 之后单侧几何重建约 1.6 ms，命中省下的是
   这笔钱，换成搬设备的约 1.07 ms。真实运行的 99.9% 源命中率与 92.9% 目标命中率说明机制正确，
   但要拿到数量级收益必须缓存的**不是几何而是编码**（T02 测得联合编码上界 10.4%）——
   这正是 T06/T07 的内容，T05 是它们的前置。
2. **命中路径的瓶颈是搬运**：缓存里几何约 2.3 MB/条，其中 int64 索引张量占大头。若 T07 需要更高的
   命中收益，可评估两种独立方案：(a) 编码缓存放 GPU（零搬运、但吃显存）；(b) 索引张量在缓存内以
   int32 存储、载入时转回 int64（减半字节与搬运，转换精确但需逐位验证）。两者都属 T07 决策，
   本卡按任务卡要求存 **CPU** 张量。
3. 目标几何也会被缓存，但**只对完全相同的重复请求命中**（同一掩码在同一配置下被请求多次）；
   不同掩码之间不会互相复用（key 含内容），符合任务卡 §一"不能将上一掩码的目标特征复用"。
   本流程每个掩码请求 12 次 → 命中率高；若将来每个掩码只请求一次，目标命中率退化为 0，源不变。
4. `fps` 采样不缓存（含 RNG）；当前生效采样恒为 `voxel`（T02/T04 已实测），该分支不执行。
5. 缓存条目**只读且独立存储**：`put()` 存的是 CPU 副本并**显式 clone**（`Tensor.to()` 同设备时是
   空操作、不复制，若不 clone，view 会把调用方的大数组一直留在内存里）；有单元测试断言
   缓存条目与调用方张量不共享存储。`join_geometries` 只做拼接（cat），不原地修改几何张量。
6. 服务端进程的缓存统计是**累计值**；`server_cache_stats` 每个请求记一次，工具按"最后一次请求后"
   解读。缓存容量可在 `--runtime-config` 里用 `geometry_cache_mb` 调整（0 = 关闭）；
   服务已在运行时改配置只告警、不重启（不打断在飞请求）。
7. 本卡改动后重新验证：单元测试 147 项通过；缓存开启的微基准与改动前**命中数、未命中数、条目数、
   字节数完全相同**（140/4/4/12 098 144），72/72 预测哈希一致。

## 七、复现路径（`/xiangyux/claude_c_work/demo_reg_cases/`，不进仓库）

| 产物 | 说明 |
|---|---|
| `t05_ab_result.txt` / `run_t05_ab.sh` | 四轮 off/on 交叉微基准的原始输出与脚本 |
| `t05_bench_off` / `t05_bench_on` / `t05_bench_off2` / `t05_bench_on2` | 四轮产物（各 72 个预测 + `server_timing.jsonl` + `t00_report.json` 内含缓存统计） |
| `t05_bench_on3` | `copy=True`（独立存储）改动后的缓存开启复验 |
| `out_t05` / `t05_result.txt` / `t05_e2e_run.sh` | 端到端产物、时间账+缓存小节的工具输出、运行脚本 |
