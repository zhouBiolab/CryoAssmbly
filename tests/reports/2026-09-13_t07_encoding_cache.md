# T07：精确 scale 源编码缓存（实测报告）

日期：2026-09-13　代码：`feat/runtime-metrics`，工作区基线 `3033a1f`（T06）+ 本卡改动
（T07 diff sha1 `d2f95f761da21d2b`）
输入：`tests/cases/registration_manifest.json`（1 源 7335 点 + 3 个掩码 8006/3565/1672 点）
GPU：A800 **MIG 7g.80gb 切片**。

**本卡结论**：源编码缓存实现完成且**结果逐位等价**（微基准缓存开/关 72/72 哈希一致；
真实运行与 T06 的 split 无缓存运行**最终 CIF md5 完全相同**）。微基准里它把"单侧编码 + 双侧配准"
从比旧路径慢 9% 变成快 17–20%；但**真实运行只做到"与旧路径持平"**（模型侧耗时都在
128–140 ms/次配准的波动带内），因此**默认仍保持 `joint`**，本卡交付"已验证、可选"的配置。

## 一、本卡做了什么

| 变更 | 位置 |
|---|---|
| `encoding_cache_key()`：几何指纹 + **精确 scale 位模式**（`struct.pack(">f")`，无分桶）+ 权重/配置指纹 + dtype + 编码版本 | `fitting/cloud_encoding.py` |
| `object_tensor_bytes()`：递归字节计费（dataclass/张量/列表），避免新增字段后账面低估 | 同上 |
| `EncodingCache`：复用 T05 的 `ByteLruCache`；**只缓存源侧**（目标编码只保留当前请求）；GPU 预算默认 256 MiB、0 = 关闭；单条超预算时正常使用但不缓存；不缓存 attention 大矩阵/逐层激活/hypotheses（这些本就不在 `EncodedCloud` 里）；无磁盘持久化 | 同上 |
| `model_fingerprint(model)`：state_dict 键名/形状/dtype + 全部权重字节（加载后算一次，≈0.2 s）；权重变化即失效 | `fitting/parenet/model.py` |
| `CloudGeometry.fingerprint()` 增加**记忆**（只覆盖不可变部分：点/特征/配置），避免每次配准重复哈希 | `fitting/cloud_encoding.py` |
| 缓存由 PARENet 常驻服务进程拥有（跨请求复用），`encoding_cache` 显式穿过 `run_inference → process_single_pair`；仅 `inference_mode="split"` 时创建（joint 模式不做单侧编码） | `fitting/demo_mask.py` |
| 配置/透传：`RuntimeConfig.encoding_cache_mb`（256）+ `--encoding-cache-mb`（服务端与 benchmark）+ `configure_encoding_cache()` | `runtime/config.py`、`fitting/parenet_client.py`、`pipeline.py`、`tools/benchmark_registration.py` |
| 埋点：`server_encode_cache`（查缓存耗时 + `src_hit`）、`server_encode_src`（`cached=False`）、`server_encode_store`、`server_register`（`src_cache_hit`）、每请求 `server_encoding_cache_stats` | `fitting/demo_mask.py` |
| 测试：`tests/test_encoding_cache.py`（14 项） | `tests/` |

## 二、验收对照（任务卡 T07）

| 验收项 | 证据 | 结论 |
|---|---|---|
| 同 scale 命中 | 微基准：hits 69 / misses 3（95.8%），entries 3（对应 3 个不同 scale）；单测 `test_put_get_roundtrip` | 通过 |
| 不同 scale 失效 | 单测 `test_scale_change_misses`、`test_exact_scale_bits`（1.5 与 1.5000001 位模式不同 → 不同 key，无分桶/无近似） | 通过 |
| 旋转不改缓存 | 单测 `test_rotated_geometry_is_not_a_hit`：旋转后点集不同 → key 不同 → 失效（不假命中） | 通过 |
| 候选与最终输出等价 | 微基准 **缓存开/关 72/72 预测哈希逐位一致**（两轮）；单测 `test_hit_equals_miss_bitwise`：`estimated_transform`/候选索引/`matching_scores`/`corr_scores`/`hypotheses` 逐位一致 | 通过 |
| 容量不超预算 | T05 的 LRU 单测（逐次断言 `bytes <= capacity`、`peak <= capacity`）+ `test_oversized_entry_not_stored` + 真实运行占用见第六节 | 通过 |
| 权重失效 | key 含 `model_fingerprint`（state_dict 全量字节）；`test_each_component_invalidates` 覆盖权重/几何/scale/dtype/版本任一项变化即失效 | 通过 |
| 命中跳过源 backbone（含独立 self-attention） | 本模型的 transformer 是**双侧 cross-attention**（无源侧独立 self-attention 阶段），因此命中跳过的是**源 backbone + 节点分区**；实测 `model_backbone` 2.21 → 1.29 s / 36 次、`model_node_partition` 0.06 → 0.03 s | 通过（与 T06 拆分结构一致） |
| 低命中率如实报告 | 见第六节真实运行命中率；不做尺度美化 | 通过 |

## 三、微基准 A/B/C（各两轮，顺序交替；几何缓存关闭）

A = `joint` + 框架默认精度（本卡之前的默认）；B = `split` + 编码缓存 256 MiB；C = `split` 无编码缓存。

| 对比 | 预测哈希一致 | 墙钟（全部重放，72 次配准） |
|---|---|---|
| A 轮内（a1 vs a2，同代码） | **72 / 72** | 49.10 → 50.12（+2.1%，噪声） |
| B 轮内（b1 vs b2） | **72 / 72** | 40.58 → 40.24（−0.8%） |
| C 轮内（c1 vs c2） | **72 / 72** | 50.60 → 52.34（+3.4%） |
| A → B | 0/72（精度模式不同，预期） | 49.10 → 40.58（**−17.4%**）；50.12 → 40.24（**−19.7%**） |
| B → C | **72 / 72**（缓存不改结果） | 40.58 → 50.60（**+24.7%**）；40.24 → 52.34（**+30.1%**） |

阶段归因（每 36 次配准，B 冷 / C 冷）：

| 阶段 | B（有缓存） | C（无缓存） | Δ |
|---|---|---|---|
| `server_encode_src` | 0.14 | 1.69 | **−1.55** |
| └ `server_encode_cache`（查缓存） | 0.04 | — | +0.04 |
| `model_backbone`（两侧） | 1.29 | 2.21 | **−0.93** |
| `model_node_partition` | 0.03 | 0.06 | −0.03 |
| `server_encode_tgt` | 1.82 | 1.73 | +0.10（噪声） |
| `server_register` | 9.31 | 7.58 | +1.73（跨运行波动） |

**直接可归因的收益 ≈ 2.5 s / 36 次配准**（源编码 + 源 backbone + 分区）。
总墙钟差（≈10 s / 72 次）大于该归因值，余额未完全解释（怀疑与"不再反复分配源编码张量"的
显存分配 churn 有关），登记为待查项，**不计入收益**。

## 四、结果等价性（缓存开关）

- 微基准：B（缓存开）与 C（缓存关）**72/72 预测哈希、文件名集合、overlap 列表逐位一致** ✓
- 单测（GPU）：命中路径与重算路径的 `register_pair` 输出（位姿、候选索引、匹配分数、
  `hypotheses`）**逐位一致** ✓
- 目标侧不缓存（`test_target_is_not_cached`）：目标编码只保留当前请求，符合任务卡"优先当前源"。

## 五、与 T06 的组合效果

| 配置 | 微基准墙钟（72 次） | 相对旧默认 |
|---|---|---|
| joint + TF32（T06 之前的默认） | 49.10 / 50.12 s | — |
| split + 无编码缓存（T06 状态） | 50.60 / 52.34 s | +3% / +4%（更慢） |
| **split + 编码缓存 256 MiB** | **40.58 / 40.24 s** | **−17.4% / −19.7%** |

即：**编码缓存把 T06 拆分的负收益翻转成明显正收益**（源编码复用率越高收益越大；
微基准里同一 scale 被复用 12 次）。

## 六、真实运行（`test/1`，split + 编码缓存 256 MiB + TF32 off）

运行配置 `rt_t07_split_cache.json`：`{"blas_threads": 1, "inference_mode": "split",
"encoding_cache_mb": 256}`；产物 `out_t07_split_cache`，日志 `t07_e2e_result.txt`，墙钟 **1096.6 s**。

| 项 | 结果 |
|---|---|
| 三个 CIF md5 | `f07db3ba…` / `f07db3ba…` / `36ed8123…` |
| 与 T06 的 split **无缓存**运行对比 | **md5 完全相同** → 编码缓存对最终产物零影响 ✓ |
| 与旧冻结基线（joint + TF32 on）对比 | 不同（这是 TF32 精度模式造成的，不是缓存造成的；T06 报告已量化） |
| 关键决策 | `Chain A rejected (cc=0.4188)`、`Chain B accepted (cc=0.4228)`、`Chain A: assembled from 2 domains` —— 与 T06 split 运行一致 |
| 源编码缓存命中 | **2335 / 2338 次（99.6%）**；entries 1、bytes 3.19 MiB / 256 MiB、0 淘汰、0 超预算拒绝 |
| 池/进程 | 池创建 1 次、无残留进程、未归因 1.27 s |
| 显存 | `allocated` 17–33 MiB（不增长）、`max_allocated` 1586 MiB（与 T05/T06 同量级）；`max_reserved` 7192 MiB（编码缓存 + 缓存分配器驻留，切片内安全） |

**三次真实运行的"模型侧耗时"对比**（同一 2338 次配准工作量，TF32 off）：

| 运行 | 模型侧合计 | 每次配准 |
|---|---|---|
| T06 joint（`server_forward`，2296 次配准） | 293.9 s | 128.0 ms |
| T06 split 无编码缓存 | 428.6 s | 183.3 ms |
| **T07 split + 编码缓存** | **327.6 s** | **140.1 ms** |
| （参考）T05 joint + TF32 on | 326.9 s | 139.8 ms |

- 编码缓存把拆分的模型侧开销从 **183 → 140 ms/次（−24%）**，**抵消了 T06 拆分的全部额外开销**；
- 但它**没有超过旧路径**：140.1 ms 落在 joint 路径的运行间波动带（128–140 ms）内，两轮墙钟
  1096.6 s vs 1054 s（+4%）同样在噪声内；
- 微基准（目标点云 8006 点、同 scale 复用 12 次）显示 −17…−20%，真实运行（掩码更小、源只有
  3008–7335 点、源编码本身便宜）只有"持平"——**小批量下拆分的固定开销占比更高**是主因。

## 七、默认配置决策（依实测）

| 配置 | 微基准 | 真实运行 | 结果 | 结论 |
|---|---|---|---|---|
| `joint` + TF32（框架默认） | 49.1 / 50.1 s | 1054 s（T06 joint） | 旧数值（旧基线有效） | **保持默认** |
| `split` + 无编码缓存 | 50.6 / 52.3 s | 1181 s | TF32 off 数值 | 不推荐（比默认慢 ~10%） |
| `split` + 编码缓存 256 MiB | **40.6 / 40.2 s** | 1096.6 s | TF32 off 数值（与上一行 md5 相同） | **可选**：微基准明显更快，真实运行持平；待 T10 用固定 manifest 复测后决定是否提升 |

因此本卡**不改变默认**（避免"用未确证的收益换取结果变化"），只交付：
`--inference-mode split --encoding-cache-mb 256`（或运行配置里两个键）这一**已验证、可回退**的配置。

## 八、限制与后续卡指向

1. 命中率取决于"同一 scale 被复用次数"：本流程每个掩码被请求 12 次，源编码复用 11 次；
   真实运行命中 99.6%（源与 scale 基本恒定）。若生产环境每个掩码只请求一次，命中率会显著下降。
2. 缓存只覆盖源侧；目标编码逐掩码重算（任务卡要求）。
3. 直接归因收益（微基准 2.5 s / 36 次）小于总墙钟差（≈10 s / 72 次），余额未完全解释；
   真实运行又显示"仅持平"——**小批量固定开销**与**跨运行波动**是当前无法分离的两个因素，
   T10 需要固定 manifest 的多轮复测来定论。
4. 提升 `split` 为默认意味着同时接受 TF32 关闭带来的结果变化（T06 实测端到端坐标漂移 0.151 Å）
   与基线重做；本卡不做该变更。
5. 偏差记录：首次 T07 真实运行因请求 JSON 里 `inference_mode=None` 被服务端判为非法值而**全部
   请求失败**（0 组件、50 s 结束）；已在客户端"None 时不写该键"+ 服务端 `or 默认值` 两侧修复并
   重跑。该缺陷由端到端运行暴露（单元测试未覆盖请求负载契约），已记入方案修订记录。

## 九、复现路径（`/xiangyux/claude_c_work/demo_reg_cases/`，不进仓库）

| 产物 | 说明 |
|---|---|
| `t07_ab_result.txt` / `run_t07_ab.sh` | A/B/C 六轮微基准原始输出与脚本 |
| `t07_bench_a1|b1|c1|a2|b2|c2` | 六轮产物（各 72 个预测 + 阶段计时 + 缓存统计） |
| `out_t07_split_cache` / `t07_e2e_result.txt` / `rt_t07_split_cache.json` | 真实运行（split + 编码缓存）产物、日志与运行配置 |
| `t06_bench_default` / `t06_bench_split_final` | T06 的 joint / split 对照（用于第五节比较） |
