# 老任务卡收口汇总（S0–S7 / R1 / R2(O4) / O5 / P1–P5 / O6）

日期：2026-09-14　分支：`feat/old-card-closeout`（自 `552d3e9` 起）　文档版本：`ENGINEERING_HARDENING_PLAN.md` v4.11

**结论**：老任务卡 **S0–S7、R1/R2/O4/O5、P1–P5 与 O6 已收口**。新任务卡 T00–T10 已完成并冻结在
`feat/runtime-metrics`（本轮不再扩大范围）。剩余范围见第五节。

## 一、本分支的七步（一步一次提交，逐步同步文档）

| 步 | 提交 | 内容 | 验收 |
|---|---|---|---|
| 1 | `af02707` | 文档口径校正 + **接口定稿**（附录 D）：三份基线用途分开、计时覆盖非加速、P3 池范围、删除 P4 收益上限、O6 升格 | 三份基线可追溯；附录 D.1–D.4 冻结 |
| 2 | `c8ede8c` | P3 池生命周期验证：`tools/pool_probe.py`（构造/就绪/首批/暖/回收/异常回收）+ `pool_start` 记录父进程 `threads`/`cuda_initialized` | fork 0.027–0.030 s、就绪 0.008–0.009 s、异常回收后存活 0；真实运行 `threads=1, cuda=false` → **维持 fork**；单测 8 → 11 |
| 3 | `592d617` | **O6 修复**：服务端候选台账（`request_id` + 连续整数 id + `ok/filtered/error` + `end{ok/error/cancelled}`）+ 客户端固定 ID 区间批次消费（批内原策略、首个达标即早停、`end` 到达后仍按批、缺 end/失败抛错、早停后确认请求结束）；掩码按名排序；父进程随机源固定 | 单测 14 项；全量 201 |
| 4 | `60ecaa2` | O6 实跑验收 + 冻结 `tests/cases/baseline_after_o6.md5`；`candidate_scan` → `candidate_stream`（口径修正） | 4 次运行逐位一致且等于 O6 前基线 |
| 5 | `e8d4f99` | **P4** 有界评分缓存（`DensityMapContext` + 结构坐标；128 MiB/进程共享；显式版本与失效） | 单测 11 项；读取 12 → 1、命中 91.7%、占用 28.3 MB、单次 −4.5%；`test/1` 与基线一致；全量 212 |
| 6 | `1ad9646` | **P5** SQLite TM 缓存（失败抛错、0.0 可缓存、两层同键、XDG 默认路径） | 单测 14 项；真实 USalign 抽查；`test/1` 与基线一致（父进程侧 hits 0/misses 3/writes 3）；全量 226 |
| 7 | `44f889f` + 本报告 | 集成验收（暖缓存跨运行回归）+ 收口声明 | **审计发现并修复"指纹含路径导致跨运行永不命中"**；修复后冷/暖两次运行验证命中与结果重复性；全量 227 |

## 二、统一验收矩阵（本轮 9 次真实 `test/1` 运行）

| # | 运行 | 配置 | 墙钟 | 三个 CIF md5 | TM 缓存（父进程侧） |
|---|---|---|---|---|---|
| 1 | `out_o6_a` | 默认（10 worker） | 1248 s | `76638d0f…`/`bd281f40…` | 未启用（P5 之前） |
| 2 | `out_o6_b` | 默认（重复） | 1260 s | 同上 | — |
| 3 | `out_o6_c` | `--num-processes 1` | 2009 s | 同上 | — |
| 4 | `out_o6_d` | `tail_pipeline=true` | 1199 s | 同上 | — |
| 5 | `out_p4` | + P4 评分缓存 | 1327 s | 同上 | — |
| 6 | `out_p5` | + P5 TM 缓存（冷库） | 1211 s | 同上 | misses 3 / writes 3 / rows 3 |
| 7 | `out_warm` | 同 6，复用 TM 库 | 1309 s | 同上 | **hits 0** ↔ 缺陷暴露（指纹含路径） |
| 8 | `out_p5b` | 修复后冷库 | 1259 s | 同上 | misses 3 / writes 3 / rows 3 |
| 9 | `out_warm2` | 修复后复用库 | 1268 s | 同上 | **hits 4 / misses 0 / writes 0 / rows 3** ✓ 跨运行复用 |

- 候选消费序列 sha1 在全部运行中一致：`de02d03105ebff9f`；台账一致：3 请求 24/28/115 候选、id 连续、全 `ok`、`end=ok`。
- **worker=1 与 10 一致、tail 开关不再改变产物** → O6 目标达成。
- 同轨迹运行间墙钟波动 ±5%（1199–1327 s；`--num-processes 1` 与 `tail=true` 另计）。
- **第 7 步审计发现并修复两个缺陷**（都由"暖缓存真实运行"暴露）：
  1. `structure_fingerprint()` 把绝对路径写进指纹 → TM 缓存跨运行永不命中（修复：纯内容 `<size>:<sha256>`）；
  2. `_ScoreCache` 用两条独立 LRU 各按上限计费 → 实际占用 113 MB + 44 MB > 128 MiB（修复：单 LRU 共享预算 + 单测）。
- **运行次数说明**：计划 7 次，实际 9 次（第 7 次暴露缺陷 → 修复后重做冷库 + 暖库各一次）。如实记录。

## 三、按任务卡的统一验收口径

| 口径 | 结论 |
|---|---|
| 几何索引/特征逐位或容差 | 沿用 T04–T08 结论；本分支未改几何/编码路径 |
| R/t、CC、候选顺序与接受结果 | 7 次运行三条决策与三个 CIF md5 一致；候选顺序由台账 id 决定（可审计） |
| 记录墙钟/阶段/缓存命中/峰值 | `performance.jsonl` + `server_timing.jsonl` + `tools/summarize_timing.py`（新增候选流与缓存小节） |
| 不把冷暖差异当加速、不把减少尝试当等价 | P4/P5 报告均只报命中/读取/耗时，不宣称端到端加速；O6 的 +15% 墙钟如实记录 |
| 默认项 | `geometry_cache_mb=512`、`inference_mode=joint`、`allow_tf32` 跟随、`encoding_cache_mb=256`（仅 split）、`hypothesis_chunk=0`、`tail_pipeline=false`、`blas_threads=1`、`seed=7351`、`score_cache_mb=128`、`tm_cache=auto` |

## 四、交付物

- 代码：`feat/old-card-closeout`（自 `552d3e9` 起 7 个提交）
- 报告：`tests/reports/2026-09-14_p4_scoring_cache.md`、`2026-09-14_p5_tm_cache.md`、
  `2026-09-13_o6_candidate_order.md`、`2026-09-13_p3_pool_lifecycle.md`、`2026-09-14_closeout_summary.md`（本文件）
- 基线：`tests/cases/baseline_after_o6.md5`（= 现行有效）；历史基线 `baseline_final_md5.txt`（T01 前）、
  `baseline_after_t01_fix.md5`（保留）
- 工具：`tools/pool_probe.py`、`tools/score_cache_probe.py`、`tools/summarize_timing.py`（更新）
- 测试：全量 **226 项通过**

## 五、剩余范围（明确不在本轮）

| 项 | 说明 |
|---|---|
| P9 GPU 真批处理 | 后续实验 |
| 增量 mask / bbox 剪枝 / 低分辨率评分替代 | 后续实验（P4 明确不做） |
| Step4/Step5 自建池（`homo_chain_refine` 5 处、`chain_enumerator`、`VoxEM`） | 只文档化未接入（P3 范围） |
| Step4/5 组件优化的 TM 计算（`refine_energy._calculate_tm_score`） | 独立实现，P5 未覆盖 |
| 6lu9 复合物端到端 | 本分支未跑（T10 已定"不作为必跑项"） |
| `MASK_REGISTRATION_OPTIMIZATION_TASKS.md` | 仍未跟踪入库（等你决定） |
