# demo_reg 工程化硬化与运行时优化实施方案（v2）

版本：2026-09-13 **v4.11**（老卡收口：接口定稿；T00–T10 完成；取代 v2.1–v4.10/v2/v1）　基线提交：`5b016f5`（含 3 处未提交改动）　目标分支：`feat/engineering-hardening`
状态：**实施中**。T00–T10 已完成（新卡到此为止，不再扩大）；老任务卡按 v4.11 与**附录 D（接口定稿）**在分支 `feat/old-card-closeout` 上收口：文档口径校正 → P3 生命周期验证 → O6 → O6 新基线 → P4 → P5 → 集成验收。每张卡/每步的实测结论写入"修订记录"，与代码同一提交。

事实分级（全文标注）：

- **【已核实】** 有代码位置或实测输出作为证据；
- **【待验证】** 需要实测或真实运行才能确认，不得当作已知结论；
- **【决策】** 已定稿的执行决定，实施时不再重新讨论。

---

## 0. 定位、维护规则与数据管理

### 0.1 阶段划分

| 阶段 | 内容 | 来源 | 前置 |
|---|---|---|---|
| **阶段一**（准备步骤 S0 + 七个实施步骤 S1–S7） | 边界与格式契约硬化：CLI 解析、入口校验、点云 TXT 统一读写、链号匹配与池、可移植性 | 本文第 3 节 | 无 |
| **阶段二**（P1–P11） | 运行时与性能：配置/线程、Metrics、复用池、评分与 TM 缓存、协议与恢复、GPU 实验 | 裁剪自 `INFERENCE_ENGINEERING_PLAN.md`，见第 4 节 | 阶段一完成且真实 case 对比通过 |

阶段一包含三类性质不同的工作，验收方式**不同**，不得用一句话概括（v1 的错误）：

| 类型 | 例子 | 验收方式 |
|---|---|---|
| A 等价重构 | argparse、重复解析收敛、路径参数化、TXT 读写归一 | 合法输入下行为与旧实现一致 |
| B 明确错误修复 | 多链残基对齐（S4a）、复合物域链号（S4b） | **允许结果变化**，用构造案例证明修正 |
| C 失败策略变化 | worker 失败不再变低分、丢链要报、>容量要报错 | 断言失败被正确报告，不再静默降级 |

### 0.2 维护规则【决策】

1. 每次修改本方案必须同时更新：顶部版本号、`修订记录` 章节、并在回复中给出**修改清单**与 **diff 摘要**。
2. 阶段一以本文为准；`INFERENCE_ENGINEERING_PLAN.md` 只覆盖阶段二，其顶部已加注说明并保留简短修订记录。
3. 事实分级标注不得删除；实施中把【待验证】变成【已核实】时，须在同一提交里更新对应标注与证据。

### 0.3 数据管理【决策】

- 真实输入（`test/`、`/xiangyux/test_data/**`）**不随代码提交**；`test/` 目前未被 Git 跟踪【已核实】。
- 入库的只有：测试代码、fixture 生成器、输入清单（`tests/cases/*.md`）、精简对比报告（`tests/reports/*.md`）。
- `.gitignore` 忽略：真实数据软链接、运行输出、临时基线目录；**不整体忽略测试数据目录**（后续小型固定 fixture 仍需入库）。

---

## 修订记录

### v4.10 → v4.11（老卡收口：口径校正与接口定稿）

> 本版只做两件事：**校正历史表述**（让每条结论与它当时的计时范围/基线一致）与**定稿 O6/P4/P5 的接口**（避免实现里留下隐含分支）。
> 不新增实验阶段；新任务卡 T00–T10 到此为止。接口细节见 **附录 D**；实施分支 `feat/old-card-closeout`（从 `552d3e9` 起，保留 T01 修复与 T05–T09 可选件）。

| # | 位置 | v4.10 的状态 | v4.11 的结论 |
|---|---|---|---|
| 90 | 基线口径【决策】 | 第 48 条只说"两份基线" | 三份并存、用途分开：**老卡原始基线**（T01 前）`2497fefc…`/`dbc937ef…` 只用于 S0–P3 的历史验收；**当前默认路径基线**（T01 后）`76638d0f…`/`bd281f40…` 用于本轮修改前后对比；**O6 后新基线** `tests/cases/baseline_after_o6.md5` 用于 P4/P5 等价性。老卡各步骤今后**不得**引用 T01 之后的基线 |
| 91 | 第 37 条措辞【已核实】 | "未归因时间从 260 s 降到 1.13 s" | 改为 **"计时覆盖完善"**：`final_select`（267.6 s）原先未埋点，补埋点后未归因降到 1.13 s。这是**口径修复，不是加速**；任何卡都不得把它计入收益 |
| 92 | 第 38/161 条的绝对数字【已核实】 | "postprocess 485 s > forward 375 s > 写盘 92 s" | 该组数字取自 **P2 时期未限线程**的配置；P1 限线程后同一 case 实测 postprocess ≈127 s、forward ≈331 s（T02/T09）。引用时必须写明配置 |
| 93 | P3 口径【已核实】 | "整个运行一个池" | 改为 **"已接入的主拟合路径复用一个池"**。已接入：`fitting/pipeline.py::_batch_cc`、`fitting/local_optimizer.py`、`assembly/orchestrator.py::_pre_screen_cc`、`core/similarity.py::prefill_tm_cache`。**未接入**：`assembly/homo_chain_refine.py`（5 处 `pool.map`，Step4/Step5）、`assembly/refine/chain_enumerator.py`、`sampling/extract_points/VoxEM.py` |
| 94 | 第 35 条的"P4 上限约 4%"【决策】 | 预估值 | **删除数字**：P4 只承诺"减少重复读取与解析"，收益以实测为准，不做倍数承诺 |
| 95 | O6 定位【已核实】 | 开放项"需你定" | 升格为**完整流水线可重复性问题**并已定位机制：客户端按"本轮新出现的 `pred_*.pdb`"凑 `batch_size` 批（`--batch-size` 默认 10、每轮 `sleep(2.5)`），而候选文件由服务端"每掩码最优改名 + 删除其余"产生 → 顺序取决于写盘/改名时机；`_PARENET_DONE` 写在 `finally` 里，失败也写 done。接口定稿见附录 D |
| 96 | 验收口径【决策】 | 第 5 节泛化"先微基准再真实 case" | 本轮真实运行共 **7 次**（O6×4：固定配置连续两次、`--num-processes 1`、`tail=on`；P4×1；P5×1；**第 7 次为暖缓存重复运行**，复用第六次的持久 TM 缓存以验证跨运行命中与结果重复性），纯运行 **≈2–2.5 h**（不按 17 min/次统一估算：`--num-processes 1` 明显更慢）。旧的 `out_t10_off` 只能作为**历史行为**证据，**不得**用于本轮速度归因。同一提交内不机械重复同一份微基准 |
| 97 | 待核查项【已核实】 | — | ① `mask_data_list` 的顺序**原先没有依据**：`utils.find_mask_files` 直接返回 `glob` 的目录顺序 → 已改为**按名排序**（候选 id 的"稳定生成顺序"以此为前提）；② 随机数：池内 worker（`_density_copy_worker`）**不用随机数**；父进程唯一的随机消费者是局部优化的回退（`local_optimizer.ScipyFitter` 的随机重启初值），而父进程原先**从未设种子**（NumPy 由 OS 熵初始化）→ 已用 `apply_seed(RuntimeConfig.seed)` 固定。`test/1` 的历史运行里该回退**从未触发**（日志 0 次），故这是消除潜在漂移，不是已发生的差异 |
| 98 | O6 实施与验收【已完成】 | 附录 D 接口定稿 | 代码：`fitting/candidate_ledger.py`（新）、`fitting/candidate_consumer.py`（新）、`fitting/pipeline.py`、`fitting/demo_mask.py`、`fitting/parenet_client.py`、`fitting/utils.py`、`runtime/config.py`（`apply_seed`）；单测 14 项；全量 201 项通过。**实跑验收（4 次）**：默认配置连续两次（1248/1260 s）、`--num-processes 1`（2009 s）、`tail_pipeline=true`（1199 s）→ **三个 CIF md5、三条决策、候选台账（24/28/115，id 连续，全 ok，end=ok）与消费序列 sha1（`de02d03105ebff9f`）四次完全一致**，且与 O6 前的冻结基线相同 → 冻结 `tests/cases/baseline_after_o6.md5`（同值：**O6 未改变默认路径产物**）。**worker=1 与 worker=10 现在一致**，P3 时期登记的那条失败验收项闭环。报告 `tests/reports/2026-09-13_o6_candidate_order.md` |
| 99 | O6 的代价【已核实】 | — | 新顺序下同 case 墙钟 1074 → 1240 s（**+15%**）：链 B 在该顺序下没有候选达到早停阈值 → 消费完全部候选并执行 `final_select`（旧到达顺序恰好在更早批次命中）。**产物完全相同**，差别只是"用多少候选换到同一结果"；单 case 观察，不推广为"O6 必然更慢" |

### v4.9 → v4.10（T10 完整回归与交付）

| # | 位置 | v4.9 的状态 | v4.10 的结论 |
|---|---|---|---|
| 83 | 任务卡 T10 | 待实施 | **已完成**（无代码改动，纯回归/判定/文档）：固定 manifest 五配置逐项复测 + 交叉复测（A/B/C 交替各两轮、F 几何缓存 off 两轮）+ 端到端 `test/1` 五次（off×2、on×3，含顺序反转与重复）；报告 `tests/reports/2026-09-13_t10_full_regression.md` |
| 84 | 配准层面等价【已核实】 | T05–T09 各自的开关等价 | **一个矩阵里同时成立**：A 默认 / B `chunk=64` / C `tail=on` / D 两者 / F 几何缓存 off 的 **72/72 预测逐位一致**（同一聚合 md5 `f6c942de…`）；E（`split` + 编码缓存）**改变结果**：best overlap 0.3268/0.0527/0.058 → **0.3283/0.0426/0.0736**，且没有速度收益 → 明确不推荐用于交付路径 |
| 85 | 微基准的可分辨度【已核实】 | T09 用微基准判定 tail"慢 9.5%" | **该判定无效**：交叉复测里 tail 反而"快 3.2%"，而**同配置两轮之差可达 2.5 s**（A1 41.67 vs A2 39.21）→ 微基准（单客户端 72 次配准）不足以判定 tail 的收益，只适合判**等价**（逐位）与显存；收益判定以 10 worker 的真实运行为准 |
| 86 | T09 的"md5 一致"【已核实·修正】 | T09 报告写"on 轮三个 CIF md5 与冻结基线完全一致" | **不可复现**：on 共 3 次运行 → 1 次复现基线、**2 次给出另一个产物**（`76638d0f…`→`51009d69…`、`bd281f40…`→`1ca5f6f9…`），那 2 次彼此逐位一致（CC 0.4189/0.4230/0.4431 vs 基线 0.4192/0.4235/0.4430）；已在该报告加"补遗" |
| 87 | 首个偏差的定位【已核实】 | O6 只登记"worker 数会改变结果" | **机制定位**：`fitting/pipeline.py` 的候选消费循环按"**新出现的文件**"成批评估（`sorted(glob('pred_*.pdb'))` 取差集、`len(new_files) >= 10` 才评估、每轮 `sleep(2.5)`）→ **文件出现时机决定批次切分 → 决定候选枚举顺序**。证据：链 A 的 24 个 `pred_*.pdb` 两次运行**逐位相同**，但 `0.301508` 在 off 是候选 `#8`、在 on 是 `#11`；最终 `results/pred_A_1.pdb` 不同 → 链 B 掩码差 2 个点 → `cc 0.4235→0.4230` → 最终 CIF 不同。尾部流水线改变了写盘时机，正好扰动这个边界 |
| 88 | 默认项选择【决策】 | T07/T08/T09 各自"默认不变" | **T10 统一确认：所有默认项一律不提升**——`geometry_cache_mb=512`（F 轮 +0.8%，噪声内）、`inference_mode=joint` + 框架 TF32、`encoding_cache_mb=256`（仅 split 生效）、`hypothesis_chunk=0`（64 为已验证可选：逐位一致 + 峰值显存 −15.6% + 微基准 −5.4% 两轮同向，但端到端未验证）、`tail_pipeline=false`（−9% 墙钟但改变产物）。交付物是**可回退的可选配置**，不是新默认 |
| 89 | 规格外结论：O6 未收口 | 开放项 O6"需你定" | 本卡给出机制但**未改代码**：要让结果对任何时序扰动都逐位可复现，需把候选枚举改成"请求完成后按稳定键整体排序"，那会**改变现有基线**（现在 off 路径 10+ 次运行都复现冻结基线）→ 属独立卡片，需重新冻结基线后再做 |

### v4.8 → v4.9（T09 有界 CPU 尾部流水线）

| # | 位置 | v4.8 的状态 | v4.9 的结论 |
|---|---|---|---|
| 76 | 任务卡 T09 | 待实施 | **已完成**：`runtime/tail_pipeline.py::TailPipeline`（单 worker、深度 1、FIFO、幂等关闭、异常显式抛出、关闭时就地执行）；`process_single_pair` 的"后处理 + 写盘"抽成尾部任务提交；`run_inference` 在每个消费点前 `wait()`；配置/CLI/benchmark 开关（**默认 False = 原路径**） |
| 77 | 验收口径 | — | 单测 8 项（顺序/深度/开关等价/异常传播/释放/上下文管理器）+ 全量 184 项通过；开关等价：微基准四轮 off/on **72/72 预测聚合 md5 相同**（`f6c942de…`）、真实运行 A/B 三个 CIF md5 与全部决策逐条一致；报告 `tests/reports/2026-09-13_t09_bounded_pipeline.md` |
| 78 | 偏差记录 | — | 首次 T09 微基准 `--tail-pipeline` 全部失败：`_record_timing(..., stage=stage)` 与形参 `stage` 重名（`TypeError: got multiple values`）→ 改为字段名 `where=`。同 T07 的 `inference_mode=None`，属"埋点/负载契约"类缺陷，由开启开关后的实跑暴露 |
| 79 | 与 T07/T08 的机会对照 | — | 服务端 CPU 尾部（本卡 off 轮 `postprocess` 127.8 s + `write_pred` 93.3 s ≈ 20% 墙钟）是当前最大的单块 CPU 开销；T09 目标即把它与下一次配准的 GPU 工作重叠 |
| 80 | T09 实测【已核实】 | 预期"CPU 准备与 GPU 执行重叠" | **端到端（test/1，10 worker，同配置）1092 → 994 s（−9.0%）**，`pipeline_total` 1082.95 → 985.19 s，产物逐位一致；`server_tail_wait` 17.13 s / 尾部 270.6 s → **94% 被重叠**（167 次消费点等待，均值 103 ms）；代价：`server_forward` 331.6 → 416.8 s（+25%）、`postprocess` +30%、`write_pred` +12%（与常驻尾部线程争 CPU） |
| 81 | 新事实：收益依赖并发【已核实】 | "重叠一定更快" | **单客户端微基准里 on 反而慢 9.5%**（四轮交叉：off 35.88/36.98 s vs on 40.09/39.70 s，72/72 预测 md5 相同）。原因：单客户端时 GPU 不被压满，重叠没有可换取的等待，只剩 CPU/GIL 争抢 → 本卡**不提升默认**（与 T07/T08 一致：交付已验证、可回退的可选配置） |
| 82 | T10 待办 | — | 用固定 manifest 复测端到端 A/B（现有 −9.0% 为单样本，历史同配置波动带 1054–1097 s），并据此决定是否把 `tail_pipeline` 提升为默认；同时记录重叠比例与 `server_forward` 膨胀，评估 O6（worker 数敏感性）→ **已由 v4.10 第 86–88 条回答：速度收益复现（−8.4…−9.1%），但产物不稳定 → 不提升默认** |

### v4.7 → v4.8（T08 位姿假设评分分块）

| # | 位置 | v4.7 的状态 | v4.8 的结论 |
|---|---|---|---|
| 72 | 任务卡 T08 | 待实施 | **已完成**：热点 = LGR/HypothesisProposer 的两处整批假设评分；**不修改共享 pareconv**，用子类覆盖单个方法 + `select_best_hypothesis()` 分块；**chunk=64 与 chunk=0 的 72/72 预测哈希逐位一致**，峰值 `max_allocated` 1194 → **1008 MiB（−15.6%）**、`max_reserved` −31%，墙钟 −3…−7%；报告 `tests/reports/2026-09-13_t08_hypothesis_chunking.md` |
| 73 | 默认配置 | `hypothesis_chunk` 无此配置 | 新增 `RuntimeConfig.hypothesis_chunk`，**默认 0（原路径，任务卡要求）**；64 作为已验证可选配置（`--hypothesis-chunk 64`），是否提升默认由 T10 复测决定 |
| 74 | 新事实：分块大小必须逐位验证 | T06 只证明"TF32 依赖张量形状" | **chunk=1 结果会变**（24/72 哈希一致、target 0/2 的 overlap 改变）且慢 63%：1 元素批次的 `apply_transform` 在 TF32 下走不同内核；关闭 TF32 时 chunk=1 与整批逐位一致（单测）。即"块越小越安全"不成立 |
| 75 | 上游包装纪律 | T06 起保留 `forward` 对照 | 本卡的子类复制了上游两个方法体（逐行一致，唯一差别是分块调用）；上游改动必须同步并重跑 `tests/test_hypothesis_chunking.py`（该测试直接比较分块与整批输出） |

### v4.6 → v4.7（T07 精确 scale 源编码缓存）

| # | 位置 | v4.6 的状态 | v4.7 的结论 |
|---|---|---|---|
| 69 | 任务卡 T07 | 待实施 | **已完成**：精确 scale 源编码缓存（GPU 256 MiB、只缓存源侧、超预算不缓存、无磁盘持久化）；缓存开/关结果**逐位一致**（微基准 72/72 哈希；真实运行与 split 无缓存运行 **CIF md5 完全相同**）；真实运行源编码命中 **2335/2338 = 99.6%**；模型侧耗时 183 → **140 ms/次**（split 无缓存 → 有缓存，−24%）；报告 `tests/reports/2026-09-13_t07_encoding_cache.md` |
| 70 | 默认配置 | joint + 框架默认精度（T06 结论） | **维持不变（依实测）**：真实运行里 split+缓存 140.1 ms/次 落在 joint 的波动带（128–140 ms）内、墙钟 1096.6 s vs 1054 s（+4%，噪声内）→ 不提升默认，交付 `--inference-mode split --encoding-cache-mb 256` 这一已验证可回退配置；微基准（大目标 + 同 scale 复用 12 次）显示 −17…−20%，说明收益高度依赖掩码规模与复用率 → 由 T10 用固定 manifest 复测定论 |
| 71 | 偏差记录 | — | 首次 T07 真实运行**全部请求失败**（`inference_mode=None` 写进请求 JSON，被服务端判为非法值 → 0 组件、50 s 结束）：已在客户端"None 时不写该键"+ 服务端 `or 默认值` 两侧修复并重跑。该缺陷由端到端运行暴露，单元测试未覆盖请求负载契约 |

### v4.5 → v4.6（T06 单侧编码与双侧配准）

| # | 位置 | v4.5 的状态 | v4.6 的结论 |
|---|---|---|---|
| 64 | 任务卡 T06 | 待实施 | **已完成**：`encode_cloud()`/`register_pair()`/`EncodedCloud` + 兼容对照 `forward()`；真实输入逐层对照位姿 max\|Δ\|=3.81e-06、候选索引/掩码/原始点 10/10 逐位一致；报告 `tests/reports/2026-09-13_t06_encoding_split.md` |
| 65 | 精度模式（新事实） | 未登记：框架默认 TF32 打开 | **TF32 使数值依赖张量形状**：同一拆分在 TF32 下位姿差 0.127（特征相对差 3.4%），关闭后 3.81e-06；关闭**没有可测减速**。新增 `inference_mode`（joint 默认 / split）+ `allow_tf32`（None = 跟随模式）+ `effective_allow_tf32()`：`split` + TF32 属非法组合，构造配置即报错 |
| 66 | 默认行为与基线 | `baseline_after_t01_fix.md5`（joint + TF32 打开） | **不变**：默认仍是 joint + 框架默认精度，实测与 T05 基线 **72/72 预测哈希逐位一致**，旧 CIF 基线继续有效；拆分路径为已验证的可选路径（TF32 off），若 T07 决定切换默认，需同时冻结新基线（端到端坐标漂移 0.151 Å） |
| 67 | T07 预期 | "固定 src 只编码一次"，收益上界 ~10% | **修正**：拆分把 backbone 由 1.19 → 2.33 s/36 次（≈×2，小批量效率损失），源编码 ≈45 ms/次（2338 次 ≈105 s），同时多付 transformer/拆分开销 ≈40–70 s → **净收益必须按真实运行实测**，不能按阶段外推 |
| 68 | 判据口径 | T03–T05 用 md5 逐位判等 | T06 起：位姿/候选索引/几何用逐位 + 1e-3 判据；`pred_*.pdb`/CIF 因 `%.3f` 舍入与流水线放大改用**坐标容差**（`tools/compare_pred_pdbs.py`），同代码重复运行仍要求 0 Å |

### v4.4 → v4.5（T05 有界 CPU 几何缓存）

| # | 位置 | v4.4 的状态 | v4.5 的结论 |
|---|---|---|---|
| 60 | 任务卡 T05 | 待实施 | **已完成**：`ByteLruCache`（按字节计费、0 = 关闭、超容量不缓存）+ `GeometryCache`/`acquire_geometry`（缓存开关同一份构建代码，只缓存确定性采样）；缓存由服务进程拥有、跨请求复用；四轮 off/on 交叉 A/B **72/72 预测哈希一致**，命中率 140/144 = 97.2%，几何阶段 0.44 → 0.13 s（每 72 次配准），缓存占用 12.1 MiB/512 MiB；报告 `tests/reports/2026-09-13_t05_geometry_cache.md` |
| 61 | 收益口径 | T02 估计"源侧可复用上界 10.4%" | **T05 只占其中很小一块**：T04 之后单侧几何重建约 3 ms，命中省下 4.3 ms/次（两侧合计），真实运行约 1%；T05 的意义是把源几何变成"按内容寻址、有界计费、可跨请求复用"的单元，为 T06/T07 的源**编码**缓存铺路 |
| 62 | 配置口径 | `RuntimeConfig` 覆盖线程/种子/池启动方式 | 新增 `geometry_cache_mb`（默认 512，0 关闭），由 `run_pipeline` 转交 `parenet_client.configure_geometry_cache`；服务已在运行时只告警不重启（不打断在飞请求）。常量放在 `runtime/config.py` 以保持该模块"无重依赖"（P1 两段式导入） |
| 63 | 目标侧缓存 | 任务卡表述"目标仍独立构建" | **实测澄清**：目标几何按**内容 key** 缓存，只在同一掩码被重复请求时命中（不同掩码之间不复用）；本流程每个掩码请求 12 次 → 命中率高；若每次掩码只请求一次，目标命中率退化为 0，源不变 |

### v4.3 → v4.4（T04 单侧几何拆分）

| # | 位置 | v4.3 的状态 | v4.4 的结论 |
|---|---|---|---|
| 56 | 任务卡 T04 | 待实施 | **已完成**：新增 `cloud_encoding.py`（`CloudGeometry`/`NodePartition` + 构建 + 联合适配 + 哨兵规则 + 指纹）；真实输入与 3 组合成输入逐阶段**按位**一致（含两侧节点分区）；微基准 72/72 预测哈希一致；几何阶段 1.35 → 0.21 s，但适配层 +0.55 s，总墙钟 −1.7%（噪声量级，不作加速结论）；报告 `tests/reports/2026-09-13_t04_geometry_split.md` |
| 57 | T02 遗留"标签 vs 生效配置" | 只记录不解释 | **已解释并显式化**：pareconv 从模块级全局变量取配置、本仓库无人设置 → 恒为 `config 0 + voxel`；T04 起由 `effective_sampling_config()` 读出传参并写入 `server_pair_info`（`effective_sampling`/`voxel_sizes`），生产行为不变 |
| 58 | 隐性显存操作 | 未登记 | 旧入口 `registration_collate_fn_stack_mode` 内部含 `torch.cuda.empty_cache()`（共享项目代码，不可改）→ 单侧路径不再调用它；T03 的显存结论不受影响，T05/T07 的命中率统计以此路径为基线 |
| 59 | 后续卡 T06 | 单侧编码拆分 | 单侧节点分区已随几何记录并可缓存，但**尚未接线**给 `forward`（本卡不给模型加第二条代码路径）；T06 负责去掉重复计算 |

### v4.2 → v4.3（T03 推理内存与中间回传）

| # | 位置 | v4.2 的状态 | v4.3 的结论 |
|---|---|---|---|
| 52 | 任务卡 T03 | 待实施 | **已完成**：`@torch.no_grad()` 覆盖推理全路径、输出白名单（`INFERENCE_OUTPUT_FIELDS`）、删除递归 `release_cuda` 与逐对 `empty_cache`、GT 对应仅训练分支计算；A/B（`00a97e8`+探针树 vs 本卡，冷/暖各 36 对）72/72 预测哈希与全部 overlap 一致；`max_allocated` **4.85 GiB → 1.15 GiB**、前向返回瞬间 `allocated` 3.75 GiB → 23.5 MiB、`server_forward` −3.82 s（冷）；报告 `tests/reports/2026-09-13_t03_inference_memory.md` |
| 53 | T02 遗留判断 | "forward 内约 19% 主机侧开销 + 4.85 GiB 峰值值得整理" | **已定位并处理**：峰值主要来自 autograd 计算图与全量 `output_dict` 驻留（非算子本身）；删除的显式工作实测 2.38 s/36 对（`empty_cache` 1.43 + `release_cuda` 0.75 + GT 对应 0.20），`no_grad` 另使模型内 CUDA 阶段合计下降 2.74 s/36 对 |
| 54 | 显存口径 | T07 GPU 缓存预算 256 MiB ≈ 峰值 5% | 峰值降到 1.15 GiB 后该预算约占 22%；`reserved` 因不再 `empty_cache` 停留在高水位（示例 1.5–3.0 GiB），属可复用块而非泄漏（`allocated` 平在 ~12 MiB） |
| 55 | 后续卡 | T04 单侧几何拆分 | 不变，仍为下一项；`no_grad`/白名单不影响 T04 的几何与索引契约 |

### v4.1 → v4.2（T02 热点与可复用比例）

| # | 位置 | v4.1 的状态 | v4.2 的结论 |
|---|---|---|---|
| 49 | 任务卡 T02 | 待实施 | **已完成**：CUDA event 三阶段 + 服务端细分 + 冷/暖各 36 对；**源侧可复用上界 10.4%**（几何+分区 2.8%）、几何重复率 98.6%、scale 命中机会 95.8%（基准口径，生产需复测）、`model_lgr` 为模型内最大头（占 68%）、峰值 `max_allocated` 4.85 GiB；报告 `tests/reports/2026-09-13_t02_hotspots_and_reuse.md` |
| 50 | 口径 | — | 自查发现并修正 `server_postprocess` 与 `server_forward` 重叠相加的错误计时（偏差处理项） |
| 51 | 后续卡预期 | — | T05/T07 收益上界约 10%，且 scale 命中率须按真实掩码序列复测；T08 应优先关注 `model_lgr`；T03 关注 forward 内 ~19% 主机侧开销与 4.85 GiB 峰值 |

### v4.0 → v4.1（T01 坐标契约与修复）

| # | 位置 | v4.0 的状态 | v4.1 的结论 |
|---|---|---|---|
| 47 | 任务卡 T01 | 待实施 | **已完成**：确认并修复"位姿在点云质心系、写出绕原子质心"的坐标错误（偏移 `(I-R)(c_atom-c_src)`）；8 项契约测试；修复后基线冻结 `tests/cases/baseline_after_t01_fix.md5`（旧基线被取代）；组件结构不变、CC 差 ≤0.0005 |
| 48 | 基线口径 | 单一冻结基线 | 现在有两份：`baseline_final_md5.txt`（T01 之前）与 `baseline_after_t01_fix.md5`（当前有效）。后续卡一律与后者比较 |

### v3.2 → v4.0（转入掩码配准优化任务卡：T00 完成）

| # | 位置 | v3.2 的状态 | v4.0 的进展 |
|---|---|---|---|
| 44 | 总体 | 阶段二 P3 收口 | 老任务卡（时间账 → P3 → 依测量定优先级）已全部收口；转入 `MASK_REGISTRATION_OPTIMIZATION_TASKS.md` 的 T00–T10 |
| 45 | 任务卡 T00 | — | **已完成**：manifest + 依赖来源实测（pareconv 来自共享项目、未用仓库副本；GPU 为 MIG 切片）；重放两次成功；报告 `tests/reports/2026-09-13_t00_registration_baseline.md` |
| 46 | 后续卡约束 | — | T04/T06/T07 若需改 `pareconv` 内部，必须先解决"实际加载别处"；所有对比需冷/暖重复并记录负载（同输入两次差 29%%） |

### v3.1 → v3.2（P3 复用池完成）

| # | 位置 | v3.1 的状态 | v3.2 的结论 |
|---|---|---|---|
| 41 | §4.1 P3 | 待实施 | **已完成**：池创建 79 → 1（串行 0）；同配置（10 worker）三个 CIF md5 与**当时的冻结基线**（T01 前 `2497fefc…`/`dbc937ef…`）逐字节一致；报告 `tests/reports/2026-09-13_p3_shared_pool_acceptance.md`。**v4.11 校正**：只覆盖已接入的主拟合路径（见第 93 条），未接入位置已列出 |
| 42 | §3 P3 验收项 | "worker=1 与当前配置返回相同排序与分数" | **实测不成立**：1 worker 会改变最终结果；用冻结基线代码复现同 md5 → 属既有配置敏感性（不是 P3 引入），登记为开放项 **O6**（v4.11 第 95 条：已定位机制，本分支修复） |
| 43 | §4.2 P4/P5 | 上限须重估、P5 暂缓 | 维持：先统计全部评分调用点再评估 P4；P5 暂缓；服务端 CPU 后处理与写盘为当前最高杠杆 |

### v3.0 → v3.1（P2 细化时间账 + 优先级修订）

| # | 位置 | v3.0 的状态 | v3.1 的结论 |
|---|---|---|---|
| 37 | §4.1 P2 | 已实施，约 260 s 未归因 | **口径修复（非加速）**：`final_select` 267.6 s 原先未埋点，补埋点后未归因 260 s → 1.13 s —— 属**计时覆盖完善**，不计入任何收益（v4.11 第 91 条）。报告 `tests/reports/2026-09-13_p2fine_time_account.md` |
| 38 | §4.2 P3–P5 优先级 | P3 优先、P4 上限约 4%、P5 暂缓 | **修订**：服务端 CPU 后处理 > forward > 写盘 是最高杠杆（当时数字 485/375/92 s 取自 **未限线程** 的 P2 配置；限线程后为 ≈127/331/93 s，见 v4.11 第 92 条）；P3 建池成本仅 4.34 s（0.4%）仍做但收益有限；**P4 上限须重估**（final_select 内也有评分）；P5 继续暂缓 |
| 39 | §4.1 P1 结论措辞 | "约 1.64 倍" | 收紧为"**本次配对运行**观察到约 1.64 倍提升，输出一致"，不作普遍收益承诺 |
| 40 | `timed_pool` 口径 | — | 自查发现首版把 `pool_close` 记成池存活时长（与上层重叠）；已改为只计 close+join |

### v2.9 → v3.0（P2 完成 + 依测量的优先级）

| # | 位置 | v2.9 的状态 | v3.0 的结论 |
|---|---|---|---|
| 34 | §4.1 P2 | 待实施 | **已完成**：8 个阶段 + 拟合内部 3 项全部埋点；`test/1` 两组实测（报告 `tests/reports/2026-09-13_p2_staged_metrics_and_thread_effect.md`） |
| 35 | §4.2 P3/P4/P5 顺序 | 按原文档顺序 P3→P4→P5 | **改按测量**：gpu_wait 46–52% > local_optimize 18–21% > cc_batch 3.6–4.8% > tm_prefill ≈0。故 **P3 优先**；**P4 上限不设数字**（v4.11 第 94 条：以实测为准）；**P5 在本 case 可推迟**；并新增"细化 `fit_request` 内约 260 s 未归因时间 + GPU 等待"这一调查项（收益最大） |
| 36 | §4.1 P1 | 已实施 | **实测确认**：1 线程默认让 `test/1` 从 1680.8 s 降到 1023.7 s（≈1.6×），且三个最终 CIF md5 与冻结基线一致 |

### v2.8 → v2.9（P1 运行配置与线程控制）

| # | 位置 | v2.8 的状态 | v2.9 的实施 |
|---|---|---|---|
| 32 | §4.1 P1 | 待实施 | **已实施**：`RuntimeConfig` + 两段式入口 + `--runtime-config`；实测确认"导入前设 env 才生效"（默认 128/112 → 应用后为设定值）；测试 7 项 |
| 33 | §4.1 P2 | 待实施 | 下一步；P1 之前记录的 956 s 属"现有配置"基线，**不得**把线程调整的收益算到 P3–P5 头上 |

### v2.7 → v2.8（O5 收口）

| # | 位置 | v2.7 的状态 | v2.8 的结论 |
|---|---|---|---|
| 30 | 附录 C O5 | 残留分支待查可达性 | **不可达**：`refine_step._backfill_chains_as_domains` 跳过 `type != "chain"`，复合物为 `type="complex"`；真正消费位置已在 R1 修复并真实验证 |
| 31 | §1.6 V3 | 部分销项 | **已销项**：引入点为中间产物，可达的最终输出路径均恢复真链号 |

### v2.6 → v2.7（O4 空结果输出契约）

| # | 位置 | v2.6 的状态 | v2.7 的处理 |
|---|---|---|---|
| 28 | 附录 C 开放项 O4 | 无组件被接受时仍写出 0 链、不可解析的 `assembled_complex*.cif` | **已修复**：不写出 + 清理旧产物 + 返回 `None` + 摘要状态 + Step4 门控；报告 `tests/reports/2026-09-13_o4_empty_output_contract.md` |
| 29 | §5.4 通过标准 | 未提"无结果"情形的验收 | 补充：无接受组件时必须满足"不产生结构文件、摘要可读、Step4/5 跳过" |

### v2.5 → v2.6（R1：域链合并丢链修复）

| # | 位置 | v2.5 的状态 | v2.6 的处理 |
|---|---|---|---|
| 25 | §1.2 同源问题 | 未记录域链合并丢链 | 新增 F：复合物多链域合并必然失败 → **已修复**（补 `is_complex` + 链号恢复），4 项定向测试 + 同配置真实运行验证 |
| 26 | §1.3 / O5 | S4b 残留分支待查可达性 | **不可达**：`_backfill_chains_as_domains` 只处理 `type=="chain"`；真正消费位置是 `assemble_domain_chains`，已在 R1 中修复并验证真链号恢复 |
| 27 | §1.6 V3 | 部分销项 | 维持"引入点为中间产物"，并把可达消费者路径的结论写入 R1 报告 |

### v2.4 → v2.5（S4b 定向探针）

| # | 位置 | v2.4 的状态 | v2.5 的结论 |
|---|---|---|---|
| 23 | §1.3 S4b / §1.6 V3 | 仅静态定位，待端到端确认 | **已执行三次定向探针**（派生 `Q/R` 复合物 CIF + `--complex-domain-opt`）：引入点确认，`final_results` 出口**无泄漏**；残留分支未证伪，按约定未改代码（报告 `tests/reports/2026-09-12_s4b_chain_space_probe.md`） |
| 24 | 附录 C | 无空复合物相关条目 | 新增开放项 **O4**（无接受组件时写出 0 链不可解析 CIF，既有行为）与 **O5**（S4b 残留分支） |

### v2.3 → v2.4（S7 收尾）

| # | 位置 | v2.3 的状态 | v2.4 的结论 |
|---|---|---|---|
| 19 | §1.6 V4 | `test/1` 完整耗时未知 | **已销项**：基线 967 s（≈16 min），新版 956 s |
| 20 | §5.4 通过标准 | 需真实 case 等价性证据 | **已达成**：`test/1` 三个最终 CIF + 摘要与固定基线逐字节一致（报告 `tests/reports/2026-09-12_test1_baseline_vs_new.md`） |
| 21 | §1.6 V5 / S4b | 复合物域优化待验证 | **未执行**：`6lu9` + `--complex-domain-opt` 属开放项（O1），S4b 仅完成静态定位 |
| 22 | §1.6 V2 | 数字占位链号下游 | 维持"部分销项"（DomainParser/domain split 未覆盖） |

### v2.2 → v2.3（S6 实施期间同步）

| # | 位置 | v2.2 的说法 | v2.3 的更正 |
|---|---|---|---|
| 16 | §3 S6 | `parenet/config.py` 的 `metadata_root` 保持仓库相对并加注释 | 已实施；`dataset_root` 直接删除（无读取者）；文件顶部加"推理不使用数据集配置"说明 |
| 17 | §3 S6 | `DomainParser.py` 改为列表参数 | 已实施并**实测**：真实链数据 `split_domains` 成功，PDB 产物与基线逐字节一致；域 TXT 仅第 0 行由 `2.0` 改为原文 `2.000000`（数值相同） |
| 18 | §5 数据管理 | 自研代码不得含服务器绝对路径 | 已由 `tests/test_portability.py` 静态断言固化（`pareconv_src/` 的 4 处 vendored 训练路径按约定保留、排除在断言外） |

### v2.1 → v2.2（S4a/S5 实施期间同步）

| # | 位置 | v2.1 的说法 | v2.2 的更正 |
|---|---|---|---|
| 13 | §1.6 V2 | 数字占位链号是否破坏下游"待实测" | **部分销项**：`read_structure` / `calculate_cc_mask` / USalign 实测可用（`tests/test_chain_ids.py`）；DomainParser/domain split 未单独覆盖，容量按原决定保留 62 |
| 14 | §1.2 A/B | 静默丢链与 worker 降级"待处理" | 已实施：跳过输入进 `skipped_inputs` 并写入运行摘要；两个 CC worker 失败改为带文件名抛错 |
| 15 | §1.1 #6 | 池的命名与容量 | 已实施：`logical_chain_ids()` / `pdb_placeholder_ids()` 两个函数；超容量 `ValueError`；删除"超过 52 走 CIF"的错误论断相关表述 |

### v2 → v2.1（S3 实施期间同步）

| # | 位置 | v2 的说法 | v2.1 的更正 |
|---|---|---|---|
| 11 | §1.1 #4 | 点云解析重复记为 **5 处** | 实施中新发现 2 处同类实现（`fitting/masker.py:_read_points_with_lines`、`assembly/orchestrator.py:_target_has_points`），已一并统一，共 **7 处**；`fitting/demo_mask.py:load_mask_points` 契约不同（容忍 `#` 注释头、返回体素坐标），**保留不合并**并在架构文档记录原因 |
| 12 | 附录 A【待验证 V1】 | 头部第 1 行轴序待实测 | 已销项：轴序**不可观测**（见附录 A）——`Sample` 会把盒子裁剪成立方并重设 origin，line1 恒为三数相等；契约改为 line1/2/4 原样保留、不解析 |

### v1 → v2


| # | 位置 | v1 的问题 | v2 的处理 |
|---|---|---|---|
| 1 | S4 | 只按 `(chain, resseq, icode)` 匹配，未处理"真链号 / 占位链号"两个空间；并错误声称"单链且编号唯一时结果不变" | 新增 §1.3 链号空间审计；接口加 `mob_chain_map`；映射必须取自原始 `chain_records`；补非恒等映射测试 |
| 2 | S3 | 接口只返回数组，却要求"原样回写"；float32 重格式化无法保住文本精度 | 改为数据记录（数组 + `sample`/`origin` + 原始 5 行头 + 原始数据行对）+ 两个写出函数；表头轴序标为【待验证】并补实测 |
| 3 | S2 | 放行 `contour=None` | `run_pipeline()` 必须收到有限数值；**不新增**自动 contour 功能路径 |
| 4 | S5 | 把占位池容量 62 降到 52，并声称"超过 52 走 CIF 路径" | 保留 62；两个池分别命名；补数字占位符下游实测；删除不存在的 CIF 通路说法 |
| 5 | §0/§5 | 笼统承诺"阶段一不改变任何语义"，与 S4 修复冲突 | 拆成 A/B/C 三类工作分别定验收 |
| 6 | S0/S7 | 基线到 S7 才构建，届时未提交改动已被提交，`git diff` 取不到 | S0 动手**之前**冻结基线快照 + 隔离副本；S7 只引用固定基线 |
| 7 | S1 | `parse_args()` 配 `nargs="*"` 会拒绝交错参数 | 改用 `parse_intermixed_args()`，并补交错参数测试 |
| 8 | §2 | 边界写死"五处"，`.get`/`hasattr` 表述过绝对 | 改为"输入 / 格式 / 进程 / 持久化边界各验证一次"；措辞调整 |
| 9 | 第 5 节 | 只用单链 PDB 集验证，不覆盖复合物与占位映射 | case 分三档（`test/1` 日常回归、`test_5kem` 较大回归、`fiting_lg/6lu9` 复合物域优化）+ 派生非恒等映射副本 |
| 10 | 全文 | 无版本维护约定 | 新增 §0.2 维护规则 |

---

## 1. 事实核对

### 1.1 用户列出的 7 条（全部【已核实】）

| # | 判断 | 结论 | 证据（文件:行） | 归属 |
|---|---|---|---|---|
| 1 | `main.py` 手写参数解析；缺文件/参数抛 IndexError/ValueError；未知选项静默忽略 | 属实 | `main.py:44-50` 未知 `--x` 直接跳过（带值时其值会落进 positional 造成错位）；`main.py:108` `find_files(dir,".mrc")[0]` 无 .mrc → IndexError；`main.py:110-111` + `core/io.py:123` 缺 `resolution.txt`/`contour_level.txt` → FileNotFoundError；`main.py:119` `float(positional[2])` → ValueError；`main.py:19-32` `_parse_float_opt` 缺值静默回默认 | S1 |
| 2 | `run_pipeline()` 进入体素化前不校验输入 | 属实 | `pipeline.py:47-231` 直接 `os.makedirs` + `setup_logging`；不检查 mrc 存在与后缀、结构列表非空、分辨率有效、contour 有限 | S2 |
| 3 | `sample_density_map()` 用 `os.system()` 字符串拼接 | 工作树已修，HEAD 未修 | HEAD：`cmd = f"{SAMPLE_BINARY} -a {mrc_path} ... > {sample_file}"` + `os.system`；工作树：`subprocess.run([...])` + 返回码 + 空输出检查。遗留：`sampling/extract_points/Sample_based_VoxEM.py:22`、`assembly/domain_parser/DomainParser.py:184,229,278`（`shell=True`） | S0 / S6 |
| 4 | `load_sample_points()` 依赖固定行号与奇偶行 | 属实且更严重 | `core/io.py:67-94` 用 `i % 2` 分奇偶；某行字段 <4 时**只跳过该侧**，points 与 normals 静默错位。同一格式在 **7 处**重复：`core/io.py:67`、`fitting/demo_mask.py:92`、`fitting/sw_mask.py:65`、`sampling/extract_points/Supporting.py:69`、`assembly/domain_parser/domain_pdb_txt.py:47`（后在库函数里 `sys.exit(1)`），以及 S3 实施中新发现的 `fitting/masker.py:_read_points_with_lines`、`assembly/orchestrator.py:_target_has_points`；`fitting/demo_mask.py:load_mask_points` 契约不同（容忍 `#` 注释头、返回体素坐标），保留不合并 | S3 |
| 5 | `align_by_resid()` 只用残基编号做键 | 属实 | `core/structure.py:113-144`：键 `res.get_id()[1]`，不含 chain 与 insertion code；`ref_ca` 为 dict，重复编号互相覆盖。调用点：`chain_fitter.py:172,222`、`refine_step.py:134`、`orchestrator.py:621` | S4a |
| 6 | `cif_to_pdb_placeholders()` 占位池含数字 | 属实 | `core/structure.py:86-87` 池为 A–Z+a–z+digits，`pool[i]` 超 62 链 → IndexError；同文件 `structure.py:42` 的 `chain_id_pool()` 是 A–Z/a–z/两字母，两套规则并存。调用点 `orchestrator.py:212-215` 外层 `except Exception: continue` | S5 |
| 7 | 硬编码服务器路径 | 属实（9 处） | `fitting/parenet/config.py:35` `/xiangyux/PARENet-main/data/demo`（目录不存在、全仓库无读取者 → 死配置）；`sw_mask.py:616,617,633`、`refine_energy.py:992` 在 `__main__`/示例内；`pareconv_src` 内 4 处为 vendored 训练路径（不动） | S6 |

### 1.2 同源问题（【已核实】，纳入阶段一）

| # | 问题 | 证据 | 处理 |
|---|---|---|---|
| A | 结构解析失败 / 找不到 txt 时静默丢链，装配在缺链情况下继续 | `orchestrator.py:196-199`、`214-215`、`220-221`、`224-225` | S5：warning 带原因 + 计数；全部不可用 → `RuntimeError`；部分成功要进 summary |
| B | 并行 worker 把异常伪装成"分数低" | `fitting/pipeline.py:40-44`（→ `cc=None`）、`orchestrator.py:787-790`（→ `0.0`） | S5：worker 返回失败标记，主进程汇总后 `raise` |
| C | 库函数里 `sys.exit(1)` | `domain_pdb_txt.py:107,110,113,148,159` | S3：改抛异常，`__main__` 统一转退出码 |
| D | 无 `tests/`；README 的 `example2` 不在仓库 | `ls tests` 不存在；`find /xiangyux -name example2` 无 | S7：新建 `tests/`；README 改写为 `<case_dir>` |
| E | vendored 第三方与自研代码同包（`pareconv_src`、`refine/`） | 目录结构 | 不搬迁，只在文档标明边界 |
| F | **复合物多链的域链合并必然失败并被静默丢链**：`assemble_domain_chains` 未传 `is_complex`，两条链的域（残基号各自从 1 起）并进同一条链 → 残基 ID 重复 → 整链丢弃 → 空复合物 | S4b probe3 日志 `merge_domains error: (' ', 1, ' ') defined twice` + `chain dropped`；`domain_assembler.py:44` 调用缺参 | **已修复（R1）**：补传 `is_complex` + `chain_map` 恢复真链号；见 `tests/reports/2026-09-13_r1_complex_domain_assembly_fix.md` |

### 1.3 链号空间审计（v2 新增，**S4 的核心**）

**【已核实】存在两个链号空间**：

| 空间 | 定义 | 出现位置 |
|---|---|---|
| **R**（真链号） | 源结构文件内的链号 | `read_chain_ids()`、`rec["chain_id"]`、`fitted_cif` 内容 |
| **P**（占位链号） | CIF → PDB 时的单字符占位 | `cif_to_pdb_placeholders()` 产物 `rec["pdb_file"]`、由其派生的域 PDB `drec["pdb_file"]` |

`fitted_cif` 恒为 R【已核实】：`orchestrator.py:563-569` —— 复合物 + `chain_map` 走 `write_structure_with_chain_map`（P→R）；单链走 `pdb_to_cif(..., chain_id=cid)`；PDB 复合物保留原链号。
域 PDB 为 P（CIF 输入时）【已核实】：`orchestrator.py:283-285`（复制 `rec["pdb_file"]` 后切分）、`orchestrator.py:319`（复合物走 `split_structure_to_chains(rec["pdb_file"])`）。

**四个调用点的空间表【已核实】**：

| 调用点 | ref | ref 空间 | mob | mob 空间 | 需要映射 |
|---|---|---|---|---|---|
| `chain_fitter.py:172` `try_domains_via_chain_pose` | `fitted_chain_pdb`（拟合 pose，由 `demo_mask.transform_pdb` 变换输入 PDB 生成，链号保留） | P/R | `drec["pdb_file"]` | 同 ref | 否 |
| `chain_fitter.py:222` `try_improve_chain_with_domains` | `fitted_pdb` | P/R | `drec["pdb_file"]` | 同 ref | 否 |
| `refine_step.py:134` `_backfill_chains_as_domains` | `rec["fitted_cif"]` | **R** | `drec["pdb_file"]` | **P** | **是** |
| `orchestrator.py:621` `_attach_domain_details` | `rec["fitted_cif"]` | **R** | `drec["pdb_file"]` | **P** | **是** |

**映射来源【决策】**：`accepted_chains` 里的是摘要记录，**没有** `chain_map`。两个调用点都必须从原始 `chain_records` 取：

- `refine_step._backfill_chains_as_domains` 已取得 `cr = _chain_record(orch, cid)`（`refine_step.py:116`）→ 用 `cr["chain_map"]`；
- `orchestrator._attach_domain_details` 需按组件 ID 从 `self.chain_records` 查找；
- 建议在 orchestrator 上提供一个明确的查找方法（如 `chain_record(cid)`），两处共用，**不复制整套映射状态**。

**【实施状态，S4a】** 已按上述实现：`AssemblyOrchestrator.chain_record(cid)` 是唯一查找入口，
`refine_step._backfill_chains_as_domains` 与 `orchestrator._attach_domain_details` 共用它；
`chain_fitter` 的两处调用（ref/mob 同空间）保持 `mob_chain_map=None`。
构造测试已覆盖：单链叠合、多链同编号（含旧逻辑 RMSD > 5 Å 的取证）、占位链号映射
（真 `Q/R` → 占位 `A/B`：无映射返回 False 且不写文件，有映射成功）、匹配不足、只读第一个 model。
**端到端复合物域优化验证仍属 S4b。**

**【S4b 结论，2026-09-12/13：三次定向探针，未观测到泄漏；未改代码】**

- 引入点**已确认**：复合物域优化路径的逐域微调产物 `work/chain_improve_<cid>/domain_*.cif` 带占位链号
  （真链号 `Q/R` → 占位 `A/B`），来自 `chain_fitter.py:237-240` 的 `chain_id_for_cif = src_cid`。
- 出口**已核实无泄漏**：`final_results` 全部产物链号均为真链号（`assembled_complex.cif` /
  `assembled_complex_all.cif` / `chains/complex_Q+R_01.cif` = `Q/R`）；`_accept_chain` 对复合物应用
  `chain_map` 完成恢复。
- **残留未覆盖分支（未证伪）**：`refine_step._backfill_chains_as_domains` 复制 `cr["domain_cifs"]`
  （占位标注）后由 Step 4 合并；触发需"链被接受且记录 domain_cifs + 发生过域拟合 + 有同源链"三者同时成立，
  本派生 case 未能构造（三次运行的 Step 4 分别因"无域拟合""无同源链"被门控）。
- 按约定**未修改 `chain_id_for_cif`**；证据、命令与复现条件见
  `tests/reports/2026-09-12_s4b_chain_space_probe.md`。
- 顺带发现（既有行为，非本阶段改动）：无组件被接受时写出的复数 CIF 为 0 链、Bio.PDB 无法解析，
  已列为开放项 O4。

**v1 的错误声明**：v1 写"单链且编号唯一时结果不变"不成立。反例【已核实】：`chain_B_2.cif` 真链号 B → 占位 A（`cif_to_pdb_placeholders` 按文件内顺序分配），`fitted_cif` 为 B，域 PDB 为 A → 严格按链号匹配将无交集。旧代码能对齐，纯粹因为旧匹配忽略链号。

**S4b 传播链【待验证】**（不得直接改 `chain_id_for_cif`）：

```text
域 CIF（可能是 P） → merge_domains → improved PDB → _accept_chain（恢复为 R） → 最终 CIF
```

要查清：(a) 哪些出口绕过了 `_accept_chain` 的恢复；(b) 疑似点 `chain_fitter.py:237-240`（`chain_id_for_cif = src_cid if (is_complex and src_cid) else cid`，`src_cid` 来自 `split_structure_to_chains(rec["pdb_file"])`，CIF 复合物时为 P）；(c) `complex_builder.build_complex` 直接合并 `fitted_cif` 的路径是否有 P 泄漏。**先补非恒等映射测试定位出口，再在确认的边界修复。**

### 1.4 对 `INFERENCE_ENGINEERING_PLAN.md` 的更正（v2 已同步修改该文件）

【决策】按约定执行：

1. 顶部加注：阶段一以本文 v2 为准；阶段二按本文第 4 节分档执行。
2. §4 表格中 `core/scoring.py`、`core/similarity.py` 标注为**"已存在，改造"**（两个文件确实已存在：147 行 / 97 行），避免被读成"新建文件"——v1 把它写成"与现状不符"属于指控过重，实际是**表格标签歧义**，后续步骤本身写的就是改造。
3. 该文档补充一段简短修订记录，两份方案不再互相冲突。

### 1.5 环境与基线事实【已核实】（2026-09-12 实测）

| 项 | 值 |
|---|---|
| 项目 | `/xiangyux/claude_c_work/demo_reg`（`X:\claude_c_work\demo_reg`），`main...origin/main`，HEAD `5b016f5` |
| 远端 | `git@github-zhouxglab:zhouxglab/CryoAssembly-preview.git` |
| 未提交 | `M protassem/fitting/pipeline.py`、`M protassem/sampling/sampler.py`、`?? protassem/core/performance.py`、`?? INFERENCE_ENGINEERING_PLAN.md`、`?? test/` |
| Python | `/root/miniconda3/envs/point/bin/python` = 3.8.20 |
| GPU | NVIDIA A800 80GB PCIe |
| 自研代码 | `protassem/`（除 `pareconv_src`）约 6.5k 行 |
| 采样 TXT 格式 | `Sample` 二进制实测：5 行头 + `(index x y z)`/`(vx vy vz d)` 交替；`index = x + y*nx + z*nx*ny`（见附录 A） |
| 日常回归 case | `test/1/`：`EMD-8436.mrc` 192×192×192、voxel 1.31 Å、28 MB；`chain_a_1.pdb` 链 A、`chain_b_2.pdb` 链 B，各 4596 原子 / 564 CA；`resolution.txt`=5.6；`contour_level.txt`=0.04；目录内无历史输出 |
| 缺失 | 无 `tests/`；`example2` 不存在；`/xiangyux/PARENet-main/data/demo` 不存在 |

### 1.6 待验证问题清单（实施中必须逐条销项）

| # | 问题 | 验证方式 | 归属 |
|---|---|---|---|
| V1 | 采样 TXT 头部第 1 行的轴序是 `nz ny nx` 还是 `nx ny nz`（v1 的写法是**推断**） | 用 6×8×10 非立方网格跑 `Sample`，读第 1 行 | S3 前置 |
| V2 | 数字占位链号是否真的破坏下游 | **部分销项**：`read_structure`/`calculate_cc_mask`/USalign 实测通过（`tests/test_chain_ids.py`）；DomainParser/domain split 未覆盖，容量保留 62 | S5 |
| V3 | S4b 传播链中哪个出口泄漏占位链号 | **已销项**：引入点是 `work/chain_improve_*` 中间产物；可达的最终输出路径（整链接受 / 域链组装）均恢复真链号，R1 修复后由真实运行验证 | S4b/R1 |
| V4 | `test/1` 完整流水线耗时与基线结果 | **已销项**：基线 967 s、新版 956 s，输出逐字节一致 | S7 |
| V5 | 复合物域优化分支（`--complex-domain-opt`）在默认参数下的触发条件与耗时 | 显式加该参数运行 | S7 |

### 1.7 实施决策清单【决策】

| 项 | 决定 |
|---|---|
| 分支与提交 | 新建 `feat/engineering-hardening`，每步单独提交，**不 push**；回退一律 `git revert` |
| 现有未提交改动 | 保留，整理为两次提交（S0a/S0b），不回退 |
| 基线 | S0 动手前冻结快照与隔离副本，S7 复用固定基线 |
| 边界校验 | 只在输入/格式/进程/持久化边界各验证一次；不新增通用校验模块 |
| CLI | `argparse` + `parse_intermixed_args()`，参数名与默认值不变 |
| TXT 读写 | 新增单一模块，替换 5 处解析 + 2 处写出；过滤器保原始行 |
| contour | `run_pipeline()` 必须收到有限数值；阶段一**不新增**自动 contour |
| 链号池 | 保留现有容量（62）；两个池分别命名；补下游实测 |
| `extract_points/` 三脚本 | 保留（README 未提及不作为删除依据），只修外部调用与统一解析接口 |
| `tools/compare_runs.py` | 新增；以清楚、完整、无重复为准，不设行数目标 |
| 测试数据 | 真实输入不提交；提交生成器、输入清单、精简报告 |
| 依赖 | 阶段一零新增依赖；阶段二若需 `threadpoolctl` 则显式登记版本 |

---

## 2. 设计约束

### 2.1 边界：各验证一次，其余不验证

| 边界 | 位置 | 校验内容 | 失败方式 |
|---|---|---|---|
| 输入 | `main.py`、`run_pipeline()` | 位置参数个数、文件存在与后缀、结构列表非空、`resolution > 0` 且有限、`contour` **有限数值**、`voxel_size > 0` | `parser.error`（退出码 2）/ `FileNotFoundError` / `ValueError`，消息含路径与期望 |
| 格式 | `core/points_txt.py`、`read_structure()` | 头部 5 行、数据成对、字段数、数值可解析、后缀支持 | `ValueError`，消息含 `文件:行号:期望 vs 实际` |
| 进程 | `sample_density_map()`、USalign 调用、PARENet 服务 | 可执行文件存在、返回码、stderr、非空产出 | `FileNotFoundError` / `RuntimeError`，附 stderr 尾部 |
| 持久化 | 后续新增的缓存 / checkpoint（阶段二） | 版本、指纹、完整性 | 阶段二定义 |

**非边界不验证**：内部函数参数、worker 入参、算法中间量。不使用默认值、宽泛捕获或静默跳过来掩盖本应成立的条件；`hasattr` / `.get(k, 默认值)` 本身不是问题，**用它隐藏本该成立的前提**才是问题。

### 2.2 代码风格（可读性优先）

1. Python 3.8 语法；不用 3.9+ 特性。
2. 一行一个操作；显式 `import`；不做压缩写法（`core/performance.py` 是反例，S0b 重写）。
3. 不新增抽象层：不加服务类、工厂、插件接口、单例；只有持有资源或状态的对象才用类。
4. 阶段一零新增依赖；测试用标准库 `unittest` + `tempfile`。
5. 公开函数写 docstring：输入、输出、单位（Å / 体素 / 秒）、修改的状态。
6. 三类工作分别提交、分别验证、分别报告（见 §0.1）。

---

## 3. 阶段一实施

每步：先写测试 → 实现 → 跑测试 → `git diff --check` → 提交。**测试随步骤提交，不集中到最后。**

### S0　冻结基线 → 构建隔离基线 → 整理现有改动

**S0-pre　冻结基线（必须是第一件事，早于任何代码改动）**

```bash
cd /xiangyux/claude_c_work/demo_reg
mkdir -p /xiangyux/claude_c_work/demo_reg_baseline_meta
git diff > /xiangyux/claude_c_work/demo_reg_baseline_meta/tracked.patch
cp protassem/core/performance.py /xiangyux/claude_c_work/demo_reg_baseline_meta/performance.py
git rev-parse HEAD > /xiangyux/claude_c_work/demo_reg_baseline_meta/HEAD
git status --short --branch > /xiangyux/claude_c_work/demo_reg_baseline_meta/status.txt
sha256sum /xiangyux/claude_c_work/demo_reg_baseline_meta/tracked.patch \
          /xiangyux/claude_c_work/demo_reg_baseline_meta/performance.py \
          > /xiangyux/claude_c_work/demo_reg_baseline_meta/SHA256SUMS
```

同时登记日常回归 case 的输入指纹【决策】：`test/1/` 五文件的 `sha256sum` 与字节数写入同一目录 `case_test1_inputs.txt`。

**S0-build　隔离基线副本**

```bash
git worktree add --detach /xiangyux/claude_c_work/demo_reg_baseline 5b016f5
cd /xiangyux/claude_c_work/demo_reg_baseline
git apply /xiangyux/claude_c_work/demo_reg_baseline_meta/tracked.patch
cp /xiangyux/claude_c_work/demo_reg_baseline_meta/performance.py protassem/core/performance.py
```

该副本**只在 S7 做对比运行**，不参与开发。用完 `git worktree remove` 清理。

**S0a / S0b　整理未提交改动**（与 v1 相同，不动逻辑）

1. `protassem/sampling/sampler.py`：保留 `subprocess.run` 版本；拆分超长行；`sample_file` 命名改用 `os.path.splitext`。
2. `protassem/core/performance.py`：按 §2.2 重写（行为不变：一次 `performance.jsonl` + 一次 `performance_summary.json`，`run_id` 唯一）；`configure_cpu_threads` 保留，注释说明"NumPy 已导入后设置不保证生效，真正控制见阶段二 P1"。
3. `protassem/fitting/pipeline.py`：保留 `Metrics` 埋点与 `fit_request` 计时，仅拆行；`global` 三变量的改造属阶段二。

**提交**：`fix(sampling): invoke Sample via subprocess and verify returncode/output`、`chore(metrics): tidy performance.py and keep fit_request timing`
**验收**：`compileall` 通过；`git diff --check` 无输出；`git status` 只剩 `INFERENCE_ENGINEERING_PLAN.md`、`ENGINEERING_HARDENING_PLAN.md`、`test/` 未跟踪。

---

### S1　`main.py` 改用 argparse（参数名与默认值不变）

**改动文件**：`main.py`、`core/io.py`（`read_param_file`）。

**设计**：与 v1 相同的参数表，但解析入口改为：

```python
def parse_args(argv):
    parser = build_parser()
    args = parser.parse_intermixed_args(argv)   # 允许选项夹在位置参数中间
    if not args.paths:
        parser.print_help()
        return None
    if len(args.paths) not in (1, 4, 5):
        parser.error("expected 1 or 4-5 positional arguments, got %d" % len(args.paths))
    return args
```

**依据【已核实】**：Python 3.8 实测，`parse_args()` 对 `['a','b','--log','c']`、`['a','--log','b','c','d']` 均 `SystemExit(2)`，与旧入口行为不一致；`parse_intermixed_args()` 分别得到 `paths=['a','b','c']`、`['a','b','c','d']`，与旧行为一致。不使用 `parse_known_args`（会放行拼错的选项）。

其余要点：`--no-domain-split` → `frozenset(strip 后非空项)`；`--log-file` 优先于 `--log`；自动模式缺 `.mrc` / 参数文件 → `parser.error` 指明路径；`read_param_file` 的 `ValueError` 带文件名与内容；`main(argv=None) -> int` + `raise SystemExit(main())`。

**不做**：不新增选项、不改默认值、不改 `run_pipeline` 签名。

**测试**（随本步提交，`tests/test_cli.py`）：默认值全表；每个开关两态；`--no-domain-split A,B`；未知选项 → 2；缺值 → 2；非数字 → 2；位置参数 0/1/2/3/4/5/6 个；**交错参数三例**（`a b --log c`、`a --log b c d`、`--log a`）；与旧实现 `assembly_kwargs` 逐项相等。

**提交**：`refactor(cli): replace hand-written argv parsing with argparse`

---

### S2　`run_pipeline()` 入口校验

**改动文件**：`pipeline.py`（`_validate_inputs`）、`core/io.py`。

**contour 契约【决策】**：`run_pipeline()` 要求 `contour` 为**有限数值**。依据【已核实】：`core/scoring.py:133` 是 `np.where(exp_map > contour, ...)`，`None` 会 TypeError。阶段一**不新增自动 contour 功能**；自动目录模式仍读 `contour_level.txt`（`main.py` 现状）。仅当确已存在自动 contour 入口时，才在入口解析一次并把同一数值贯通传递。

校验项：mrc 存在且 `.mrc`；结构列表非空且文件都存在；`resolution > 0` 且有限；`contour` 有限；`voxel_size > 0`。全部在 `os.makedirs` / `setup_logging` 之前。

**测试**（随本步提交）：各错误分支的类型与消息关键字；断言失败时输出目录未被创建。

**提交**：`feat(pipeline): validate inputs at the run_pipeline boundary`

---

### S3　统一点云 TXT 读写

**前置**：先做 V1 实测（6×8×10 非立方网格），确定第 1 行轴序后再冻结契约。

**新增文件**：`protassem/core/points_txt.py`。

**数据记录【决策】**（v2 修正 v1 的接口缺陷）：

```python
PointCloud = collections.namedtuple(
    "PointCloud",
    "points vectors densities indices sample origin header_lines data_lines")
# points/vectors/densities/indices: numpy 数组（坐标=体素坐标*sample+origin）
# sample, origin: 头部解析值
# header_lines: 原始 5 行头部（保文本精度）
# data_lines: 原始数据行对 [(coord_line, vector_line), ...]
```

**两个写出函数【决策】**（不塞进可选参数）：

```python
def write_filtered(path, cloud, keep_indices):
    """按保留索引写回原始行对，保留原文本精度（供域 TXT 过滤）。"""

def write_point_cloud(path, points, vectors, densities=None, indices=None,
                      sample=1.0, origin=(0.0, 0.0, 0.0)):
    """由数组按明确格式生成新点云（供掩膜点云等新产物）。"""
```

**解析规则**：头部固定 5 行；数据区**按顺序成对**（不再用绝对行号奇偶）；尾部空行允许，中间空行报错；字段数严格 `== 4`；错误消息含 `文件:行号`。**精度说明【已核实】**：`core/io` 返回 float64、`demo_mask` 结构化数组为 float32，只有 `write_filtered` 能做到字节级一致，v1 声称的"往返逐字节一致"只对该函数成立。

**替换范围**：`core/io.py:67`（薄包装，签名保留）、`demo_mask.py:92`、`sw_mask.py:65`（+`:587` 写出）、`Supporting.py:69`、`domain_pdb_txt.py:47`（返回记录；删除 5 处 `sys.exit`，`__main__` 统一转退出码）、`domain_pdb_txt.py:118`（改 `write_filtered`）。

**不做**：不改头部含义、不增列、不做宽容解析；`sw_mask` 写出的 `sample=1.0 / origin=0` 绝对坐标变体保持不变（满足同一契约）。

**测试**（随本步提交）：正常 3 点逐值比对；**错位反例**（第 2 个点行少一字段 → 新实现 `ValueError`，旧实现返回长度不等的数组）；尾部空行、中间空行、头部不足、非数字、空文件；`write_filtered` 过滤后**逐字节**等于原始行子集；`write_point_cloud → read_point_cloud` 数值往返。

**提交**：`refactor(core): single point-cloud TXT reader/writer`

---

### S4a　`align_by_resid()` 链感知匹配 + 映射

**改动文件**：`core/structure.py`、`assembly/refine_step.py`、`assembly/orchestrator.py`。

**接口【决策】**：

```python
def align_by_resid(ref_file, mob_file, output_file, mob_chain_map=None):
    """按 (链号, 残基号, 插入码) 匹配 CA，把 mob 叠合到 ref。

    mob_chain_map: 把 mob 的链号映射到 ref 的链号空间（如占位链号 -> 真链号）。
                   只在构造匹配键时使用，不修改结构本身；None 表示两边同一空间。
    """
```

实现要点：只读第一个 model（与 `read_chain_ids`、`read_structure` 一致，避免跨 model 同键互相覆盖）；`shared = sorted(set(ref_keys) & set(mob_keys))`；`< min_pairs(3)` → `log.warning` 带 `ref/mob/shared` 计数并返回 `False`；不加"匹配不足退回序列对齐"的隐式兜底。

**调用点改造【决策】**：

- `chain_fitter.py:172,222`：同一空间，传 `None`（保持现状语义）；
- `refine_step.py:134`：`cr = _chain_record(orch, cid)` 已存在 → `mob_chain_map=cr.get("chain_map")`；
- `orchestrator.py:621`：从 `self.chain_records` 按 `cid` 查找；**统一查找方法**（如 `AssemblyOrchestrator.chain_record(cid)`），`refine_step` 复用之，不复制状态。

**测试**（随本步提交）：单链 PDB（两边同空间）与旧实现逐坐标一致（1e-6）；两链同编号结构按链匹配、链 A RMSD≈0，而旧键逻辑的大 RMSD 作为对照；**非恒等映射三例**：① 单链真 B → 占位 A → 恢复 B（无映射失败、有映射成功，两个断言都要有）；② 两条链合并为一个 CIF，真链号改为 `Q/R`，验证复合物链号处理；③ 多字符真链号（≥53 链场景另见 S5）；匹配 <3 → 返回 False 且不写文件。

测试副本**从 `test/1` 派生，原始输入不动**（生成脚本入库，产物放 `tests/data/` 且被忽略）。

**提交**：`fix(structure): match residues by chain id with explicit chain mapping`

---

### S4b　复合物域优化中的链号空间一致性（先验证，后修复）

**目标**：确认 `域 CIF → merge_domains → improved PDB → _accept_chain → 最终 CIF` 里哪些出口会把占位链号带到最终输出。

**步骤**：

1. 先用非恒等映射测试（S4a ②③ + `--complex-domain-opt`）把链路走通，记录每一步的链号；
2. 找出绕过 `_accept_chain` 恢复的出口（疑似 `chain_fitter.py:237-240`、`complex_builder` 的合并路径）；
3. **确认后**才在对应边界修复；`chain_id_for_cif` 那一行**不得**直接改成真链号（可能造成重复映射，或让多字符真链号提前进入 PDB 写出）；
4. 独立提交，并在提交信息里写明"泄漏出口 + 修复边界 + 测试"。

**若验证结果是没有泄漏**：把结论和证据写入本方案（V3 销项），**不产生代码改动**。

**提交**（视结果）：`fix(assembly): keep chain-id space consistent for complex domain outputs`

---

### S5　链号池与失败策略

**设计【决策】**（v2 修正 v1 的容量误判）：

1. 两个池分别命名并写清用途：`logical_chain_ids()`（A–Z、a–z、AA…ZZ，用于去重重排）与 `pdb_placeholder_ids()`（A–Z、a–z、0–9，**保留现有 62 容量**）。
2. `cif_to_pdb_placeholders()` 用 `pdb_placeholder_ids()`；超过容量 → `ValueError`，消息给出链数与容量。
3. 删除 v1 的错误论断："超过 52 必须走 CIF 路径"不成立【已核实】——`orchestrator._prepare_chains` 对每个 `.cif` 都做占位转换，**没有纯 CIF 通路**；容量上限即硬上限。
4. V2 实测：数字占位链号在 `read_structure` / `calculate_cc_mask` / USalign / domain split 上是否可用；**只有测出真破坏**才限制为 52，并作为兼容性变化单独提交。
5. 丢链与降分：`_prepare_chains` 收集跳过原因并 `log.warning`，全部不可用 → `RuntimeError`；CIF→PDB 转换失败向上抛；`_cc_worker` / `_pre_screen_cc_worker` 返回失败标记，主进程汇总后 `raise`。
6. **部分成功要可见**【决策】：运行摘要写出输入数、处理数、跳过数及原因；只在日志里 warning 不算交付。

**测试**（随本步提交）：两个池前 54 项顺序；62 链占位可用、63 链报错；数字占位结构在下游四个函数上的实测结果（记录为测试或报告）；不可解析结构 + 一个正常结构 → 1 条跳过记录且 summary 计数正确；全部不可解析 → `RuntimeError`；worker 异常 → 主进程 `raise`（断言异常类型与消息）。

**提交**：`fix(assembly): split chain-id pools and surface skipped chains`

---

### S6　可移植性

| 位置 | 动作 |
|---|---|
| `fitting/parenet/config.py:35` | 删除死配置 `dataset_root`（无读取者、目录不存在），顶部注释说明推理不使用数据集配置 |
| `fitting/parenet/config.py:36` | `metadata_root` 保持仓库相对，补一行来源注释 |
| `fitting/sw_mask.py:616-633` | `__main__` 改 `argparse`（`--source`/`--target`/`--out`） |
| `assembly/refine/refine_energy.py:992` | `example_usage(case_dir)`；`__main__` 用 `argparse` |
| `sampling/extract_points/Sample_based_VoxEM.py:22` | 改 `subprocess.run([...])` + 返回码检查（脚本**保留**） |
| `assembly/domain_parser/DomainParser.py:184,229,278` | 改列表参数（路径含空格/全角字符/中文时字符串拼接会真实出错） |
| `pareconv_src` 内 4 处 | 不动；README 与本文标注 "vendored，勿改" |

**测试**（随本步提交，`tests/test_portability.py`）：排除 `pareconv_src/` 后断言无 `"/xiangyux/` 字面量、无 `os.system(`、无 `shell=True`。

**提交**：`chore(portability): drop hardcoded server paths and shell=True calls`

---

### S7　集成验证、文档与报告

1. `tests/` 结构（stdlib `unittest`）：

```text
tests/
├── __init__.py
├── fixtures.py          # 合成 TXT / 小 MRC / 假 Sample 可执行 / 小 PDB / 两链 CIF / 非恒等链号副本生成
├── cases/               # 输入清单（数量、来源、sha256），不含真实数据
├── reports/             # 精简 Markdown 报告（入库）
├── test_cli.py
├── test_pipeline_inputs.py
├── test_points_txt.py
├── test_align_by_resid.py     # 含非恒等映射三例
├── test_chain_ids.py
├── test_sampler.py
└── test_portability.py
```

2. `tools/compare_runs.py`【决策】：比较两个 `output_dir` 的组件集合与顺序、每个组件 `cc_mask`、最终 CIF 的链/域构成、坐标差异（PDB 精度 1e-3 Å）；输出精简 Markdown 报告。以清楚、完整、无重复为准，不设行数目标。

3. 真实 case 分档【决策】：

| 档位 | case | 覆盖 | 何时跑 |
|---|---|---|---|
| 日常回归 | `test/1`（192³、2 条单链 PDB、A/B） | CLI、输入检查、采样、拟合、评分、装配、默认精修 | 每个实施步骤后跑一次；与 S0 固定基线对比 |
| 较大回归 | `/xiangyux/test_data/lg_diff/test_5kem/5kem`（4 链 A–D） | 多链、去重、队列调度 | S4a/S5 完成后各一次 |
| 复合物路径 | `/xiangyux/test_data/fiting_lg/6lu9/6lu9.cif`（4 链 A/B/C/D + `EMD-0979.mrc` + 参数齐全） | 占位映射、**显式 `--complex-domain-opt`**、域回填、S4b | S4b 定向测试通过后跑一次；**不作为每步必跑项**，`fiting_lg/7sjx` 留作更小备选 |

4. 运行约定【决策】：基线与新版使用**不同输出目录**；同 seed、同线程数、同 GPU 状态；报告写 `tests/reports/<date>_<case>.md`，内容包括 commit、命令、耗时、差异清单、未验证项。

5. 阶段一总体验收：

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate point
cd /xiangyux/claude_c_work/demo_reg
python -m unittest discover -s tests -v
python -m compileall -q main.py protassem tools
git diff --check
git status --short --branch
```

**提交**：`test: add boundary regression tests, compare tool and case reports`

---

## 4. 阶段二：运行时与性能（裁剪自 `INFERENCE_ENGINEERING_PLAN.md`）

启动前置：阶段一完成、`test/1` 基线对比通过、旧文档更正已合并。

### 4.1 必须做

| 编号 | 内容 | 原文档 | 修正 |
|---|---|---|---|
| P1 | 配置与早期线程控制（导入 NumPy/Torch 前设环境；`threadpoolctl` 前后验证；资源解析） | §5 步骤 1 | 原 `configure_cpu_threads` 在 NumPy 导入后调用，不保证生效 |
| P2 | Metrics 定型（事件字段、单 writer、JSONL + summary、GPU 事件可选） | §5 步骤 2 | S0b 只做可读化，本步才做字段与并发 |
| P3 | 复用受控 CPU 池（spawn、惰性、不嵌套），替换临时池 | §5 步骤 3 | 需先去掉 `_NUM_PROCESSES`/`_BATCH_SIZE`/`_METRICS` 全局量 |
| P4 | `DensityMapContext` + 坐标 CC + 有界评分缓存（key 含路径/size/mtime_ns/contour，动态图含 mask_version） | §5 步骤 4 | `core/scoring.py` **已存在** → 改造 |
| P5 | SQLite TM 缓存（父进程读写，key 含内容指纹 + USalign 指纹 + 参数） | §5 步骤 5 | `core/similarity.py` **已存在**（含进程内缓存）→ 加持久层 |
| P7' | **服务退出检测**：poll 同时检查进程退出状态，服务崩溃立即失败，不无限等 DONE | §5 步骤 7 部分 | 从"可选"提升为必须（运行可靠性） |

### 4.2 可选

| 编号 | 内容 | 原文档 |
|---|---|---|
| P6 | 候选评分与优化预算（`max_local_candidates`、`local_time_budget_s`，默认 None；shadow 剪枝统计） | §5 步骤 6 |
| P7 | 请求协议：`request_id` 取代 DONE 语义、`wait(timeout)` 真正等待、原子 rename 发布 | §5 步骤 7 其余 |
| P8 | 阶段恢复 v1（标准化/体素化/采样/TM/已完成推理请求） | §5 步骤 8 第一版 |

### 4.3 实验（默认不做）

| 编号 | 内容 | 前置 |
|---|---|---|
| P9 | GPU 真批处理（`forward_batch`，样本隔离审计，batch=2 起步） | P3/P4 完成 + 单样本稳定 |
| P10 | 增量 mask | 先测出 mask I/O 占比显著 |
| P11 | 装配轮次 checkpoint 恢复（`RoundState`） | P8 完成 |

### 4.4 不可宣称的边界（沿用原文档 §3、§6）

不把"请求排队 + 逐一 forward"称为 GPU batch；不用回旋半径/bbox/序列长度直接判定 TM-score=0；无上界证明不做早停；缓存冷热差异不得归因于 batching；真实数据未跑时写"未验证"，不写"已加速"。

---

## 5. 验证方案

### 5.1 验收按工作类型分列【决策】

| 类型 | 验收标准 |
|---|---|
| A 等价重构 | `test/1` 结果与 S0 固定基线一致（组件集合、顺序、`cc_mask`、最终 CIF 链/域构成；坐标差 ≤1e-3 Å） |
| B 错误修复（S4a/S4b） | 构造案例证明修正（非恒等映射三例）；若真实 case 结果变化，逐项解释来源，**不与 A 类混报** |
| C 失败策略变化（S5） | 断言失败被报告：异常类型/消息、summary 计数、非零退出；不出现"静默降级为 0 分/丢链" |

### 5.2 需要 GPU 与不需要 GPU

| 类别 | 覆盖 |
|---|---|
| 无 GPU | 全部单元测试（CLI、输入校验、TXT 读写、链号、对齐、假采样器、可移植性断言）；秒级 |
| 需 GPU | `test/1` 回归、5kem、6lu9（含 `--complex-domain-opt`） |

### 5.3 合成 fixture（`tests/fixtures.py`）

`make_txt` / `make_broken_txt`（错位）/ `make_mrc`（小图，含 6×8×10 非立方）/ `make_fake_sample_bin`（成功、非零、空输出）/ `make_two_chain_pdb` / `make_two_chain_cif_with_ids(ids)`（非恒等映射）/ `derive_single_chain_cif(src, chain_id)`（B → 单链 CIF）。

### 5.4 通过标准

1. 无 GPU 测试全过；`compileall`、`git diff --check` 无输出；
2. A 类改动在 `test/1` 上与固定基线一致；
3. B/C 类改动有对应构造测试，报告写清"改了什么、为什么允许变化"；
4. 阶段一不报告任何加速；未跑的 case 一律写"未验证"。

---

## 6. 提交顺序

```bash
cd /xiangyux/claude_c_work/demo_reg
git switch -c feat/engineering-hardening     # S0a 之前
```

| 顺序 | 提交 | 主要文件 |
|---|---|---|
| 0-pre | 不提交：基线快照 + worktree（`/xiangyux/claude_c_work/demo_reg_baseline*`，仓库外） | — |
| 0a | `fix(sampling): invoke Sample via subprocess and verify returncode/output` | `sampling/sampler.py` + 测试 |
| 0b | `chore(metrics): tidy performance.py and keep fit_request timing` | `core/performance.py`、`fitting/pipeline.py` |
| 1 | `refactor(cli): replace hand-written argv parsing with argparse` | `main.py`、`core/io.py`、`tests/test_cli.py` |
| 2 | `feat(pipeline): validate inputs at the run_pipeline boundary` | `pipeline.py`、`tests/test_pipeline_inputs.py` |
| 3 | `refactor(core): single point-cloud TXT reader/writer` | `core/points_txt.py`、5 处调用点 + `tests/test_points_txt.py` |
| 4a | `fix(structure): match residues by chain id with explicit chain mapping` | `core/structure.py`、`refine_step.py`、`orchestrator.py` + 测试 |
| 4b | 视 V3 结果：`fix(assembly): keep chain-id space consistent for complex domain outputs` 或"无改动 + 结论入档" | 待定 |
| 5 | `fix(assembly): split chain-id pools and surface skipped chains` | `core/structure.py`、`orchestrator.py`、`fitting/pipeline.py` + 测试 |
| 6 | `chore(portability): drop hardcoded server paths and shell=True calls` | 6 个文件 + `tests/test_portability.py` |
| 7 | `test: add boundary regression tests, compare tool and case reports` | `tests/**`、`tools/compare_runs.py`、`README.md`、`.gitignore` |

禁止：`git reset --hard`、`git clean -fd`、`git push`、递归删除/移动。回退一律 `git revert`。

---

## 7. 风险与不做的事

| 风险 | 处理 |
|---|---|
| S3 改动面最大（5 处解析 + 2 处写出） | 先做 V1 轴序实测与错位反例测试，再替换；替换后立刻跑 `domain_pdb_txt` 单域/多域分支 |
| S4a 会改变多链/CIF 输入的数值 | 单独提交；构造非恒等映射三例；真实 case 变化逐项解释 |
| S4b 误判（内部占位未必是错误） | 先验证传播链再改；不得直接改 `chain_id_for_cif`；无泄漏就只销项不改代码 |
| S5 报错可能让此前"能跑完"的 case 失败 | 预期行为；报错与 summary 列出被跳过文件与原因 |
| `test/1` 完整耗时未知【待验证】 | S0 冻结基线后先跑一次记录耗时；超预算则先报耗时再决定 |
| 6lu9 体量与耗时未知 | 只在 S4b 定向测试通过后跑一次；不设每步必跑 |
| 测试数据混入 Git | `.gitignore` 只忽略真实数据链接/输出/临时基线；生成器、清单、报告入库 |

**明确不做**：不升级 Python/PyTorch/CUDA；不换权重；不重构 vendored `pareconv_src`；不改默认阈值、候选顺序、输出目录结构；阶段一不做任何性能改动；不新增自动 contour 功能。

---

## 8. 开工前 checklist

1. 第 1 节 7 条 + 同源问题 A–E + §1.3 链号空间审计已逐条确认；
2. S0-pre 基线快照与 S0-build 隔离副本已建立并校验 sha256；
3. `test/1` 输入指纹已登记；Output 目录规划为基线与新版分开；
4. `tests/` 可运行（`python -m unittest discover -s tests -v`）；
5. 每步完成后的回复格式：改了什么 / 跑了什么 / 结果 / 未验证项 / 是否保留任务外改动 / 本方案版本是否需同步更新。

---

## 附录 A　采样 TXT 格式实测

【已核实】用 8×8×8、voxel=2.0、origin=(1,2,3) 的合成 MRC 调用仓库内 `protassem/sampling/Sample`：

```text
2.000000                                  <- 行0  sample（= 采样体素边长）
8 8 8                                     <- 行1  盒子尺寸（轴序【待验证】）
9.000000 10.000000 11.000000              <- 行2  未使用
1.000000 2.000000 3.000000                <- 行3  origin
169.100255 1407.526926 183.433056         <- 行4  未使用
146 2 2 2                                 <- 行5  点：index = 2 + 2*8 + 2*64 = 146
0.577350 0.577350 0.577350 138.800858     <- 行6  法向量 + 密度
210 2 2 3
0.688412 0.688412 0.228423 160.823456
```

【已核实，V1 销项】第 1 行轴序**不可观测**：用 6×8×10 非立方输入实测，`Sample` 会按密度包围盒把盒子裁剪成**立方**（实测 8³ / 6³ / 12³）并重设 origin（例：输入 shape (nz,ny,nx)=(6,8,10)、origin (1,2,3) 时输出 line1=`6 6 6`、line3=`5 4 3`），因此 line1 三个数恒相等；点坐标只依赖 line0(sample) 与 line3(origin)，而 `index = x + y*nx + z*nx*ny` 的 nx 取 line1 首项（立方时与轴序无关）。契约据此定为：line1/2/4 原样保留、不解析。

## 附录 B　服务器命令模板

```bash
ssh my-server 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate point && cd /xiangyux/claude_c_work/demo_reg && <COMMAND>'
```

（DSH 内同一主机亦可用别名 `xiangyux`；两者都是 10.8.4.222:30170。注意：`X:\` 挂载视图只读可用，DSH 文件写入后端在 SSHFS 上会 `EPERM`，所有写操作走 SSH 侧。）

## 附录 C　已定稿决策与仍开放项

### C.1 已定稿【决策】

| 项 | 决定 |
|---|---|
| 回归 case | case A=5kem（较大，4 链）；日常回归另用 `test/1`；case B=`fiting_lg/6lu9`（复合物域优化），`7sjx` 备选 |
| 输入目录 | 仓库外干净输入目录 + 软链接清单文件；每次运行独立输出目录 |
| 基线 | `git worktree` 隔离副本；S0 修改前冻结 diff/未跟踪代码/HEAD/指纹 |
| `extract_points/` 三脚本 | 保留，只修外部调用与统一解析接口 |
| `.gitignore` | 忽略真实数据链接、运行产物、临时基线；不整体忽略测试数据 |
| `tools/compare_runs.py` | 新增，无行数目标 |
| 旧文档 | 已同步更正 §4 表格标签 + 顶部关系说明 + 简短修订记录 |

### C.2 仍开放

| # | 事项 | 需要谁定 |
|---|---|---|
| O1 | 6lu9 完整端到端是否在阶段一内执行（当前定为"定向测试通过后跑一次"） | 用户可按耗时再定 |
| O2 | 5kem 是否纳入阶段一（当前定为 S4a/S5 后各一次） | 同上 |
| O3 | 阶段二启动时间与 P1–P11 的取舍 | 阶段一验收后 |
| O4 | 无组件被接受时的空结果输出契约 | **已修复（R2）**：不写空 CIF、清理旧产物、返回 `None`、摘要状态字段、Step4 门控；6 项测试 + 真实集成验证 |
| O6 | **完整流水线可重复性问题**：`--num-processes` / 尾部流水线开关 / 机器负载都会改变最终 `assembled_complex.cif` | **本分支修复中（接口见附录 D）**：机制 = 客户端按"本轮新出现的 `pred_*.pdb`"凑批（`--batch-size` 默认 10、轮询 2.5 s），候选由服务端"每掩码最优改名 + 删其余"产生 → 顺序取决于写盘/改名时机；`_PARENET_DONE` 失败也写 done。修法 = 服务端候选台账（`request_id` + 连续整数 id + 终态）+ 客户端按固定 ID 区间批次消费 |
| O5 | S4b 残留分支（refine backfill 的链号出口） | **已收口**：`refine_step._backfill_chains_as_domains` 只处理 `type == "chain"`，而复合物记录是 `type="complex"` → 该路径**对复合物不可达**；真正消费复合物域产物的位置是 `assemble_domain_chains` → `merge_domains(is_complex=True)` → `_save_domain_chain` → `build_complex`，已在 R1 修复（补 `is_complex` + `chain_map` 恢复真链号）并用真实运行验证（链号 `Q`,`R`） |

---

## 附录 D　老卡收口接口定稿（v4.11）

> 目的：**不留下隐含分支**。O6/P4/P5 的接口、失败语义、编号与验收口径以本附录为准；实现不得自行扩大或收窄。
> 实施分支 `feat/old-card-closeout`；提交顺序：docs 校正 → P3 生命周期验证 → O6 → O6 新基线 → P4 → P5 → 集成收口。

### D.1 O6：候选台账协议（服务端 → 客户端）

**D.1.1 台账文件与记录**

- 位置：请求输出目录 `<request_out>/candidates.jsonl`（**每个请求一份**，与 `server_timing.jsonl` 同目录、同风格：追加写）。
- 每条记录必须带 `v`（协议版本）与 `request_id`（与服务端请求一一对应）；**新请求不得读到旧台账**（`request_id` 不匹配即视为非法记录）。
- 记录类型（JSONL，一行一条）：

| kind | 字段 | 含义 |
|---|---|---|
| `candidate` | `id`（连续整数）、`state`、`name`、`overlap`、`source{mask,config,sampling}` | 一个候选的终态 |
| `end` | `status`（`ok`/`error`/`cancelled`）、`count`、`error?` | 请求结束（唯一权威结束标记） |

- `state` 三态【决策】：

| 情况 | state | 客户端处理 |
|---|---|---|
| 正常计算并产出有效候选 | `ok` | 纳入批次消费 |
| 正常计算但**没有有效候选**（如最小覆盖过滤） | `filtered` | 计数、记录原因、**跳过**，不视为失败 |
| 文件/模型/外部程序**执行失败** | `error` | 记录 `error` 与 stderr 摘要；**该候选不参与**，并按请求级状态处理 |
| 客户端已找到达标结果主动早停 | 请求级 `cancelled` | **正常控制流程**，不当作运行失败 |

- **发布顺序**：候选文件**完整写出**（临时名 → 原子 rename）→ 追加台账记录 → `flush + fsync`。台账里出现的 `name` 必须已经存在且不会再被改名/删除（`utils.py` 的"删其余 11 个"必须在发布前完成）。
- **编号**：协议里的 `id` **永远是连续整数**（0 起、按稳定生成顺序分配）；`mask/config/sampling` 只作为来源字段。`use_mask=False` 路径用自己的稳定生成顺序分配整数 id —— **客户端不感知两套编号**。
- **顺序稳定性是前提，必须实测核对**：`mask_data_list` 的顺序不得靠变量名推定（核查集合遍历、并行完成顺序、随机状态是否影响）。若不稳定，先固定该顺序再谈 id。
- 请求级状态：`ok`＝正常结束；`error`＝失败（必须带 `error` 文本）；`cancelled`＝客户端主动早停。**服务端异常/进程消失不得产生 `status="ok"`**（现有 `finally` 里写 `_PARENET_DONE` 的语义按此修正；`_PARENET_DONE` 保留为兼容标记）。

**D.1.2 客户端消费契约**

1. **只消费完整行**：只处理以换行符结束的 JSONL 记录；读到文件末尾的半行时**等待后续数据**，不得判为损坏、不得丢弃。
2. **固定批次**：批次成员由**固定 ID 区间**决定（`[0,batch_size)`、`[batch_size,2·batch_size)`…，`batch_size` 仍取 `--batch-size`，默认 10）。不得再用"本次轮询发现的文件集合"当批次。
3. **批内策略不变**【决策】：批次成员固定后，**保持原有候选排序与逐个局部优化策略**；**第一个达到阈值的候选即触发早停**，不要求把整批优化完。
4. **end 到达后仍按批消费**【决策】：即使 `end` 已到、全部候选已生成，客户端**仍必须按固定批次依次消费**；不得切换回"服务端结束了，剩余候选一起 `final_select`"的旧分支。
5. `final_select` **只在**"全部批次消费完 且 未早停"时执行，规则维持原样；并列时最终排序键 = `(score desc, id asc)`。
6. **失败语义**：未见 `end` 而请求句柄已结束 → **抛错**；`end.status="error"` → **抛错**；`end.status="cancelled"` → 正常返回；候选级 `error`/`filtered` → 跳过并计入摘要。
7. **早停后的资源契约**：确认请求**已结束**（句柄已终止 + 台账 `end` 已见）之后，才可进入下一次会改写输入或复用目录的操作。
8. 客户端不再依赖文件名规则选候选；`_is_valid_pred` 只保留为二次校验。

**D.1.3 假生产者验收矩阵**（`tests/test_candidate_ledger.py` + `tests/test_candidate_ordering.py`）

- 发布节奏：快/慢/随机延迟；worker 数：1/2/10；边界：最后一个不足额批次（含 `end` 已到才凑齐的场景）；分数并列；结束态：正常 / 服务失败 / 取消（含"客户端早停"）。
- 判据：**同一候选 id、同一接受决策**；`error` 必须抛；`filtered` 必须跳过且不阻塞后续批次；半行记录不得导致丢候选或误判。
- **随机性核查**【待验证】：若局部优化使用随机初始化，必须确认每个**逻辑任务**获得与 worker 调度无关的同一种子（不能继承各 worker 的 RNG 消费进度）。**先核查，存在依赖才改**；不无依据扩大重构。

### D.2 P4：有界评分缓存接口

- `DensityMapContext` 保存：密度数组、`voxel_size`、`origin`、`shape`、`contour`、密度版本/指纹。**保持原评分路径的 dtype 与阈值处理顺序**；类型转换类优化**不纳入本步**（等价性取决于原 dtype 与 NumPy 行为，不做默认强转）。
- 接口分层：`score_coords(ctx, coords, elements, resolution)` 为坐标数组入口；`calculate_cc_mask(density_mrc, structure_file, resolution, contour)` **签名不变、退化为薄包装**。
- 预算【决策】：`score_cache_mb` 默认 **128 MiB / 进程**，**密度上下文与结构坐标共享该预算**，0 = 关闭；报告须注明多 worker 下缓存总量按进程数放大。
- 失效：`(abspath, size, mtime_ns, contour, SCORING_VERSION)`；**动态密度**（会被同名覆盖的 `current_density.mrc` 等）必须带**显式密度版本**或在更新后**显式失效** —— 不以"同路径覆盖测试通过"代替版本契约。
- 只读约定：缓存数组只读，需要修改时使用独立数组（`.copy()`）。
- worker 私有缓存（模块级 holder）允许保留；但**配置变化（预算/版本/路径）必须清空或重建**，串行多次运行不得沿用上一次的预算与上下文。
- 不做：低分辨率评分替代、bbox 硬剪枝、CC 公式调整、局部优化目标函数统一、无界保存候选结构。

### D.3 P5：SQLite TM 缓存接口

- `_run_usalign_tm(...) -> (ok, value, stderr)`：超时、非零退出、解析失败 ⇒ `ok=False`（**不返回伪 0.0**）。
- `calculate_tm_score(...)`：**合法 `TM-score=0` 正常返回且可缓存**；失败则**抛出明确异常**（含两个输入路径、命令行、stderr 摘要）。若确有允许失败的调用点，在该调用点显式处理，**不在公共评分函数里统一变 0**。
- `TMStore`（父进程 sqlite3，WAL）：键 = 两个结构内容 `sha256` **排序后** + USalign 指纹（二进制 sha256 + size）+ 实际参数 + `TM_CACHE_VERSION`；**只父进程查询/写入**，worker 只执行 USalign；失败不写；合法 0.0 写。
- **两层缓存同键**【已核实】：旧的进程内 `_TM_CACHE` 目前只按路径做键。P5 必须让它与 SQLite 使用**同一套有效指纹**（或直接由 `TMStore` 取代内存层）；**禁止"先命中仅按路径的内存缓存、再查库"**。
- 默认路径【决策，本版新增】：`$XDG_CACHE_HOME/protassem/tm.sqlite3`，未设置该变量时用 `~/.cache/protassem/tm.sqlite3`；开关 `--tm-cache PATH` / `--no-tm-cache`，配置项 `RuntimeConfig.tm_cache`。**默认开启与默认路径是本版新增决策**（不是此前已确认的选择）。
- 不做：未经证明的几何预筛。

### D.4 验收约定（本轮）

| 项 | 约定 |
|---|---|
| 真实 `test/1` 次数 | **7 次**：仅 O6 ×4（固定配置连续两次、`--num-processes 1`、`tail_pipeline=on`）、P4 ×1、P5 ×1、**第 7 次 = 暖缓存重复运行**（复用第六次的持久 TM 缓存，验证跨运行命中与最终结果重复性） |
| 时间预算 | 纯运行 **≈2–2.5 h**（不按 17 min/次统一估算：`--num-processes 1` 已测更慢） |
| 旧产物用途 | `out_t10_off`（1083 s，`76638d0f…`）只作**历史行为**证据，**不得**用于本轮速度归因 |
| 微基准 | 按**代码版本与相关改动**执行即可；同一提交内每次长跑前不机械重复同一份微基准 |
| 基线 | O6 通过后冻结 `tests/cases/baseline_after_o6.md5`（并给出新旧候选序列 diff 作为差异归因）；P4/P5 均与该基线比对 |
| 收口口径 | 全部通过后才可写"S0–S7、R1/R2/O4/O5、P1–P5 与 O6 已收口"；剩余范围（P9 GPU 批处理、增量 mask 等）明确列为后续实验 |
