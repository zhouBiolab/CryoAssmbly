# demo_reg 任务总账

日期：2026-09-15　代码状态：`29c2019`（`feat/old-card-closeout` = `main`；远端已同步）

**本文用途**：把四份任务来源压缩成一份总账，回答"一共做过哪些任务、各自什么状态、还剩什么"。
细节仍以各自的方案与报告为准，本文只做**索引 + 状态 + 结论**，不重复推导过程。

**一句话结论**：老任务卡（S0–S7 / R1 / R2(O4) / O5 / P1–P5 / O6）与两轮审计（7 + 4 项）**全部收口**；
新任务卡 T00–T10 **全部完成并冻结**；剩余的都是明确"不在本轮"的后续实验与决策项（见 §5）。

---

## 0. 一页速览

| # | 任务来源 | 权威文档 | 范围 | 状态 |
|---|---|---|---|---|
| 1 | **老任务卡** | `ENGINEERING_HARDENING_PLAN.md`（v4.13） | 用户 7 条缺陷 + 6 条同源问题 + 阶段一 S0–S7 + 阶段二 P1–P11 | ✅ S0–S7、R1、R2(O4)、O5、P1–P5、O6 全部收口；P6/P8 未做；P9–P11 属后续实验 |
| 2 | **新任务卡** | `MASK_REGISTRATION_OPTIMIZATION_TASKS.md` | T00–T10 掩码配准优化（源侧缓存、推理显存、流水线） | ✅ T00–T10 全部完成，冻结在 `552d3e9`；**所有默认项均未提升** |
| 3 | **审计第一轮** | `tests/reports/2026-09-14_audit_fixes_acceptance.md` | 7 项功能缺陷（P1-1…P1-4、P2-5…P2-7） | ✅ 7/7 已修，各有定向测试 + 真实运行验收 |
| 4 | **审计第二轮** | 同上「一bis」 | 4 项复核（#1–#4） | ✅ 4/4 已修；单测 228 → **259** |
| 5 | **剩余开放项** | 本文 §5 | 未做的 P6/P8/P9–P11、6lu9 端到端、Step4/5 池与 TM、默认项提升决策 | ⏸ 均明确"不在本轮"，需另行决策 |

**当前硬指标**

| 项 | 值 |
|---|---|
| 单元测试 | **259 项全绿**（`python -m unittest discover -s tests -t .`） |
| 冻结基线 | `tests/cases/baseline_after_o6.md5` = `76638d0f97d4ee5775110c66b15e73c6`（`assembled_complex.cif` / `_all.cif`）+ `bd281f40e25352455ba34d41e6bbc9f7`（`refined_complex.cif`） |
| 候选台账（test/1） | 3 请求 24 / 28 / 115 候选，id 连续，全 `ok`，`end=ok` |
| 候选消费序列 sha1 | `de02d03105ebff9f`（9 次真实运行全部一致） |
| `test/1` 墙钟 | 1199–1327 s（运行间波动 ±5%） |

---

## 1. 老任务卡

### 1.1 用户列出的 7 条缺陷（全部【已核实】）

| # | 缺陷 | 归属 | 状态 |
|---|---|---|---|
| 1 | `main.py` 手写参数解析：缺文件/参数抛 `IndexError`/`ValueError`，未知选项静默忽略（带值时其值落进 positional 造成错位） | S1 | ✅ argparse + `parse_intermixed_args()`，退出码 2 + 原因 |
| 2 | `run_pipeline()` 进入体素化前不校验输入（不查 mrc 存在/后缀、结构列表非空、分辨率、contour 有限） | S2 | ✅ 边界校验前移到 `os.makedirs` 之前 |
| 3 | `sample_density_map()` 用 `os.system()` 字符串拼接（路径含空格即出错） | S0 / S6 | ✅ `subprocess.run([...])` + 返回码 + 空输出检查 |
| 4 | `load_sample_points()` 依赖固定行号与奇偶行；某行字段 <4 时**只跳过该侧** → points 与 normals 静默错位（同一格式在 **7 处**重复实现） | S3 | ✅ 统一到 `core/points_txt.py`，按顺序成对 + 严格 4 列 + `文件:行号` |
| 5 | `align_by_resid()` 只用残基编号做键（不含链号与插入码），重复编号互相覆盖 | S4a | ✅ 键改为 `(链号, 残基号, 插入码)` + 显式 `mob_chain_map` |
| 6 | `cif_to_pdb_placeholders()` 占位池含数字，超 62 链 → `IndexError`；与 `chain_id_pool()` 两套规则并存 | S5 | ✅ 拆成 `logical_chain_ids()` / `pdb_placeholder_ids()`；超容量抛 `ValueError` |
| 7 | 硬编码服务器路径（9 处） | S6 | ✅ 删死配置、`__main__` 改 argparse、去掉 `shell=True`；`pareconv_src` 内 4 处标注"vendored 勿改" |

### 1.2 同源问题（6 条，纳入阶段一）

| # | 问题 | 处理 | 状态 |
|---|---|---|---|
| A | 结构解析失败/找不到 txt 时静默丢链，装配在缺链情况下继续 | S5：warning 带原因 + `skipped_inputs` 计数；全不可用 → `RuntimeError` | ✅ |
| B | 并行 worker 把异常伪装成"分数低"（`cc=None` / `0.0`） | S5：worker 返回失败标记，主进程汇总后 `raise` | ✅ |
| C | 库函数里 `sys.exit(1)`（5 处） | S3：改抛异常，`__main__` 统一转退出码 | ✅ |
| D | 无 `tests/`；README 的 `example2` 不在仓库 | S7：新建 `tests/`；README 改写为 `<case_dir>` | ✅ |
| E | vendored 第三方与自研代码同包（`pareconv_src`、`refine/`） | 不搬迁，只在文档标明边界 | ✅ |
| F | **复合物多链域链合并必然失败并被静默丢链**（残基 ID 重复 → 整链丢弃 → 空复合物） | R1：补传 `is_complex` + `chain_map` 恢复真链号 | ✅ 已修复 |

### 1.3 阶段一 S0–S7

| 步 | 内容 | 状态 |
|---|---|---|
| S0 | 冻结基线 → `git worktree` 隔离副本 → 整理未提交改动（S0a 采样器、S0b `performance.py` 可读化、S0c 文档入库） | ✅ |
| S1 | `main.py` 改 argparse（参数名与默认值不变，选项可交错） | ✅ 17 项测试 |
| S2 | `run_pipeline()` 入口校验（含 `contour` 必须为有限数值） | ✅ 11 项测试 |
| S3 | 统一点云 TXT 读写到 `core/points_txt.py`，替换 7 处解析 + 2 处写出 | ✅ 13 项测试 |
| S4a | `align_by_resid()` 链感知匹配 + `AssemblyOrchestrator.chain_record(cid)` 唯一查找入口 | ✅ 5 项测试 |
| S4b | 复合物域优化链号空间一致性（**先验证后修复**） | ✅ 3 次定向探针**未观测到泄漏**，按约定未改代码；残留分支由 O5 收口 |
| S5 | 链号池拆分 + 失败策略（跳链可见、部分成功进 summary） | ✅ 9 项测试 |
| S6 | 可移植性（删死配置、去 `shell=True`、去硬编码路径） | ✅ 静态断言测试 |
| S7 | 集成验证、`tests/` 骨架、`tools/compare_runs.py`、README、`.gitignore` | ✅ 端到端逐字节一致 |
| **R1** | 复合物域链合并丢链修复（同源问题 F） | ✅ |
| **R2 (O4)** | 空结果输出契约：0 组件时不写空 CIF、清旧产物、返回 `None`、摘要 `final_status` | ✅ 6 项测试 + 真实集成验证 |
| **O5** | S4b 残留分支收口：`refine_step` 只处理 `type=="chain"`，复合物记录是 `type="complex"` → 该路径**对复合物不可达** | ✅ 以可达性论证销项 |

### 1.4 阶段二 P1–P11

| 编号 | 内容 | 档位 | 状态 |
|---|---|---|---|
| P1 | 配置与早期线程控制（两段式导入，导入 NumPy/Torch **之前**设环境） | 必须做 | ✅ 实测 OpenBLAS 128 → 设定值 |
| P2 | Metrics 定型（事件字段、单 writer、JSONL + summary）+ P2-fine 时间账细化 | 必须做 | ✅ 补埋点后未归因 260 s → 1.13 s |
| P3 | 复用运行级受控 CPU 池（spawn/fork 显式、惰性、不嵌套） | 必须做 | ✅ 池创建 **79 → 1**；生命周期探针 fork 0.027–0.030 s |
| P4 | `DensityMapContext` + 坐标 CC + 有界评分缓存 | 必须做 | ✅ 128 MiB/进程共享；读取 **12 → 1**、命中 91.7%；端到端无可测收益（评分仅占 2.7%） |
| P5 | SQLite TM 缓存（父进程独占读写、失败抛错、合法 0.0 可缓存） | 必须做 | ✅ 暖缓存跨运行 hits 4 / misses 0（修掉"指纹含路径"缺陷后） |
| P7′ | **服务退出检测**（从"可选"提升为必须） | 必须做 | ✅ 由审计 P1-3 实施：`poll()` 三态 + `describe_failure()` |
| P6 | 候选评分与优化预算（`max_local_candidates` / `local_time_budget_s`） | 可选 | ⏸ 未做 |
| P7 | 请求协议：`request_id` 取代 DONE 语义、原子 rename 发布、`wait(timeout)` | 可选 | 🟡 大部分由 O6 覆盖（`request_id` + 原子发布）；`wait(timeout)` 未单独实施 |
| P8 | 阶段恢复 v1（标准化/体素化/采样/TM/已完成请求） | 可选 | ⏸ 未做 |
| P9 | GPU 真批处理（`forward_batch`、样本隔离审计） | 实验 | ⏸ 未做（用户决定不纳入本轮） |
| P10 | 增量 mask | 实验 | ⏸ 未做 |
| P11 | 装配轮次 checkpoint 恢复（`RoundState`） | 实验 | ⏸ 未做 |

### 1.5 O6：完整流水线可重复性（本轮最大的一项）

**问题**：`--num-processes`、尾部流水线开关、机器负载都会改变最终 `assembled_complex.cif`。
**机制**：客户端按"本轮新出现的 `pred_*.pdb`"凑批（轮询 2.5 s / 批 10），候选由服务端"每掩码最优改名 + 删其余"产生 → 顺序取决于写盘/改名时机；且 `_PARENET_DONE` 在失败时也写。

**修法（附录 D.1 定稿接口）**

- 服务端**候选台账** `fitting/candidate_ledger.py`：`request_id` + 连续整数 `id` + 终态 `ok/filtered/error` + `end{ok/error/cancelled}`；**先完整写出候选文件（临时名 → 原子 rename）→ 再追加记录 → `flush + fsync`**。
- 客户端 `fitting/candidate_consumer.py`：**固定 ID 区间**批次 `[0,batch_size)`、`[batch_size,2·batch_size)`…；批内保持原策略（CC 降序 + id 次序 → 逐个局部优化 → **首个达标即早停**）；`end` 到达后仍按批消费；`final_select` 仅在"未早停且全部批次消费完"时执行。
- 失败语义：未见 `end` 或 `end.status=error` → **抛错**；`cancelled` 属正常控制流。
- 附带修复：`find_mask_files` 改按名排序；`apply_seed` 固定父进程随机源。

**验收**：4 次真实运行（默认 ×2、`--num-processes 1`、`tail_pipeline=true`）三个 CIF md5、三条接受决策、台账（24/28/115）与消费序列 sha1 全部逐位一致 → **worker=1 与 10 一致、tail 开关不再改变产物**。冻结 `baseline_after_o6.md5`（O6 未改变默认路径产物）。代价：如实记录 **+15% 墙钟**。

### 1.6 待验证问题清单

| # | 问题 | 结论 |
|---|---|---|
| V1 | 采样 TXT 头部第 1 行轴序 | ✅ 销项（S3 前置实测） |
| V2 | 数字占位链号是否真破坏下游 | 🟡 部分销项：`read_structure`/`calculate_cc_mask`/USalign 实测通过；DomainParser/域切分未覆盖，容量保留 62 |
| V3 | S4b 传播链哪个出口泄漏占位链号 | ✅ 销项（引入点在中间产物；可达最终输出均恢复真链号） |
| V4 | `test/1` 完整流水线耗时与基线 | ✅ 销项（967 s → 956 s，输出逐字节一致） |
| V5 | 复合物域优化分支触发条件与耗时 | ✅ 由 S4b 三次探针与 6lu9 备选路径覆盖 |

---

## 2. 新任务卡：T00–T10（掩码点云配准优化）

**边界**：固定 src，依次配准大量不同 tgt 掩码；不重训练、不升级 Torch/CUDA、不修改共享 `/xiangyux/PARENet-main`。
**总结果**：T00–T10 全部完成；**所有默认项一律未提升**（凡改变产物的开关都保持关闭）。

| 卡 | 内容 | 关键结论 | 状态 |
|---|---|---|---|
| T00 | 固定基线、依赖与测试输入 | 实测确认 pareconv **从共享项目加载**、未用仓库副本；GPU 为 **MIG 7g.80gb 切片**；同输入两次重放差约 29%（需冷/暖重复） | ✅ |
| T01 | 坐标/旋转中心契约 | 修复 `transform_pdb(..., center=c_src)`；合成实测偏移 0.310 Å；修复后**重冻基线** | ✅ |
| T02 | 热点与源侧可复用比例 | 源侧可复用上界 **10.4%**；几何重复率 98.6%；scale 命中机会 95.8%；`model_lgr` 占模型 68% | ✅ |
| T03 | 推理内存与中间回传 | `@torch.no_grad()` + 输出白名单；`max_allocated` **4.85 GiB → 1.15 GiB** | ✅ |
| T04 | 单侧几何拆分 | 逐阶段按位一致；72/72 预测哈希一致；总墙钟 −1.7%（噪声，不作加速结论） | ✅ |
| T05 | 有界 CPU 几何缓存 | 开关等价（72/72）；命中率 97.2%；几何阶段 0.44 → 0.13 s | ✅ |
| T06 | 单侧编码与双侧配准 | 拆出 `encode_cloud`/`register_pair`；显式 `inference_mode`/`allow_tf32`（TF32 使数值依赖张量形状） | ✅ |
| T07 | 精确 scale 源编码缓存 | 缓存开关 72/72 哈希一致；源编码命中 **99.6%**；默认不变（joint） | ✅ |
| T08 | 位姿假设评分分块 | chunk=64 与 0 **72/72 逐位一致**；峰值 −15.6%；默认仍为 0 | ✅ |
| T09 | 有界 CPU 尾部流水线 | 单 worker、深度 1、FIFO；真实运行 1092 → 994 s（−9.0%）；**默认不提升**（T10 复测证明其改变产物） | ✅ |
| T10 | 完整回归与交付 | 配准层面 A/B/C/F **72/72 逐位一致**；E（split）改变 overlap 且无收益；端到端 on 3 次中 **2 次产物不同** → **所有默认项一律不提升** | ✅ |

**⚠️ 重要修正（T10 对 T09 的推翻）**：T09 声称"tail 开关产物一致"**不可复现** —— on 共 3 次，1 次复现基线、2 次给出 `51009d69…`。
根因是候选枚举顺序的时序敏感性（与模型输出无关）→ **这就是 O6 的问题来源**，已在老卡收口阶段修复。

---

## 3. 两轮审计

### 3.1 第一轮：7 项（各自一次提交，只做功能修复）

| 审计项 | 提交 | 修复要点 |
|---|---|---|
| **P1-1** 复合物逐域改善后重复恢复链号 | `f9e3ef1` | 固化三个链号空间 + "占位→真实**只在最终输出映射一次**"；新增 `_restore_chain_ids()` 唯一恢复点 |
| **P1-2** 复合物只接受一个域时链号没恢复 | `f9e3ef1` | `_handle_single_domain` 从内部空间出发按 `source_chain_id` 做单条映射 |
| **P1-3** 服务意外退出 → 客户端永久等待 | `bd92f45` | 句柄持有服务进程；`poll()` 区分"完成/已退出/仍在跑"；消费者**快速失败**（≤4 次轮询） |
| **P1-4** 无掩码路径发布后才改名 | `9b5947f` | 先 `_drain_tail` + `rename_pdb_files_by_ranking()`，再按生成顺序一次性发布 |
| **P2-5** 失败被静默跳过、请求仍报成功 | `4b5ad7e` | 异常写入结果集；请求级状态优先级明确；客户端消费到 `error` 立即抛错 |
| **P2-6** 台账半行 UTF-8 截断 | `9b6c3c6` | `LedgerReader` 按**字节**缓冲，切完整行后再解码 |
| **P2-7** 动态密度没有版本契约 | `8f1d2ce` | 每轮写 `current_density_mNN.mrc`（**路径即版本**） |

**证据分级**（复核要求）：每条分开标注 **(a)** 旧接口不兼容 / **(b)** 旧行为缺陷复现 / **(c)** 新行为验收；
凡声称"旧代码失败"的，必须真的跑出错误结果（如 `PDBConstructionException: C defined twice`、`UnicodeDecodeError`）。

### 3.2 第二轮：4 项复核

| 复核项 | 提交 | 修复要点 |
|---|---|---|
| **#1** 同一 mask 内部分评估失败仍报成功 | `28d4440` | `_publish` 改**错误优先**；异常写入 `mask_results` 与 `all_results`，日志/`Failed`/台账三处口径一致 |
| **#2** 客户端抛错未收口请求 | `28d4440` | `run()` 包装 `_run()`：异常时先 `on_cancel()` 再 `_await_request_end()`，清理异常不得覆盖原始错误 |
| **#3** `error_policy="skip"` 名不副实 | `28d4440` | **移除该参数**（服务端仍写 `end.status="error"`，不构成宽松模式）；候选级 `error` 一律抛错 |
| **#4** 评分缓存统计读旧字段 | `60ea283` | 统计抽成 `_record_score_cache()` / `_record_tm_cache()`，占用/峰值取**顶层共享字段** |

**测试计数**：228 → **259**（拆分测试文件后复核，避免同一组用例被计两次）。
**文档同步**：`PROJECT_ARCHITECTURE.md`、`ENGINEERING_HARDENING_PLAN.md`（v4.12 第 106–115 条、v4.13 第 116–119 条）。

---

## 4. 验收口径（贯穿全部任务）

| 口径 | 做法 |
|---|---|
| 证据三类 | (a) 旧接口不兼容 / (b) 旧行为缺陷复现 / (c) 新行为验收；不混用 |
| 等价性 | 先比 `R/t`、CC、候选顺序、接受结果；PDB 坐标 1e-3 Å；特征 atol=1e-6、rtol=1e-5 |
| 偏差处理顺序 | 固定输入/配置/随机状态 → 关新功能复现基线 → 找首个偏差阶段 → 只修本卡 → 重跑依赖测试 |
| 不宣称的边界 | 不把"请求排队 + 逐一 forward"称 GPU batch；不把冷暖差异当加速；不把减少随机尝试当等价优化 |
| 真实数据 | 真实输入与运行产物**从不入库**（`test/` 由 `.gitignore` 覆盖，跟踪文件数 0） |
| 每步节奏 | 一次提交一步 + 同步 `PROJECT_ARCHITECTURE.md` 变更行 + `tests/reports/` 报告 |

---

## 5. 剩余与开放项（**现在还没做的**）

| # | 项 | 说明 | 需要谁定 |
|---|---|---|---|
| 1 | **P6** 候选评分与优化预算 | `max_local_candidates` / `local_time_budget_s`（默认 `None`）+ shadow 剪枝统计 | 阶段一验收后 |
| 2 | **P8** 阶段恢复 v1 | 标准化/体素化/采样/TM/已完成推理请求的断点续跑 | 同上 |
| 3 | **P9** GPU 真批处理 | `forward_batch` + 样本隔离审计，batch=2 起步 | 用户已决定**不纳入本轮** |
| 4 | **P10** 增量 mask | 需先测出 mask I/O 占比显著 | 同上 |
| 5 | **P11** 装配轮次 checkpoint | `RoundState`，依赖 P8 | 同上 |
| 6 | **6lu9 复合物端到端** | `/xiangyux/test_data/fiting_lg/6lu9/6lu9.cif` + `EMD-0979.mrc`，res 8.8 / contour 0.316；需干净输入目录 | 未跑；耗时由用户定 |
| 7 | **Step4/Step5 自建池未接入 P3** | `homo_chain_refine.py`（5 处 `pool.map`）、`refine/chain_enumerator.py`、`sampling/extract_points/VoxEM.py` | 只文档化，未接入 |
| 8 | **Step4/5 组件优化的 TM 计算** | `refine_energy._calculate_tm_score` 是独立实现，**P5 未覆盖** | 未覆盖 |
| 9 | **`tail_pipeline` 是否提升为默认** | 实测 −4%（post-O6）且产物等价，但 T10 曾观察到它改变产物 | 待决策；**当前默认 false** |
| 10 | **`--hypothesis-chunk 64` 是否提升** | 预测逐位一致、峰值 −15.6%，但端到端收益未验证 | 待决策；**当前默认 0** |
| 11 | **`inference_mode="split"`** | 改变 overlap 且无收益 | **不推荐**；当前默认 `joint` |
| 12 | **V2 残留** | 数字占位链号在 DomainParser / 域切分上未覆盖 | 保留 62 容量 |
| 13 | **`assembly_summary.txt` md5 逐次不同** | 只因时间戳/路径，非产物差异 | 已解释，无需处理 |

---

## 6. 复现：命令、默认值与产物

### 6.1 环境与常规命令

```bash
cd /xiangyux/claude_c_work/demo_reg
source /root/miniconda3/etc/profile.d/conda.sh && conda activate point
```

| 用途 | 命令 |
|---|---|
| 全量测试（259） | `python -m unittest discover -s tests -t .` |
| 日常回归 test/1（≈21 min） | `python main.py test/1/EMD-8436.mrc test/1 5.6 0.04 <out_dir> --log --runtime-config /tmp/rt.json` |
| 配准微基准（≈40 s） | `python tools/benchmark_registration.py run --manifest tests/cases/registration_manifest.json --out-dir <dir> --repeat 2` |
| 计时汇总 | `python tools/summarize_timing.py <out_dir>` |
| 两次运行对比 | `python tools/compare_runs.py <dir_a> <dir_b>` |

**运行纪律**：不在同一 MIG 切片上并发跑两条流水线。

### 6.2 默认值（末次确认）

| 配置 | 默认 | 配置 | 默认 |
|---|---|---|---|
| `blas_threads` | 1 | `hypothesis_chunk` | 0（原路径） |
| `seed` | 7351 | `tail_pipeline` | **false** |
| `pool_start_method` | null（自动） | `score_cache_mb` | 128 MiB/进程共享，0 = 关 |
| `geometry_cache_mb` | 512 | `tm_cache` | `"auto"`（`$XDG_CACHE_HOME/protassem/tm.sqlite3`） |
| `inference_mode` | `"joint"` | `encoding_cache_mb` | 256（仅 split 模式使用） |
| `allow_tf32` | null → 跟随模式 | CLI | `--num-processes 10 --batch-size 10` |

### 6.3 产物判读

| 文件 | 含义 |
|---|---|
| `assembled_complex.cif` / `assembled_complex_all.cif` | 主产物（过滤版 / 完整版） |
| `refined_complex.cif` | Step4 精修产物 |
| `assembly_summary.txt` | 运行摘要（`final_status`、接受域数、合并链数、跳过输入） |
| `performance.jsonl` / `server_timing.jsonl` | 客户端/服务端埋点 |
| `<request_out>/candidates.jsonl` | O6 候选台账（**权威候选顺序**） |

---

## 7. 代码、标签与文档索引

### 7.1 分支与标签（均已推送）

| ref | commit | 含义 |
|---|---|---|
| `main` = `feat/old-card-closeout` | `29c2019` | 当前交付 |
| `checkpoint/audit2` | `6c4ed00` | 第二轮审计后的**已验证代码态** |
| `checkpoint/closeout` | `decc52f` | 老卡 S0–S7/P1–P5 收口完成 |
| `checkpoint/t10` | `552d3e9` | 新任务卡 T00–T10 完成 |
| `checkpoint/phase1` | `087a013` | 阶段一功能加固完成 |
| `checkpoint/baseline` | `5b016f5` | **最初版本**（加固前） |
| `feat/engineering-hardening` | `087a013` | 阶段一分支 |
| `feat/runtime-metrics` | `552d3e9` | 新任务卡分支 |

回退：`git switch --detach checkpoint/<name>`；
独立对照：`git worktree add <dir> checkpoint/baseline`。

### 7.2 关键提交（收口阶段，时间顺序）

`af02707` 接口定稿 → `c8ede8c` P3 生命周期 → `592d617` O6 修复 → `60ecaa2` O6 验收+基线 →
`e8d4f99` P4 → `1ad9646` P5 → `44f889f` TM 指纹修复 → `decc52f` 收口汇总 →
`f9e3ef1` P1-1/P1-2 → `bd92f45` P1-3 → `9b5947f` P1-4 → `4b5ad7e` P2-5 → `9b6c3c6` P2-6 →
`8f1d2ce` P2-7 → `aee3e78` 审计验收 → `28d4440` 审计2 #1–#3 → `60ea283` 审计2 #4 →
`7517f11` 报告准确性 → `6c4ed00` 审计2 回归文档

### 7.3 文档地图

| 文档 | 管什么 |
|---|---|
| `ENGINEERING_HARDENING_PLAN.md`（v4.13） | **老任务卡权威方案**：事实核对、S0–S7、P1–P11、附录 C 开放项、**附录 D 接口定稿** |
| `MASK_REGISTRATION_OPTIMIZATION_TASKS.md` | **新任务卡权威方案**：T00–T10、统一验收、第一轮不做 |
| `PROJECT_ARCHITECTURE.md` | **架构变更台账**（每步一行：变更/影响面/验证） |
| `RUNTIME_ASSEMBLY_LOGIC_REVIEW.md` | 运行逻辑与组装规则逐条核对 + 源码定位索引 |
| `INFERENCE_ENGINEERING_PLAN.md` | 阶段二原始方案（已被 `ENGINEERING_HARDENING_PLAN.md` §4 与附录 D 取代/更正） |
| `ALGORITHM.md`、`STEP4_ASSEMBLY_NOTES.md`、`组装算法逻辑.md` | 算法背景（既有） |
| `tests/reports/*.md` | 每一步的验收报告（24 份） |

### 7.4 主要报告

| 报告 | 内容 |
|---|---|
| `2026-09-14_closeout_summary.md` | 老任务卡收口汇总（9 次真实运行矩阵） |
| `2026-09-14_audit_fixes_acceptance.md` | 两轮审计 11 项的修复与验收（含证据分级） |
| `2026-09-13_o6_candidate_order.md` | O6 候选消费确定性 |
| `2026-09-14_p4_scoring_cache.md` / `2026-09-14_p5_tm_cache.md` | P4 / P5 |
| `2026-09-13_p3_pool_lifecycle.md` | P3 池生命周期 |
| `2026-09-13_t00…t10_*.md`（11 份） | 新任务卡逐卡验收 |
