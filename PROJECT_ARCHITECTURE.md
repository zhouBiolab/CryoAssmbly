# protassem 项目架构文档

> ⚠️ **组装(Step 3)拟合的当前权威逻辑见 `组装算法逻辑.md`（全局轮次模型）。**
> 下文"统一队列（逐个弹出）/每链轮次"等描述属更早架构，已被全局轮次重写取代。

输入实验冷冻电镜密度图 + 若干链的结构模板，自动完成
**体素化 - 点云采样 - PARENet 配准拟合 - 统一队列组装 - 同源域精修 - 同源链精修**，
输出组装好的复合物 CIF。

## 架构变更记录（工程化硬化，阶段一）

> 只记录对**接口、边界行为、模块职责**有影响的变更；算法流程、阈值、候选顺序不变。
> 方案与验收标准见 `ENGINEERING_HARDENING_PLAN.md`（v2）；分支 `feat/engineering-hardening`；
> 每步一次提交，提交号见 `git log --oneline feat/engineering-hardening`。

| 步骤 | 变更 | 影响面 | 验证 |
|---|---|---|---|
| S0a | 采样器调用外部 `Sample` 改用 `subprocess.run([...])`（不再 `os.system` 字符串拼接），检查返回码与非空输出；新建 `tests/`（`test_sampler.py`） | `sampling/sampler.py`：函数签名与返回值不变 | `tests/test_sampler.py` 5 项；真实二进制烟测（输出目录含空格）通过 |
| S0b | `core/performance.py` 重写为可读版（行为不变：`performance.jsonl` + `performance_summary.json`）；`fitting/pipeline.py` 拆行、删除未用的 `nullcontext` 导入 | `core/performance.py`、`fitting/pipeline.py` | `tests/test_metrics.py` 5 项 |
| S0c | 两份方案文档入库（`ENGINEERING_HARDENING_PLAN.md` v2、`INFERENCE_ENGINEERING_PLAN.md` v1.1） | 仅文档 | — |
| S1 | `main.py` 手写 argv 解析 → `argparse` + `parse_intermixed_args()`：选项可位于位置参数之前/之间/之后；未知选项、缺值、非数字、位置参数个数错误统一退出码 2 且打印原因；`core/io.read_param_file` 的格式错误带文件名与内容 | `main.py`、`core/io.py`：选项名、默认值、开关语义不变 | `tests/test_cli.py` 17 项；CLI 烟测 5 类错误退出码 2 |
| T00 | **固定基线、依赖与测试输入**：新增 `tools/benchmark_registration.py`（`manifest` 生成 + `run` 重放，只做固定配准、产物写独立目录）；manifest 记输入 sha256/点数/参数/顺序/seed + 依赖来源；实测确认 **pareconv 实际从共享项目 `/xiangyux/PARENet-main` 加载、未使用仓库副本**，GPU 为 **MIG 7g.80gb 切片** | `tools/benchmark_registration.py`（新）、`tests/cases/registration_manifest.json`（新） | 重放两次成功：3 个目标（8006/3565/1672 点）各 12 个预测，best overlap 0.3268/0.0527/0.058；同输入两次重放差约 29%%（需冷/暖重复）；报告 `tests/reports/2026-09-13_t00_registration_baseline.md` |
| P3 | **复用运行级 CPU 池**：新增 `runtime/execution.py`（惰性共享池、顺序归并、幂等释放、workers<=1 串行）；`runtime/pool.py` 提供 `open_pool`/`close_pool`；`context` 显式穿过拟合调用链；删除 4 类临时建池（每批 CC 池、每批局部优化池、TM 预填池、预筛池）；新增 `pool_start_method` 配置 | `runtime/execution.py`（新）、`runtime/pool.py`、`runtime/config.py`、`pipeline.py`、`assembly/orchestrator.py`、`assembly/chain_fitter.py`、`assembly/domain_fitter.py`、`fitting/pipeline.py`、`fitting/local_optimizer.py`、`core/similarity.py`、`tests/test_runtime_execution.py`（新）、`tools/pool_bench.py`（新） | 池创建 **79 → 1**（串行 0）；同配置（10 worker）三个 CIF md5 与冻结基线**一致**；1 worker 的结果差异经冻结基线代码复现（同 md5），判定为既有配置敏感性而非 P3 引入；无残留进程；fork ≈25 ms vs spawn ≈52 ms 单独记录 |
| P2-fine | **时间账细化**：客户端新增 `request_submit` / `first_pred` / `candidate_scan` / `cc_verify` / `cc_candidate_initial` / `final_select` / `save_result` / `analyze_sources`；服务端（`demo_mask`）新增 `server_queue_wait` / `server_request_total` / `server_preprocess` / `server_masks` / `server_mask_preprocess` / `server_to_gpu` / `server_forward` / `server_postprocess` / `server_write_pred`（写入请求目录的 `server_timing.jsonl`）；新增 `runtime/pool.py::timed_pool` 记录 `pool_start`/`pool_close`；新增汇总工具 `tools/summarize_timing.py` | `protassem/runtime/pool.py`（新）、`fitting/pipeline.py`、`fitting/local_optimizer.py`、`fitting/demo_mask.py`、`core/similarity.py`、`assembly/orchestrator.py`、`assembly/chain_fitter.py`、`tools/summarize_timing.py`（新）、`tests/test_runtime_pool.py`（新） | 84 项测试；`test/1` 实测：未归因时间从 260 s 降到 1.13 s（缺口即 `final_select` 267.6 s）；服务端 postprocess 485 s > forward 375 s > 写盘 92 s；池创建 79 次仅 4.34 s |
| P2 | **分阶段计时**：`Metrics` 迁移并扩展到 `runtime/metrics.py`（事件字段 + `summary()` + `total_wall_s`）；删除 `core/performance.py`；埋点覆盖 standardize / voxelization / sampling / domain_split / tm_prefill / prescreen / assembly_rounds / mask / domain_assembly / build_complex / refine_step4–5 / pipeline_total，拟合内部为 gpu_wait / cc_batch / local_optimize；摘要附 `performance_summary` 路径 | `protassem/runtime/metrics.py`（新）、`pipeline.py`、`assembly/orchestrator.py`、`assembly/chain_fitter.py`、`assembly/domain_fitter.py`、`fitting/pipeline.py`、`tests/test_metrics.py` | `tests/test_metrics.py` 6 项；`test/1` 实测两组（128 / 1 线程）分阶段拆解与产物等价性，见 `tests/reports/2026-09-13_p2_staged_metrics_and_thread_effect.md` |
| P1 | **运行配置与线程控制**：新增 `protassem/runtime/config.py`（`RuntimeConfig`：JSON 加载、未知键报错、`apply_thread_env()`、`describe_effective_threads()`）；`main.py` 改为**两段式导入**（先解析参数并应用线程 env，再导入 NumPy/Torch）；新增 `--runtime-config <json>`；`run_pipeline` 记录实测生效线程数；`requirements.txt` 登记 `threadpoolctl==3.5.0` | `protassem/runtime/config.py`（新）、`main.py`、`pipeline.py`、`requirements.txt` | `tests/test_runtime_config.py` 7 项（含子进程实测 BLAS=1、main 导入不加载 numpy/torch）；实测默认 OpenBLAS 128 线程 / torch 112 → 应用配置后为设定值 |
| R2 (O4) | **空结果输出契约**：`build_complex` 在 0 组件时删除同名旧文件、不写出、返回 `None`；新增 `clear_stale_outputs()` 清理程序管理产物；`orchestrator` 过滤版为空时保留完整版并把回退链改为 `homo or refined or complex or all`；运行摘要新增 `assembled_complex_all/_filtered`、`final_status` 与 接受域数/合并链数/过滤数量；`refine_step` 在无组装产物时明确跳过、备份日志指向实际文件 | `assembly/complex_builder.py`、`assembly/orchestrator.py`、`assembly/refine_step.py` | `tests/test_complex_output_contract.py` 6 项；真实集成验证（单链 + 0.99 阈值 + 预置旧产物）见 `tests/reports/2026-09-13_o4_empty_output_contract.md` |
| R1 | **复合物域链合并丢链修复**：`assemble_domain_chains` 补传 `is_complex`（否则两条链的域被并进同一条链、残基 ID 重复、Biopython 报错、整链丢弃 → 空复合物）；`merge_domains` 复合物分支新增 `chain_map_of()`，按原始 `chain_records` 的 `chain_map` 把占位链号恢复为真链号（真链号原样通过，不二次映射）；`_filtered_domain_cif` 同步透传 `is_complex` | `assembly/domain_assembler.py` | `tests/test_domain_assembly.py` 4 项（含旧行为取证）；同配置真实运行 421 s：`assembled from 5 domains (cc=0.4490)`，最终产物链号 `Q`,`R`，摘要 `As domain chain: 1` |
| S7 | 集成验证与归档：`tests/`（9 个测试文件 + `fixtures.py` + `cases/` + `reports/`）、`tools/compare_runs.py`、README 去私有路径并补 CLI/校验说明、`.gitignore` 忽略真实数据与运行产物 | `tests/**`、`tools/compare_runs.py`、`README.md`、`.gitignore` | `test/1` 端到端对比：三个最终 CIF 与 `assembly_summary.txt` **逐字节一致**（md5 相同，967 s → 956 s，不作加速结论）；单元测试 63 项全过 |
| S4b | 链号空间一致性：**三次定向探针**（派生 `Q/R` 复合物 CIF + `--complex-domain-opt`）：引入点确认在 `chain_fitter.py:237-240`（逐域微调产物 `work/chain_improve_<cid>/domain_*.cif` 带占位链号）；`final_results` 全部产物为真链号 `Q/R`（`_accept_chain` 应用 `chain_map` 恢复），**未观测到泄漏**；残留分支（backfill + 同源 Step4）未证伪，按约定未改代码 | 仅文档 + `tests/reports/2026-09-12_s4b_chain_space_probe.md` | 3 次真实运行（2290 s / 176 s / 372 s）
| S6 | 可移植性：删除 `parenet/config.py` 中无读取者的 `dataset_root='/xiangyux/PARENet-main/data/demo'`（目录不存在、推理不读取；文件顶部注明推理不使用数据集配置）；`sw_mask.py` 与 `refine_energy.py` 的 `__main__` 调试示例改为 argparse 参数；`Sample_based_VoxEM.py` 的 `os.system` 改 `subprocess.run` 列表参数（脚本保留）；`DomainParser.py` 三处 `shell=True` 改为列表参数 + `cwd=tmp_dir` + `env['DSSP_PATH']` | `fitting/parenet/config.py`、`fitting/sw_mask.py`、`assembly/refine/refine_energy.py`、`sampling/extract_points/Sample_based_VoxEM.py`、`assembly/domain_parser/DomainParser.py` | `tests/test_portability.py` 静态断言（排除 `pareconv_src/`、`tests/`、`test/`）；真实链数据域切分烟测：`split_domains` 成功、PDB 产物与基线**逐字节一致**、域 TXT 仅第 0 行由 `2.0` 变为原文 `2.000000`（数值相同，其余逐字节一致） |
| S5 | 链号池拆成两个用途明确的函数：`logical_chain_ids()`（A-Z/a-z/AA..ZZ，链号去重重排用）与 `pdb_placeholder_ids()`（A-Z/a-z/0-9，**保留现有 62 容量**）；`cif_to_pdb_placeholders` 超容量抛 `ValueError`（当前没有纯 CIF 替代通路，不静默截断）；`_prepare_chains` 把跳过的输入记入 `skipped_inputs` 并在无可用链时 `RuntimeError`（CIF→PDB 占位失败不再丢链）；`_cc_worker` 与 `_pre_screen_cc_worker` 失败改为带文件名抛 `RuntimeError`（不再返回 `cc=None` / `0.0`）；运行摘要新增 `input_files` / `processed_chains` / `skipped_inputs` / `skipped_detail` | `core/structure.py`、`pipeline.py`、`assembly/orchestrator.py`、`fitting/pipeline.py` | `tests/test_chain_ids.py` 9 项：两个池的顺序与容量、62 链满容量（数字占位在使用）、63 链报错、跳过输入记录、全不可用 `RuntimeError`、两个 worker 带文件名抛错、数字占位链号在 `read_structure`/`calculate_cc_mask`/USalign 上可用 |
| S4a | `align_by_resid()` 改为按 `(链号, 残基号, 插入码)` 匹配、只读第一个 model；新增 `mob_chain_map`（mob 链号 → ref 链号空间，**只用于构造匹配键，不修改结构**）；匹配不足时 `log.warning` 并返回 False（不写文件）。链号映射从原始 `chain_records` 取：新增 `AssemblyOrchestrator.chain_record(cid)` 作为唯一查找入口，`refine_step._backfill_chains_as_domains` 与 `orchestrator._attach_domain_details` 共用（删除 refine_step 的模块级 `_chain_record`）；`chain_fitter` 的两处调用 ref/mob 同空间，保持默认 `None` | `core/structure.py`、`assembly/refine_step.py`、`assembly/orchestrator.py` | `tests/test_align_by_resid.py` 5 项：单链叠合、多链同编号（新实现按链匹配，旧逻辑 RMSD > 5 Å 取证）、占位映射（无映射 → False 且不写文件；有映射 → 成功）、匹配不足、只读 model 1 |
| S3 | 点云 TXT 读写统一到 `protassem/core/points_txt.py`：解析按**顺序成对**（不再用绝对行号奇偶），字段数严格 4 列，错误带 `文件:行号`；新增 `write_filtered`（按原始行回写，保文本精度）与 `write_point_cloud`（由数组生成）。替换 7 处解析与 2 处写出：`core/io.load_sample_points`、`demo_mask.load_sample_points`、`sw_mask.load_sample_points`/`_save_point_cloud_as_txt`、`Supporting.load_sample_points`、`domain_pdb_txt.load_sample_points_with_info`/`save_domain_txt`、`masker.mask_fitted_region`（删除 `_read_points_with_lines`/`_save_filtered_txt`）、`orchestrator._target_has_points`；`domain_pdb_txt` 库函数内 5 处 `sys.exit` 改为抛异常 | `core/points_txt.py`（新）、`core/io.py`、`fitting/demo_mask.py`、`fitting/sw_mask.py`、`fitting/masker.py`、`sampling/extract_points/Supporting.py`、`assembly/domain_parser/domain_pdb_txt.py`、`assembly/orchestrator.py` | `tests/test_points_txt.py` 17 项：数值/头部、错位反例（旧实现点法向量错位、新实现报错）、成对校验、头部不足、非数字、过滤写回逐字节一致 |
| S2 | `run_pipeline()` 增加入口校验：密度图存在且为 `.mrc`、结构列表非空且文件都存在、`resolution > 0` 且有限、`contour` 为**有限数值**（`None` 被拒绝）、`voxel_size > 0`；校验发生在 `os.makedirs`/`setup_logging` 之前。内部函数不再重复校验 | `protassem/pipeline.py`：新增 `_validate_inputs`；`contour=None` 由采样器兜底改为入口报错 | `tests/test_pipeline_inputs.py` 11 项（含"失败时不建输出目录"） |

---

## 总体流程

```
密度图(.mrc) + 链结构(.pdb/.cif)
  |
  0. 输入标准化              读取内部 chain ID，多链文件保留为复合物；链号重复则自动去重重排（A-Z/a-z/AA..ZZ，>52 多字符走 CIF）
  |     入口边界：main.py 用 argparse 解析并校验用法；run_pipeline 在建目录前校验 mrc/结构/分辨率/contour
  |
  1. 体素化 voxelize        每条链/复合物模板 -> 模拟密度(.mrc)
  |
  2. 点云采样 sampling       密度图(实验+模拟) -> 带法向量的点云(.txt)
  |
  3. 全局轮次拟合+组装 assembly
  |     - 预筛 pre-screen（并行 cc_mask，已就位链直接接受+掩码）
  |     - PARENet 配准（GPU 常驻服务）
  |     - 两阶段局部优化（CPU，多进程 CC + 多副本）
  |     - 链和域按回旋半径全局分轮拟合（达标即收+掩码）
  |     - 复合物独立阈值 + 可选复合物域优化
  |
  4. 同源域精修 refine（自动触发，--no-refine 关闭）
  |
  5. 同源链精修 homo-chain-refine（可选，--homo-chain-refine）
  |
  复合物(.cif)
```

---

## 一、目录结构

```
demo_reg/
+-- main.py                      命令行入口
+-- compute_cc_mask.py           独立算 cc_mask
+-- check_clash.py               CA/P 原子重叠检测（numpy/torch）
+-- geo_sym_refine.py            独立同源 refine CLI
+-- geo_test.py                  对称 refine 评测脚手架
+-- requirements.txt
+-- protassem/                   主包
    +-- pipeline.py              三步总流程编排（含输入标准化）
    +-- runtime/                 阶段二：RuntimeConfig（config.py）、分阶段计时（metrics.py）、
    |                            共享池与运行上下文（execution.py / pool.py）
    +-- core/                    公共底层
    |   +-- scoring.py           cc_mask 计算
    |   +-- structure.py         PDB/CIF 读写、域切片对齐
    |   +-- similarity.py        USalign 封装(TM-score、Seq_ID)
    |   +-- numba_kernels.py     numba 加速核函数
    |   +-- constants.py         原子序数、范德华半径
    |   +-- io.py
    |   +-- points_txt.py        点云 TXT 单一读写（5 行头 + 成对数据行）
    |   +-- USalign              外部二进制
    +-- voxelize/                Step 1
    |   +-- mol_to_mrc.py
    +-- sampling/                Step 2
    |   +-- sampler.py
    |   +-- extract_points/      VoxEM
    |   +-- Sample               外部二进制
    +-- fitting/                 Step 3a 拟合
    |   +-- pipeline.py          两阶段局部优化+监控早停+多进程CC
    |   +-- parenet_client.py    PARENet 常驻服务客户端（文件信号IPC）
    |   +-- demo_mask.py         PARENet 推理（含 --server 模式）
    |   +-- local_optimizer.py   密度梯度+scipy 多副本并行
    |   +-- masker.py            已接受区域掩膜
    |   +-- sw_mask.py / utils.py
    |   +-- parenet/             vendored PARENet 模型+权重
    |   +-- pareconv_src/        pareconv 源码（需编译）
    +-- assembly/                Step 3b 组装
        +-- orchestrator.py      组装总入口（准备+域分割+复合物域优化+收尾）
        +-- unified_queue.py     全局轮次主循环（链域按回旋半径分轮拟合）
        +-- chain_fitter.py      链/复合物拟合（接受/拒绝判定+域微调+链姿态降域）
        +-- domain_fitter.py     域拟合一次（同源宽松/候选/平台期接受）
        +-- domain_assembler.py  域链组装（merge_domains+单域/多域/复合物域合并）
        +-- assembly_opt.py      优化辅助（AssemblyOptConfig+multi_accept+clash）
        +-- complex_builder.py   合并CIF+报告
        +-- domain_splitter.py   DomainParser 切域
        +-- domain_parser/       DomainParser+dssp 二进制
        +-- refine_step.py       Step4 门控+backfill
        +-- homo_chain_step.py   Step5 门控
        +-- homo_chain_refine.py Step5 核心(并行)
        +-- refine/              vendored 同源域枚举精修
```

---

## 二、拟合模块 fitting

### 2.1 PARENet 配准（GPU 常驻服务）

parenet_client.py 管理单例服务进程 demo_mask.py --server：
- get_server() 懒启动，模型只加载一次，常驻 GPU
- 文件信号 IPC：请求=JSON stdin, 完成=_PARENET_DONE, 早停=_PARENET_STOP
- ParenetRequest 模拟 Popen 接口（poll/terminate）
- 服务日志 -> parenet_server.log，atexit 自动关闭
- 主进程不初始化 CUDA -> fork 并行安全

### 2.2 两阶段局部优化（fitting/pipeline.py）

- _batch_cc 支持 multiprocessing.Pool 并行算 CC
- CC 不重复计算：run_fitting 返回已验证 cc_mask，orchestrator 直接复用
- 触发阈值：链 0.29 / 域 0.28（代码常量）

阶段1：攒够 batch_size 个 pred -> 并行 CC -> 高于阈值的逐个局部优化 -> 达标则早停

阶段2：取 CC 前5 + 混合分数前5（不重叠）-> 逐个优化 -> 选最高

混合分数：cc_w * cc_mask + 1.0 * overlap（cc < 0.2 时 cc_w = 0.5，否则 1.0）

### 2.3 局部优化器（fitting/local_optimizer.py）

6 副本并行密度梯度上升 -> CC 下降则 scipy -> 精细优化 -> 下降则回退
返回 (success, path, cc)

---

## 三、组装模块 assembly

### 3.0 模块化架构

组装模块拆分为五个文件，各有明确职责：

| 文件 | 行数 | 职责 |
|------|------|------|
| `orchestrator.py` | ~560 | 薄入口：准备链、预筛、域分割、复合物域优化、收尾（构建复合物+精修调度） |
| `unified_queue.py` | ~240 | 全局轮次主循环：每轮按回旋半径走一遍待处理项，达标即收+掩码+break；轮末单选+降阈值 |
| `chain_fitter.py` | ~250 | 链/复合物拟合：相似分组、接受/拒绝判定、域微调(improve_accepted)、链姿态降域 |
| `domain_fitter.py` | ~60 | 域拟合一次：达标(同源宽松)→接受+掩码 / 未达标→候选 / 平台期→直接接受 |
| `domain_assembler.py` | ~160 | 域链组装：merge_domains（单链残基序合并 / 复合物多链合并） |

共享状态通过 `orchestrator` 实例传递（chain_records, domain_records, accepted_chains 等），各模块不维护额外全局状态。

### 3.1 核心设计：全局轮次

#### 设计动机

旧架构是两阶段：Phase 1 拟合所有链 → Phase 2 拟合所有域。问题是：
- 大域（回旋半径大于某些小链）必须等所有链拟合完才能开始，期间密度可能被低质量链占用
- 链和域之间无法按"大优先"统一调度

新架构用**全局轮次**替代两阶段：每轮把当前所有待处理项（pending 链 + available 域）按回旋半径降序走一遍。链拟合失败时其域**即时入池**，下一轮即可与剩余链按回旋半径混排。

#### 核心策略：大片段优先占密度

回旋半径大的片段包含更多空间信息，配准更准确。让大片段先拟合，先从实验密度中占位并扣除（mask），能减少后续小片段的搜索空间。

```
[Round N] 待处理项(按回旋半径降序): [复合物A+B(大), 链C(中), 链D(小)]
  |
  复合物A+B 拟合 -> cc≥complex_threshold(0.35) -> 接受+掩码 -> 本轮 break
  |
  [Round N+1] 链C 拟合 -> 失败 -> C的域置 available 入池
  待处理项重排: [C-d1(中), 链D(小), C-d2(小)]  <- 按回旋半径
  |
  某项达标 -> 接受+掩码+break；无人达标 -> 轮末单选+降阈值
```

#### 关键：达标即收 + 掩码 + 重排

每接受一项就 break、重算剩余密度，下一轮在掩码后的新密度上重新评估剩余项。
域"即时入池"而非"批量延迟"——失败链的域下一轮就能和小链按回旋半径混排，一个大域可能排在小链前面。

### 3.1.1 预筛（pre-screen，默认开，`--no-pre-screen` 关）

全局轮次开始前，`_pre_screen_chains` 并行计算各链在**模板原始位姿**下的 cc_mask：
≥ 接受阈值（复合物用 complex_threshold）的链先互相做 clash 仲裁（重叠 > clash_overlap_thr 时保 CA 多者），
胜者按回旋半径大者优先依次接受 + 掩码；对已就位的链省去一次完整配准，输的链留待全局轮次正常拟合。

### 3.2 组装优化辅助（assembly_opt.py）

AssemblyOptConfig dataclass 集中优化参数：

| 参数 | 默认 | 含义 |
|------|------|------|
| chain_similar_relax | 0.03 | 同源组已有接受时链阈放宽 |
| domain_similar_relax | 0.025 | 相似域已接受时域阈放宽 |
| clash_overlap_thr | 0.10 | CA 重叠比 clash（保 CA 多者） |

辅助函数：
- ca_count(pdb) -- CA/P 原子数
- ca_overlap(pdb_a, pdb_b) -- 复用 check_clash
- select_round_end_candidate(candidates, ca_margin) -- 轮末单选：cc 最高；相差≤ca_margin 偏 CA 多

### 3.3 链/复合物拟合（chain_fitter.py）

#### 流程

1. **相似分组**（USalign TM >= similarity_threshold）
2. **探针跳过**：与已失败链相似 且 该组无成功 -> 域加入队列
3. **拟合 + 阈值判定**：
   - 普通链：`chain_threshold`（默认 0.45）
   - 复合物：`complex_threshold`（默认 0.35，独立参数）
   - 同源组已有接受 -> 阈值 - chain_similar_relax（0.03）
4. **接受后逐域微调**（默认开启，`--no-improve-accepted` 关闭）
5. **拒绝处理**：
   - 有域 -> 先尝试 `try_domains_via_chain_pose`（利用链姿态切域优化）
   - 仍失败 -> 域加入统一队列

#### 为什么复合物需要独立阈值

复合物由多条链组成，体积大、灵活性高，cc_mask 天然低于同等大小的单链。
若用 chain_threshold=0.45 判定，很多合理的复合物拟合会被拒绝。
complex_threshold 默认 0.35，允许复合物以更低的 CC 被接受。

#### try_improve_chain_with_domains 策略

链被接受后，如果有多个域（>=2），尝试"逐域微调"：
1. 每个域按链姿态切出（align_by_resid）
2. 每个域单独 local_optimize
3. 所有域合并回完整链
4. 如果合并后的 CC 优于原链 → 替换

这比整链优化更精细，因为各域在密度中可能有微小的相对位移。

### 3.4 域拟合（domain_fitter.py）

`fit_domain_once` 在全局轮次里拟合单个域一次，返回 accepted / candidate / failed：

- **同源宽松**：与某已接受域同源（`_is_homolog_accepted_domain`，group_id 查表）→
  有效阈值 = 当前阈值 − domain_similar_relax(0.025)，下限 min_domain_threshold。
- **达标**（cc ≥ 有效阈值）→ 接受 + 掩码，返回 accepted（主循环本轮 break）。
- **平台期直接接受**：连续 ≥3 轮 cc 变化 < 0.02 → 判定到顶，按当前结果接受（不再空耗后续轮）。
- 否则记为本轮 **candidate**（带 cc/pdb，留作轮末候选）。

> 阈值衰减、轮末单选、降阈值都在主循环 `unified_queue` 里：前 3 轮 −0.015、之后 −0.020，下限 domain_min_cc；
> 一轮无人达标时在候选里选 cc 最高的一个（相差 ≤0.03 偏 CA 多）。每链独立阈值/轮次的旧 DomainRoundTracker 已移除。

### 3.5 复合物域优化（`--complex-domain-opt`）

#### 设计动机

复合物作为整体拟合可能 CC 可接受，但内部各链的位姿不一定最优。
启用此选项后，复合物在域分割阶段被拆分为内部单链，各自做域分割，
接受后逐域微调时能单独优化每个域的位置。

#### 实现流程

```
complex_A+B.pdb → split → chain_A_1.pdb, chain_B_1.pdb
                        → 各自 voxelize + sample + DomainParser
                        → 域记录含 source_chain_id 字段
```

- 复合物拟合接受后，improve_accepted 对各域单独优化后按 source_chain_id 多链合并
- 若合并后 CC 更高则替换原拟合结果

#### 命名约定

为了保证 DomainParser 输出的 TXT 文件与 PDB 文件能被 `find_domain_files` 正确配对，
内部链被复制到子目录并简化命名为 `chain_{cid}_1.pdb`。
这样 TXT 前缀 `chain_{cid}_1` 与 PDB 文件名前缀一致，regex `(chain_[^_]+_\d+)` 能匹配。

### 3.6 域链组装（domain_assembler.py）

失败链走域路径，不回退整链：
- 收集该链已接受的域 -> 按残基序合并（merge_domains）
- 0 个域 -> 该链不进复合物
- 1 个域 -> 直接作为单域链
- 2+ 个域 -> 按残基顺序合并为单链 CIF
- 复合物域 -> 按 source_chain_id 创建多链 CIF（BioPython Structure 中每个 source_chain_id 一个 Chain 对象）

### 3.7 复合物构建 + 质量闸门（complex_builder.py）

build_complex 合并已接受组件，产出两个结果：
- assembled_complex_all.cif（完整版，含全部已接受域，不过滤）
- assembled_complex.cif（过滤版，按 complex_min_cc 逐结构域剔除 cc 低的域；整链/复合物不做域级过滤）
输入阶段已去重链号，正常无需 remap（仍保留 remap 兜底）
assembly_summary.txt：每组件整体 cc + 逐域 cc + 排除域列表

### 3.8 Step 4 -- 同源域精修（refine_step.py，自动触发）

触发条件：存在同源链 + 已分域 + 已做域拟合
backfill：整链接受的链按域切片落到 fitted_domains/
调 vendored ChainEnumerator（TM-score 找同源域组 -> 跨链对齐 -> 穷举组合 -> 连接能量最优）
产物 refined_complex.cif。--no-refine 关闭。

### 3.9 Step 5 -- 同源链精修（homo_chain_step.py，可选 --homo-chain-refine）

触发：同源链（Seq_ID 分组）且组内 cc 有分化
**残基保护**：候选丢失 >10% 残基时跳过（防止不完整链替换完整链）
好链当模板：序列叠合搬到差链位姿 + 密度优化 + clash 门控
**域补回**：clash 门控后检测同源组内残基覆盖差异，从完整链补回缺失域
（序列叠合 → 合并缺失残基 → density 优化 → clash 门控，不卡 CC 下降）
四处并行：分组 USalign / cc_mask / 候选优化 / 域补回
产物 homo_chain_refined_complex.cif

---

## 四、独立工具

### geo_sym_refine.py（独立同源 refine CLI）

与 Step 5 类似但完全独立，可单独运行：
1. 拆单链 -> Seq_ID 同源分组（并行 USalign）
2. 并行重算链 cc_mask
3. 好链保留不动；**残基保护**：候选丢失 >10% 跳过
4. 差链 x 每个 cc 更高的 donor：序列叠合+local_optimize
5. 顺序 clash 门控，优于原链+eps 才替换
6. **缺失域补回**
7. 重组输出

### geo_test.py（评测脚手架）

两套指标：
1. 复合物级主指标：USalign -mm 1 多链对齐，整体 TM-score + RMSD
2. per-chain 诊断：拆单链 vs native 匈牙利分配，逐链 TM/RMSD

### check_clash.py（重叠检测）

calculate_overlap_ratio_numpy(pdb1, pdb2, clash_distance=3.0)
被 assembly_opt、homo_chain_refine、geo_sym_refine 复用

---

## 五、外部工具

| 工具 | 位置 | 用途 | 调用方式 |
|------|------|------|---------|
| PARENet | pip pareconv + parenet/ | 点云配准 | 常驻服务GPU |
| USalign | core/USalign | TM-score/Seq_ID | subprocess |
| DomainParser | assembly/domain_parser/ | 切域 | subprocess |
| Sample | sampling/Sample | 点云采样 | subprocess |
| cc_mask | core/scoring.py | Pearson 相关 | import |
| 局部优化 | local_optimizer.py | 密度梯度+CC | import |
| check_clash | check_clash.py | CA 重叠比 | import |

---

## 六、阈值参数

### CLI 参数

| 参数 | 默认 | 含义 |
|------|------|------|
| --chain-threshold | 0.45 | 普通链接受阈值 |
| --complex-threshold | 0.35 | 复合物接受阈值（独立于链阈值，因复合物 CC 天然偏低） |
| --domain-threshold | 0.45 | 域起始接受阈值(每链独立衰减) |
| --domain-min-cc | 0.35 | 域拟合绝对阈值：衰减下限 + 多域同接门槛 |
| --complex-min-cc | 0.25 | 收尾高置信度筛选下限（与拟合无关） |
| --similarity-threshold | 0.85 | TM 相似判定 |
| --refine-tm | 0.75 | Step4 同源域分组 |
| --num-processes | 10 | 并行进程数 |
| --batch-size | 10 | 监控批次 |

### 代码内常量

| 参数 | 值 | 位置 |
|------|-----|------|
| cc_threshold（链/域） | 0.29 / 0.25 | fitting/pipeline.py |
| chain_similar_relax | 0.03 | assembly_opt.py |
| domain_similar_relax | 0.025 | assembly_opt.py |
| clash_overlap_thr | 0.10 | assembly_opt.py |
| 轮末 CA 优先 margin | 0.03 | unified_queue.py (CA_TIEBREAK_MARGIN) |
| 阈值衰减(前3轮/之后) | 0.015 / 0.020 | unified_queue.py |
| 平台期接受(Δcc/轮数) | 0.02 / 3轮 | domain_fitter.py |

---

## 七、安装

1. conda create -n point python=3.8 && conda activate point && pip install -r requirements.txt
2. pareconv 编译：cd protassem/fitting/pareconv_src && pip install -e .
   cd pareconv/extensions/pointops/ && python setup.py install
3. chmod +x protassem/core/USalign protassem/sampling/Sample
   protassem/assembly/domain_parser/domainparser2.LINUX
   protassem/assembly/domain_parser/dssp

## 八、使用

python main.py <data_dir> --log
python main.py <density.mrc> <结构目录> <分辨率> <contour> [输出目录] --log

- 选项可放在位置参数之前、之间或之后（`parse_intermixed_args`）。
- 入口校验：`run_pipeline()` 在建目录之前检查密度图存在且为 `.mrc`、结构文件列表非空且都存在、
  `resolution > 0`、`contour` 为有限数值、`voxel_size > 0`；`contour=None` 会被拒绝
  （`core/scoring.py` 直接执行 `exp_map > contour`）。自动目录模式仍从 `contour_level.txt` 读数值。
- 用法错误（未知选项、选项缺值、非数字、位置参数个数不是 1/4/5）退出码为 2，并打印具体原因。

## 九、输出

```
<output>/
+-- pipeline_<时间戳>.log
+-- voxelized/ sampled/ sampled_sources/
+-- assembly/
    +-- work/
    |   +-- chain_fit/chain_fit_N/
    |   +-- domain_fit/domain_fit_N_cid_dN/
    |   +-- temp_chain/temp_chain_N/
    |   +-- fitted_domains/<链>/
    |   +-- domains/<链>/
    +-- final_results/
        +-- chains/ domain_chains/
        +-- assembled_complex.cif
        +-- refined_complex.cif           (Step4)
        +-- homo_chain_refined_complex.cif (Step5)
        +-- assembly_summary.txt
```
