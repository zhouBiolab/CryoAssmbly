# O6：候选消费确定性（老卡收口第 3–4 步）

日期：2026-09-13　代码：`feat/old-card-closeout`（实现提交 `592d617`；本报告 = 第 4 步验收）
输入：`test/1`（`EMD-8436.mrc` + 两条链，res 5.6，contour 0.04），1 BLAS 线程。
对照基线：**当前默认路径基线** `assembled_complex.cif` = `76638d0f97d4ee5775110c66b15e73c6`、
`refined_complex.cif` = `bd281f40e25352455ba34d41e6bbc9f7`（T01 后冻结；O6 前默认路径多次复现）。

**结论**：O6 通过。四次真实运行（默认配置两次、`--num-processes 1`、`tail_pipeline=true`）
**三个 CIF md5、三条接受决策、候选台账与候选消费序列全部逐位一致**，且与 O6 前的冻结基线相同
→ 已冻结 `tests/cases/baseline_after_o6.md5`（与 T01 基线同值，因为 O6 **没有改变默认路径的产物**）。

## 一、问题与机制（T10 已定位，本步修复）

- 旧消费循环：每 2.5 s `glob(registration/pred_*.pdb)`，取"本轮新出现的文件"凑 `--batch-size`（默认 10）批；
- 候选文件由服务端"每掩码 12 次评估 → 选最优 → 改名 → 删除其余 11 个"产生 → 文件出现时机
  由写盘/改名节奏决定（尾部流水线开关、机器负载、worker 数都会动它）；
- 结果：同一组候选、不同消费顺序 → 早停点与 `final_select` 的挑选不同 → 产物不同
  （T10：tail on 的 3 次运行里 2 次给出 `51009d69…`，off 的 10+ 次运行都复现旧基线）。

## 二、修复（接口见 `ENGINEERING_HARDENING_PLAN.md` 附录 D.1）

| 环节 | 变更 |
|---|---|
| 服务端 | 请求级候选台账 `candidates.jsonl`（`request_id` + 连续整数 `id` + 终态 `ok/filtered/error` + `end{ok/error/cancelled}`）；掩码分支按掩码序号、无掩码分支按 `(config, sampling)` 顺序发布；**先完整写出文件、再追加记录并 fsync** |
| 顺序依据 | `find_mask_files` 改为按名排序（原先取 `glob` 目录顺序，顺序无依据） |
| 客户端 | `CandidateConsumer`：固定 ID 区间批次；批内保持 CC 降序 + id 次序的逐个优化；**首个达标即早停**；`end` 到达后仍按批消费；`final_select` 仅在无早停且批次消费完后执行；缺 `end`/`end=error` 抛错；`cancelled` 正常；早停后确认请求结束再返回 |
| 请求身份 | `start_request(..., request_id=…)`（`task-seq`），发请求前清掉旧台账；客户端按 `request_id` 拒绝旧记录 |
| 随机源 | 父进程 `apply_seed(RuntimeConfig.seed)`（唯一父进程随机消费者是局部优化回退 `ScipyFitter`，`test/1` 历史运行从未触发） |
| 埋点修正 | `candidate_scan`（原意：glob 轮询耗时）→ **`candidate_stream`**（整个候选流消费时长）；`tools/summarize_timing.py` 同步 |

## 三、单测（假生产者：`tests/test_candidate_ordering.py` + `tests/test_candidate_ledger.py`，14 项）

| 场景 | 结论 |
|---|---|
| 快发 / 慢发 / 随机延迟 | 批次划分均为 `[(0,10),(10,20),(20,24)]`，优化序列一致 |
| worker 1 / 2 / 10 | 优化序列一致（`context.map` 按输入顺序归并） |
| 记录与 end 都在 t=0（服务端早已结束） | **仍按固定批次消费**（回归：不得一次全消费） |
| 分数并列 | 取最小候选 id；达标即停（不优化同批其余候选） |
| `filtered` / `error` 候选 | 跳过且不阻塞批次，计数与原因入库 |
| 缺 `end` / `end=error` | 抛错（不静默用部分结果） |
| `end=cancelled` | 正常返回 |
| 台账协议 | 半行缓冲、归属校验（拒绝旧台账）、重复 id / 未知状态 / 版本不符 → 抛错 |

## 四、真实运行矩阵（四次，全部与冻结基线逐位一致）

| 运行 | 配置 | 墙钟 | `assembled_complex.cif` | `refined_complex.cif` | 决策（A / B / 域） |
|---|---|---|---|---|---|
| a | 默认（10 worker） | 1248 s | `76638d0f…` | `bd281f40…` | 0.4192 拒 / 0.4235 受 / 2 域 0.4430 |
| b | 默认（10 worker，重复） | 1260 s | `76638d0f…` | `bd281f40…` | 同上 |
| c | `--num-processes 1` | 2009 s | `76638d0f…` | `bd281f40…` | 同上 |
| d | `tail_pipeline=true` | 1199 s | `76638d0f…` | `bd281f40…` | 同上 |

台账与消费序列（四次完全相同）：

| 请求 | 候选数 | id 连续 | 状态 | end | 消费候选数 | 消费序列 sha1 |
|---|---|---|---|---|---|---|
| `chain_fit_1/…/registration` | 24 | ✔ | 全 `ok` | `ok` | 58（三次请求合计） | `de02d03105ebff9f` |
| `chain_fit_2/…/registration` | 28 | ✔ | 全 `ok` | `ok` | 同上 | 同上 |
| `domain_fit/round03/A_d1/registration` | 115 | ✔ | 全 `ok` | `ok` | 同上 | 同上 |

- **worker=1 与 worker=10 现在结果一致** —— 这正是 P3 时期登记为 O6 的那个失败验收项，已闭环。
- **tail 开关不再改变产物**（d 与 a/b/c 完全一致），O6 前它曾 2/3 次改变产物。

## 五、性能观察（诚实记录，不作定论）

| 阶段 | O6 前（T10 off） | O6 后（run a） | 说明 |
|---|---|---|---|
| `pipeline_total` | 1074 s | 1240 s | **+15%** |
| `local_optimize` | 24 次 / 159.9 s | 28 次 / 208.9 s | 多优化 4 个候选 |
| `final_select` | 208.3 s | 271.4 s | 链 B 未早停 → 走了 final_select |
| `gpu_wait` | 610.0 s | 652.5 s | 批次对齐后等待更长 |

原因已定位：**搜索轨迹变了**。新顺序下链 B 的 24 个候选在同一批内按 CC 降序优化后没有候选达到早停阈值
（全部 ≈0.418x），于是消费完全部候选并执行 `final_select`（它的 5+5 次额外优化给出了最终的 0.4235）；
旧顺序恰好在更早的批次里命中了阈值，直接早停。两者**产物相同**（0.4235 与三个 CIF 一致），
差别只是"用多少候选换到同一个结果"。单case 观察，不推广为"O6 必然更慢"；`--num-processes 1`
（2009 s）与 O6 前同配置（1470 s，P3 记录）相比同样更慢，机制相同。

## 六、复现路径（`/xiangyux/claude_c_work/demo_reg_cases/`，不进仓库）

| 产物 | 说明 |
|---|---|
| `o6_result.txt` / `run_o6_matrix.sh` | 四次运行 + 汇总的原始输出 |
| `out_o6_{a,b,c,d}` | 四次运行产物（每个请求含 `candidates.jsonl`） |
| `rt_o6_10w.json` / `rt_o6_1w.json` / `rt_o6_tail.json` | 运行配置 |
| `/tmp/o6_summary.py` | 台账与消费序列汇总脚本 |
