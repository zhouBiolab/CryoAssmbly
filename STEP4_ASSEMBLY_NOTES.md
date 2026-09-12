# Step4 装配算法笔记（测试版 + 参考 build_chain_models）

记录时间：2026-06-24
适用项目：demo_reg（cryo-EM 蛋白复合物装配）

---

## 0. 目标与上下文

把"拟合好的结构域"装配成"连通、贴密度、互不重叠"的多链复合物。
当前正式 Step4 = `protassem/assembly/refine/chain_enumerator.py`（ChainEnumerator）。
本笔记讨论：仿照 `init_modeling.py` 的 `build_chain_models()` 写的**独立测试版 Step4**
（`test_step4_assign.py`），它遇到的问题，以及参考算法的关键点。

## 1. 关键名词

- 槽 slot：复合物里一条链的位置；同源 N 聚体有 N 个槽。
- 实例 instance：某条链的某个域的一份已拟合位姿（`work/fitted_domains/<链>/pred_<链>_d_<n>.cif`）。
- cc_mask：单个域摆在当前位姿和密度图的相关性（贴密度好坏）。
- 断开：相邻域接缝处 CA-CA 距离过大（正常肽键约 3.8 Å）。
- 空槽：某个（槽 × 域）格子上没有现成已拟合实例（该链少拟合了这个域）。
  例：example5 中 B 只有域 1/2/3，缺域 4，所以 B 槽的域 4 是空槽。

## 2. 拟合之后的流水线（现状）

1. assembled_complex_all.cif —— 全量（所有已接受域，不过滤）
2. assembled_complex.cif —— 过滤版（按 complex_min_cc 逐域剔除 cc 低的域）
3. refined_complex.cif —— Step4（ChainEnumerator 同源域枚举）
4. homo_chain_refined_complex.cif —— Step5（同源链精修）

过滤已在 Step4 之前生效（`refine_step._filter_fitted_domains`），cc 阈值默认 0.15。

---

## 3. 测试版 Step4 算法（test_step4_assign.py）—— 我现在用的

本质：不重拟合，只把"已拟合好的域实例"**重排/指派**到各链槽，使链连通且不抢槽。

输入：`work/fitted_domains/<链>/*.cif` 的已拟合域实例（载入 CA 坐标 + 算 cc_mask）。

步骤：
1. 按域号分组（同源组）；槽数 N = 链数（example5 为 4）。
2. 种子 = 实例最多、cc 最高的域；其各实例按 cc 降序占据各槽（定义链骨架）。
   - example5 实测：种子 = 域 3（4 实例，cc 0.515/0.484/0.461/0.404）。
3. 其余域按"距种子由近到远"处理；对每个域用匈牙利
   `scipy.optimize.linear_sum_assignment` 把实例最优指派到还缺该域的槽，
   代价 = 候选实例与该槽已放相邻域的"接缝 CA-CA 距离"。
4. 物理阈值卡连通：接缝距离 > 阈值的配对设为不可指派（拒绝）。
   当前阈值（错误，见问题②）：阈值 = 3.8 × 残基gap + 20。
5. 空槽：**Step4 不补**，只列出来，交给 Step5 的缺失域补回。
6. 按"槽 -> 链"组装，输出 test_step4_complex.cif。

复用函数：`calculate_cc_mask`、Biopython 读 CA、`linear_sum_assignment`、Kabsch（自写）。

example5 实测结果（不补空槽）：
- 整体 cc = 0.4182（约等于 refined 0.4171）；两两 clash 最大 0.0294（无重叠）。
- 空槽：C 的 d2/d4、D 的 d2/d4。
- 每槽 CA：A=753 / B=749 / C=410 / D=584（C、D 残缺）。

---

## 4. 测试遇到的问题（关键）

### 问题①：纯重排修不了"散的拟合"
这批已拟合域本身是散的：连相邻域 d2-d3 接缝都 19~47 Å，跨域更是 36~59 Å。
匈牙利只能在散数据里挑"最不坏"的配对，结果每个槽混进了好几条原链的域，
接缝普遍 19~59 Å（远超 3.8 Å）—— 链实际还是断的，只是聚合 cc 和 clash 看着还行。
结论：**纯重排（不重定位）创造不出本来不存在的连通性**。
连通性必须靠"重定位"（密度fit 把域挪到对的位置），这正是我们 Step5
用"逐域锚定 + local_optimize"才解决的那点。

### 问题②：连通阈值按"残基 gap"算会放飞
当中间域缺失时（例：d1 跨过缺失的 d2 直接连 d3，残基 gap = 139），
阈值变成 3.8×139+20 约 548 Å，等于不设防 -> d1 被乱配到 36~59 Å。
正确做法见参考算法：按"缺失的域数 + 缺失域跨度"算，而不是按残基 gap。

---

## 5. 参考算法：build_chain_models()（init_modeling.py:968-1751）

定位：init_modeling.py 的 Part 2"把拟合好的域装配成链"，对应我们的 Step4。
本质：cc_mask（密度）当指挥棒，最信任的域当骨架，往外拼，既贴密度又连得通又不抢槽。

### 5.1 跨 fasta 同源池（_homolog_swap_cross_fasta，547+）
Union-Find 把任意 TM-score > 0.85 的域并成同源池（不分链）。

### 5.2 同源放置取 cc 最高（_hs_place_entity，664+）—— 与我们 Step5 逐域锚定同构
把一个域放进槽时，试两个位姿取 cc 高的：
- 位置1：TM 对齐到目标槽（_hs_tmalign）-> cc_tm
- 位置2：在 TM 位姿基础上再做密度fit（_hs_domainfit）-> cc_fit
- if cc_fit > cc_tm: 用 fit 否则用 tm
处理顺序：按 cc_mask 降序；B 与 C 距离硬阈值 60 Å。

### 5.3 主装配（cc 优先 + Union-Find + 两遍）
1. 所有域按 cc_mask 降序（_domain_max_cc）。
2. cc 最高的域 = 种子（seed_did）；其每个实例 = 一个链槽。
3. Pass 1 片段连接：共享同一 trace 片段的域实例直接 Union 成簇（强证据，防距离贪心乱配）。
4. Pass 2 匈牙利最优分配（linear_sum_assignment）：按从种子出发的 BFS 顺序，
   把"域簇"双射指派到"链槽"，代价 = 簇到槽的边界距离；
   超物理阈值的配对设为拒绝。
   - 物理阈值 = Σ（中间缺失域的 AF2 跨度）+ 3.8 ×（缺失域数 + 1）+ 20 Å
   - 保证：每槽每域只放一个（不抢位/不重叠）；连不通的不指派。

### 5.4 "密度优先"具体指什么（核心概念）
cc_mask 全程当排序/取舍依据，体现在 5 处：
1. 处理顺序按 cc 降序；
2. cc 最高的域当种子（定链数与骨架）；
3. 种子簇按 cc_max 降序填槽；
4. 放置时 TM 对齐 vs 密度fit 取 cc 高的；
5. 哲学：最信任贴密度最好的域，以它为锚往外拼可信度低的域。

---

## 6. 对照表：build_chain_models vs 我们 Step4（ChainEnumerator）

| 维度 | build_chain_models | 我们 Step4（chain_enumerator.py） |
|---|---|---|
| 同源分组 | Union-Find, TM>0.85 | TM 分组（tm_threshold=0.75） |
| 选择依据 | cc_mask（密度）优先 + 连通 | 纯连通 score，完全不看密度/cc_mask |
| 放置 | TM 对齐 + 密度fit，可重定位 | 只把同源变体对齐到已有帧，不做密度优化、不重定位 |
| 防同槽冲突 | 匈牙利全局最优双射 | 贪心 _update_occupied_positions |
| 连通约束 | 物理阈值（Σ缺失域span + 肽键 + 20Å） | tight/loose/断开 距离分档（3.8/5.3/loose_max） |
| 片段证据 | Pass1 trace 桥接 | 无 |

关键证据：chain_enumerator.py 选最优 config 用 `config[score] > best_score`
（energy 并列裁决，约 769-771 行），无任何 cc_mask/density。
这就是它修不了散架链的原因：连通分数最高的组合可能根本不贴密度，
而且它只在已拟合（可能散开）的位姿里选，不会把域往正确密度位置拉。

---

## 7. Step5 现状（已实现并验证，homo_chain_refine.py）

我们已把"密度fit 重定位 + 防重叠"做进了 Step5：
1. 同源分组（Seq_ID>0.9）；断开数用 Step4 的 `calculate_chain_connection_score`
   （阈值 loose_max=10 Å）统计（`_count_breaks`）。
2. 每组选最连续、cc 高的链当模板（group_best）。
3. 逐域锚定放置（`_anchor_cand_worker`）：把模板分别锚到目标链每个同源域上，
   各做 局部对齐 + local_optimize，取 cc 最高的落点。
4. 取舍/防重叠：cc 高占位、cc 低让位换槽，clash 为硬门（<=clash_thr=0.1）；
   贪心按最佳锚定 cc 从高到低放置，每条只选不与已放置链冲突的落点。
5. 缺失域补回（空槽填补）：从同源链借域、按相邻域锚定搬入，clash 合格才接受。

example5 验证（阈值 0.15）：整体 cc 0.4153（保住），断开 B/D 由 4 降到 1，
两两 clash 最大 0.0133（无重叠）。

注意：Step5 的逐域锚定 = build_chain_models 的 _hs_place_entity 同构（精简版）。

---

## 8. 关键结论与待定方向

结论：
- 装配的本质 = 换位置使链连通；cc 已由拟合保证，难点是连通的指派/重定位。
- 纯重排不够，必须有"重定位"（密度fit）才能修散架。
- 连通阈值要按"缺失域跨度"算，不能按残基 gap（否则缺域时放飞）。

待定方向（暂不改，等决策）：
- A：先把测试版阈值改对（只允许相邻域连接，缺域时用域跨度上界），再测。
- B：Step4 也带密度fit 重定位（会与 Step5 锚定重叠）。
- C：Step4 只做粗指派，连通性全交给 Step5 锚定（已验证 Step5 能把散链拉回）。

---

## 9. 相关文件与函数

- 参考：init_modeling.py
  - build_chain_models()                 968-1751
  - _homolog_swap_cross_fasta()          547+
  - _hs_place_entity()                   664+（TM对齐 + 密度fit 取 cc 高）
- 正式 Step4：protassem/assembly/refine/chain_enumerator.py
  - 选 config：约 759-779（纯 score/energy）
  - 连通评分：protassem/assembly/refine/refine_energy.py
    calculate_chain_connection_score / calculate_chain_connection_energy
- Step5：protassem/assembly/homo_chain_refine.py
  - _anchor_cand_worker（逐域锚定）、_count_breaks、贪心+clash 取舍、缺失域补回
- 测试版 Step4：test_step4_assign.py（cc种子 + 匈牙利 + 阈值；空槽不补）
- 冒烟脚本：rerun_post_fitting.py（复用中间文件复算 1-4 步）
