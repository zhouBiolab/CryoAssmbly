# Step 3 组装算法（含拟合内核与 Step 4/5 精修）

> ⚠️ **组装拟合的当前权威逻辑以代码为准（`unified_queue.run_unified_assembly` 的全局轮次模型）。**
> 本文部分章节描述的是更早的"统一队列/每链独立轮次"架构，已被**全局轮次重写**取代，
> 阅读时请以 `protassem/assembly/` 下的实现为准。

代码位置：orchestrator.py（总入口）+ unified_queue.py（队列调度）
+ chain_fitter.py（链拟合）+ domain_fitter.py（域拟合）+ domain_assembler.py（域合并）
+ fitting/pipeline.py（拟合内核）+ assembly_opt.py（优化辅助）
+ refine_step.py（Step4）+ homo_chain_step.py/homo_chain_refine.py（Step5）

---

## 总体流程

```
run_assembly(target_txt, source_dir, density_mrc, resolution, contour)
|
+-- _setup()                  复制目标点云和密度图到工作目录
+-- _prepare_chains()         查找链文件，计算回旋半径，排序
+-- _run_domain_splitting()   对每条链调用 DomainParser 做域分割
|                             (复合物 + --complex-domain-opt → 拆内部链再分域)
+-- _reorder_chains()         按回旋半径降序排列（不分组）
|
+-- 统一队列拟合               unified_queue.run_unified_assembly()
|     链和域按回旋半径交替拟合，链失败时域即时入队
|
+-- 域链组装                   domain_assembler.assemble_domain_chains()
|
+-- build_complex()           合并接受的组件 -> assembled_complex.cif
+-- _attach_domain_details()  补逐域 cc_mask
+-- create_report()           生成 assembly_summary.txt
+-- refine_step.maybe_refine()     Step 4 同源域精修
+-- homo_chain_step.maybe_homo_refine()  Step 5 同源链精修
+-- [可选] 清理临时文件
```

---

## 准备阶段

### 输入标准化（pipeline.py）
1. 读取每个结构文件内部的 chain ID（不依赖文件名）
2. 多链文件保留为复合物（chain_id="A+B"），标记 is_complex=True
3. 单链文件 chain_id 取自内部链 ID

### 链准备 _prepare_chains()
1. 扫描 source_dir 的所有 .pdb/.cif 文件，从文件内部读取 chain ID
2. 为每条链配对 .txt 点云文件，读取点云算回旋半径
3. 按回旋半径降序（大链优先拟合）

### 域分割 _run_domain_splitting()

对每条链/复合物：

- `--no-domain-split` 指定的链 → 跳过，domain_count=0
- 普通链 → 复制 txt+pdb → DomainParser(subprocess) → 解析域残基范围与域间邻接
- 复合物 + `--complex-domain-opt` → 特殊流程（见下文"复合物域优化"）
- 复合物 + 无 `--complex-domain-opt` → 跳过，domain_count=0

### 复合物域优化的域分割（_split_complex_domains）

启用 `--complex-domain-opt` 时，复合物不作为整体分域，而是拆为内部单链分别分域：

```
complex_A+B.pdb
  → split_structure_to_chains → chain_A, chain_B
  → 对每条内部链（在独立子目录中）:
      1. 简化命名为 chain_{cid}_1.pdb（保证 TXT/PDB 前缀匹配）
      2. pdb2vol → chain_{cid}_1.mrc（模拟密度）
      3. sample_density_map → chain_{cid}_1.txt（点云）
      4. DomainParser → 域 PDB + TXT
      5. find_domain_files → 配对域文件
      6. 每条域记录增加 source_chain_id 字段（标记所属内部链）
```

域编号跨内部链累加（避免冲突）。所有域记录汇总到 `domain_records[complex_cid]`。

### 链排序 _reorder_chains()

所有链按回旋半径降序排列，不做分组。

> **设计意图**：旧架构将多域链排在前面，但这造成小链被人为推迟。
> 新架构统一按大小排列，大片段先占密度，减少后续搜索空间。

---

## 拟合内核 run_fitting（链/域共用，两阶段）

fitting/pipeline.py。一次拟合 = PARENet 配准（GPU）+ 两阶段局部优化（CPU）。

### PARENet 配准（常驻服务）
- parenet_client.py 懒启动单例 demo_mask.py --server，模型只加载一次
- 文件信号 IPC：请求=JSON stdin, 完成=_PARENET_DONE, 早停=_PARENET_STOP
- ParenetRequest 模拟 Popen 接口
- 主进程不初始化 CUDA -> fork 并行安全

### 触发与早停阈值
- cc_threshold（触发局部优化）：链 0.29，域 0.28
- stop_threshold（早停）= 当前接受阈值

### 阶段 1：批次监控 + 早停
```
每攒够 batch_size 个新 pred -> 并行算 cc_mask（_batch_cc，multiprocessing.Pool）
  -> 挑出 cc_mask > cc_threshold 的逐个局部优化
  -> 某个优化后 cc >= stop_threshold -> 早停
```

### 阶段 2：最终策略
```
未优化文件中：
  (1) cc_mask 前5
  (2) 混合分数 前5（排除(1)已选）
  合并（<=10个）逐个局部优化 -> 选 optimized_cc 最高
```
混合分数：cc_w * cc_mask + 1.0 * overlap（cc<0.2 时 cc_w=0.5）

### 局部优化 local_optimize
- 多副本并行（6 副本，multiprocessing.Pool）密度梯度上升
- 梯度用预计算密度梯度场 + 解析 Euler 链式法则（力臂 q=x−c，不用 torque）
- 取 CC 最高者（未改动的原始位姿始终在候选集里）再精细优化（250步），下降则回退
- 无 scipy 回退；返回 (success, output_path, final_cc)

---

## 统一队列拟合（unified_queue.py）

### 核心思想

所有链和域共享一个按回旋半径降序排列的队列。逐个弹出：
- 是链/复合物 → 调用 chain_fitter.fit_chain_item
- 是域 → 调用 domain_fitter.fit_domain_item

链拟合失败时，其域**即时加入队列并重排**，不等其他链拟合完。

### 为什么不用两阶段

两阶段（先链后域）的问题：
1. 一个大域（回旋半径大于某些小链）必须等所有链拟合完才能开始
2. 在等待期间，密度可能被低质量小链占用，大域拟合时反而受干扰
3. 无法利用"大片段先拟合、先 mask"的策略优势

统一队列保证：全局最大的片段总是最先被拟合，无论它是链还是域。

### 退出条件

- 队列为空
- 目标密度点云耗尽（所有点都被 mask 掉）

---

## 统一迭代拟合模型（最新，权威）

链拟合、域拟合本质同一套迭代；差异只在链多了"域优化"和"完全失败→域拟合"。

### 通用迭代（链/域都适用）
1. 按回旋半径排队，逐个拟合。达标（可放宽/可递减阈值）→ 接受+掩码；没达标但有结果 → 搁置（进"可重试池"），去拟下一个。
2. **每次接受+掩码后，密度变了 → 把池里的失败结构在新密度上重拟**（对不相似的也成立）：
   `a(0.41✗) → b(0.50✓掩码) → 重拟 a(0.42✗) → c(0.51✓掩码) → 重拟 a → ...`
3. **智能接受**：某结构连续两轮 cc 几乎不变 = 到极限 → 算它与已接受结构的 **clash，小就直接接受**。
4. **进度门控**：重拟只由"新接受"触发；没有新接受就停 → 必然终止。

### 重激活只在域层做
- **链不做跨结构重拟**：链拟一次，达标接受（可域优化），不达标 → 落到域拟合（尽量相信域拟合）。
- **域做跨链全局重激活**：某域没达标 → 进 `retriable_domains` 池；任何结构接受+掩码后，把池里（非"结构疑似有问题"的）域重置入队重拟。结构疑似有问题的（与已失败相似且无相似接受）暂留池中，等有相似接受了再放出来。

### 单域链
- **单域链直接走域拟合**（不做链拟合、不跑域优化），在统一队列里以"域"身份按回旋半径参与。

### 相似结构（通用迭代的特例）
- 失败 + 相似的有失败过 + 还有不相似可试 → 跳过相似的，先试不相似的（疑结构问题）。
- **只剩一种相似**（无不相似可退）→ 不跳过，挨个全试（无次数上限）。
- **已有相似达标过**（含"部分域达标"）→ 不跳过，迭代所有相似的；失败的靠上面的重激活在掩码后重拟。

### 同源组参考 + 多拷贝（修 examplezzy 全不接受）
- **组参考 cc**：同源组里第一个达标(≥0.35)的 cc 作参考；若整组都没到 0.35 → **兜底取最高(带 clash 检查)接受为第一份**并建立参考（`_salvage_retriable_groups`）。
- **逐份多拷贝**：之后同源拷贝 `cc ≥ 参考 − gap(0.025)` 且无 clash → 接受+掩码（又一份拷贝）；比参考**骤降(< 参考−gap)** → 不接受，掩码后重拟（同源优先），直到无拷贝接近参考即停。
- **0.35 拟合下限不变**；`0.25` 只是收尾 `complex_min_cc` 逐结构域过滤，与拟合无关。
- 同源宽松、骤降 gap 均取 **0.025**；兜底取最高都带 **clash 检查**。

---

## 链/复合物拟合（chain_fitter.py）

### 流程

```
对每个弹出的链/复合物:
+-- [同源分组] calculate_tm_score() TM>=0.85 归同组
+-- [相似跳过] 见下"相似链处理（三种情况，无回灌）"
+-- [拟合] run_fitting(mode="chain")
+-- [阈值选择]
|       普通链: chain_threshold(0.45)
|       复合物: complex_threshold(0.35)   ← 独立参数
|       同源组内已有接受 -> 阈值 - chain_similar_relax(0.03)
+-- cc >= 阈值 -> 接受（整链达标）
|   +-- [可选] 域优化（见"功能开关"中的独立域优化开关）
|   +-- 保存 PDB→CIF，掩掉密度区域
|   +-- 该链所有域标记 rejected
+-- cc < 阈值 -> 域优化兜底
    +-- 有域 → try_domains_via_chain_pose
    |   对每个域：按链姿态切出 → 局部优化 → CC >= domain_threshold(可宽松) → 接受+mask
    |   ★ 部分域达标 = 链达标（accepted_via_domains），剩下没接受的域无所谓
    +-- 一个域都没达标 + 有域 → ★ 域加入统一队列做域拟合 ★
    +-- 无域 → rejected_no_domain
```

### 链达标的两种方式

1. 整链 cc >= 链阈值；**或**
2. 域优化后**部分域达标** → 也算链达标（accepted_via_domains），剩下没接受的域无所谓。
   该链算"有相似达标过"，相似链仍可继续拟合（进入下面情况 2）。

### 相似链处理（三种情况，无回灌）

判相似：TM-score >= 0.85 归同组。某条链没达标时，是否跳过它的相似链分三种：

1. **第一次拟合就失败，且之前没有相似的达标过**
   → 失败就失败，**其余相似的本轮先跳过**，优先去试**不相似**的（怀疑是结构质量问题）。
2. **之前已有相似的达标过**（含上面"部分域达标"那种）
   → 本轮这个没达标，就**继续试另一个相似的**（结构已证明没问题，只是这次拟合没拟好）；
   等本轮再没有"能像之前那样成功的相似结构"了，就**回到正常顺序**往下走。
3. **这是最后一种相似的了**（剩下全和它相似，没有不相似可退）
   → **不要全跳过**，**再多给一次尝试**（拟合有时不准），**只多一次**；多域链此时也可转域拟合。

> **结构域同理**：域拟合用同一套三种情况。
> **此外还有宽松阈值机制**：同源组/相似项已有接受时，链阈值 −chain_similar_relax(0.03)、域阈值 −domain_similar_relax(0.025)，让相似的兄弟更容易通过。

> 不采用"掩码后回灌"机制：只有 B、A 两个且相似时，B 先失败 → 情况 3 试 A 成功掩码
> → B 不再重拟（已确认接受此结果）。

### 复合物阈值

复合物由多条链组成，体积更大、内部柔性更高，在实验密度中的 cc_mask 天然低于同等大小的单链。
`complex_threshold`（默认 0.35）独立于 `chain_threshold`（默认 0.45），避免合理的复合物拟合被误拒。

### 逐域微调（try_improve_chain_with_domains）

链接受后（CC 已达标），如果有 >= 2 个域，尝试更精细的优化：
1. 每个域按链姿态切出（align_by_resid：按残基编号对齐）
2. 每个域单独 local_optimize（密度梯度+CC）
3. 所有域合并回完整链（merge_domains / 复合物按 source_chain_id 多链合并）
4. 合并后 CC > 原链 CC → 替换

> **策略本质**：整链优化只有一组 6DOF（刚体旋转+平移），各域的微小相对位移被平均掉。
> 逐域优化给每个域独立的 6DOF，能更好地适应实验密度中的局部偏差。

### 链姿态降域（try_domains_via_chain_pose）

链拟合虽未达标，但 PARENet 给出的位姿可能对域有价值。
对每个域：按链姿态切出 → 如果初始 CC > 0.10 → 局部优化 → 达域阈值则直接接受。
即使域未达标，也存为 chain_pose 候选供后续比较。

> **策略本质**：链整体 CC 不够（可能某个域拟合差），但某些域在链姿态下已经很好，
> 不需要从头拟合。

> ⚠️ **待核实（#1）**：当某域靠链姿态就达到域阈值、链被标 `accepted_via_domains` 并 return 时，
> 该已接受的域目前**没有加入 `needs_domain_assembly`**，可能不会拼进最终复合物。
> 常见路径（走 needs_domain_assembly）不受影响，此稀有分支待自测确认后再修。

---

## 域拟合（domain_fitter.py）

### 每链独立的轮次管理：DomainRoundTracker

**每条链的域有独立的 threshold 和轮次计数**（各链互不影响，不共享递减状态）。

```python
DomainRoundTracker:
  per_chain[cid] = {
    threshold: float,          # 当前阈值，从 initial_domain_threshold 开始
    threshold_decreases: int,  # 衰减次数
    pending_count: int,        # 本轮剩余待处理域数
    round_results: [],         # 本轮结果
    round_failed_pdbs: [],     # 本轮失败的域 PDB
  }
```

**为什么要独立**：
- 链 A 的域可能在第 1 轮就全部失败，threshold 递减到 0.435
- 链 B 的域可能在链 A 域拟合中途才加入队列（因为链 B 刚失败），起始 threshold 还是 0.45
- 如果共享状态，链 B 的域会"享受"链 A 造成的低阈值，这不合理

### 域拟合流程

```
对每个弹出的域:
+-- [相似跳过] 与链拟合相同的三种情况（见上"相似链处理"），域同理
+-- [本轮去重] 与本轮已失败域相似 -> 跳过（不消耗 pending_count）
+-- [拟合] run_fitting(mode="domain")
+-- eff_threshold = per_chain[cid].threshold
|     与已接受域相似 -> max(0.35, eff - domain_similar_relax(0.025))   ← 宽松仍保留，下限0.35
|     与上轮 force-accepted 域相似 -> min(eff, max(0.35, 上轮cc - next_round_float))
+-- [智能接受] 连续 >=3 轮 CC 变化 < 0.015 且 >= 拟合绝对阈值(0.35) -> 接受
+-- cc >= eff 或 智能接受 -> 接受该域 + mask
+-- cc < eff -> 记录到 round_results
|
+-- pending_count 减 1
+-- pending_count == 0 -> 触发轮次结束
```

### 智能接受

当一个域连续 >=3 轮 CC 变化 < 0.015，说明密度中没有更好的位置，这个域已到达"平台期"。
只要 CC >= 拟合绝对阈值(0.35)，就直接接受，避免无意义的继续尝试。

### 轮次结束处理

当一条链的所有域在本轮都已处理（pending_count=0）时触发：

```
+-- 有域被接受 -> 对该链的剩余 available 域重入队列开启下一轮
+-- 无域被接受 -> select_multi_accept:
|   +-- best + 差值 <= margin(0.02) 且 >= 0.35 的域
|   +-- 按 CC 降序逐个加入，clash(CA重叠>10%) 时保 CA 多者
|   +-- 有域被 force-accept -> mask + 剩余域重入队列
|   +-- 仍选 cc 最高的掩码进下一轮；best < 0.35 -> 不接受（排除）
+-- 阈值衰减: 前3轮 -0.015，之后 -0.020，下限 = 0.35（拟合绝对阈值）
+-- 剩余 available 域（如果有）重入统一队列
```

### 阈值衰减策略

| 衰减次数 | 每次递减 | 累计阈值（从0.45起） |
|----------|---------|---------------------|
| 1 | -0.015 | 0.435 |
| 2 | -0.015 | 0.420 |
| 3 | -0.015 | 0.405 |
| 4 | -0.020 | 0.385 |
| 5 | -0.020 | 0.365 |
| ... | -0.020 | ... |
| 下限 | — | 0.350 (拟合绝对阈值) |

> **策略本质**：前期温和递减（-0.015），给域多几次机会在高阈值下被接受；
> 后期加速递减（-0.020），避免长时间卡在中等阈值上。

### multi-accept 策略

当一轮全部失败时，检查是否有多个域 CC 接近且都可接受：
1. 取本轮最高 CC 域（best）
2. 与 best 差值 <= 0.02 且 CC >= 0.35 的域纳入候选
3. 按 CC 降序逐个加入，两两检查 clash（CA 原子重叠 > 10%）
4. 发生 clash 时保 CA 数多者（更完整的域）

> **为什么需要 multi-accept**：两个相邻域 CC 都不错但差一点达标，分别看都不够好，
> 但合起来能覆盖链的大部分密度。一起接受比等下一轮（阈值更低、可能被其他链 mask）更好。

---

## 域链组装（domain_assembler.py）

统一队列拟合完成后，对所有 needs_domain_assembly 中的链执行域组装：

```
对每条失败链:
  收集 status="accepted" 的域
  0 个 -> 该链不进复合物
  1 个 -> 直接作为单域链保存
  2+ 个 -> merge_domains 按残基顺序合并为单链 CIF
  复合物域 -> 按 source_chain_id 分组创建多链 CIF
```

### merge_domains 合并策略

普通链：所有域残基按序号排列，放入同一个 BioPython Chain 对象。
复合物：按 source_chain_id 分组，每组一个 Chain 对象，残基 ID 保证组内唯一。

---

## 组装收尾

### build_complex（complex_builder.py）
合并已接受组件。chain ID 冲突自动 remap。

**产出两个结果**（已实现）：
- ① 完整版 `assembled_complex_all.cif`：保留全部已接受域，不过滤。
- ② 过滤版 `assembled_complex.cif`（主输出）：按 `complex_min_cc` **逐结构域**剔除 cc_mask < 阈值的域；整链/复合物不做域级过滤；域链保留其余达标域（全部低于阈值则该链不进复合物）。

> 命名保持 `assembled_complex.cif` 为主输出，外部接口不变；完整版另存 `_all`。

### 报告 assembly_summary.txt
- 每组件：整体 cc_mask + 逐域 cc_mask
- 末尾：Excluded domains 列表

---

## Step 4：同源域精修（refine_step.py，自动触发）

- 触发条件：存在同源链 + 已分域 + 已做域拟合
- backfill：整链接受的链也按域切片落到 fitted_domains/
  （复用逐域微调的最优结果 chain_rec.domain_cifs，否则 align_by_resid 切片）
- 调 ChainEnumerator：USalign 找同源域组 -> 跨链对齐 -> 穷举组合 -> 连接能量最优
- 产物 refined_complex.cif 作为新主输出
- --no-refine 关闭

## Step 5：同源链精修（homo_chain_step.py + homo_chain_refine.py，可选）

- 开关 --homo-chain-refine（默认关）
- 触发：同源链（Seq_ID 分组）且组内 cc 有分化（链 cc 差>chain_eps 0.02 + 域级确认）
- **残基保护**：候选丢失 >10% 残基时自动跳过，防止不完整链替换完整链
- 好链当模板：序列叠合搬到差链位姿 + 密度优化（local_optimize，并行度由运行级共享池 context 决定）
- clash 门控选最优，优于原链+eps 才替换。好链不动
- **域补回**：clash 门控后检测同源组内残基覆盖差异，从完整链补回缺失域
  （序列叠合 → 合并缺失残基 → density 优化 → clash 门控，不卡 CC 下降）
- 四处并行：分组 USalign / cc_mask / 候选优化 / 域补回
- 产物 homo_chain_refined_complex.cif

---

## 阈值参数汇总

| 参数 | 默认 | 用途 |
|------|------|------|
| chain_threshold | 0.45 | 普通链接受阈值 |
| complex_threshold | 0.35 | 复合物接受阈值（独立于链阈值） |
| chain_similar_relax | 0.03 | 同源放宽 |
| initial_domain_threshold | 0.45 | 域初始接受 |
| domain_similar_relax | 0.025 | 域相似放宽 |
| 拟合绝对阈值 | 0.35 | 域拟合下限+衰减下限+多域同接门槛（低于不接受） |
| 后处理过滤 complex_min_cc | 0.25 | 组装后**逐结构域**去掉 cc<此值 的域，产出过滤版 assembled_complex.cif |
| similarity_threshold | 0.85 | TM 相似判定 |
| refine_tm | 0.75 | Step4 分组 |
| multi_accept_margin | 0.02 | 多域同接 |
| next_round_float | 0.015 | 下轮阈跟随 |
| next_round_max_gap | 0.03 | 跟随下限 |
| clash_overlap_thr | 0.10 | clash |
| cc_threshold 链/域 | 0.29/0.28 | 触发局部优化 |
| 衰减(前3/后) | 0.015/0.020 | 域阈值每轮递减 |
| 智能接受 | 0.015/3轮 | 平台期检测 |

---

## 功能开关

| CLI 参数 | 行为 |
|----------|------|
| --complex-domain-opt | 复合物域优化：拆内部链→各自域分割→接受后逐域微调→合并比较 CC |
| --no-improve-accepted | 关闭逐域微调（默认开启，链接受后逐域优化取更高 CC） |
| --no-domain-opt | 关闭域优化（**默认开启**）：不管链拟合达标与否都跑域优化，保留域达标的；与逐域微调(--no-improve-accepted)解耦 |
| --no-domain-split <ids> | 指定不拆域的链 ID（逗号分隔），这些链只能整链拟合或丢弃 |
| --no-pre-screen | 关闭装配前原始位姿并行预筛（默认开启） |
| --save-all-attempts | 保存所有拟合尝试 |
| --cleanup | 完成后删临时目录 |
| --no-refine | 关闭 Step4 |
| --homo-chain-refine | 开启 Step5 |

---

## 已知问题 / 待修正

| 编号 | 问题 | 状态 |
|------|------|------|
| #1 | `accepted_via_domains` 分支已接受域补进 `needs_domain_assembly`，不再丢域 | ✅ 已修 |
| #6 | 拆出独立域优化开关 `--no-domain-opt`（默认开），与逐域微调解耦 | ✅ 已实现 |
| #7 | mask 目录改独立单调计数器 `_mask_iter` | ✅ 已改 |
| 阈值 | 拟合绝对阈值改 0.35（min_domain_threshold）；两输出已加（置信度 `assembled_complex.cif` / 完整 `assembled_complex_all.cif`） | ✅ 已实现 |
| #3 | 只剩一种相似 → 去掉"≤2次"上限，挨个全试 | ✅ 已改 |
| 迭代 | 单域链直接域拟合；域跨链全局重激活（retriable 池 + 进度门控）；smart-accept 加 clash | ✅ 已实现 |
| 链姿态降域 | 链没达标→降域接受部分域后，**剩余没达标的域仍送去域拟合**（原先 return None 丢掉） | ✅ 已修 |
| 链号去重/多字符 | 输入链号重复→重排唯一（A-Z/a-z/AA..ZZ）；内部 PDB 用单字符占位、真实(可两字符)号只在 CIF 输出还原 | ✅ 已实现 |
| overlap 精度 | 预测 pose 文件名重叠率 5 位 → 6 位（减少撞名覆盖） | ✅ 已改 |
| 同源全不接受 | 低质量同源单域链都没到 0.35 → 原先全 excluded(Accepted 0)；现"组参考+兜底取最高(clash)+近参考逐份多拷贝" | ✅ 已修 |
| 阈值语义 | 0.35 拟合下限不变；同源宽松/骤降 gap=0.025；complex_min_cc=0.25 仅收尾高置信度筛选 | ✅ |
| force-accept | 域轮末没人达标：不同源→强制接受 cc 最高(+clash)；同源→2 个取最高建参考。**不再因低于 0.35 剔除**(修 examplezzy Accepted 0) | ✅ 已修 |
