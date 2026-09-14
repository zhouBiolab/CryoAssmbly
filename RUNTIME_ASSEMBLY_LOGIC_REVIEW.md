# demo_reg 新版本运行逻辑与组装规则核对

核对日期：2026-09-14。代码：`feat/old-card-closeout`，提交 `6c4ed00`。
服务器项目：`/xiangyux/claude_c_work/demo_reg`。

本文根据当前代码阅读整理，描述“现在实际怎么运行”，不是改造计划。此次没有执行 GPU 推理、修改算法或重新运行测试。正常 case 的历史验证不能代替本文指出的分支级验证。

## 1. 先看整体：三层不同的选择

```text
密度图 + 结构 + resolution/contour
  → 输入检查、结构链号标准化
  → 每个结构生成模拟密度 → 实验图/模拟图采样
  → 准备链、域拆分、相似分组、原位姿预筛
  → 全局轮次队列：整链与结构域拟合
       → 每个 mask 内选最优配准（overlap）
       → 客户端固定批次 CC + 局部优化 + 请求早停
       → 装配层接受/拒绝（原图 CC 或链姿态降域路径的当前图 CC）
       → 接受后掩膜，更新目标点云与密度图
  → 将接受的域拼成链
  → 输出完整/过滤复合物
  → Step4 同源域替换与连接几何枚举（有条件）
  → Step5 同源链锚定修复及补残基（默认关闭）
  → 返回优先级最高的有效产物
```

三个“最优”不能混为一谈：服务端每 mask 最优主要看 overlap；拟合候选看 CC/局部优化；Step4 主要看连接几何评分和能量。最终文件名含 refined 不等于最终全复合物 CC 必然更高。

## 2. 输入和数据处理

### 2.1 CLI 和默认参数

入口 `main.py:build_parser/parse_args/main`。使用 `parse_intermixed_args`，选项可夹在位置参数之间。

- 自动模式：`python main.py <data_dir>`。取该目录按排序得到的第一张 `.mrc`，读取全部 `.pdb/.cif`，读取 `resolution.txt`、`contour_level.txt`。
- 手动模式：`python main.py <mrc> <struct_dir> <resolution> <contour> [output_dir]`。
- 自动模式不是智能实验图识别：有多张模拟/实验 MRC 时仍取第一张；目录里的历史预测 PDB 也可能被收为结构输入。应使用干净输入目录。
- 默认输出为实验图所在目录的 `output/`；指定独立输出目录更便于保留对照。
- `--runtime-config` 是 JSON 配置文件路径。线程环境在导入重依赖之前设置，主进程随后应用随机 seed、缓存及服务配置。

| 参数 | CLI 默认 | 实际用途 |
|---|---:|---|
| chain_threshold | 0.45 | 普通整链接受基础阈值 |
| complex_threshold | 0.35 | 多链复合物整体拟合基础阈值 |
| initial_domain_threshold | 0.45 | 全局域阈值起点 |
| min_domain_threshold | 0.35 | 阈值衰减/同源放宽的下限，注意不是所有接受分支的硬门槛 |
| similarity_threshold | 0.85 | 装配阶段整链 TM 分组及 Step4 入口同源检查 |
| complex_min_cc | 0.25 | 域链过滤版及 Step4 已评分域输入过滤 |
| refine_tm | 0.75 | Step4 跨链域 TM 同源判断 |
| num_processes / batch_size | 10 / 10 | CPU 工作进程数 / 固定候选批次大小，不是模型 batch size |
| improve_accepted / domain_opt | 开 / 开 | 达标链接受前逐域改善 / 链失败后姿态降域兜底 |
| pre_screen / do_refine | 开 / 开 | 原位姿预筛 / 有条件 Step4 |
| complex_domain_opt / homo_chain_refine | 关 / 关 | 复合物拆域 / Step5 |

表格是 CLI 默认；直接构造 `AssemblyOrchestrator` 的默认值可能不同，不能把 Python API 与 CLI 默认视为完全相同。

### 2.2 标准化与链号空间

`protassem/pipeline.py:run_pipeline` 在产生主要输出前校验输入。结构身份来自文件内部链号，而非用户原文件名。

1. 单链文件保留为单链；多链文件保留为一个复合物组件。
2. 多输入链号重复时，按遍历顺序保留已经使用的 ID，冲突者取下一个空闲逻辑链号，并写 CIF。
3. 标准化结果命名为 `chain_<id>_<n>` 或 `complex_<id+id>_<n>`。
4. 装配准备阶段的 CIF 转内部 PDB，占位池 A–Z/a–z/0–9，单个文件容量 62；保存占位→真实映射。
5. 内部 PDB/域来源链保持占位空间，组件 ID 如 `Q+R` 用于记录和目录；最终输出恢复真实链号。

注意：标准化时的内容判断与装配准备阶段的文件名前缀约定配套。若绕过总流水线直接调用装配接口，要核对 source_dir 的命名契约。

### 2.3 体素化、采样、域拆分

`voxelize/mol_to_mrc.py:pdb2vol`：读取结构原子，按元素权重累加高斯密度。无参考图时网格间距为 resolution/3，边缘留 3×resolution；不代表输出点云采样间距也等于它。

`sampling/sampler.py:sample_density_map`：用 Sample 外部二进制采样，流水线点云采样间距默认 2.0 Å。实验图使用输入 contour；模拟图未给 contour 时采用 3×标准差。

`core/points_txt.py`：统一读五行头部及点/向量行对。世界坐标 = 文件坐标×sample+origin；过滤文件保留原始行信息，避免重新格式化导致误差。

`assembly/domain_splitter.py`：调用 DomainParser/DSSP 相关流程，在统一坐标网格中按域原子掩膜筛出域点云，并保存残基范围。域拆分在队列拟合之前进行，并非“整链失败才开始运行 DomainParser”。

- `--no-domain-split` 指定的组件不拆。
- 复合物默认不拆；开启 `--complex-domain-opt` 后，先拆内部链，再分别域拆分，域记录保存 `source_chain_id`，域序号作偏移以在组件内唯一。
- 域邻接/范围虽已读取，主全局队列并没有据此强制相邻域一起拟合；残基范围主要在合并、回填和后续精修中使用。

## 3. 两张密度图与评分语义

| 数据 | 是否变化 | 用途 |
|---|---|---|
| original_density_mrc | 不变 | 原位姿预筛、拟合最终复核、最终域链 CC、逐域改善合并后评估、Step5 |
| current_density_mrc | 接受后变化 | 当前候选 CC、局部优化、链姿态降域 |
| current_target_txt | 接受后变化 | 下一个 PARENet 请求的剩余目标点云 |

`core/scoring.py` 的标准 CC_mask：规范化 MRC 轴/原点；实验值不高于 contour 的位置替为 -1；按原子生成模拟密度及软化后阈值 mask；在 mask 内计算 Pearson 相关。不是简单的全图相关，也不是 overlap。

P4 缓存只复用密度上下文/结构解析，不能被理解为复用所有候选的最终分数。动态密度每轮换文件名，路径区分版本；worker 有各自的有界缓存。

`fitting/masker.py:mask_fitted_region`：按原子 VDW 半径+1.1 Å 生成占用掩膜，将该区域密度清零，同时去掉落在掩膜中的目标采样点。其占用掩膜与 CC_mask 的软化评分掩膜不是同一个算法。

## 4. 相似结构如何分组

### 4.1 装配前整链分组

`orchestrator._precompute_similarity` 先预计算整链两两 TM；`core/similarity.py` 调 USalign，取双向长度归一 TM 的较小值，默认 >=0.85 为相似。合法结果使用内存/SQLite 缓存，工具失败明确抛错。

整链分组是按当前链顺序进行的**代表链贪心分组**：依次比较已有代表，进入第一个达标的组，否则新建组。不是聚类距离矩阵的全局优化，也不保证同组所有成员两两都 >=0.85。

### 4.2 主队列的域分组

域 group_id 直接由 `整链组号.域序号` 构成。代码会预填同组同序号域的 TM，但没有用这些 TM 重新拆分 group_id。

因此当前主队列采用的假设是：同源整链的同序号域同源。它不是经过所有域两两验证后的分组；不同拆域边界/域编号可使假设失效。与 Step4 真正计算跨链域 TM 的分组方法不同。

### 4.3 相似性带来的动作

| 情况 | 实际动作 |
|---|---|
| 相似整链组已有接受记录 | 当前整链接受阈值减 0.03；仍自己生成并评估位姿 |
| 相似整链组没有接受记录且失败计数达到 2 | 其余整链跳过链级拟合，有域则转域，无域则拒绝 |
| 相似域组已接受过域 | 域阈值减 0.025，但不低于 0.35（默认） |
| 相似域组未接受过域 | 一轮最多尝试该组两个域，其他留待下轮 |
| 上一轮刚接受某类域 | 下一轮同 group_id 的域优先，再按回旋半径排序 |
| 相似整链全部高分但位置重叠 | 预筛 clash 筛选会保留部分，其余继续拟合 |

相似不等于同一物理位置，也不代表可以删除拷贝。正常主流程仍为每个组件寻找独立位姿，接受后掩掉已占区域。

## 5. 原位姿预筛

`_pre_screen_chains` 默认开启，不调用 PARENet。先对所有输入原位姿在原始实验图上并行算 CC：普通链用 0.45，复合物用 0.35。

通过后进行 CA/P 重叠检查：距离 3 Å 的重叠比例 >0.10 算冲突，倾向保留 CA/P 数更多者。代码按顺序找到第一个冲突对象后比较，不是全局最优的冲突集合求解。

保留对象可先做逐域改善，然后接受并掩膜，其所属域标 rejected，防止再重复作为独立域接受。预筛可以一次接受多个组件。

若设置 `--pre-screen-by-domain`，则用域原位姿预筛，阈值为 initial_domain_threshold；按回旋半径、链号、域号排序，与已接受位姿 clash 才跳过。有域预筛接受的父链转入域路线；完全没接受域的父链仍待整链拟合。这不是先做整链预筛、再追加域预筛，而是二选一。

## 6. 主队列：每轮到底做什么

入口 `assembly/unified_queue.py:run_unified_assembly`。

1. 预筛接受的链登记相似组。
2. 普通单域链直接转域，不再单独跑整链拟合；复合物不适用这条。
3. 收集 pending 链及 available 域；父链仍 pending 时，其域不进入可执行池。
4. 与上一轮接受域相似的域优先，其余主要按回旋半径从大到小排序。
5. 遍历：一个组件被接受后结束本轮，更新目标，进入下一轮。链姿态降域函数内部可能一次接受多个域，不能理解为全局始终严格“一轮只收一个域”。
6. 若本轮没有达标接受、但存在域候选：取最高 CC 附近 0.03 内的候选，选 CA/P 数最多者直接接受并掩膜；然后降低域阈值。
7. 阈值前 3 次降低各 0.015，以后各 0.02，最低为 min_domain_threshold。只在上述轮末补选时降低，不是每一轮都降。
8. 没有可处理项或目标点云不足时结束。当前 `_target_has_points` 还包含文件行数条件，不能只从函数名理解为“只要至少一个点就继续”。

**重要实际行为：轮末补选没有再次执行 `cc >= min_domain_threshold`，也没有统一调用 clash 检查。** 所以“域最低接受 CC=0.35”不是当前所有分支的事实。记录到 accepted 不代表必然进入过滤版最终结构。

## 7. 整链拟合的分支

`assembly/chain_fitter.py:fit_chain_item`：

- 首先处理同源组“两次失败”规则。
- 用当前剩余点云和密度启动 `run_fitting(mode="chain")`；内部早停阈值传基础阈值，尚未减同源放宽值。
- 拟合成功返回后，CC 已经在原始密度上复核。
- 普通链接受阈值默认 0.45；已有接受相似链时 0.42。复合物对应 0.35/0.32。
- 达标：可逐域改善，随后接受、掩膜，并拒绝其独立域候选。
- 未达标且有多个域、domain_opt 开启：尝试按整链位姿对齐各域，再对域单独优化。
- 降域收到了部分域：父链记 accepted_via_domains，剩余域仍要继续拟合。
- 降域失败但有域：进入域组装路线；无域则 rejected_no_domain。

链姿态降域：域按残基键对齐到整链拟合姿态；当前图初始 CC<=0.10 则跳过优化；优化后按初始域阈值（有同源放宽）接受，接受后立即掩膜。此处记录的 CC 来自当前图，与独立域拟合返回的原图复核 CC 不完全同口径。

### 相似组状态的注意点

预筛初始化写入了 `accepted_groups`，但没有同步写 `accepted_group_ids`。独立域队列接受更新的是 orch 的域组集合，也不会自动更新 ChainFitState 的链组成功集合。因而“同组已经通过任何路径接受过组件，就一定不会再触发两次失败跳过”并非当前代码的保证。这是应进一步专项验证的状态一致性问题。

## 8. 独立域拟合的三种接受方式

`assembly/domain_fitter.py:fit_domain_once`：

1. **阈值接受**：原图复核 CC >= 当前域阈值，已有同源域则用 max(min_domain_threshold, threshold-0.025)。
2. **平台期接受**：未达标的尝试累计 >=3，最近三次 CC 的极差 <0.02；从所有历史尝试按 CC 降序选不与已接受结构过度 clash 的位姿，接受它。
3. **轮末补选**：交回全局队列，若这一轮无人达标，从候选中补选一个。

无可用拟合文件则域标 rejected；仍有候选但未接受则保持可再尝试状态。

平台期接受也没有额外检查 bcc >= min_domain_threshold。平台期选的是原图评分最高的可行历史位姿，因此可以跨轮比较，但 clash 只在这个特定路径显式检查。

注意：普通达标、链姿态降域、轮末补选并没有全部统一接到 `_clashes_with_accepted`。不要将“项目有 clash 函数”理解成“所有接受都已通过 clash 硬门”。

## 9. PARENet 候选与 O6 请求早停

主要文件：`fitting/demo_mask.py`、`candidate_ledger.py`、`candidate_consumer.py`、`fitting/pipeline.py`。

- 默认单个常驻模型服务；当前模型默认 joint 路径。`--batch-size=10` 控制客户端候选消费，不代表 GPU 一次 forward 十个结构。
- 掩码路径按 mask、config、voxel/fps 顺序评估，当前 all 配置通常每 mask 12 次。每个 mask 内按 overlap 保留最优预测，只有这个最终结果进入候选台账。
- 客户端按稳定 ID 区间消费，批内先 CC 降序，再按 candidate_id 排序；不按文件出现速度决定成员。
- chain 模式 CC>0.20、domain 模式 CC>0.25 才进入实时批内局部优化。这里是“触发优化阈值”，不是装配接受阈值。
- 某优化候选在当前密度上达到 stop_threshold，就停止本拟合请求；不是终止整个复合物任务。
- 没有早停时，未成功优化的候选取 CC top5，再取不重复的 hybrid top5，逐个优化，达标可停止。
- hybrid = CC权重×CC + overlap；CC>=0.2 权重1，否则0.5。
- 最后从优化结果选最优；没有有效优化结果时允许用原始候选。`success=True` 表示有拟合产物，不等于装配层接受。
- chain 模式最多尝试 min(8,源文件数) 个源；首个产出成功即返回，不会为了寻找全局更高 CC 把所有源都跑完。主装配通常只把一个组件放入临时源目录。

异常状态与正常不达标不同：filtered 是正常无候选；执行失败走 error，不能改称低 CC；cancelled 表示主动早停。最新代码含失败后的请求清理，不能沿用旧版“只有进程退出才清理”的描述。

## 10. 局部优化实际执行

`fitting/local_optimizer.py:local_optimize`：

1. 有 initial_cc 则复用，否则计算原位姿 CC。
2. 从同一输入创建 6 组密度梯度搜索，初始步长 `[1.25,3.0,3.5,4.5,5.5,6.0]`，默认最多 2000 步，使用共享池并行这些副本。
3. DensityFitter 的目标是原子位置采样密度均值，交替估计平移/旋转梯度；无改善计数达到400、步长过小或迭代耗尽停止。不是每一步都算完整 CC_mask。
4. 每个副本写结构并计算 CC，用 CC 选优，原始输入也保留为候选。
5. 如果最好的密度梯度副本 CC 仍低于原始值，再运行 ScipyFitter：L-BFGS-B、4 次随机初始化、默认 maxiter=1500。
6. 从原始/六副本/Scipy候选中选好者，再做250步、步长0.5的细密度优化。
7. 细优化后重新计算标准 CC，低于选中候选则回退。

**限定：** ScipyFitter 内部 CC 使用另一套 mask/数据预处理，不与标准 calculate_cc_mask 完全相同。因此代码“原始位姿始终参与选优”是事实，但所有候选完全同口径比较、最终标准 CC 必不下降，需要额外验证，不能只引用注释保证。

逐域改善是另外一层：链接受前，将每个域对齐到链姿态后分别 local_optimize，合并域，再在原图上评估合并 CC；只有合并 CC>传入 original_cc 才替换整个链。不是无条件用拆域结果替换。

## 11. 是否有重复判断/重复计算

| 看起来重复的地方 | 是否必要/实际差异 |
|---|---|
| 原位姿预筛后又 PARENet | 不同位姿；预筛未接受才继续，通常必要 |
| 当前图候选 CC → 原图最终 CC | 密度不同，是评分复核，不是同输入重复 |
| 请求达到 stop_threshold 后装配再判断 | 两层职责，可能原图/当前图不同且同源放宽不同，不能简单删除 |
| 标准 CC 初算传 initial_cc 给优化 | 已减少一次相同初始评分 |
| local_optimize 在多个路径被调用 | 候选优化、链改善、降域、Step5等位姿/目标不同，不能按函数名判重复 |
| TM 预填后分组再调用 calculate_tm_score | 逻辑重复查询，缓存开启时通常避免重跑 USalign；关闭缓存时可能实际重复计算 |
| 装配域分组与 Step4 再算域 TM | 分组规则和用途不同，不能直接合并；Step4 自己的 TM 包装未统一接入 P5 |
| 最终域链过滤与 Step4 输入过滤 | 前者决定初始复合物；后者决定精修输入，作用对象不同 |
| 成功优化候选再次进入 final_select | 成功优化文件用集合排除；失败优化不在成功列表，后面可能重试 |

**阈值不一致造成的额外搜索：** 同源链最终可按0.42接受，但内部请求仍按0.45早停；同源域最终可按更低有效阈值接受，但请求仍按未放宽阈值早停。这可能多算，不一定改变最终接受能力。未经对照不能直接把内部阈值改低，因为会改变候选停止点。

## 12. 已接受域如何组回链及最终复合物

`assembly/domain_assembler.py`：去重 needs_domain_assembly，仅收 status=accepted 的域。

- 一个域：走单域路径，从内部位姿出发恢复来源真实链号并输出 CIF。
- 多域：按域残基范围拆成片段，按起始残基序号排序，复制对应残基。
- 普通链拼入同一链；复合物按 source_chain_id 分链，合并期间保持内部链号，域链最终落盘再恢复真实 ID。
- 合并是按残基范围拼接，不会自动创建缺失连接肽，不会运行 PARENet重新拟合整条域链。
- 合并错误当前仍可返回 None 并丢掉该域链，所以日志中必须检查 domain merge failed，而不是只看最终 exit code。

域链输出两份：全量保留全部接受域；过滤版仅保留记录 CC>=complex_min_cc 的域。整链直接接受的 `type=chain/complex` 不在 build_complex 内逐域过滤，两版都取它的 fitted_cif。因此 README 中“所有组件都逐域过滤”的笼统说法不准确。

`assembly/complex_builder.py:build_complex` 将已选 CIF 的链复制到统一 model，遇到最终链 ID 冲突会重命名并记录。这里不是再做位姿优化，也没有统一的最终碰撞筛选。

没有组件则不写空 CIF、返回 None。存在全量但过滤为空时保留全量，并允许后续步骤/最终返回回退到它。

## 13. 结构相似的结构域组装：Step4

这是独立于主域拟合的“同源域变体枚举”，不是所有输入都运行。

### 13.1 启动条件全部满足

- do_refine 开启；
- 全量或过滤复合物至少一个存在；
- 有域拆分记录；
- accepted_domain_pdbs 非空（实际至少有域被接受，不只是尝试过域拟合）；
- 已接受组件对应原始模板中存在 TM>=similarity_threshold 的相似对。

因此仅有整链接受、从未接受独立域时 Step4 跳过，即便输入链相似。只有一个复合物组件内部有两条相似链，也不自动等价于两个已接受模板满足门控。

### 13.2 回填与过滤

整链接受的 `type=chain` 会回填域 CIF：优先使用此前 domain_cifs，否则从接受姿态对齐各原始域；需要时传占位→真实的 mob_chain_map。`type=complex` 不进入这个 backfill 分支。

Step4 再按 complex_min_cc 过滤有域评分记录的输入；没有记录 CC 的整链回填域保留。不能理解成对所有回填域重新算 CC 再过滤。

### 13.3 跨链域同源关系

`refine/chain_enumerator.py:step2_find_homologous_domains` 对不同链的域运行自己的 `_calculate_tm_score`，取 min(tm1,tm2)，默认阈值 refine_tm=0.75；用并查集合并同源域组。

这与主队列的 `链组.域序号` 不同：Step4 实际比较跨链域。并查集具有传递性，组内任意两域未必都直接达阈值。

完全同源链还要求域数量相同、各对应位置的域达到阈值；这决定后面使用渐进式还是全局枚举。

### 13.4 生成变体并选择组合

每个目标域位置保留原始变体0；其他链中同源域通过序列对齐/叠合生成替代变体。同链来源不作为跨链替代。

每条链对各位置的变体做笛卡尔积，计算连接评分与连接能量。评分是几何量：理想 CA 连接距离3.8 Å，±1.5 Å（2.3–5.3）得1分，(5.3,20] Å得0.1分，其余0分；函数还按片段残基序号间隔筛选参与评分的连接。具体能量由 refine_energy 的连接能量函数计算，不是全复合物密度 CC。

- 有完全同源链组：优先按链顺序渐进选择，分数高优先、能量低次之，更新已占用来源域位置。属于贪心/渐进策略，不保证整个组合空间全局最优。
- 无完全同源链组：枚举跨链配置组合，检查来源域位置唯一使用，最大化总连接评分，能量作次级选择。
- “来源位置唯一”是来源链/域索引约束，不等同于空间原子 clash 检测。
- 配置数量大于100000时只是警告，不自动截断到top-K，可能组合爆炸。

### 13.5 精修输出的边界

枚举器产生 `optimization_workspace/final_results/final.cif` 后，包装层复制为 `refined_complex.cif`。包装层没有将整复合物 CC 与精修前比较后再决定是否替换。

所以 Step4 的含义是“在候选域变体中改进连接几何”，不是“保证实验密度拟合更好”。其完整性也依赖输入回填；混合整链/复合物/域链的复杂场景应专项验证。

## 14. Step5 同源链精修（可选）

`homo_chain_step.py` 仅在 `--homo-chain-refine` 开启且存在输入复合物时调用。

当前函数体的实际顺序：

1. 拆最终复合物为链，以 USalign 对齐区间序列同一性 >=0.9 分组；不是 Step4 的 TM>=0.75。
2. 原始密度上并行算链 CC，同时根据域范围估算断开数。
3. 每组选择断开最少、同等情况下 CC最高的链作为模板。
4. 差链触发：断开更多，或者可比较域比模板差超过 domain_eps=0.02。
5. 用模板按目标链各域作锚定生成候选；原位姿也参加选择。残基数小于原链90%的锚定候选丢弃。
6. 候选与已占位链计算 clash；可行者优先断开少，再按 CC量化排序；无可行者取 clash 最少者。默认 clash_thr=0.1、CC量化 eps=0.003。
7. 同源组按残基编号集合寻找缺失残基，从覆盖缺失最多的供体尝试补回；clash达标可接受补回，没有额外要求整链 CC 必须上升。
8. 有改变才输出 homo_chain_refined_complex.cif；失败包装层记录警告并回退原输入。

注意：模块顶部注释写“链CC差<=chain_eps跳过、优于原链才替换”，但当前函数体的核心门控是断开数/域差异，chain_eps 没有在该函数选择逻辑中使用。不要据这段旧注释推断严格CC单调提高。

## 15. 早停与终止汇总

| 层级 | 条件 | 停止什么 |
|---|---|---|
| 单次PARENet拟合请求 | 某局部优化候选达 stop_threshold | 停该请求并等待取消收口；后续组件仍可继续 |
| final_select | 顺序优化时首个达标 | 停后续精选候选优化 |
| 密度梯度优化 | max_iter、无改善patience、最小步长 | 停该优化副本 |
| 同源链两次失败 | 未登记成功组且失败计数>=2 | 跳同组后续整链拟合，域可继续 |
| 同源域每轮两次 | 未接受组本轮已试两个 | 暂缓本轮其余同组域，下轮计数重置 |
| 全局轮次 | 接受了链或域 | 本轮结束，更新mask后下一轮 |
| 全局组装 | 目标不足或无可处理项 | 结束拟合队列，进入输出/精修 |
| Step4 | 门控条件不满足 | 跳过精修，不终止整个任务 |
| Step5 | 未开启/无同源组/最终无改变 | 不生成新精修产物 |

O6 只固定请求候选消费；不能据此声称 Step4/Step5 的所有并行枚举、集合遍历和并列取舍都已经具备同样的确定性保证。

## 16. 最终应该看哪个文件

程序返回的优先级：

```text
homo_chain_refined_complex.cif
  > refined_complex.cif
  > assembled_complex.cif
  > assembled_complex_all.cif
  > None
```

其中 assembled_complex 是过滤域链后的初始组装；all 是所有已接受组件；refined 是Step4几何枚举产物；homo是Step5产物。查看日志最后的 `Complex:` 或 run_pipeline 返回的 complex_cif，不要固定认为过滤版永远是最后交付件。

assembly_summary 在Step4/5之前生成，主要描述初始接受/域链组装，不能单靠它推断精修后每条链的最终CC。

## 17. 本次阅读中需要特别留意的行为

这些是代码阅读结论，不是本轮新跑实验确认的失败；本轮不修改它们。

1. 域阈值下限不是所有接受路径硬门：平台期和轮末补选可以绕过最低CC。
2. clash 检查没有统一覆盖所有接受分支。
3. 主队列的域同源组由整链组+域编号推定，域TM预填并不改变该组划分。
4. 链同源成功状态由多个集合记录，预筛/独立域成功与“两次失败跳过”集合并不完全同步。
5. 请求早停阈值与装配同源放宽阈值不同，且当前图/原图不同，不可直接删除复核判断。
6. Step4 按连接几何选结果，没有最终全复合物CC提升门；完全同源组采用渐进式，不是穷举全局最优。
7. Step5 的实际断开/补全优先规则与顶部“只提高CC”的旧注释不一致。
8. `domain_cifs` 在整链改善是否最终获胜之前就保存；Step4回填会优先读取它，需确认这是否符合预期的“保留最佳链姿态”语义。
9. 部分辅助函数仍有异常降级/返回None的行为；“候选推理失败明确报错”不等于整个项目所有步骤统一fail-fast。

## 18. 源码定位索引

下列路径相对于 `/xiangyux/claude_c_work/demo_reg`；以提交6c4ed00函数名定位，避免后续行号移动误读。

| 内容 | 文件/函数 |
|---|---|
| CLI和参数 | main.py / build_parser、assembly_kwargs_from、main |
| 预处理/主调用 | protassem/pipeline.py / run_pipeline |
| 结构链号和匹配 | protassem/core/structure.py / cif_to_pdb_placeholders、align_by_resid |
| 体素化/点云 | protassem/voxelize/mol_to_mrc.py、sampling/sampler.py、core/points_txt.py |
| 准备/相似/预筛/掩膜/输出顺序 | protassem/assembly/orchestrator.py / run、_precompute_similarity、_pre_screen_chains、_pre_screen_domains、_mask_region |
| 全局轮次 | protassem/assembly/unified_queue.py / run_unified_assembly、_active_items |
| 整链/降域/改善 | protassem/assembly/chain_fitter.py / fit_chain_item、try_domains_via_chain_pose、try_improve_chain_with_domains |
| 域阈值/平台期 | protassem/assembly/domain_fitter.py / fit_domain_once |
| 轮末/clash | protassem/assembly/assembly_opt.py / select_round_end_candidate、ca_overlap |
| PARENet/O6 | protassem/fitting/demo_mask.py、candidate_consumer.py、candidate_ledger.py、pipeline.py |
| 局部优化 | protassem/fitting/local_optimizer.py / DensityFitter、ScipyFitter、local_optimize |
| CC与TM | protassem/core/scoring.py、similarity.py |
| 域拼接/复合物 | protassem/assembly/domain_assembler.py、complex_builder.py |
| Step4门控/回填 | protassem/assembly/refine_step.py |
| Step4枚举/评分 | protassem/assembly/refine/chain_enumerator.py、refine_energy.py |
| Step5门控/修复 | protassem/assembly/homo_chain_step.py、homo_chain_refine.py |

本说明不替代结构质量评价；它用于解释日志、手动检查输出及定位后续规则调整的位置。
