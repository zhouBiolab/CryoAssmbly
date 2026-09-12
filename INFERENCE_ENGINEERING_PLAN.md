# demo_reg 推理加速与工程化实施方案

版本：2026-09-12。目标读者：接手实施的工程师或模型。本文是实施规范，不代表所列功能已经实现。

> **修订说明（2026-09-12，v1.1）**：本文只覆盖**阶段二**（运行时与性能）。阶段一（边界与格式契约硬化）以
> `ENGINEERING_HARDENING_PLAN.md`（v2）为准；阶段二请按该文档第 4 节的"必须 / 可选 / 实验"三档选择执行。
> 本次更正：§4 模块表中 `core/scoring.py`、`core/similarity.py` 标注为"已存在，改造"，避免被读成新建文件。
> 修订记录见文末。

## 1. 目标、现状与交付边界

目标按顺序为：代码可读、模块化、易维护和移植；保持科学计算语义；减少重复计算；用实测证明加速。

项目位于服务器 `/xiangyux/claude_c_work/demo_reg`，SSH 别名 `my-server`，Conda 环境 `point`。Windows 映射为 `X:\claude_c_work\demo_reg`；当前会话看不到 X 盘。所有 Python、Git 和测试在服务器运行。不要将本地空仓库当成服务器仓库。

审查基线提交为 `5b016f5`。实施前重新读取 Git 状态和项目指令；如果基线已变化，先核对差异，再应用本文。不能覆盖后续用户修改。

### 已经发生的修改

| 文件 | 当前变更 | 实际完成程度 |
|---|---|---|
| `protassem/core/performance.py` | 新增 Metrics 和 configure_cpu_threads | 初步模块，代码过于压缩，需要重写整理 |
| `protassem/fitting/pipeline.py` | fit_request 总耗时、metrics 输出、调用线程配置 | 仅粗粒度计时，未完成共享池、缓存或独立配置 |
| `protassem/sampling/sampler.py` | os.system 改 subprocess.run，检查返回码和空输出 | 已实现，仍需路径及失败场景测试 |

上述变更没有提交。编译、导入与 `git diff --check` 通过；没有跑真实完整 GPU 基准。不能把这些检查表述为完整正确性验证或性能提升。

需要修正此前陈述：`setdefault` 不会覆盖已有线程设置，而且当前调用发生在 NumPy 导入后，不能保证已加载 BLAS 实际只使用一个线程。当前并没有真正完成 CPU 线程控制。

### 本方案覆盖

- CLI 到推理、CC、局部优化、掩膜、结果输出的主路径。
- 运行配置、计时、持久池、有限缓存、服务生命周期、任务身份、阶段恢复。
- GPU 真批处理作为独立实验阶段，有明确的前置验证，不冒充已完成的批量功能。
- 不升级 Python/PyTorch/CUDA，不换模型权重，不全面重构第三方 pareconv，不改变默认阈值。

## 2. 编码要求：简洁优先

1. Python 3.8 兼容。配置和记录用标准库 dataclass；不要用 Python 3.10 的联合类型语法。
2. 普通转换用函数；只有持有资源或状态的对象使用类。不要为每个步骤建立服务类、工厂或插件接口。
3. 一行一个操作，显式 import，清晰变量名。移除单行 try/finally、多个分号和压缩 JSON 写入。
4. 业务参数显式传递。删除 `_NUM_PROCESSES`、`_BATCH_SIZE`、`_METRICS` 全局可变状态，不换成隐藏单例或线程局部全局状态。
5. 不做重复的防御性编程：内部受控字段直接访问，不到处 `.get(..., 默认值)`、`hasattr` 或 `except Exception: pass`。
6. 输入、外部进程、持久文件是必要边界：这些地方验证一次并给出清晰错误。内部错误直接传播，不把异常变成 CC=0 或成功。
7. 只在资源上下文中使用 try/finally，保证池、文件、进程关闭。进程协议入口可以捕获异常以回传失败，必须保留 traceback 并返回 failed。
8. 不创建推测性抽象。模块职责分开，但不用九个服务类包裹原有函数。
9. 每个公开函数写清输入、输出、单位和修改的状态；注释解释算法约束，避免复述代码。
10. 默认启用等价优化。会改变候选集合、搜索预算或数值路径的功能必须显式开启，并记录在配置中。

## 3. 必须保持的科学语义与纠正项

### 3.1 接受和掩膜仍按原顺序串行

现有队列通常在第一个达标组件被接受后立即结束本轮。不得把所有达标结构一起接受，也不得按 worker 完成顺序接受。

并行任务可以基于同一不可变密度快照计算，但接受决策使用原排序、原阈值和原 tie-break。一次接受成功后更新 mask_version；旧版本推测任务作废。最初版本直接丢弃旧结果，不设计未经验证的廉价复核。

### 3.2 预筛没有 GPU 推理

当前 pre_screen 是对输入原始位姿计算 CPU CC，然后处理 clash。不得为“GPU 批量预筛”新增 PARENet 调用；这会增加工作并改变预筛含义。这里只复用 CC 池和密度上下文。

### 3.3 请求排队不是真正模型 batching

PARENet forward 使用单对点云布局，例如 `lengths[*][0]`，部分代码存在 batch=1 假设。把 JSON 请求组成列表、逐一 forward，只是队列协议，不是多样本 GPU batch。

真批处理需要分离邻居索引、点分区和配准过程，防止样本间互相匹配。未完成这些改动及等价验证前，模型 batch size 保持 1。

### 3.4 不使用未经证明的筛选条件

回旋半径、bounding box 和序列长度只能作为候选排序特征，不能未经证明直接判定 TM-score=0。构象变化可能使这些特征不可靠。

同源整链的同序号域不一定同源。当前代码已有这种假设，性能改造不能进一步扩大；保留现状并单独记录为算法验证事项。

没有可证明上界时，不实现“剩余候选不可能超过最佳”的早停。此前 1.5~3 倍、2~5 倍只是设想，不是验收承诺。

## 4. 最小模块布局与接口

新增模块控制在以下范围，原有算法函数继续留在现有模块（标“已存在，改造”的是改造现有文件，不是新建）：

| 模块 | 职责 | 不负责 |
|---|---|---|
| `protassem/runtime/config.py` | RuntimeConfig、JSON 加载、资源解析 | 业务阈值重写 |
| `protassem/runtime/metrics.py` | 计时事件、资源快照、汇总 | 调度和算法 |
| `protassem/runtime/execution.py` | 运行级上下文、CPU 池生命周期 | 组件接受 |
| `protassem/runtime/manifest.py` | 指纹、阶段状态、恢复 | pickle 整个 orchestrator |
| `protassem/core/scoring.py`（已存在，改造） | DensityMapContext、坐标 CC | 任务调度 |
| `protassem/core/similarity.py`（已存在，改造） | USalign、SQLite 缓存 | 掩膜状态 |
| 现有 parenet_client/demo_mask | 请求协议和模型进程 | 装配接受 |

`RuntimeConfig` 至少包含：`cc_workers`, `tm_workers`, `local_opt_workers`, `blas_threads`, `candidate_batch_size`, `poll_interval_s`, `scoring_cache_mb`, `tm_cache_path`, `seed`, `request_timeout_s`, `max_local_candidates`, `local_time_budget_s`, `resume`。

默认值：候选 batch 维持 CLI 当前 10；轮询初期维持 2.5 秒；BLAS 线程 1；评分缓存每 worker 128 MiB；seed 7351；request timeout 不设置硬时限，服务退出检测始终启用；局部候选上限和时间预算默认 None；resume 默认 false。

新增 `--runtime-config <json>`。JSON 未给出的值取 RuntimeConfig 默认；显式 CLI 参数覆盖 JSON；旧 `--num-processes` 在没有各阶段 worker 配置时作为兼容上限。不要复制旧的所有 CLI 解析缺陷：新增参数用 argparse，业务阈值集中保留当前 CLI 默认。

运行级 ExecutionContext 用 with 管理，含 config、metrics 和一个惰性 CPU 池。CC、TM 和局部优化当前分时使用同一池；每次 map 的同时在途任务不超过对应阶段 worker 数。明确禁止在池 worker 内再创建池。

`run_pipeline(..., runtime_config=None)` 在根部建立上下文，传入 `run_assembly(..., context=...)`，再传到 `run_fitting(..., context=...)`。保留旧参数调用兼容；独立调用 run_fitting 时在入口创建自己的上下文并在返回前关闭。

## 5. 分步实施任务

每步完成后先通过本步测试，再进入下一步。禁止把所有步骤合并成一个无法定位问题的大补丁。

### 步骤 0：记录基线和输入

操作：读取三处现有修改、父目录指令和运行环境。确认一个小型 fixture 和一个真实数据集；README 中 example2 当前未随仓库提供，不能假定存在。真实数据集缺失时完成其余实现与合成测试，将真实基准标记为未验证，不虚构结果。

记录 Git commit/dirty diff 指纹、输入和权重 SHA256、实际包版本、CUDA、CPU affinity、GPU 型号/可用显存。先保存当前有修改版本的基线；历史 HEAD 的对比在隔离副本完成，不能 reset 工作树。

预期：得到唯一的 baseline manifest；后续能分清代码改变和输入改变。

验收：基线包含所有输入、配置、工具指纹；不上传数据、不启动不受控训练任务。

### 步骤 1：配置与早期线程控制

修改：建立 runtime/config.py，入口在导入 NumPy/SciPy/Torch 前设置子进程环境。CPU 可用量取 affinity 与系统 CPU 上限，物理核心检测可用时再取较小值；检测不到物理核心时明确记录采用逻辑核心数，不伪装物理核心。

资源默认：有效 C 核下 CC=min(旧 worker 上限, max(1,C*3//4))，TM=min(旧上限,max(1,C//2))，local=min(旧上限,6,C)。当前阶段分时执行，不把这三项相加创建三套池。

库调用可能已加载 BLAS：优先使用环境中已有 threadpoolctl 的上下文限制实际线程，同时在依赖文件显式登记经当前 Python 验证的兼容版本。只改当前进程和子进程，不改系统配置。

预期：两个独立 run_fitting 调用不会互相覆盖配置；实际 BLAS 线程限制可检测。

验收：测试默认/JSON/CLI 优先级、C=1、worker 超 CPU 上限；使用 threadpool_info 检查真实限制。超预算显式报参数错误，不偷偷猜用户意图。

### 步骤 2：重写 Metrics，细化计时

修改：把初稿 performance.py 迁移到 runtime/metrics.py，使用多行可读代码，清除未使用 nullcontext。运行根目录一个 writer，由主进程写入；worker 返回耗时字段，不并发写同一文件。

每条事件包含 run_id、stage、task_id、chain_id/domain_id（无则 null）、attempt_id、mask_version、started_at、elapsed_s、status、candidate_count、accepted、cc_mask。上下文异常时 status=failed，随后原异常继续传播。

埋点：标准化、逐结构体素化、采样、DomainParser、TM、GPU 请求等待、模型 forward、CC batch、局部优化、mask、结果写出和总流程。GPU forward 用 CUDA event 在 profiling 模式测量；不要默认每个算子 synchronize。

GPU 利用率可通过每秒一次 nvidia-smi 可选采样；不存在时记录 unavailable，不使 CPU 测试失败。显存区分进程 PyTorch 峰值和设备级显存，不能混写。CPU RSS 区分父进程与 worker，不简单相加作唯一真实内存。

输出 performance.jsonl 和 summary.json，assembly_summary.txt 附性能摘要路径。汇总区分 elapsed wall time 与并行 worker 累计时间，不能把重叠阶段相加当总耗时。

预期：知道时间花在哪个阶段，而不只是 fit_request 总耗时。

验收：成功和异常均生成事件；两次运行 run_id 不重复；JSONL 无交错；汇总可由事件重新计算。

### 步骤 3：复用一个受控 CPU 池

修改：用 ExecutionContext 的 map 接口替换 `_batch_cc()`、prefill_tm_cache、预筛与 local_optimize 中的临时池。worker 函数保持顶层可序列化；局部优化仍按当前六组 step_sizes 运行，候选选择顺序不变。

采用 spawn 上下文，避免从已经初始化 CUDA 的进程 fork。池惰性创建，整个 run_pipeline 内复用；standalone fitting 在自己的 with 中释放。pool worker 只完成单个 CC/TM/gradient copy，不调用会创建新池的 local_optimize。

预期：候选 batch 数增加不再线性增加池启动次数；无嵌套并行，保持原结果顺序。

验收：同一运行多批任务只创建一个池；异常后 worker 退出；worker=1 和 2 返回相同排序与分数；CPU 等待时间、任务时间分别报告。确认 GPU server 与 CPU 池启动顺序不导致继承句柄或卡死。

### 步骤 4：有界评分缓存与数组接口

修改：在 scoring.py 添加 DensityMapContext，保存规范化数据、voxel_size、origin、shape、contour 后的实验数组。提取 `calculate_cc_mask_coords(context, coords, elements, resolution)`，保留原文件接口作为薄包装。

只缓存确定可复用的数据，不把依赖候选坐标的 mask 当公共静态缓存。每进程缓存有字节上限；单张图大于上限时直接计算不缓存。key 包含绝对路径、size、mtime_ns、contour；动态 current_density.mrc 还要使用显式 mask_version，防止同路径覆盖命中旧数据。

缓存数组只读。local_optimizer.DensityMap 需要阈值修改时用独立数组，不能改共享原图。结构坐标缓存同样有容量限制和文件指纹；临时优化 PDB 会覆盖/删除，不能永久按文件名缓存。

局部优化保留目前 ScipyFitter 的目标函数；它与最终 CC_mask 不完全相同，不得在“缓存重构”中偷偷统一评分公式。

预期：同一批候选只需每个 worker 读取一次当前密度，减少 I/O，CC 数值一致。

验收：合成 MRC 对比旧/新 CC 绝对差 <=1e-6；覆盖非零 origin、轴映射、各向异性 voxel、空 mask、同路径文件替换。不能顺手修正轴/掩膜公式造成基线变化；发现公式问题另列提交。缓存内存不超过预算。

### 步骤 5：SQLite TM 缓存

修改：父进程负责 SQLite 查询和批量写入，worker 只运行 USalign。使用标准库 sqlite3，不引入数据库服务。

缓存表存 key、score 和创建时间。key 来自对称排序的两结构内容指纹、USalign 可执行文件内容指纹、完整参数及缓存版本。文件 hash 在一个运行内按 stat 记忆，避免每个结构对重复扫描文件；跨运行验证真实内容。

USalign 超时、非零返回、输出无法解析均不写合法分数；保留 stderr 并传播明确失败。合法分数 0 可以缓存。默认不加基于 Rg/bbox 的硬过滤。

预期：同一数据重复运行不再重复执行相同 USalign 请求，换权重无关但换 USalign 或输入会正确失效。

验收：对称 key 命中；文件/二进制改变失效；失败未缓存；并行结果由父进程一次写入；第二次测试外部调用数为零。比较新旧同源分组一致。

### 步骤 6：候选评分与优化预算

修改：首先显式 CandidateRecord，字段包括编号、path/pose、overlap、CC、mask_version、优化状态。保留当前实时 batch 选择、最终 CC top5 + hybrid top5 和第一个达标早停。

把 max_iterations=2000、六个 step_sizes、fine_iterations=250、fine_step=0.5 等当前常量放入 LocalOptimizationConfig；Scipy 参数按真实代码提取，默认不改。

max_local_candidates、local_time_budget_s 默认 None。开启时在下一候选启动前检查预算；不强杀正在写结果的优化任务。输出注明 budget_exhausted，不能叫模型失败。候选级并行作为后续模式，不与六-copy 并行叠加。

廉价 bbox/低分辨率评分先以 shadow 模式记录，不丢候选。经过真实集验证再新增显式 experimental_pruning 配置，默认关闭。

预期：默认质量路径不变；用户可以显式控制速度/质量预算。

验收：默认候选 ID 和顺序与旧代码一致；预算=0 不启动优化且仍可选择已有原始候选；预算耗尽状态可追溯；固定随机种子按 task_id 派生，不依赖 worker 完成顺序。确认 SciPy 随机初始化可复现。

### 步骤 7：可靠的单 GPU 服务协议

修改：保留常驻单模型，但以 request_id 定位任务，不再仅靠目录内 DONE 文件。请求包含协议版本、request_id、mask_version、输入及输出路径、seed；完成文件包含 status、error、candidate_count 和计时。

客户端 poll 同时检查 server.poll；服务器死亡立即报失败，不能永远等待 DONE。服务返回失败时不当 success。STOP 后需要收到本请求 cancelled/done，再复用可能变化的输入路径。

预测文件先写临时文件，再原子 rename 为完整文件；发布候选事件/索引只发生在文件完成后，避免监视器读半个 PDB。服务日志放本次 run/logs，不写所有任务共用的仓库根目录日志。

wait(timeout) 真正等待或抛 TimeoutError，不能像当前接口立即 poll 伪装等待。可选 retry 默认 0，只有明确可恢复的进程启动失败允许有限重试；格式错误/OOM 不无限重试。

预期：取消、崩溃和推理失败不会挂起装配，也不会读到前一次请求结果。

验收：用假 server 测成功、失败、取消、进程中途退出、旧 marker、两个请求、路径含空格、半文件。测试不需要 GPU。

### 步骤 8：任务身份、mask 版本和阶段恢复

修改：AssemblyState 先只持有 mask_version、task records、已完成阶段。不要一次迁移所有 orchestrator 字典。组件成功接受并且 mask 更新成功后提交一次新版本；mask 更新失败中止，不能继续用旧 mask 假装完成。

需要推测执行时，将 target/density 保存为不可变版本路径，不给 GPU server 指向将被覆盖的 current_* 文件。所有结果核验版本；旧版本结果作废。未启用推测执行时不额外复制全图。

manifest 采用 JSON，记录 schema_version、输入/配置/代码/工具指纹、阶段状态、产物 hash、任务状态和最终路径；同目录临时文件+os.replace 原子更新。配置也用 JSON，避免为 YAML 多加依赖。

第一版恢复支持标准化、体素化、采样、TM 和完整已完成的推理请求；装配中断时从装配阶段重跑。文档必须明确不能恢复任意接受中间点。

第二版装配恢复仅在完整轮次边界：同时持久化 chain/domain records、accepted lists、队列 threshold/decreases/round_num、last_accepted_pdbs、ChainFitState、计数器、mask 和 target 版本、随机状态、候选文件。将 unified_queue 中这些局部变量收进显式 RoundState，再序列化。输出结果和状态必须属于同一个 checkpoint；不能只存 accepted 列表就声称恢复。

预期：第一版可靠省略已完成预处理；第二版才具备轮次恢复。

验收：中断后恢复与同 seed 连续运行一致；修改输入、配置、代码或工具后拒绝复用旧阶段；缺失或 hash 不符产物重新执行该阶段及依赖阶段；失败阶段不标 completed。

### 步骤 9：GPU 优化与真正 batching（独立实验）

先检查 eval/no_grad 和 GT correspondence 的使用链。只有证明 GT 输出在推理所有消费者中不用，才能把训练专用计算移入训练分支。保持权重格式不变；不直接用当前 PyTorch 不支持的 torch.compile。

优先在同一个 fitting 请求内部对独立 mask/config 子任务做预处理预取，CPU 与 GPU 重叠；保持原候选顺序和 seed。这比跨装配组件推测更容易保持结果。

真 batching 前逐层审计：collate 点云布局、lengths、邻居偏移、point_to_node_partition、attention、matching、hypothesis registration。确保样本间隔离；不能仅 padding 后调用原 forward。建立 forward_batch(pairs) 输出按 request_id 分组，单样本 forward 仍可直接运行。

按两侧总点数分桶（<5k、5k~15k、>15k），桶边界只是调度参数。显存预算依据实测不同点数峰值拟合，不能用“显存字节/点数”这种单位不成立的公式。先最大 batch=2 验证，再放大；超预算提前拆分，OOM 后报告并释放本请求资源，不无限缩小重试。

跨组件批处理只有在前述 RoundState/mask_version 正确后加入，默认关闭。同一轮按原顺序消费预计算结果，第一个接受后取消或丢弃剩余旧结果。记录浪费的推测计算量；加速不成立则保留单任务默认。

验收：batch1/2 同 seed 坐标与分数对比；不同点数、取消、OOM、样本间零交叉匹配；完整装配接受顺序和结果对比。未通过时不得宣布真 batching 完成。

### 步骤 10：增量 mask（最后实施）

先测量 mask I/O 占比；仅当耗时显著时实施。复用 mask_fitted_region 的真实几何条件和 TXT 元数据更新，逐 voxel/逐 point 与旧实现对比。

维护数组和 active_points，不反复删除数组造成索引漂移；checkpoint 时序列化。MRC header、origin、voxel size、axis mapping 和 TXT 坐标/法向量顺序保持一致。只有等价验证通过才改变默认存储方式。

验收：连续接受至少三个组件后的密度数组、点序列与旧实现一致；边界原子、重叠 mask、清空点云及恢复后追加 mask 均通过。该阶段不能与评分公式修复混成一个提交。

## 6. 测试、性能对比及验收

### 无 GPU 测试

使用 unittest 和 tempfile，fixture 自动生成，不依赖个人 /xiangyux/test_data 路径。覆盖配置、线程控制、pool 复用/清理、计时、缓存失效、SQLite、假推理服务、状态版本、阶段恢复。采样器测试用假可执行文件检查含空格路径、stderr、非零返回和空输出。

服务器命令：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate point
cd /xiangyux/claude_c_work/demo_reg
python -m unittest discover -s tests -v
python -m compileall -q main.py protassem tools
git diff --check
git status --short --branch
```

### 科学等价性

- 默认优化：CC 绝对差 <=1e-6，输出坐标 <=1e-3 Å（考虑 PDB 写出精度），组件身份、顺序及状态一致。
- 处于阈值附近的微小浮点差异如果改变接受决策，也算待调查失败，不能只因 CC 差小就通过。
- 队列与最终产物完整比较；Step4/Step5 启用时同样纳入，不以主装配结束替代最终结果。
- 剪枝、预算和真 batching 单独报告是否发生质量变化，不和等价缓存优化混报。

### 性能实验

同一代码基线、数据、seed、硬件和配置；先冷启动，再至少两次暖启动。记录 CPU affinity 和其他 GPU 负载。分别对比总 wall time、候选数、CC 次数、USalign 次数、优化次数、读图次数、pool 启动次数、RSS 和 GPU 峰值。

不能把缓存冷热差异归因于 GPU batching；不能把减少候选导致的更快叫等价加速。汇总同时给速度与质量，不预先承诺倍数。

通过标准：功能测试通过、默认结果等价、有明确减少的重复工作、代表性数据总耗时不发生可重复的明显退化。真实数据缺失时写“未验证”，不写“已加速”。

## 7. 提交顺序、风险与接手检查单

建议每步单独提交：baseline/docs → runtime config → metrics → reusable pool → scoring cache → TM cache → budgets → request protocol → stage resume → round checkpoint → GPU experiments → mask experiments。先整理现有三处未提交修改，避免重复实现。

| 常见错误 | 必须采用的处理 |
|---|---|
| 多进程传入 Metrics/Pool 导致 pickle 失败 | worker 只接收小型纯数据参数 |
| 每 worker 缓存完整大图造成内存翻倍 | 强制缓存字节预算；记录总 worker 数 |
| current_density 路径不变导致旧分数 | mask_version 加入上下文/cache key |
| completion 顺序改变首次接受 | 按原候选顺序归并 |
| server 崩溃无限等待 | poll 同时检查进程退出状态 |
| CUDA fork 死锁 | CPU worker 使用 spawn |
| 只写阶段标记就跳过损坏文件 | 验证产物指纹及依赖阶段 |
| 为了“简洁”吞异常 | 边界报告失败，内部直接传播 |
| 大规模重构后无法解释结果差异 | 每步比较，算法变更独立实验 |

接手模型开始前逐项确认：

1. 已读取实际代码及现有 diff，知道哪些只是方案。
2. 配置与资源由根上下文传递，没有新增模块全局状态。
3. 无批处理=单任务列表循环之类虚假交付。
4. 每步交付说明修改内容、实测结果、尚未验证事项和下一依赖。
5. 不删除用户数据、不自动 push、不升级 CUDA 环境。
6. 文档仅是起点：若实际代码与描述不符，先记录具体位置，调整最小实现，不凭不存在的函数名继续写。

最终交付必须包含代码、测试、默认配置示例、性能对比和恢复说明。只完成 metrics 或 compileall 不能宣称整套方案完成。


## 修订记录

- 2026-09-12 v1：初版。
- 2026-09-12 v1.1：顶部加入与阶段一 `ENGINEERING_HARDENING_PLAN.md` v2 的关系说明；§4 模块表把 `core/scoring.py`、`core/similarity.py` 标注为“已存在，改造”（消除“新增”歧义）。正文其余内容未改。
