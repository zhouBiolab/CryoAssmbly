# protassem -- 蛋白质结构组装流水线

**简体中文** | [English](README.md)

输入实验密度图 + 若干链的结构文件，自动完成 **体素化 -> 点云采样 -> PARENet 配准拟合 -> 统一队列组装 -> 精修**，
输出组装好的复合物 CIF。

```
密度图(.mrc) + 链结构(.pdb/.cif)  ->  [体素化] -> [采样] -> [统一队列拟合+组装] -> [精修]  ->  复合物(.cif)
```

---

## 一、环境依赖

### 1. Python 包

```bash
conda create -n point python=3.8
conda activate point
pip install -r requirements.txt
```

### 2. pareconv（关键依赖，源码随仓库附带，需本机编译）

PARENet 推理依赖 pareconv 包（含 CUDA 编译扩展），源码已随仓库放在
protassem/fitting/pareconv_src/：

```bash
cd protassem/fitting/pareconv_src
pip install -e .
cd pareconv/extensions/pointops/
python setup.py install
```

编译前提：已装好 torch（如 1.10.0+cu113）+ nvcc + gcc/g++。

验证：python -c "import torch; import pointops_cuda; from pareconv.modules.ops import index_select; print('OK')"

### 3. 可执行文件（项目已内置，首次使用需赋权）

```bash
chmod +x protassem/core/USalign
chmod +x protassem/sampling/Sample
chmod +x protassem/assembly/domain_parser/domainparser2.LINUX
chmod +x protassem/assembly/domain_parser/dssp
```

### 4. 模型权重（已内置）

PARENet 权重 epoch-18.pth.tar（6.6MB）已放在 protassem/fitting/parenet/weights/。

---

## 二、运行

### 快速测试

仓库不含示例数据（`example2` 不在仓库内），请用任意数据目录，目录内需包含
密度图 `.mrc`、结构文件 `.pdb`/`.cif`、`resolution.txt`、`contour_level.txt`：

```bash
cd <项目根目录>
python main.py <case_dir> --log
```

结果在 `<case_dir>/output/`（手动模式可指定独立输出目录）：

```
example2/output/
+-- pipeline_<时间戳>.log
+-- voxelized/
+-- sampled/  sampled_sources/
+-- assembly/
    +-- final_results/
    |   +-- assembled_complex.cif      最终复合物（过滤版，按 complex_min_cc 逐结构域剔除 cc 低的域）
    |   +-- assembled_complex_all.cif  完整版（含全部已接受域，不过滤）
    |   +-- refined_complex.cif        Step4 精修（若触发）
    |   +-- assembly_summary.txt       摘要
    |   +-- chains/                    链结果
    |   +-- domain_chains/             域组装结果
    +-- work/                          中间过程文件
```

### 自动模式

数据目录下放：密度图 .mrc、结构文件 .pdb/.cif、resolution.txt、contour_level.txt

> **自动标准化**：程序自动读取结构文件内部的 chain ID（不依赖文件名），
> 多链复合物自动保留为复合物（chain_id="A+B"形式）。文件名可以任意命名。
> **链号去重**：若各输入（含复合物内部链）链号有重复，自动重排为唯一链号
> （保留首次出现，冲突者顺延到下一空闲号：A–Z、a–z、然后两字母 AA…ZZ），并以 CIF 写出。
> 链号用到两字母（总链数 > 52）时全程以 CIF 处理（PDB 单字符列存不下）。

```bash
python main.py <data_dir> --log
```

### 手动模式

```bash
python main.py <density.mrc> <struct_dir> <resolution> <contour> [output_dir] --log
```

选项可以放在位置参数之前、之间或之后。用法错误（未知选项、选项缺值、非数字、位置参数
个数不是 1/4/5）退出码为 2，并打印具体原因。入口校验在建立输出目录之前完成：密度图必须
存在且为 `.mrc`、结构文件列表非空且文件都存在、`resolution > 0`、`contour` 为有限数值、
`voxel_size > 0`（`contour` 不接受缺省，自动目录模式仍从 `contour_level.txt` 读取）。

### 参数说明

#### 开关参数（加上=开启，不加=关闭）

| 参数 | 默认 | 含义 |
|------|------|------|
| `--log` | 关 | 写日志文件到输出目录（pipeline_<时间戳>.log） |
| `--log-file <path>` | 无 | 指定日志文件路径（覆盖 `--log` 的自动命名） |
| `--no-improve-accepted` | — | 加上=**关闭**逐域微调；不加=默认**开启**（链接受后逐域优化取更高 CC） |
| `--no-domain-opt` | — | 加上=**关闭**域优化兜底；不加=默认**开启**（链没达标时用链姿态降域，把达标的域接受；与逐域微调解耦） |
| `--complex-domain-opt` | 关 | 复合物域优化：拆内部链→各自域分割→接受后逐域微调→按链合并比较 CC |
| `--homo-chain-refine` | 关 | Step 5 同源链精修：残基保护 + 域补回 + 用好链模板修复差链 |
| `--no-pre-screen` | — | 加上=**关闭**装配前原始位姿并行预筛；不加=默认**开启** |
| `--save-all-attempts` | 关 | 保存所有拟合尝试到 all_attempts/（调试用） |
| `--cleanup` | 关 | 完成后删除 work/ 下的临时目录 |
| `--no-domain-split <ids>` | 无 | 指定不拆域的链 ID（逗号分隔，如 `--no-domain-split A,B`），这些链只能作为整链拟合 |
| `--no-refine` | — | 加上=**关闭** Step 4 同源域精修；不加=Step 4 默认**开启** |

#### 数值参数（后跟一个数字）

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--chain-threshold` | 0.40 | 普通链接受的 cc_mask 阈值 |
| `--complex-threshold` | 0.35 | 复合物接受的 cc_mask 阈值（独立于链阈值，因复合物 CC 天然偏低） |
| `--domain-threshold` | 0.40 | 域接受的起始 cc_mask 阈值（每条链的域独立衰减） |
| `--domain-min-cc` | 0.35 | 域拟合绝对阈值：衰减下限 + 多域同接门槛（拟合阶段低于此值不接受） |
| `--complex-min-cc` | 0.25 | 收尾高置信度筛选：cc<此值的组件不进最终复合物（与拟合阶段无关） |
| `--similarity-threshold` | 0.85 | 链/域间 TM-score 相似判定阈值 |
| `--refine-tm` | 0.75 | Step 4 同源域分组的 TM-score 阈值 |
| `--num-processes` | 8 | 并行进程数（CC 计算 / 局部优化） |
| `--batch-size` | 8 | 监控循环每攒多少 pred 做一次评估 |
| `--mask-radius-factor` | 1.35 | PARENet 掩码半径因子（掩码半径 = 回转半径 × 该因子） |
| `--min-point-distance-factor` | 0.32 | 掩码内最小点间距因子（最小点间距 = 掩码半径 × 该因子） |

#### 使用示例

最简运行（全部默认）：

```bash
python main.py example2
```

推荐生产配置（开日志 + 同源精修）：

```bash
python main.py <data_dir> --log --homo-chain-refine
```

含复合物的数据（降低复合物阈值 + 复合物域优化）：

```bash
python main.py <data_dir> --log --complex-threshold 0.35 --complex-domain-opt
```

全开 + 自定义阈值：

```bash
python main.py <data_dir> \
  --log \
  --complex-domain-opt \
  --homo-chain-refine \
  --chain-threshold 0.40 \
  --complex-threshold 0.30 \
  --domain-threshold 0.40 \
  --num-processes 16
```

只做拟合+组装，关闭 Step 4 精修：

```bash
python main.py <data_dir> --no-refine --log
```

指定部分链不做域分割（只能整链拟合）：

```bash
python main.py <data_dir> --no-domain-split A,B --log
```

---

## 三、组装策略简述

组装采用**统一队列**：链和域按回旋半径（大优先）排入同一个队列。

1. **链拟合**：PARENet 配准 → 局部优化 → CC 达阈值则接受并 mask 密度
2. **链失败**：该链的域**立即加入队列**，与剩余链/域按回旋半径混排
3. **域拟合**：每条链的域有独立的阈值和轮次管理
4. **域链合并**：所有域拟合完后，按残基序合并回链
5. **复合物构建**：合并已接受的链/域链 → assembled_complex_all.cif（完整）+ assembled_complex.cif（域级过滤）

详细算法见 [ALGORITHM.md](ALGORITHM.md)。

---

## 四、项目结构

```
demo_reg/
+-- main.py                          入口（参数解析）
+-- compute_cc_mask.py               独立算 cc_mask
+-- check_clash.py                   CA 重叠检测
+-- geo_sym_refine.py                独立同源 refine CLI
+-- requirements.txt
+-- ALGORITHM.md                     Step 3 组装算法详解
|
+-- protassem/
    +-- pipeline.py                  三步流水线总调度
    +-- core/                        共享工具
    |   +-- scoring.py               cc_mask（numba）
    |   +-- structure.py             PDB/CIF 读写/对齐
    |   +-- similarity.py            USalign 封装（TM-score、Seq_ID）
    |   +-- numba_kernels.py         numba 核函数
    |   +-- constants.py / io.py
    |   +-- USalign                  (可执行文件)
    +-- voxelize/                    步骤1：体素化
    +-- sampling/                    步骤2：采样
    |   +-- Sample                   (可执行文件)
    +-- fitting/                     步骤3a：拟合
    |   +-- pipeline.py              统一拟合入口（两阶段+监控+多进程CC）
    |   +-- parenet_client.py        PARENet 常驻服务客户端
    |   +-- demo_mask.py             PARENet 推理引擎
    |   +-- local_optimizer.py       局部优化
    |   +-- masker.py                掩膜
    |   +-- sw_mask.py / utils.py
    |   +-- parenet/                 PARENet 模型+权重
    |   +-- pareconv_src/            pareconv 源码
    +-- assembly/                    步骤3b：组装
        +-- orchestrator.py          组装总入口（准备+域分割+复合物域优化+收尾）
        +-- unified_queue.py         统一队列调度器
        +-- chain_fitter.py          链/复合物拟合逻辑
        +-- domain_fitter.py         域拟合+轮次管理
        +-- domain_assembler.py      域链合并
        +-- assembly_opt.py          优化辅助（参数集中+多域同接+clash）
        +-- complex_builder.py       复合物拼接+报告
        +-- domain_splitter.py       域分割
        +-- domain_parser/           DomainParser(可执行文件)
        +-- refine_step.py           Step4 同源域精修
        +-- homo_chain_step.py       Step5 同源链精修 门控
        +-- homo_chain_refine.py     Step5 同源链精修 核心
        +-- refine/                  vendored 同源域枚举
```

---

## 五、外部依赖一览

| 依赖 | 类型 | 处理方式 |
|------|------|----------|
| pareconv | CUDA 编译包 | 源码随仓库，需本机编译 |
| PARENet 权重 | 模型文件 6.6MB | 已内置 parenet/weights/ |
| USalign | 可执行文件 | 已内置 core/ |
| Sample (VoxEM) | 可执行文件 | 已内置 sampling/ |
| DomainParser / dssp | 可执行文件 | 已内置 assembly/domain_parser/ |
| config/model/backbone | PARENet 模型代码 | 已内置 parenet/ |

除 pareconv（CUDA 编译包，须按目标机编译）外，其余依赖均已内置。

---

## 六、独立工具

### 独立算 cc_mask
```bash
python compute_cc_mask.py <结构.pdb/cif> <密度.mrc> <分辨率> [contour]
```

### 独立同源 refine（不经主流程）
```bash
python geo_sym_refine.py <case_dir>
python geo_sym_refine.py --complex a.cif --density b.mrc --resolution 3.5
```

### CA 重叠检测
```bash
python check_clash.py <PDB目录> [碰撞距离] [重叠比例阈值]
```
