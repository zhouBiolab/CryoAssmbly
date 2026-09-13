# T00：固定基线、依赖和测试输入（任务卡第一卡）

日期：2026-09-13　代码：`feat/runtime-metrics` @ `e8a6906`
（`git status` 为 dirty：未跟踪 `MASK_REGISTRATION_OPTIMIZATION_TASKS.md` 与本卡新增的 `tools/benchmark_registration.py`）

## 一、环境与依赖来源（实际加载路径，不是"应该"）

| 项 | 实测值 |
|---|---|
| Python / Torch / CUDA | `3.8.20` / `1.10.0+cu113` / `11.3` |
| GPU | `NVIDIA A800 80GB PCIe MIG 7g.80gb`（**MIG 切片**，非整卡 A800） |
| 权重 | `protassem/fitting/parenet/weights/epoch-18.pth.tar`，sha256 `dcb187df47e3f7b3…` |
| **pareconv 实际导入** | `/xiangyux/PARENet-main/pareconv/__init__.py` |
| 仓库副本 | `/xiangyux/claude_c_work/demo_reg/protassem/fitting/pareconv_src/pareconv/__init__.py` |
| **是否使用仓库副本** | **`False`** ← 任务卡列出的风险项，实测命中 |
| CUDA 扩展实际导入 | `/root/miniconda3/envs/point/lib/python3.8/site-packages/pointops-0.0.0-py3.8-linux-x86_64.egg/pointops_cuda.cpython-38-x86_64-linux-gnu.so` |

**含义**：`pareconv` 来自共享项目 `/xiangyux/PARENet-main`，**修改仓库内 `pareconv_src` 不会生效**；
而 `protassem/fitting/parenet/model.py`、`backbone.py` 在仓库内，改动会生效。
按任务卡"偏差处理"要求：不替换共享依赖、不重装 CUDA、不换权重，只如实记录。

## 二、固定输入（manifest：`tests/cases/registration_manifest.json`）

- 源：`sampled_sources/chain_A_1_2.00.txt`，7335 点，sha256 `bf589f7d91a48151…`；配套 PDB `chain_A_1.pdb`。
- 目标掩码（来自 `out_p3_a2` 的装配过程产物，按点数从大到小，order 0/1/2）：

| order | 文件 | 点数 | sha256（前 16） |
|---|---|---|---|
| 0 | `/xiangyux/claude_c_work/demo_reg_cases/out_p3_a2/assembly/work/mask_1/filtered.txt` | 8006 | `d31d7f2dc2cb456b` |
| 1 | `/xiangyux/claude_c_work/demo_reg_cases/out_p3_a2/assembly/work/mask_2/filtered.txt` | 3565 | `40645d454d140d24` |
| 2 | `/xiangyux/claude_c_work/demo_reg_cases/out_p3_a2/assembly/work/mask_3/filtered.txt` | 1672 | `b0870e2727d42fc3` |

- 参数：`configs=all`（6 个配置 × 2 种采样 = 12 对/目标）、`use_mask=False`（固定配准，不生成掩码）、`seed=100000`、
  `mask_radius_factor=1.35`、`min_coverage=0.15`、`min_point_distance_factor=0.32`。

## 三、重放证据（微基准只做固定配准，不跑装配）

命令：
```bash
python tools/benchmark_registration.py run --manifest tests/cases/registration_manifest.json \
    --out-dir /xiangyux/claude_c_work/demo_reg_cases/t00_bench --repeat 1
```

| order | 目标点数 | 源点数 | wall (s) | 预测数 | best overlap |
|---|---|---|---|---|---|
| 0 | 8006 | 7335 | 15.73 | 12 | 0.3268 |
| 1 | 3565 | 7335 | 3.80 | 12 | 0.0527 |
| 2 | 1672 | 7335 | 3.90 | 12 | 0.058 |

- 每个目标写出 12 个 `pred_*.pdb`（6 配置 × 2 采样），与参数一致 ✓
- 两次重放同一 manifest 都成功：目标 0 分别 20.28 s / 15.73 s（**运行间波动约 29%%**，含首次预热与 MIG 争用），
  因此 T02 必须做冷/暖重复并记录 GPU 负载，不能把单次差值当算子加速。
- 工具修正记录：`run_inference` **没有返回值**（只写文件与日志），最初按返回值统计得到 `predictions=0`；
  已改为从产物文件名解析 overlap（命名携带 overlap），修正后与写出的 12 个文件一致 ✓

## 四、验收对照（任务卡 T00）

| 验收项 | 结果 |
|---|---|
| 同一 manifest 能重放 | ✓（两次重放均成功，产物写独立目录 `t00_bench/`） |
| 报告含实际模块路径 | ✓（pareconv / CUDA 扩展 / 权重路径与 hash 均实测记录） |
| 报告含有效参数 | ✓（configs=all → 12 对/目标；seed、半径参数、点数、顺序） |
| 依赖来源可复查 | ✓（含"未使用仓库副本"这一关键结论） |
| 不重装 CUDA / 不替换共享依赖 / 不换权重 | ✓ 未做任何环境改动 |

## 五、留给后续卡的注意项

1. **pareconv 不在仓库内**：T04（单侧几何拆分）、T06/T07（编码缓存）若需要改 `pareconv` 内部代码，
   必须先把加载来源切到仓库副本，否则改动无效；切源属环境改动，需要单独决定。
2. **GPU 是 MIG 7g.80gb 切片**：绝对耗时与显存上限不代表整卡；T02/T03 的峰值显存结论需标注该前提。
3. **基线波动大**：同一输入两次重放差 29%%，后续所有对比需冷/暖各两次并记录负载。
4. 基线代码里 `run_inference` 无返回值，微基准与后续测量都应从产物/日志取值。
