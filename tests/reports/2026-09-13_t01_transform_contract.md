# T01：坐标、旋转中心与变换组合（契约 + 修复 + 新基线）

日期：2026-09-13　代码：`feat/runtime-metrics` @ `219ce89`（修复提交 `219ce89`）

## 一、坐标空间与契约（已核实）

| 空间 | 定义 | 证据 |
|---|---|---|
| 原始坐标 | PDB 原子坐标（Å）；采样点云坐标 = 体素坐标×sample + origin（同一 Å 框架） | `core/points_txt.py`、`demo_mask` 读取 |
| 中心化坐标 | `x' = x - c_src`、`y' = y - c_ref`，c 为**点云质心** | `demo_mask.py:192-197` |
| 编码/求解空间 | 模型内部再做 `points / scale`（`parenet/model.py:236-237`），网络在此空间给出 `y' = R x' + t'` | 代码 + 本轮 overlap 实测 |
| 写出坐标 | `R (x - center) + center + t`（`StructureData.apply_transformation`） | `local_optimizer.py:52-56` |

**平移单位核实（实测）**：`demo_mask` 用**未除 scale** 的 `src_for_ov` 直接施加模型返回的 `pred_t`，
得到的 overlap = 0.3268（若单位带 scale，位移会被放大数十倍、overlap 应接近 0）→ `pred_t` 与点云同单位
（中心化但未除 scale）✓ 与任务卡"网络编码缩放不意味着平移自动需要缩放"一致。

## 二、发现的坐标错误（已确认并修复）

位姿在**点云质心**系求解，而 `transform_pdb` 默认绕**原子质心**旋转，于是写出坐标相对正确结果
偏移 `(I - R)(c_atom - c_src)`。

判定实验（合成结构，`|c_atom - c_src| = 0.69 Å`、绕 z 旋转 37°）：

| 检查 | 结果 |
|---|---|
| 理论偏移 `(I-R)(c_atom-c_src)` | (0.3098, −0.1831, 0.0)，模 0.3599 Å |
| 旧路径实测最大坐标差 | **0.3103 Å**（逐原子差与理论一致，差异来自 PDB 三位小数写出） |
| 原子质心 == 点云质心时 | 0.0009 Å（仅写出精度） |

真实 case 的质心差（`test/1` 的链 A 源点云 vs 其 PDB 原子）：
`|c_atom - c_src| = 1.5255 Å` → 该量级会随旋转角放大成同量级～数倍 Å 的位姿偏移。

**修复（只在配准输出边界）**：`transform_pdb(..., center=c_src)`；`demo_mask` 在写出预测 PDB 时
显式传 `center=c_src`，平移仍为 `t' + (c_ref - c_src)`。数学上：
`R(x - c_src) + c_src + [t' + c_ref - c_src] = R(x - c_src) + t' + c_ref` ✓ 精确等于正确提升。
**局部优化绕自身质心旋转的参数化保持不变**（`center=None` 仍是原子质心，`apply_transformation` 未改语义）。

## 三、测试（8 项，`tests/test_transform_contract.py`）

| 测试 | 断言 |
|---|---|
| 点云质心系位姿正确映射原子 | 写出坐标 = `R(x-c_src)+t'+c_ref`（≤ 1e-3 写出精度） |
| 默认绕原子质心 | `center=None` → `R(x-c_atom)+c_atom+t` ✓ 参数化未被破坏 |
| 旧路径偏移取证 | 与正确结果之差 = `(I-R)(c_atom-c_src)`（本用例模 > 0.1 Å） |
| 两质心相等时无差异 | 最大差 0.0009 Å（写出精度） |
| 已知刚体变换往返 | 正变换+逆变换回到原坐标（≤ 2×1e-3） |
| 两阶段组合约定 | `T = (R2 R1, R2 t1 + t2)` 与逐步施加一致（1e-9） |
| 旋转不原地污染 | `apply_transformation` 不修改入参、不共享内存 |
| 变换不写回源文件 | 源 PDB 内容不变 |

全量测试：100 项通过。

## 四、修复后的新基线（旧基线已被取代）

| 产物 | 修复前 md5 | 修复后 md5 |
|---|---|---|
| assembled_complex.cif | `2497fefc2e866ad65db628fb75f1ec0a` | **`76638d0f97d4ee5775110c66b15e73c6`** |
| assembled_complex_all.cif | `2497fefc…` | **`76638d0f…`** |
| refined_complex.cif | `dbc937efc071cd9d1d2611f3ad84743f` | **`bd281f40e25352455ba34d41e6bbc9f7`** |
| assembly_summary.txt | `2f184d9d…` | `0356ae49…` |

新基线记录：`tests/cases/baseline_after_t01_fix.md5`。

**变化性质（不是阈值边界翻转）**：

| 组件 | 修复前 CC | 修复后 CC | 差 |
|---|---|---|---|
| B（整链） | 0.442731 | 0.443163 | +0.00043 |
| A（域链） | 0.443079 | 0.442980 | −0.00010 |

组件数量与类型完全一致（2 个：1 整链 + 1 域链），CC 差在 ±0.0005 内 → 属"B 类明确错误修复"允许的
结果变化，且未出现阈值附近决策改变。

微基准（固定配准）对照：best overlap 完全相同（0.3268 / 0.0527 / 0.058，overlap 在点云空间计算，
不受写出路径影响）；wall 15.73→19.85、3.80→5.04、3.90→3.61 s——落在 T00 实测的 ±29%% 波动带内，
不归因于本修复。

运行时基线（修复后一次完整 `test/1`）：1062 s、池创建 1 次、未归因 1.08 s、无残留进程。
