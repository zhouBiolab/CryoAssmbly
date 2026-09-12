# 回归 case 清单（真实输入不入库）

真实数据保留在服务器上，**不提交**；本目录只记录输入清单与指纹，便于复现与对比。

## case A：`test/1`（日常回归，单链 PDB 输入）

| 项 | 值 |
|---|---|
| 目录 | `/xiangyux/claude_c_work/demo_reg/test/1`（仓库内 `test/`，未跟踪） |
| 实验密度图 | `EMD-8436.mrc`，192×192×192，voxel 1.31 Å，28 325 376 字节，sha256 `a3e82ddd10d804dd14fd191520d8dbc2a79a60f0fe379d1527f63a5087caf84f` |
| 结构 | `chain_a_1.pdb`（链 A，564 残基，4596 原子，sha256 `0e28a47f28f84a2ee5ec5fd1af346828fa844486fa1bdda4e461c2c9b815d9f1`）、`chain_b_2.pdb`（链 B，564 残基，4596 原子，sha256 `e0619a4c6632e1961bfe3ecd90a11bd818a781a7af44b0f7a08ab46e563b9e50`） |
| 参数 | `resolution.txt` = 5.6（3 字节，sha256 `5fa067c1e4cdd35a5916d6eab8a27fb0f32778ce1de6c013e44ac205822b6698`）、`contour_level.txt` = 0.04（4 字节，sha256 `a888fe9e2469182b8e3e3bca241d3189dc144349bc2d0ac64c56c444276e9763`） |
| 运行命令 | `python main.py test/1/EMD-8436.mrc test/1 5.6 0.04 <out_dir> --log` |
| 基线运行 | `/xiangyux/claude_c_work/demo_reg_cases/out_baseline_test1`（2026-09-12，967 秒） |
| 新版运行 | `/xiangyux/claude_c_work/demo_reg_cases/out_new_test1` |

## case B：`fiting_lg/6lu9`（复合物路径，需显式 `--complex-domain-opt`）

| 项 | 值 |
|---|---|
| 目录 | `/xiangyux/test_data/fiting_lg/6lu9` |
| 结构 | `6lu9.cif`（4 链 A/B/C/D） |
| 实验密度图 | `EMD-0979.mrc`；同目录含 `resolution.txt` 与 `contour_level.txt` |
| 用途 | 占位链号映射、复合物域优化、S4b 链号空间确认；**不是每步必跑项**，备选 `fiting_lg/7sjx`（2 链） |
| 注意 | 需从原始 CIF 派生出**非恒等映射**副本（真链号 `Q/R/S/T`）才能暴露占位链号与真链号混用 |

## 派生副本（不入库，由测试生成）

- 单链真 `B` → 占位 `A` → 恢复 `B`（`tests/test_align_by_resid.py` 覆盖）。
- 两链合一 CIF、真链号 `Q/R`（同上测试覆盖占位映射）。
- 复合物域优化链路（`--complex-domain-opt`）用 6lu9 派生的 `Q/R/S/T` 副本确认。
