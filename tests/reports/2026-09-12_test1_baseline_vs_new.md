# test/1 基线 vs 新版对比报告

- 日期：2026-09-12
- case：`test/1`（`EMD-8436.mrc` 192³、voxel 1.31 Å；`chain_a_1.pdb` 链 A、`chain_b_2.pdb` 链 B，各 564 残基；
  `resolution.txt` = 5.6；`contour_level.txt` = 0.04）
- 基线代码：`git worktree` @ `5b016f5` + S0 冻结的三处未提交改动（`/xiangyux/claude_c_work/demo_reg_baseline`）
- 新版代码：分支 `feat/engineering-hardening` @ `39fc426`（S1–S6 全部落地）
- 命令（两侧相同）：`python main.py test/1/EMD-8436.mrc test/1 5.6 0.04 <out_dir> --log`
- 输出目录：基线 `/xiangyux/claude_c_work/demo_reg_cases/out_baseline_test1`；
  新版 `/xiangyux/claude_c_work/demo_reg_cases/out_new_test1`

## 对比结果

```text
# 运行对比

- baseline: `/xiangyux/claude_c_work/demo_reg_cases/out_baseline_test1`
- new: `/xiangyux/claude_c_work/demo_reg_cases/out_new_test1`

```text
assembled_complex.cif    IDENTICAL  md5=2497fefc2e86
assembled_complex_all.cif IDENTICAL  md5=2497fefc2e86
refined_complex.cif      IDENTICAL  md5=dbc937efc071
assembly_summary.txt     IDENTICAL（11 行）
```
```

## 结论

- 三个最终 CIF 与 `assembly_summary.txt` **逐字节一致**（md5 相同），接受组件数一致（2）。
- wall time：基线 967 s，新版 956 s（差约 1%，属运行间波动，**不作为加速结论**）。
- 本 case 的输入是两个单链 PDB，不触发占位链号映射与复合物域优化路径，因此它验证的是
  S1/S2/S3/S5/S6 的"等价重构"属性；S4a 的链号修复由 `tests/test_align_by_resid.py` 的构造用例覆盖。

## 未验证项

- 复合物 CIF 输入 + `--complex-domain-opt`（S4b 链号空间确认，需派生的非恒等映射副本）。
- `test_5kem/5kem`（4 链）与 `fiting_lg/6lu9`（复合物）两个更大 case。
