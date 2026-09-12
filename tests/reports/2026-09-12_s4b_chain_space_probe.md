# S4b 链号空间定向探针报告

日期：2026-09-12/13　代码：`feat/engineering-hardening` @ `c1fc6eb`

## 目的

确认"复合物 CIF + 非恒等映射（真链号 `Q/R` → 占位 `A/B`）"时，占位链号是否会进入最终输出。

## 派生输入（原始数据未改动）

- `test/1/chain_a_1.pdb` + `chain_b_2.pdb` 合并为一个 CIF，链号改为 `Q`、`R`（坐标、残基不变）：
  `/xiangyux/claude_c_work/demo_reg_cases/derived_qr/complex_QR.cif`
- 实验图沿用 `EMD-8436.mrc`；分辨率 5.6、contour 0.04。
- 占位转换实测（`work/cif_conversions/complex_Q+R_1.pdb`）= `['A', 'B']` → **非恒等映射已确认生效**。
- 日志确认命中的路径：`Complex Q+R: split into 2 chains for domain opt`、`5 total domains across 2 chains`。

## 三次配置与结果

| 配置 | 关键参数 | 接受情况 | 带占位链号的文件 | 最终输出链号 |
|---|---|---|---|---|
| probe1（2290 s） | `--complex-domain-opt`（默认阈值） | Total accepted 0 | 中间 PDB `domains/Q+R/A|B/*.pdb`；`fitted_domains/Q+R/domain_*.cif` = `Q+R` | 空复合物（0 链） |
| probe2（176 s） | `--complex-domain-opt --no-refine --chain-threshold 0.30 --complex-threshold 0.25 --domain-threshold 0.30 --domain-min-cc 0.25 --complex-min-cc 0.10` | 复合物整链接受（cc=0.2943） | `work/chain_improve_Q+R/domain_*.cif` = **`A`/`B`**；`improved_Q+R.pdb` = `A`,`B` | `assembled_complex.cif` / `assembled_complex_all.cif` / `chains/complex_Q+R_01.cif` = **`Q`,`R`** |
| probe2b（176 s） | 同 probe2，去掉 `--no-refine` | 复合物整链接受 | 同上 | 同上 = **`Q`,`R`**；Step4 因 `no domain fitting was performed` 跳过 |
| probe3（372 s） | `--complex-domain-opt --domain-threshold 0.30 --domain-min-cc 0.25 --complex-min-cc 0.10`（整链/复合物阈值保持 0.45/0.35） | 域级接受 5 个组件 | `fitted_domains/Q+R/domain_*.cif` = `Q+R` | 0 链复合物；Step4 因 `no homologous chains` 跳过 |

## 结论

1. **引入点确认（已核实）**：`chain_fitter.py:237-240` 的 `chain_id_for_cif = src_cid`（`src_cid` 来自
   `_split_complex_domains` → `split_structure_to_chains(rec["pdb_file"])` 的占位空间）会让逐域微调产物
   `work/chain_improve_<cid>/domain_*.cif` 带占位链号（`A`/`B`）。
2. **未观测到泄漏（已核实，覆盖三条路径）**：所有 `final_results` 产物链号均为真链号 `Q`/`R`；
   `_accept_chain` 对复合物应用 `chain_map`（`orchestrator.py:563-569`）完成恢复。
3. **残留未覆盖分支（未证伪）**：`refine_step._backfill_chains_as_domains` 会把 `cr["domain_cifs"]`
   （占位标注）复制进 `fitted_domains/`，再由 Step 4 合并。触发它需要同时满足：链被接受**且**记录了
   `domain_cifs`、发生过域拟合、且有同源链使 Step 4 运行——本派生 case（单一复合物）未能构造出该组合
   （三次运行的 Step 4 分别因为"无域拟合"/"无同源链"被门控）。按既定约定**未修改 `chain_id_for_cif`**。
4. **复现条件（留给后续）**：构造含 ≥2 条同源链的复合物 CIF，让域拟合先发生（整链阈值 > 域阈值），
   随后整链被接受并记录 `domain_cifs`，再开启 Step 4，检查 `refined_complex.cif` 链号。

## 顺带发现（既有行为，非阶段一改动）

无任何组件被接受时，`build_complex` 仍写出 `assembled_complex.cif` / `assembled_complex_all.cif`（0 链），
Bio.PDB 解析报 `_atom_site.label_atom_id`。`complex_builder.py` 在阶段一**未改动**（`git diff 5b016f5..HEAD`
可证），属既有行为；后续需单独决定"跳过写出"还是"写合法空 CIF"。
