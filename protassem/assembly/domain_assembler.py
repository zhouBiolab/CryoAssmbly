"""Domain chain assembly -- merge accepted domains into chain CIFs.

链号空间约定（审计 P1-1/P1-2 后明确）
------------------------------------
- **内部（占位）空间**：`cif_to_pdb_placeholders()` 给出的单字符占位链号（如 A/B），
  域 PDB、`source_chain_id`、`merge_domains()` 的输出都在这个空间；
- **真实链号空间**：最终 CIF 里用户可见的链号（如 Q/R，或多字符）；
- **组件 ID**：`chain_id`（如 `Q+R`），只用于内部记录与中间文件命名。

规则：**占位 -> 真实的映射只在最终输出处做一次**，而且只在两个地方做：
  1. `orchestrator._accept_chain()`（链级路径）；
  2. 本模块的 `_restore_chain_ids()`（域链最终落盘：多域合并与单域两条路径共用）。
`merge_domains()` 一律留在内部空间，绝不提前映射 —— 否则"逐域改善"的产物会带着真链号
回流，再被最终输出映射第二次（真链号与占位链号有交集时撞名/报错）。
"""

import os
import logging
import shutil

from Bio.PDB import MMCIFParser, MMCIFIO, Structure, Model, Chain

from protassem.core.scoring import calculate_cc_mask
from protassem.core.structure import pdb_to_cif, write_structure_with_chain_map

log = logging.getLogger(__name__)


def chain_map_of(orch, chain_id):
    """取该组件的占位链号 -> 真链号映射（原始 chain_records 是唯一来源）。"""
    record = orch.chain_record(chain_id)
    return (record.get("chain_map") or {}) if record else {}


def _restore_chain_ids(orch, chain_id, cif_path, source_cid=None):
    """把**内部空间**的文件写成**最终链号空间**的文件（映射一次）。

    - 无 `chain_map`（非复合物）：链号即真实链号，直接返回原路径；
    - 给了 `source_cid`（单域路径）：只改这一条来源链，显式指定、不做 `get(id, id)` 猜测，
      因此不可能失败在其他链上、也不会二次映射；
    - 否则（多域合并）：整个组件的占位链号一起映射。
    """
    chain_map = chain_map_of(orch, chain_id)
    if not chain_map:
        # 非复合物：链号即真实链号。源可能是 PDB（单域路径的 fitted_pdb），
        # 落盘前统一转成 CIF（链号不变），避免把 .pdb 内容写进 .cif 名字。
        if str(cif_path).lower().endswith(".cif"):
            return cif_path
        out_path = str(orch.work_dir
                       / ("final_%s.cif" % os.path.basename(cif_path).rsplit(".", 1)[0]))
        pdb_to_cif(cif_path, out_path, chain_id=None)
        return out_path
    if source_cid is not None:
        real_cid = chain_map.get(source_cid)
        if real_cid is None:
            log.warning("Chain %s: source chain %s not in chain_map %s; "
                        "keeping the id as-is", chain_id, source_cid, chain_map)
            return cif_path
        mapping = {source_cid: real_cid}
    else:
        mapping = chain_map
    out_path = str(orch.work_dir / ("final_%s.cif" % os.path.basename(cif_path).rsplit(".", 1)[0]))
    write_structure_with_chain_map(cif_path, mapping, out_path)
    return out_path


def assemble_domain_chains(orch):
    """Assemble each failed chain's accepted domains into chain CIFs.

    A chain reaches here only after failing chain fitting, so we commit to
    the domain path.  A chain whose domains all fell below domain-min-cc
    simply does not enter the complex.

    每条域链同时产出两个 CIF：
      - 全量（所有已接受域）          -> 记录 fitted_cif          -> assembled_complex_all.cif
      - 过滤版（仅 cc>=complex_min_cc）-> 记录 fitted_cif_filtered -> assembled_complex.cif
    两者都在**最终链号空间**落盘（内部空间 -> 最终只映射一次）。
    """
    for cid in dict.fromkeys(orch.needs_domain_assembly):   # 去重，避免同一域链装两遍
        chain_rec = next((c for c in orch.chain_records
                          if c["chain_id"] == cid), None)
        if not chain_rec:
            continue

        fitted = [d for d in orch.domain_records.get(cid, [])
                  if d["status"] == "accepted"]
        if not fitted:
            log.info("Chain %s: no domain passed domain-min-cc; chain dropped",
                     cid)
            continue

        if len(fitted) == 1:
            _handle_single_domain(orch, cid, chain_rec, fitted[0])
            continue

        is_complex = chain_rec.get("is_complex", False)
        ranges, _ = orch.domain_adjacency.get(cid, ({}, {}))
        assembled_cif = merge_domains(orch, cid, fitted, ranges,
                                      is_complex=is_complex)
        if not assembled_cif:
            log.info("Chain %s: domain merge failed; chain dropped", cid)
            continue

        domain_cc = calculate_cc_mask(orch.original_density_mrc, assembled_cif,
                                      orch.resolution, orch.contour)
        log.info("Chain %s: assembled from %d domains (cc=%.4f)",
                 cid, len(fitted), domain_cc)
        final_cif = _restore_chain_ids(orch, cid, assembled_cif)
        filtered_cif = _filtered_domain_cif(orch, cid, fitted, ranges,
                                            assembled_cif,
                                            is_complex=is_complex)
        if filtered_cif is None:
            filtered_final = None
        elif os.path.abspath(filtered_cif) == os.path.abspath(assembled_cif):
            filtered_final = final_cif          # 无域被过滤：与全量同一份
        else:
            filtered_final = _restore_chain_ids(orch, cid, filtered_cif)
        _save_domain_chain(orch, cid, final_cif, domain_cc, fitted,
                           filtered_final)


def merge_domains(orch, chain_id, fitted_domains, domain_ranges,
                  is_complex=False, out_cif=None):
    """Merge domain CIF files into one chain CIF by residue order.

    For complexes with complex_domain_opt, creates multi-chain CIF
    using source_chain_id.  out_cif 可指定输出路径（缺省落到 work_dir）。

    **输出保持在内部（占位）链号空间**；最终链号的恢复由调用方在落盘处做一次
    （`_restore_chain_ids()` 或 `orchestrator._accept_chain()`）。
    """
    out_cif = out_cif or str(orch.work_dir / ("assembled_%s.cif" % chain_id))
    try:
        segments = []
        for drec in fitted_domains:
            dnum = drec["domain_num"]
            if dnum in domain_ranges:
                for s, e in domain_ranges[dnum]:
                    segments.append({"start": s, "end": e, "drec": drec})
        segments.sort(key=lambda x: x["start"])

        struct = Structure.Structure("assembled")
        model_obj = Model.Model(0)

        if is_complex:
            chains_map = {}
            serial = 1
            for seg in segments:
                cif = seg["drec"].get("fitted_cif")
                if not cif or not os.path.exists(cif):
                    continue
                src_cid = seg["drec"].get("source_chain_id", chain_id)
                if src_cid not in chains_map:
                    # 内部空间：**保持占位链号**（见模块开头"链号空间约定"）
                    chains_map[src_cid] = Chain.Chain(src_cid)
                chain_obj = chains_map[src_cid]
                parser = MMCIFParser(QUIET=True)
                ds = parser.get_structure("d", cif)
                for m in ds:
                    for ch in m:
                        for res in ch:
                            if seg["start"] <= res.get_id()[1] <= seg["end"]:
                                nr = res.copy()
                                for atom in nr:
                                    atom.serial_number = serial
                                    serial += 1
                                chain_obj.add(nr)
            for ch in chains_map.values():
                model_obj.add(ch)
        else:
            chain_obj = Chain.Chain(chain_id)
            serial = 1
            for seg in segments:
                cif = seg["drec"].get("fitted_cif")
                if not cif or not os.path.exists(cif):
                    continue
                parser = MMCIFParser(QUIET=True)
                ds = parser.get_structure("d", cif)
                for m in ds:
                    for ch in m:
                        for res in ch:
                            if seg["start"] <= res.get_id()[1] <= seg["end"]:
                                nr = res.copy()
                                for atom in nr:
                                    atom.serial_number = serial
                                    serial += 1
                                chain_obj.add(nr)

            if len(list(chain_obj)) == 0:
                log.warning("merge_domains: empty merge for chain %s "
                            "(ranges=%s), skipping", chain_id, domain_ranges)
                return None
            model_obj.add(chain_obj)

        struct.add(model_obj)
        io = MMCIFIO()
        io.set_structure(struct)
        io.save(out_cif)
        return out_cif
    except Exception as e:
        log.error("merge_domains error: %s", e)
        return None


def _filtered_domain_cif(orch, chain_id, fitted_domains, ranges, full_cif,
                         is_complex=False):
    """过滤版域链 CIF：仅保留 cc_mask >= complex_min_cc 的域（仍在内部空间）。

    返回：
      - full_cif：无域被过滤（复用全量，省一次合并）
      - None：所有域均低于阈值（该链不进入过滤复合物）
      - 新路径：部分域被过滤，按保留域重新合并
    """
    thr = orch.complex_min_cc
    kept = [d for d in fitted_domains if d.get("cc_mask", 0.0) >= thr]
    if len(kept) == len(fitted_domains):
        return full_cif
    if not kept:
        return None
    out = str(orch.work_dir / ("assembled_%s_filtered.cif" % chain_id))
    return merge_domains(orch, chain_id, kept, ranges, out_cif=out,
                         is_complex=is_complex)


def _handle_single_domain(orch, chain_id, chain_rec, domain_rec):
    """Single accepted domain is the chain's contribution.

    P1-2：只接受一个域时也必须恢复链号 —— 域 CIF 是以**组件 ID**（如 `Q+R`）落盘的中间产物，
    直接复制到最终目录会让它成为最终结构的链号。这里从内部空间的位姿出发，
    按该域的来源链**映射一次**（复合物），非复合物保持链号。
    """
    d_file = domain_rec.get("fitted_cif") or domain_rec.get("fitted_pdb")
    if not d_file or not os.path.exists(d_file):
        log.info("Chain %s: single domain file missing; chain dropped",
                 chain_id)
        return
    d_cc = calculate_cc_mask(orch.original_density_mrc, d_file,
                             orch.resolution, orch.contour)
    # 源位姿优先取内部空间的 fitted_pdb（其链号是 source_chain_id）；domain_cif 只作兜底
    source = domain_rec.get("fitted_pdb") or d_file
    final_file = _restore_chain_ids(orch, chain_id, source,
                                    source_cid=domain_rec.get("source_chain_id"))
    # 单域：该域 cc 达标则过滤版=全量，否则过滤版为空（链不进过滤复合物）
    filtered = (final_file if domain_rec.get("cc_mask", d_cc) >= orch.complex_min_cc
                else None)
    _save_domain_chain(orch, chain_id, final_file, d_cc, [domain_rec], filtered)


def _save_domain_chain(orch, chain_id, cif_file, cc, fitted_domains,
                       filtered_cif=None):
    orch._domain_chain_order += 1
    base = "domain_chain_%s_%02d" % (chain_id, orch._domain_chain_order)
    dst = str(orch.final_dir / "domain_chains" / (base + ".cif"))
    shutil.copy2(cif_file, dst)

    # 过滤版：与全量同源则复用 dst；为空则 None；否则单独落盘
    if filtered_cif is None:
        filtered_dst = None
    elif os.path.abspath(filtered_cif) == os.path.abspath(cif_file):
        filtered_dst = dst
    else:
        filtered_dst = str(orch.final_dir / "domain_chains"
                           / (base + "_filtered.cif"))
        shutil.copy2(filtered_cif, filtered_dst)

    orch.accepted_chains.append({
        "type": "domain_chain", "chain_id": chain_id,
        "fitting_order": orch._domain_chain_order,
        "cc_mask": cc, "fitted_cif": dst, "fitted_cif_filtered": filtered_dst,
        "domains": [d["domain_num"] for d in fitted_domains],
        "domain_details": [{"num": d["domain_num"],
                            "cc_mask": d.get("cc_mask")}
                           for d in fitted_domains],
    })
