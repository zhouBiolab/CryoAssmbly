"""Domain chain assembly -- merge accepted domains into chain CIFs."""

import os
import logging
import shutil

from Bio.PDB import MMCIFParser, MMCIFIO, Structure, Model, Chain

from protassem.core.scoring import calculate_cc_mask
from protassem.core.structure import pdb_to_cif

log = logging.getLogger(__name__)


def chain_map_of(orch, chain_id):
    """取该组件的占位链号 -> 真链号映射（原始 chain_records 是唯一来源）。"""
    record = orch.chain_record(chain_id)
    return (record.get("chain_map") or {}) if record else {}


def assemble_domain_chains(orch):
    """Assemble each failed chain's accepted domains into chain CIFs.

    A chain reaches here only after failing chain fitting, so we commit to
    the domain path.  A chain whose domains all fell below domain-min-cc
    simply does not enter the complex.

    每条域链同时产出两个 CIF：
      - 全量（所有已接受域）          -> 记录 fitted_cif          -> assembled_complex_all.cif
      - 过滤版（仅 cc>=complex_min_cc）-> 记录 fitted_cif_filtered -> assembled_complex.cif
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
        filtered_cif = _filtered_domain_cif(orch, cid, fitted, ranges,
                                            assembled_cif,
                                            is_complex=is_complex)
        _save_domain_chain(orch, cid, assembled_cif, domain_cc, fitted,
                           filtered_cif)


def merge_domains(orch, chain_id, fitted_domains, domain_ranges,
                  is_complex=False, out_cif=None):
    """Merge domain CIF files into one chain CIF by residue order.

    For complexes with complex_domain_opt, creates multi-chain CIF
    using source_chain_id.  out_cif 可指定输出路径（缺省落到 work_dir）。
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
            chain_map = chain_map_of(orch, chain_id)
            chains_map = {}
            serial = 1
            for seg in segments:
                cif = seg["drec"].get("fitted_cif")
                if not cif or not os.path.exists(cif):
                    continue
                src_cid = seg["drec"].get("source_chain_id", chain_id)
                if src_cid not in chains_map:
                    # 源链号在 CIF 输入时是占位空间（A/B），写出前映射回真链号（Q/R）；
                    # chain_map 的键只含占位链号，真链号原样通过，不会二次映射。
                    chains_map[src_cid] = Chain.Chain(chain_map.get(src_cid, src_cid))
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
    """过滤版域链 CIF：仅保留 cc_mask >= complex_min_cc 的域。

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
    """Single accepted domain is the chain's contribution."""
    d_file = domain_rec.get("fitted_cif") or domain_rec.get("fitted_pdb")
    if not d_file or not os.path.exists(d_file):
        log.info("Chain %s: single domain file missing; chain dropped",
                 chain_id)
        return
    d_cc = calculate_cc_mask(orch.original_density_mrc, d_file,
                             orch.resolution, orch.contour)
    # 单域：该域 cc 达标则过滤版=全量，否则过滤版为空（链不进过滤复合物）
    filtered = (d_file if domain_rec.get("cc_mask", d_cc) >= orch.complex_min_cc
                else None)
    _save_domain_chain(orch, chain_id, d_file, d_cc, [domain_rec], filtered)


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
