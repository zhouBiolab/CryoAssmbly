"""Final complex building and summary report."""

import os
import string
import logging
from datetime import datetime
from Bio.PDB import MMCIFParser, MMCIFIO, Structure, Model

log = logging.getLogger(__name__)

_CHAIN_ID_POOL = (list(string.ascii_uppercase)
                  + list(string.ascii_lowercase)
                  + list(string.digits))


def _next_available_id(used_ids):
    for cid in _CHAIN_ID_POOL:
        if cid not in used_ids:
            return cid
    for c1 in _CHAIN_ID_POOL:
        for c2 in _CHAIN_ID_POOL:
            cid = c1 + c2
            if cid not in used_ids:
                return cid
    raise RuntimeError("Exhausted chain ID namespace")


def build_complex(accepted_chains, output_dir, cif_key="fitted_cif",
                  out_name="assembled_complex.cif"):
    """Merge accepted chain/domain CIF files into one complex CIF.

    cif_key 决定每条记录取哪个 CIF（仅对 domain_chain 生效）：
      - "fitted_cif"           域链含其全部已接受域           -> assembled_complex_all.cif
      - "fitted_cif_filtered"  域链仅含 cc>=complex_min_cc 的域 -> assembled_complex.cif
    整链 / 复合物记录不做域级过滤，两个文件都用其 fitted_cif；
    域链过滤版为 None（所有域均低于阈值）时，该链不进入复合物。

    Args:
        accepted_chains: list of dicts (type, chain_id, fitted_cif,
                         fitted_cif_filtered, fitting_order ...)
        output_dir: directory for output
        cif_key: "fitted_cif" 或 "fitted_cif_filtered"

    Returns:
        (path to complex CIF, remap_log) or (None, [])
    """
    os.makedirs(output_dir, exist_ok=True)
    remap_log = []
    try:
        complex_struct = Structure.Structure("complex")
        model = Model.Model(0)
        complex_struct.add(model)
        added = 0
        used_ids = set()

        for rec in accepted_chains:
            cif = (rec.get(cif_key) if rec.get("type") == "domain_chain"
                   else rec.get("fitted_cif"))
            if not cif or not os.path.exists(cif):
                continue
            parser = MMCIFParser(QUIET=True)
            s = parser.get_structure("t", cif)
            for chain in s[0]:
                new_chain = chain.copy()
                original_id = new_chain.id
                if original_id in used_ids:
                    new_id = _next_available_id(used_ids)
                    log.info("Chain ID collision: %s -> %s (from %s)",
                             original_id, new_id, rec.get("chain_id"))
                    new_chain.id = new_id
                    remap_log.append((original_id, new_id, rec.get("chain_id")))
                used_ids.add(new_chain.id)
                model.add(new_chain)
                added += 1

        out = os.path.join(output_dir, out_name)
        io = MMCIFIO()
        io.set_structure(complex_struct)
        io.save(out)
        log.info("Complex CIF created: %s (%d chains)", out, added)
        return out, remap_log
    except Exception as e:
        log.error("build_complex error: %s", e)
        return None, remap_log


def create_report(accepted_chains, output_dir, config=None, excluded_domains=None):
    """Write a summary report."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "assembly_summary.txt")
    try:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "w", encoding="utf-8") as f:
            f.write("Assembly Summary Report\n")
            f.write("=" * 70 + "\n\n")
            f.write("Date: " + now_str + "\n")
            if config:
                for k, v in config.items():
                    f.write(f"{k}: {v}\n")
            f.write("\n")

            chain_count = sum(1 for a in accepted_chains if a["type"] == "chain")
            domain_count = sum(1 for a in accepted_chains if a["type"] == "domain_chain")
            complex_count = sum(1 for a in accepted_chains if a["type"] == "complex")
            f.write(f"Total accepted: {len(accepted_chains)}\n")
            f.write(f"  As chain: {chain_count}\n")
            f.write(f"  As complex: {complex_count}\n")
            f.write(f"  As domain chain: {domain_count}\n\n")

            sorted_acc = sorted(accepted_chains,
                                key=lambda x: x.get("fitting_order", 999))
            for i, a in enumerate(sorted_acc, 1):
                order = a.get("fitting_order", 0)
                cid = a["chain_id"]
                atype = a["type"]
                cc = a["cc_mask"]
                f.write(f"{i}. [{order:02d}] {cid} ({atype})\n")
                f.write(f"   CC_mask: {cc:.6f}\n")
                details = a.get("domain_details") or []
                if details:
                    f.write("   Per-domain CC_mask:\n")
                    for d in details:
                        dcc = d.get("cc_mask")
                        dcc_s = f"{dcc:.6f}" if dcc is not None else "N/A"
                        f.write(f"     - domain {d['num']}: {dcc_s}\n")
                elif a["type"] == "domain_chain":
                    f.write(f"   Domains: {a.get('domains', [])}\n")
                f.write("\n")
            if excluded_domains:
                f.write("Excluded domains (below domain-min-cc, not assembled):\n")
                for e in excluded_domains:
                    f.write(f"  - {e['domain']}: cc_mask={e['cc_mask']:.6f}\n")
                f.write("\n")
        log.info("Report created: %s", path)
    except Exception as e:
        log.error("create_report error: %s", e)
