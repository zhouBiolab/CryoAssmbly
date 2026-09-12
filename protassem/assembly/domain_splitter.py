"""Domain splitting — imports domain_pdb_txt directly (no subprocess).

Uses DomainParser binary to split chains into structural domains,
then filters point clouds by domain-specific atomic masks.
"""

import os
import re
import io
import glob
import logging
import contextlib
from pathlib import Path
from collections import defaultdict

from protassem.assembly.domain_parser.domain_pdb_txt import (
    split_txt_by_domains_unified_grid,
)

log = logging.getLogger(__name__)

# DomainParser binary location (project-internal)
_PARSER_DIR = os.path.join(os.path.dirname(__file__), "domain_parser")
_DOMAIN_PARSER_PATH = os.path.join(_PARSER_DIR, "DomainParser.py")


def split_domains(txt_file, pdb_file, work_dir, resolution):
    """Split a chain into domains using DomainParser.

    Calls split_txt_by_domains_unified_grid() directly (no subprocess).

    Args:
        txt_file: chain point cloud .txt
        pdb_file: chain structure .pdb
        work_dir: output directory for domain files
        resolution: map resolution

    Returns:
        True on success
    """
    os.makedirs(work_dir, exist_ok=True)
    # DomainParser writes DomainParser.log via a relative path (cwd). The original
    # code ran it with cwd=work_dir; replicate that so the log + domain outputs
    # land in work_dir (otherwise each chain overwrites a log at the project root).
    abs_txt = os.path.abspath(str(txt_file))
    abs_pdb = os.path.abspath(str(pdb_file))
    prev_cwd = os.getcwd()
    try:
        os.chdir(work_dir)
        # suppress the splitter's verbose print() output (terminal noise);
        # the concise per-chain summary is logged by the orchestrator
        with contextlib.redirect_stdout(io.StringIO()):
            split_txt_by_domains_unified_grid(
                input_txt=abs_txt,
                input_pdb=abs_pdb,
                domain_parser_path=_DOMAIN_PARSER_PATH,
                resolution=resolution,
            )
        return True
    except Exception as e:
        log.error("Domain splitting error for %s: %s", pdb_file, e)
        return False
    finally:
        os.chdir(prev_cwd)


def find_domain_files(search_dir):
    """Find domain txt+pdb pairs in a directory.

    Returns list of (txt_file, pdb_file, domain_num).
    """
    txt_files = glob.glob(os.path.join(search_dir, "*_domain*.txt"))
    pdb_files = glob.glob(os.path.join(search_dir, "chain*.pdb"))
    pairs = []
    for txt in txt_files:
        m = re.search(r"_domain(\d+)\.txt$", os.path.basename(txt))
        if not m:
            continue
        dnum = int(m.group(1))
        base = os.path.basename(txt).replace(f"_domain{dnum}.txt", "")
        prefix_m = re.match(r"(chain_[^_]+_\d+)", base)
        if not prefix_m:
            continue
        prefix = prefix_m.group(1)
        for pdb in pdb_files:
            pdb_stem = os.path.splitext(os.path.basename(pdb))[0]
            pat = rf"^{re.escape(prefix)}_d_{dnum}$"
            if re.match(pat, pdb_stem):
                pairs.append((txt, pdb, dnum))
                break
    pairs.sort(key=lambda x: x[2])
    return pairs


def parse_domain_ranges(work_dir):
    """Parse DomainParser.log to get domain residue ranges.

    Returns (domain_ranges, adjacency) where
      domain_ranges = {domain_num: [(start, end), ...]}
      adjacency = {domain_num: set of adjacent domain nums}
    """
    log_file = Path(work_dir) / "DomainParser.log"
    if not log_file.exists():
        return {}, {}
    try:
        with open(log_file) as f:
            line = f.readline().strip()
        parts = line.split()
        if len(parts) < 3:
            return {}, {}
        total_res = int(parts[1])
        n_domains = int(parts[2])
        if n_domains == 1 and len(parts) == 3:
            return {1: [(1, total_res)]}, {}
        if len(parts) < 4:
            return {}, {}
        domain_ranges = {}
        dnum = 1
        for part in parts[3:]:
            if part.startswith("(") and part.endswith(")"):
                segs = []
                for seg in part[1:-1].split(";"):
                    if "-" in seg:
                        s, e = seg.split("-")
                        segs.append((int(s), int(e)))
                if segs:
                    domain_ranges[dnum] = segs
                    dnum += 1
        adjacency = defaultdict(set)
        nums = sorted(domain_ranges.keys())
        for i, d1 in enumerate(nums):
            for d2 in nums[i + 1:]:
                last = domain_ranges[d1][-1]
                first = domain_ranges[d2][0]
                if abs(first[0] - last[1]) <= 2:
                    adjacency[d1].add(d2)
                    adjacency[d2].add(d1)
        return domain_ranges, dict(adjacency)
    except Exception as e:
        log.error("parse_domain_ranges error: %s", e)
        return {}, {}