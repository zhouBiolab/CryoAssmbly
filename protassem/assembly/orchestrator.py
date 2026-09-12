"""Assembly orchestrator -- unified chain+domain assembly.

Coordinates the full assembly pipeline: prepare chains, split domains,
unified fitting queue (chains and domains interleaved by gyration radius),
domain chain assembly, complex building, and refinement.
"""

import os
import re
import shutil
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
from Bio.PDB import MMCIFParser, MMCIFIO, Structure, Model, Chain

from protassem.core.io import load_sample_points, find_files
from protassem.core.points_txt import read_point_cloud
from protassem.core.structure import (
    cif_to_pdb, pdb_to_cif, calculate_gyration_radius,
    extract_chain_id, align_by_resid,
)
from protassem.core.scoring import calculate_cc_mask
from protassem.core.similarity import calculate_tm_score
from protassem.fitting.masker import mask_fitted_region
from protassem.assembly.domain_splitter import (
    split_domains, find_domain_files, parse_domain_ranges,
)
from protassem.assembly.complex_builder import build_complex, create_report
from protassem.assembly import refine_step

log = logging.getLogger(__name__)


class AssemblyOrchestrator:
    """Unified assembly: chains and domains interleaved by gyration radius."""

    def __init__(self, target_txt, source_dir, density_mrc,
                 resolution, contour, output_dir=None,
                 chain_threshold=0.40, initial_domain_threshold=0.40,
                 min_domain_threshold=0.35, similarity_threshold=0.85,
                 complex_threshold=0.35, complex_domain_opt=False,
                 improve_accepted=True, domain_opt=True,
                 save_all_attempts=False, cleanup=False, num_processes=1,
                 batch_size=20, complex_min_cc=0.15, do_refine=True,
                 refine_tm=0.75, homo_chain_refine=False,
                 no_domain_split_chains=frozenset(),
                 pre_screen=True, pre_screen_by_domain=False):
        self.original_target_txt = target_txt
        self.source_dir = Path(source_dir)
        self.original_density_mrc = density_mrc
        self.resolution = resolution
        self.contour = contour

        self.chain_threshold = chain_threshold
        self.initial_domain_threshold = initial_domain_threshold
        self.min_domain_threshold = min_domain_threshold
        self.similarity_threshold = similarity_threshold
        self.complex_threshold = complex_threshold
        self.complex_domain_opt = complex_domain_opt

        self.improve_accepted = improve_accepted
        self.domain_opt = domain_opt
        self.save_all_attempts = save_all_attempts
        self.do_cleanup = cleanup
        self.num_processes = num_processes
        self.batch_size = batch_size
        self.complex_min_cc = complex_min_cc
        self.do_refine = do_refine
        self.refine_tm = refine_tm
        self.homo_chain_refine = homo_chain_refine
        self.no_domain_split_chains = frozenset(no_domain_split_chains)
        self.pre_screen = pre_screen
        self.pre_screen_by_domain = pre_screen_by_domain

        from protassem.assembly.assembly_opt import AssemblyOptConfig
        _ocfg = AssemblyOptConfig()
        self.chain_similar_relax = _ocfg.chain_similar_relax
        self.domain_similar_relax = _ocfg.domain_similar_relax
        self.clash_overlap_thr = _ocfg.clash_overlap_thr

        self.output_dir = Path(output_dir or (self.source_dir / "assembly_output"))
        self.work_dir = self.output_dir / "work"
        self.final_dir = self.output_dir / "final_results"

        self.current_target_txt = None
        self.current_density_mrc = None
        self.chain_records = []
        self.domain_records = {}
        self.domain_adjacency = {}
        self.accepted_chains = []
        self.needs_domain_assembly = []
        self.failed_chain_pdbs = []
        self.failed_domain_pdbs = []
        self.accepted_domain_pdbs = []        # 已接受域的 CIF 路径（计数/切片/refine 用）
        self.accepted_domain_src_pdbs = []    # 对应模板 PDB 路径（相似判定，与 TM 缓存同键）
        self.accepted_domain_groups = set()   # 已接受域的 group_id 集合（同源查表 O(1)）
        self.accepted_fitted_pdbs = []        # 已接受结构(链+域)的拟合 pose PDB（clash 检测用）
        self.excluded_domains = []
        self._chain_order = 0
        self._domain_chain_order = 0
        self._chain_iter = 0
        self._domain_iter = 0
        self._mask_iter = 0

    # ==================================================================
    # Public entry point
    # ==================================================================

    def run(self):
        """Execute the full assembly pipeline."""
        self._setup()
        self._prepare_chains()
        self._run_domain_splitting()
        self._reorder_chains()
        self._precompute_similarity()

        if self.pre_screen:
            if self.pre_screen_by_domain:
                self._pre_screen_domains()
            else:
                self._pre_screen_chains()

        from protassem.assembly.unified_queue import run_unified_assembly
        run_unified_assembly(self)

        from protassem.assembly.domain_assembler import assemble_domain_chains
        assemble_domain_chains(self)

        build_complex(
            self.accepted_chains, str(self.final_dir),
            cif_key="fitted_cif", out_name="assembled_complex_all.cif")
        complex_cif, remap_log = build_complex(
            self.accepted_chains, str(self.final_dir),
            cif_key="fitted_cif_filtered", out_name="assembled_complex.cif")
        if remap_log:
            for orig, new, comp in remap_log:
                log.info("Chain ID remapped: %s -> %s (component %s)",
                         orig, new, comp)
        self._attach_domain_details()
        create_report(self.accepted_chains, str(self.final_dir),
                      excluded_domains=self.excluded_domains, config={
            "chain_threshold": self.chain_threshold,
            "complex_threshold": self.complex_threshold,
            "domain_initial_threshold": self.initial_domain_threshold,
            "domain_min_cc": self.min_domain_threshold,
            "complex_min_cc": self.complex_min_cc,
            "similarity_threshold": self.similarity_threshold,
            "resolution": self.resolution,
            "contour": self.contour,
        })

        refined_cif = refine_step.maybe_refine(self)

        from protassem.assembly import homo_chain_step
        # Step5 在 Step4（已按 complex_min_cc 过滤域集）之上做同源链精修
        homo_cif = homo_chain_step.maybe_homo_refine(
            self, refined_cif or complex_cif)

        self._cleanup_temp_files()
        log.info("Assembly complete. Accepted: %d components",
                 len(self.accepted_chains))
        final_cif = homo_cif or refined_cif or complex_cif
        if final_cif:
            log.info("Complex: %s", final_cif)
        return final_cif

    # ==================================================================
    # Setup
    # ==================================================================

    def _setup(self):
        for d in [self.work_dir, self.final_dir,
                  self.final_dir / "chains", self.final_dir / "domain_chains"]:
            os.makedirs(d, exist_ok=True)
        self.current_target_txt = str(self.work_dir / "current_target.txt")
        self.current_density_mrc = str(self.work_dir / "current_density.mrc")
        shutil.copy2(self.original_target_txt, self.current_target_txt)
        shutil.copy2(self.original_density_mrc, self.current_density_mrc)

    # ==================================================================
    # Chain preparation
    # ==================================================================

    def _prepare_chains(self):
        """Find structure files, read chain IDs from content, sort by size."""
        from protassem.core.structure import read_chain_ids
        cif_dir = self.work_dir / "cif_conversions"
        os.makedirs(cif_dir, exist_ok=True)

        structure_files = (list(self.source_dir.glob("*.pdb"))
                           + list(self.source_dir.glob("*.cif")))
        for sf in structure_files:
            ext = sf.suffix.lower()
            is_complex = sf.name.startswith("complex_")
            try:
                chain_ids = read_chain_ids(str(sf))
            except Exception:
                continue
            if not chain_ids:
                continue

            if is_complex:
                chain_id = "+".join(chain_ids)
            else:
                chain_id = chain_ids[0]

            chain_map = None
            if ext == ".cif":
                pdb_file = cif_dir / (sf.stem + ".pdb")
                try:
                    from protassem.core.structure import cif_to_pdb_placeholders
                    chain_map = cif_to_pdb_placeholders(str(sf), str(pdb_file))
                except Exception:
                    continue
            else:
                pdb_file = sf

            txt_file = self._find_txt_for_chain(sf)
            if txt_file is None:
                continue

            pts, _ = load_sample_points(str(txt_file))
            if len(pts) == 0:
                continue

            self.chain_records.append({
                "chain_id": chain_id,
                "chain_map": chain_map,
                "pdb_file": str(pdb_file),
                "txt_file": str(txt_file),
                "point_count": len(pts),
                "gyration_radius": calculate_gyration_radius(pts),
                "status": "pending",
                "cc_mask": 0.0,
                "fitted_pdb": None,
                "fitted_cif": None,
                "domain_count": 0,
                "is_from_cif": ext == ".cif",
                "is_complex": is_complex,
            })

        self.chain_records.sort(key=lambda c: c["gyration_radius"],
                                reverse=True)
        log.info("Found %d chains (sorted by gyration radius)",
                 len(self.chain_records))

    def _find_txt_for_chain(self, pdb_file):
        base = Path(pdb_file).stem
        for txt in self.source_dir.glob("%s*.txt" % base):
            if "domain" not in txt.name.lower():
                return txt
        return None

    # ==================================================================
    # Domain splitting
    # ==================================================================

    def _run_domain_splitting(self):
        """Split each chain into domains using DomainParser."""
        domain_base = self.work_dir / "domains"
        for rec in self.chain_records:
            cid = rec["chain_id"]
            if cid in self.no_domain_split_chains:
                log.info("Chain %s: domain splitting skipped (user override)",
                         cid)
                rec["domain_count"] = 0
                self.domain_records[cid] = []
                continue
            if rec.get("is_complex") and not self.complex_domain_opt:
                log.info("Complex %s: domain splitting skipped", cid)
                rec["domain_count"] = 0
                self.domain_records[cid] = []
                continue
            if rec.get("is_complex") and self.complex_domain_opt:
                self._split_complex_domains(rec, domain_base)
                continue

            ddir = domain_base / cid
            os.makedirs(ddir, exist_ok=True)

            chain_txt = ddir / os.path.basename(rec["txt_file"])
            chain_pdb = ddir / os.path.basename(rec["pdb_file"])
            shutil.copy2(rec["txt_file"], chain_txt)
            shutil.copy2(rec["pdb_file"], chain_pdb)

            split_domains(str(chain_txt), str(chain_pdb), str(ddir),
                          self.resolution)

            pairs = find_domain_files(str(ddir))
            domains = []
            for txt, pdb, dnum in pairs:
                pts, _ = load_sample_points(txt)
                domains.append({
                    "chain_id": cid, "domain_num": dnum,
                    "pdb_file": pdb, "txt_file": txt,
                    "point_count": len(pts),
                    "gyration_radius": calculate_gyration_radius(pts),
                    "status": "available", "cc_mask": 0.0,
                    "fitted_pdb": None, "fitted_cif": None,
                })
            self.domain_records[cid] = domains
            rec["domain_count"] = len(domains)

            ranges, adj = parse_domain_ranges(str(ddir))
            self.domain_adjacency[cid] = (ranges, adj)
            log.info("Chain %s: %d domains", cid, len(domains))

    def _split_complex_domains(self, rec, domain_base):
        """Split a complex into chains, then each chain into domains."""
        from protassem.core.structure import split_structure_to_chains
        from protassem.voxelize.mol_to_mrc import pdb2vol
        from protassem.sampling.sampler import sample_density_map

        cid = rec["chain_id"]
        ddir = domain_base / cid
        os.makedirs(ddir, exist_ok=True)

        chain_pairs = split_structure_to_chains(rec["pdb_file"], str(ddir))
        log.info("Complex %s: split into %d chains for domain opt",
                 cid, len(chain_pairs))

        all_domains = []
        all_ranges = {}
        domain_offset = 0

        for src_cid, chain_pdb_orig in chain_pairs:
            subdir = ddir / src_cid
            os.makedirs(subdir, exist_ok=True)

            simple_pdb = str(subdir / ("chain_%s_1.pdb" % src_cid))
            shutil.copy2(chain_pdb_orig, simple_pdb)

            chain_mrc = str(subdir / ("chain_%s_1.mrc" % src_cid))
            pdb2vol(simple_pdb, self.resolution, output_mrc=chain_mrc)
            _, _, chain_txt = sample_density_map(
                chain_mrc, voxel_size=2.0, output_dir=str(subdir))

            split_domains(chain_txt, simple_pdb, str(subdir), self.resolution)
            pairs = find_domain_files(str(subdir))

            for txt, pdb, dnum in pairs:
                adj_dnum = dnum + domain_offset
                pts, _ = load_sample_points(txt)
                all_domains.append({
                    "chain_id": cid, "domain_num": adj_dnum,
                    "source_chain_id": src_cid,
                    "pdb_file": pdb, "txt_file": txt,
                    "point_count": len(pts),
                    "gyration_radius": calculate_gyration_radius(pts),
                    "status": "available", "cc_mask": 0.0,
                    "fitted_pdb": None, "fitted_cif": None,
                })

            ranges, _ = parse_domain_ranges(str(subdir))
            for dnum, segs in ranges.items():
                all_ranges[dnum + domain_offset] = segs

            if pairs:
                domain_offset += max(dnum for _, _, dnum in pairs) + 1

        self.domain_records[cid] = all_domains
        rec["domain_count"] = len(all_domains)
        self.domain_adjacency[cid] = (all_ranges, {})
        log.info("Complex %s: %d total domains across %d chains",
                 cid, len(all_domains), len(chain_pairs))

    def _reorder_chains(self):
        """Sort all chains/complexes by gyration radius (descending)."""
        self.chain_records.sort(key=lambda c: c["gyration_radius"],
                                reverse=True)

    # ==================================================================
    # Pre-screening
    # ==================================================================

    def _pre_screen_chains(self):
        """Pre-screen: parallel cc_mask, accept chains already well-placed."""
        if not self.chain_records:
            return

        from multiprocessing import Pool
        from protassem.assembly.assembly_opt import ca_overlap, ca_count

        args_list = [
            (self.original_density_mrc, rec["pdb_file"],
             self.resolution, self.contour)
            for rec in self.chain_records
        ]
        n_workers = min(self.num_processes, len(args_list))
        log.info("=" * 60)
        log.info("Pre-screening %d chains (parallel cc_mask, %d workers)",
                 len(args_list), n_workers)
        log.info("=" * 60)

        if n_workers > 1:
            with Pool(n_workers) as pool:
                cc_values = pool.map(_pre_screen_cc_worker, args_list)
        else:
            cc_values = [_pre_screen_cc_worker(a) for a in args_list]

        for rec, cc in zip(self.chain_records, cc_values):
            rec["pre_screen_cc"] = cc
            log.info("  %s: cc_mask=%.4f", rec["chain_id"], cc)

        candidates = []
        for rec in self.chain_records:
            thr = (self.complex_threshold if rec.get("is_complex")
                   else self.chain_threshold)
            if rec["pre_screen_cc"] >= thr:
                candidates.append(rec)
                log.info("  %s PASS (%.4f >= %.3f)",
                         rec["chain_id"], rec["pre_screen_cc"], thr)

        if not candidates:
            log.info("Pre-screening: no chains above threshold")
            return

        # Clash resolution: keep the one with more CA atoms, loser stays pending
        accepted = []
        for cand in candidates:
            conflict = None
            for acc in accepted:
                if ca_overlap(cand["pdb_file"], acc["pdb_file"]) > self.clash_overlap_thr:
                    conflict = acc
                    break
            if conflict is None:
                accepted.append(cand)
            elif ca_count(cand["pdb_file"]) > ca_count(conflict["pdb_file"]):
                accepted.remove(conflict)
                log.info("  Clash: %s (CA=%d) replaces %s (CA=%d)",
                         cand["chain_id"], ca_count(cand["pdb_file"]),
                         conflict["chain_id"], ca_count(conflict["pdb_file"]))
                accepted.append(cand)
            else:
                log.info("  Clash: %s (CA=%d) loses to %s (CA=%d), will fit normally",
                         cand["chain_id"], ca_count(cand["pdb_file"]),
                         conflict["chain_id"], ca_count(conflict["pdb_file"]))

        # Sequential accept + mask (large first, already sorted)
        for rec in accepted:
            cc = rec["pre_screen_cc"]
            cid = rec["chain_id"]

            if self.improve_accepted and len(self.domain_records.get(cid, [])) > 1:
                from protassem.assembly.chain_fitter import try_improve_chain_with_domains
                improved_pdb, improved_cc = try_improve_chain_with_domains(
                    self, rec, rec["pdb_file"], cc)
                if improved_cc > cc:
                    log.info("  %s improved by domains: %.4f -> %.4f",
                             cid, cc, improved_cc)
                    cc = improved_cc
                    rec["fitted_pdb"] = improved_pdb

            rec["cc_mask"] = cc
            if not rec.get("fitted_pdb"):
                rec["fitted_pdb"] = rec["pdb_file"]
            self._accept_chain(rec, rec["fitted_pdb"], cc)
            self._mask_region(rec["fitted_pdb"])

            for d in self.domain_records.get(cid, []):
                d["status"] = "rejected"

            log.info("Pre-screen ACCEPTED: %s (cc=%.4f)", cid, cc)

        remaining = sum(1 for r in self.chain_records if r["status"] == "pending")
        log.info("Pre-screening complete: %d accepted, %d remaining for fitting",
                 len(accepted), remaining)

    def _pre_screen_domains(self):
        """Accept well-placed domains, then route their parent chains to domains.

        The original experimental density is used for this placement check, as in
        whole-chain pre-screening.  Accepted domains are masked sequentially.
        A chain without any accepted domain remains pending and therefore follows
        the normal whole-chain fitting path.
        """
        from multiprocessing import Pool

        domains = [
            drec for rec in self.chain_records
            for drec in self.domain_records.get(rec["chain_id"], [])
            if drec.get("status") == "available"
        ]
        if not domains:
            log.info("Domain pre-screening: no available domains")
            return

        args_list = [
            (self.original_density_mrc, drec["pdb_file"],
             self.resolution, self.contour)
            for drec in domains
        ]
        n_workers = min(self.num_processes, len(args_list))
        log.info("=" * 60)
        log.info("Domain pre-screening %d domains (parallel cc_mask, %d workers; "
                 "threshold %.3f)", len(args_list), n_workers,
                 self.initial_domain_threshold)
        log.info("=" * 60)

        if n_workers > 1:
            with Pool(n_workers) as pool:
                cc_values = pool.map(_pre_screen_cc_worker, args_list)
        else:
            cc_values = [_pre_screen_cc_worker(a) for a in args_list]

        candidates = []
        for drec, cc in zip(domains, cc_values):
            drec["pre_screen_cc"] = cc
            log.info("  %s domain %d: cc_mask=%.4f", drec["chain_id"],
                     drec["domain_num"], cc)
            if cc >= self.initial_domain_threshold:
                candidates.append(drec)

        # Large-first ordering makes the result deterministic.  A clashing
        # domain stays available for the usual PARENet/local-optimisation route.
        candidates.sort(key=lambda d: (-d.get("gyration_radius", 0.0),
                                       d["chain_id"], d["domain_num"]))
        routed_chain_ids = set()
        accepted_count = 0
        for drec in candidates:
            cid = drec["chain_id"]
            dnum = drec["domain_num"]
            pdb_file = drec["pdb_file"]
            if self._clashes_with_accepted(pdb_file):
                log.info("  Clash: %s domain %d will fit normally", cid, dnum)
                continue

            cc = drec["pre_screen_cc"]
            self._accept_domain(drec, pdb_file, cc)
            self._mask_region(pdb_file)
            routed_chain_ids.add(cid)
            accepted_count += 1
            log.info("Domain pre-screen ACCEPTED: %s domain %d (cc=%.4f)",
                     cid, dnum, cc)

        for rec in self.chain_records:
            cid = rec["chain_id"]
            if cid in routed_chain_ids and rec["status"] == "pending":
                rec["status"] = "routed_to_domain"
                if cid not in self.needs_domain_assembly:
                    self.needs_domain_assembly.append(cid)

        remaining = sum(1 for r in self.chain_records if r["status"] == "pending")
        log.info("Domain pre-screening complete: %d domains accepted; %d chains "
                 "routed to domain fitting; %d chains remain for whole-chain fitting",
                 accepted_count, len(routed_chain_ids), remaining)

    # ==================================================================
    # Accept / save helpers
    # ==================================================================

    def _accept_chain(self, rec, final_pdb, cc):
        self._chain_order += 1
        rec["status"] = "accepted_as_chain"
        rec["fitting_order"] = self._chain_order

        is_complex = rec.get("is_complex", False)
        cid = rec["chain_id"]
        prefix = "complex" if is_complex else "chain"
        cif_path = str(self.final_dir / "chains"
                       / ("%s_%s_%02d.cif" % (prefix, cid, self._chain_order)))
        chain_map = rec.get("chain_map")
        if is_complex and chain_map:
            from protassem.core.structure import write_structure_with_chain_map
            write_structure_with_chain_map(final_pdb, chain_map, cif_path)
        else:
            pdb_to_cif(final_pdb, cif_path,
                       chain_id=None if is_complex else cid)
        rec["fitted_cif"] = cif_path

        self.accepted_chains.append({
            "type": "complex" if is_complex else "chain",
            "chain_id": cid,
            "fitting_order": self._chain_order,
            "cc_mask": cc, "fitted_cif": cif_path,
        })
        self.accepted_fitted_pdbs.append(final_pdb)

    def _accept_domain(self, drec, final_pdb, cc):
        cid = drec["chain_id"]
        dnum = drec["domain_num"]
        temp_dir = self.work_dir / "fitted_domains" / cid
        os.makedirs(temp_dir, exist_ok=True)

        cif_path = str(temp_dir / ("domain_%d.cif" % dnum))
        pdb_to_cif(final_pdb, cif_path, chain_id=cid)
        drec["fitted_cif"] = cif_path
        drec["fitted_pdb"] = final_pdb
        drec["cc_mask"] = cc
        drec["status"] = "accepted"
        self.accepted_domain_pdbs.append(cif_path)
        self.accepted_domain_src_pdbs.append(drec["pdb_file"])
        self.accepted_fitted_pdbs.append(final_pdb)
        gid = drec.get("group_id")
        if gid is not None:
            self.accepted_domain_groups.add(gid)

    def _record_excluded_domain(self, chain_id, dkey, cc):
        log.info("Domain %s excluded from assembly (cc=%.4f < %.3f)",
                 dkey, cc, self.min_domain_threshold)
        self.excluded_domains.append(
            {"chain_id": chain_id, "domain": dkey, "cc_mask": cc})

    def _attach_domain_details(self):
        for rec in self.accepted_chains:
            if rec.get("domain_details"):
                continue
            cid = rec["chain_id"]
            chain_cif = rec.get("fitted_cif")
            domains = self.domain_records.get(cid, [])
            if (not chain_cif or not os.path.exists(chain_cif)
                    or len(domains) <= 1):
                continue
            tmp = self.work_dir / "report_domains" / cid
            os.makedirs(tmp, exist_ok=True)
            details = []
            for drec in domains:
                dnum = drec["domain_num"]
                tpdb = str(tmp / ("d%d.pdb" % dnum))
                if align_by_resid(chain_cif, drec["pdb_file"], tpdb):
                    cc = calculate_cc_mask(self.original_density_mrc, tpdb,
                                           self.resolution, self.contour)
                    details.append({"num": dnum, "cc_mask": cc})
            rec["domain_details"] = details

    def _save_attempt(self, label, rec, cc, pdb_file):
        if not self.save_all_attempts:
            return
        attempts_dir = self.output_dir / "all_attempts"
        os.makedirs(attempts_dir, exist_ok=True)
        dest = attempts_dir / ("%s.pdb" % label)
        if pdb_file and os.path.exists(pdb_file):
            shutil.copy2(pdb_file, dest)
        info = attempts_dir / ("%s.txt" % label)
        with open(info, "w") as f:
            f.write("cc_mask: %.6f\n" % cc)
            for k, v in rec.items():
                if k not in ("fitted_pdb", "fitted_cif"):
                    f.write("%s: %s\n" % (k, v))

    # ==================================================================
    # Masking
    # ==================================================================

    def _mask_region(self, fitted_pdb):
        self._mask_iter += 1
        mask_dir = self.work_dir / ("mask_%d" % self._mask_iter)
        os.makedirs(mask_dir, exist_ok=True)
        new_txt = str(mask_dir / "filtered.txt")
        new_mrc = str(mask_dir / "masked.mrc")
        ok = mask_fitted_region(self.current_target_txt, fitted_pdb,
                                self.current_density_mrc, new_txt, new_mrc)
        if ok and os.path.exists(new_txt) and os.path.exists(new_mrc):
            shutil.copy2(new_txt, self.current_target_txt)
            shutil.copy2(new_mrc, self.current_density_mrc)
            try:
                with open(self.current_target_txt) as _f:
                    _n = max(0, (sum(1 for _ in _f) - 5)) // 2
                log.info("  masked %s -> remaining target points ~%d",
                         os.path.basename(str(fitted_pdb)), _n)
            except Exception:
                pass

    # ==================================================================
    # Similarity helpers
    # ==================================================================

    def _precompute_similarity(self):
        """准备阶段相似度预计算（并行）：
        1) 并行预填 链×链 TM 缓存；贪心得到链同源分组 -> rec["group_id"]。
        2) 同源链的同序号结构域即同源：域 group_id = "链组.域序号"（省掉大量 域×域 USalign）。
        3) 并行预填 同源链之间同序号域 的 域×域 TM（让 _is_similar_to_any 命中缓存）。
        之后 2-tries / 轮间优先 用 group_id 查表，轮内不再现算 USalign。
        """
        from collections import defaultdict
        from protassem.core.similarity import prefill_tm_cache, calculate_tm_score
        chains = list(self.chain_records)
        cpairs = [(chains[i]["pdb_file"], chains[j]["pdb_file"])
                  for i in range(len(chains)) for j in range(i + 1, len(chains))]
        n1 = prefill_tm_cache(cpairs, self.num_processes)
        reps = []
        for c in chains:
            gid = None
            for idx, rp in enumerate(reps):
                if calculate_tm_score(c["pdb_file"], rp) >= self.similarity_threshold:
                    gid = idx
                    break
            if gid is None:
                gid = len(reps)
                reps.append(c["pdb_file"])
            c["group_id"] = gid
        self.domain_group_of = {}
        bygrp = defaultdict(list)
        for c in chains:
            cg = c.get("group_id", c["chain_id"])
            for d in self.domain_records.get(c["chain_id"], []):
                dg = "%s.%d" % (cg, d["domain_num"])
                d["group_id"] = dg
                self.domain_group_of[d["pdb_file"]] = dg
                bygrp[dg].append(d["pdb_file"])
        dpairs = []
        for g, pl in bygrp.items():
            for i in range(len(pl)):
                for j in range(i + 1, len(pl)):
                    dpairs.append((pl[i], pl[j]))
        n2 = prefill_tm_cache(dpairs, self.num_processes)
        log.info("[相似预计算] 链 %d 条 -> %d 同源组; 结构域 %d 同源组; "
                 "并行预填 TM 对 链%d+域%d (进程 %d)",
                 len(chains), len(reps), len(bygrp), n1, n2, self.num_processes)

    def _find_similar_group(self, pdb, groups):
        for gid, members in groups.items():
            for m in members:
                if calculate_tm_score(pdb, m["pdb_file"]) >= self.similarity_threshold:
                    return gid
        return None

    def _is_similar_to_failed(self, pdb, failed_list):
        return any(calculate_tm_score(pdb, f) >= self.similarity_threshold
                   for f in failed_list)

    def _is_similar_to_any(self, pdb, pdb_list):
        return any(calculate_tm_score(pdb, p) >= self.similarity_threshold
                   for p in pdb_list)

    def _is_homolog_accepted_domain(self, drec):
        """该域是否与某个已接受域同源。

        优先用预计算 group_id 查表(O(1))；缺 group_id 时回退到模板 PDB 的 TM 比对
        (命中 _precompute_similarity 预填的缓存，键为 PDB 路径)。
        """
        gid = drec.get("group_id")
        if gid is not None:
            return gid in self.accepted_domain_groups
        return self._is_similar_to_any(drec["pdb_file"], self.accepted_domain_src_pdbs)

    def _clashes_with_accepted(self, pdb):
        """fitted pose 与任一已接受结构(链+域)的 CA 重叠是否 > clash_overlap_thr。
        注意：须用 PDB（check_clash 按固定列解析 ATOM 的 CA/P，CIF 不适用）。"""
        from protassem.assembly.assembly_opt import ca_overlap
        return any(ca_overlap(pdb, q) > self.clash_overlap_thr
                   for q in self.accepted_fitted_pdbs)

    # ==================================================================
    # Utilities
    # ==================================================================

    def _ensure_work_files(self):
        if not os.path.exists(self.current_target_txt):
            log.warning("current_target.txt missing, re-copying from original")
            shutil.copy2(self.original_target_txt, self.current_target_txt)
        if not os.path.exists(self.current_density_mrc):
            log.warning("current_density.mrc missing, re-copying from original")
            shutil.copy2(self.original_density_mrc, self.current_density_mrc)

    def _target_has_points(self):
        """目标点云是否仍有剩余点（沿用历史阈值：至少 2 个点）。

        读取失败时按 True 处理（保守认为目标仍可用），由后续步骤决定。
        """
        try:
            return len(read_point_cloud(self.current_target_txt).points) >= 2
        except Exception:
            return True

    def _cleanup_temp_files(self):
        if not self.do_cleanup:
            return
        import glob as g
        for pattern in ["temp_chain", "temp_domain", "chain_improve_*",
                        "chain_domain_transform_*", "chain_fit", "domain_fit"]:
            for d in g.glob(str(self.work_dir / pattern)):
                if os.path.isdir(d):
                    shutil.rmtree(d, ignore_errors=True)
        log.info("Temp files cleaned up")


# ==================================================================
# Module-level helpers
# ==================================================================

def _pre_screen_cc_worker(args):
    """Worker for parallel cc_mask in pre-screening."""
    density_mrc, structure_file, resolution, contour = args
    try:
        return calculate_cc_mask(density_mrc, structure_file, resolution, contour)
    except Exception:
        return 0.0


def run_assembly(target_txt, source_dir, density_mrc, resolution, contour,
                 output_dir=None, **kwargs):
    """Convenience function wrapping AssemblyOrchestrator."""
    orch = AssemblyOrchestrator(
        target_txt=target_txt, source_dir=source_dir,
        density_mrc=density_mrc, resolution=resolution,
        contour=contour, output_dir=output_dir, **kwargs)
    return orch.run()
