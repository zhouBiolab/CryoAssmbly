# -*- coding: utf-8 -*-
# init_modeling.py - Backbone modeling: map traces to domains and connect adjacent domains
# Refactored: 3-part pipeline
#   Part 1: find_corresponding_CA  — build trace↔domain atom correspondences
#   Part 2: build_chain_models     — Union-Find based chain assembly, cc_mask priority
#   Part 3: repair_chains_with_fragments — fragment-based gap repair per chain
import time
import copy
import numpy as np
import os
from classes.globalcommunication import SharedData
from scipy.spatial import cKDTree
    # ── Import TM-align / domain-fit primitives from new_alignment ───────────
from .new_alignment import get_tm_alignment as _hs_tmalign
from .new_alignment import _do_single_domain_fit as _hs_domainfit

shared_data = SharedData()

# ── Amino acid mappings (for PDB export) ─────────────────────────────────────

AA_abb_T = {0:"A",1:"C",2:"D",3:"E",4:"F",5:"G",6:"H",7:"I",8:"K",9:"L",
            10:"M",11:"N",12:"P",13:"Q",14:"R",15:"S",16:"T",17:"V",18:"W",19:"Y"}
abb2AA = {"A":"ALA","C":'CYS',"D":'ASP',"E":'GLU',"F":'PHE',"G":'GLY',"H":"HIS","I":"ILE",
          "K":"LYS","L":"LEU","M":"MET","N":"ASN","P":"PRO","Q":"GLN","R":"ARG","S":"SER",
          "T":"THR","V":"VAL","W":"TRP","Y":"TYR"}

# SS type names for diagnostics
_SS_NAMES = {0: 'UNK', 1: 'H(helix)', 2: 'E(sheet)', 3: 'C(coil)'}
COIL_SS = 3
DIST_THRESHOLD = 2.0  # Å for CA correspondence


def record_time(func_name, start_time):
    return time.time() - start_time


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _get_domain_residue_indices(fasta_name, domain_id):
    """Return sorted list of sequence positions belonging to *domain_id*."""
    global shared_data
    if fasta_name not in shared_data.fastas:
        return []
    seq_obj = shared_data.fastas[fasta_name]
    if domain_id is None:
        return list(range(len(seq_obj.sequence)))
    if not hasattr(seq_obj, 'get_domain_label_for_position'):
        return []
    try:
        return sorted([
            i for i in range(len(seq_obj.sequence))
            if seq_obj.get_domain_label_for_position(i) == domain_id
        ])
    except Exception:
        return []


def _estimate_residue_indices_from_coords(entry, fasta_name):
    global shared_data
    n_coords = len(entry.get('fitted_coords', []))
    if n_coords == 0:
        return []
    domain_id = entry.get('domain_id', None)
    indices = _get_domain_residue_indices(fasta_name, domain_id)
    if len(indices) == n_coords:
        return indices
    if len(indices) > n_coords:
        return indices[:n_coords]
    if len(indices) > 0 and len(indices) < n_coords:
        last = indices[-1]
        while len(indices) < n_coords:
            last += 1
            indices.append(last)
        return indices
    return list(range(n_coords))


def _get_af2_ss_at_seq_pos(fasta_name, seq_pos):
    """Return AF2 secondary structure type at absolute *seq_pos*."""
    global shared_data
    if fasta_name not in shared_data.fastas:
        return None
    seq_obj = shared_data.fastas[fasta_name]
    if not hasattr(seq_obj, 'AF2_SS'):
        return None
    af2_ss = seq_obj.AF2_SS
    if seq_pos < 0 or seq_pos >= len(af2_ss):
        return None
    return int(af2_ss[seq_pos])


def _num_chains_for_fasta(fname):
    global shared_data
    if fname in shared_data.fastas:
        nc = getattr(shared_data.fastas[fname], 'num_chains', 1)
        return max(int(nc), 1)
    return 1


# ── Union-Find ───────────────────────────────────────────────────────────────

class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def make_set(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def connected(self, x, y):
        return self.find(x) == self.find(y)

    def groups(self):
        from collections import defaultdict
        g = defaultdict(list)
        for x in self.parent:
            g[self.find(x)].append(x)
        return dict(g)


# ═══════════════════════════════════════════════════════════════════════════════
# Part 1.  find_corresponding_CA
# ═══════════════════════════════════════════════════════════════════════════════

def find_corresponding_CA():
    """
    Build bidirectional mapping between trace CA atoms and domain fitted atoms.

    Results stored on shared_data:
        trace_to_domain_map : dict  (chain_idx, atom_pos) -> (domain_idx, domain_atom_pos)
        domain_to_trace_map : dict  (domain_idx, domain_atom_pos) -> [(chain_idx, atom_pos), ...]
    """
    global shared_data
    start_time = time.time()
    print(f"  [1/3] Finding corresponding CA atoms between traces and domains...")

    trace_to_domain_map = {}
    domain_to_trace_map = {}

    # Build KD-tree over all domain atoms
    all_domain_coords = []
    all_domain_indices = []

    for domain_idx, domain_entry in enumerate(shared_data.filtered_domain_list):
        dom_coords = domain_entry.get('fitted_coords', None)
        if dom_coords is None or len(dom_coords) == 0:
            continue
        dom_coords = np.asarray(dom_coords, dtype=np.float64)
        for local_pos in range(len(dom_coords)):
            all_domain_coords.append(dom_coords[local_pos])
            all_domain_indices.append((domain_idx, local_pos))

    if len(all_domain_coords) == 0:
        print("      WARNING: no domain atoms found.")
        shared_data.trace_to_domain_map = trace_to_domain_map
        shared_data.domain_to_trace_map = domain_to_trace_map
        return

    all_domain_coords = np.array(all_domain_coords, dtype=np.float64)
    domain_tree = cKDTree(all_domain_coords)
    print(f"      Domain KD-tree: {len(all_domain_coords)} atoms")

    num_mappings = 0
    num_trace_atoms = 0

    for chain_idx in range(shared_data.final_traces['num_chains']):
        chain_info = shared_data.final_traces['chains'][chain_idx]
        ca_indices = chain_info['ca_indices']

        for atom_pos in range(len(ca_indices)):
            ca_idx = ca_indices[atom_pos]
            trace_coord = shared_data.ca_pos[ca_idx]
            num_trace_atoms += 1

            dist, nearest_flat_idx = domain_tree.query(trace_coord, k=1)

            if dist < DIST_THRESHOLD:
                domain_idx, domain_atom_pos = all_domain_indices[nearest_flat_idx]

                trace_key = (chain_idx, atom_pos)
                domain_key = (domain_idx, domain_atom_pos)

                if trace_key not in trace_to_domain_map:
                    trace_to_domain_map[trace_key] = domain_key

                if domain_key not in domain_to_trace_map:
                    domain_to_trace_map[domain_key] = []
                domain_to_trace_map[domain_key].append(trace_key)

                num_mappings += 1

    shared_data.trace_to_domain_map = trace_to_domain_map
    shared_data.domain_to_trace_map = domain_to_trace_map

    elapsed = record_time('find_corresponding_CA', start_time)
    print(f"      Completed in {elapsed:.2f}s")
    print(f"      Trace atoms scanned           : {num_trace_atoms}")
    print(f"      Total correspondences (< 2A)  : {num_mappings}")
    print(f"      Trace atoms with mapping       : {len(trace_to_domain_map)}")
    print(f"      Domain atoms with mapping      : {len(domain_to_trace_map)}")

    for dom_idx, entry in enumerate(shared_data.filtered_domain_list):
        n_total = len(entry.get('fitted_coords', []))
        n_mapped = sum(1 for k in domain_to_trace_map if k[0] == dom_idx)
        print(f"        Domain[{dom_idx}] {entry.get('fasta_name','?')}"
              f"/domain{entry.get('domain_id','?')}: "
              f"{n_mapped}/{n_total} atoms mapped")


# ═══════════════════════════════════════════════════════════════════════════════
# Part 2.  build_chain_models
#   - Enrich domain entries
#   - Discover sequence-neighbor relationships
#   - Sort domains by cc_mask descending
#   - Greedily connect domains using boundary distance, Union-Find
#   - Assemble chains
# ═══════════════════════════════════════════════════════════════════════════════

def _enrich_domain_entries():
    """Add residue_indices and other bookkeeping to each filtered domain entry."""
    global shared_data
    enriched = []
    for dom_idx, entry in enumerate(shared_data.filtered_domain_list):
        fasta_name = entry['fasta_name']
        domain_id = entry.get('domain_id', None)
        residue_indices = _estimate_residue_indices_from_coords(entry, fasta_name)
        fitted_coords = np.asarray(entry['fitted_coords'], dtype=np.float64)
        # 从 filtered_domain_list 传入 R/T（new_alignment 可能未对所有域完整计算）
        raw_R = entry.get('rotation', None)
        raw_T = entry.get('translation', None)
        rotation    = np.asarray(raw_R, dtype=np.float64) if raw_R is not None else np.eye(3)
        translation = np.asarray(raw_T, dtype=np.float64) if raw_T is not None else np.zeros(3)
        enriched.append({
            'fasta_name':        fasta_name,
            'domain_id':         domain_id,
            'fitted_coords':     fitted_coords,
            'residue_indices':   residue_indices,
            'cc_mask':           entry.get('cc_mask', 0.0),
            'source_domain_idx': dom_idx,
            'rotation':          rotation,
            'translation':       translation,
        })
    return enriched


def _recompute_RT_for_domain(fasta_name, domain_id, fitted_coords, residue_indices):
    """
    通过 superpose3d 将 AF2 域坐标对齐到 fitted_coords，重新求解变换矩阵 R 和平移向量 T。

    变换约定与 new_alignment.py 一致：
        fitted ≈ AF2_domain @ R.T + T

    参数
    ----
    fasta_name      : str   — 蛋白质名称（用于在 shared_data.fastas 中查找 AF2_struct）
    domain_id       : int|None — 域标签（None 表示整条链）
    fitted_coords   : ndarray (N, 3) — EM 拟合后的 CA 坐标
    residue_indices : list[int]      — fitted_coords 各行对应的全局序列位置

    返回
    ----
    R : ndarray (3, 3)  — 旋转矩阵
    T : ndarray (3,)    — 平移向量
    若发生任何异常则返回单位矩阵和零向量。
    """
    import superpose3d
    global shared_data
    try:
        if fasta_name not in shared_data.fastas:
            return np.eye(3), np.zeros(3)
        seq_obj  = shared_data.fastas[fasta_name]
        full_af2 = np.asarray(seq_obj.AF2_struct, dtype=np.float64)

        # 取该域在全序列中的位置索引
        if domain_id is None:
            af2_abs_indices = list(range(len(seq_obj.sequence)))
        else:
            af2_abs_indices = [
                i for i in range(len(seq_obj.sequence))
                if seq_obj.get_domain_label_for_position(i) == domain_id
            ]
        if len(af2_abs_indices) < 3:
            return np.eye(3), np.zeros(3)

        af2_dom_coords = full_af2[af2_abs_indices]           # (M_af2, 3)
        af2_pos_to_lp  = {ai: j for j, ai in enumerate(af2_abs_indices)}

        # 仅保留两侧都有坐标的残基做配对
        source_pts, target_pts = [], []
        for lp, sp in enumerate(residue_indices):
            if lp >= len(fitted_coords):
                break
            if sp in af2_pos_to_lp:
                source_pts.append(af2_dom_coords[af2_pos_to_lp[sp]])
                target_pts.append(fitted_coords[lp])

        if len(source_pts) < 3:
            return np.eye(3), np.zeros(3)

        source_pts = np.asarray(source_pts, dtype=np.float64)
        target_pts = np.asarray(target_pts, dtype=np.float64)

        # superpose3d.Superpose3D(X_target, X_source) -> rmsd, R, T, _
        # 满足: X_target ≈ X_source @ R.T + T
        _, R_mat, T_vec, _ = superpose3d.Superpose3D(target_pts, source_pts)
        return np.asarray(R_mat, dtype=np.float64), np.asarray(T_vec, dtype=np.float64)

    except Exception as _exc:
        print(f"      [_recompute_RT_for_domain] 警告: {fasta_name}/域{domain_id} "
              f"R/T 重算失败 ({_exc})，回退为单位变换")
        return np.eye(3), np.zeros(3)


def _find_domain_sequence_neighbors(by_domain, enriched_entries):
    """
    For each pair of domains that are sequence-adjacent (residue gap == 1),
    collect all boundary specs.

    Returns
    -------
    neighbor_specs : dict
        key   = (did_a, did_b)
        value = list of (pos_left, pos_right, left_did, right_did)
              where pos_right == pos_left + 1
    Also returns:
    all_neighbors : dict
        key = did -> set of neighbor dids
    """
    domain_pos_set = {}
    for did, eidx_list in by_domain.items():
        ps = set()
        for eidx in eidx_list:
            ps.update(enriched_entries[eidx]['residue_indices'])
        domain_pos_set[did] = ps

    pos_to_did = {}
    for did, ps in domain_pos_set.items():
        for p in ps:
            if p not in pos_to_did:
                pos_to_did[p] = did

    raw = {}
    for did, ps in domain_pos_set.items():
        for p in ps:
            nxt = p + 1
            if nxt in pos_to_did and pos_to_did[nxt] != did:
                nd = pos_to_did[nxt]
                key = frozenset({did, nd})
                raw.setdefault(key, []).append((p, nxt, did, nd))

    neighbor_specs = {}
    all_neighbors = {}
    for key, bspecs in raw.items():
        da, db = tuple(key)
        neighbor_specs[(da, db)] = bspecs
        neighbor_specs[(db, da)] = bspecs
        all_neighbors.setdefault(da, set()).add(db)
        all_neighbors.setdefault(db, set()).add(da)

    return neighbor_specs, all_neighbors


def _compute_boundary_distance(eidx_a, eidx_b, boundary_specs, enriched_entries):
    """
    Domain-pair boundary distance: sum of Euclidean distances at each
    boundary point (pos_left in one domain, pos_right in the other).

    Example: domain2=[100..199]∪[400..599], domain4=[200..399]
      boundary points: (199,200) and (399,400)
      distance = dist(eidx_a@199, eidx_b@200) + dist(eidx_b@399, eidx_a@400)
    """
    ea, eb = enriched_entries[eidx_a], enriched_entries[eidx_b]
    did_a, did_b = ea['domain_id'], eb['domain_id']

    def _coord_map(e):
        return {int(sp): e['fitted_coords'][lp]
                for lp, sp in enumerate(e['residue_indices'])}

    cm_a = _coord_map(ea)
    cm_b = _coord_map(eb)

    total, n_valid = 0.0, 0
    for pos_left, pos_right, left_did, right_did in boundary_specs:
        if left_did == did_a and right_did == did_b:
            c_l, c_r = cm_a.get(pos_left), cm_b.get(pos_right)
        elif left_did == did_b and right_did == did_a:
            c_l, c_r = cm_b.get(pos_left), cm_a.get(pos_right)
        else:
            continue
        if c_l is not None and c_r is not None:
            total += float(np.linalg.norm(
                np.asarray(c_l, dtype=np.float64) - np.asarray(c_r, dtype=np.float64)))
            n_valid += 1

    return total if n_valid > 0 else float('inf')


def _check_boundary_already_bridged(eidx_a, eidx_b, boundary_specs, enriched_entries):
    """
    Check which boundary points between two domain instances are already
    connected by a trace fragment (i.e. both sides have domain_to_trace_map
    correspondences on the same trace chain).

    Returns
    -------
    bridged : set of (pos_left, pos_right) that are already bridged by fragment
    unbridged : set of (pos_left, pos_right) that are NOT bridged
    """
    global shared_data
    ea, eb = enriched_entries[eidx_a], enriched_entries[eidx_b]
    did_a = ea['domain_id']

    def _lp_map(e):
        return {int(sp): lp for lp, sp in enumerate(e['residue_indices'])}

    lpm_a = _lp_map(ea)
    lpm_b = _lp_map(eb)

    bridged = set()
    unbridged = set()

    for pos_left, pos_right, left_did, right_did in boundary_specs:
        if left_did == did_a:
            left_entry, right_entry = ea, eb
            left_lpm, right_lpm = lpm_a, lpm_b
        else:
            left_entry, right_entry = eb, ea
            left_lpm, right_lpm = lpm_b, lpm_a

        left_lp = left_lpm.get(pos_left)
        right_lp = right_lpm.get(pos_right)
        if left_lp is None or right_lp is None:
            unbridged.add((pos_left, pos_right))
            continue

        left_key = (left_entry['source_domain_idx'], left_lp)
        right_key = (right_entry['source_domain_idx'], right_lp)

        left_traces = shared_data.domain_to_trace_map.get(left_key, [])
        right_traces = shared_data.domain_to_trace_map.get(right_key, [])

        # Check if any trace chain spans both
        left_chains = {ci for ci, _ in left_traces}
        right_chains = {ci for ci, _ in right_traces}

        if left_chains & right_chains:
            bridged.add((pos_left, pos_right))
        else:
            unbridged.add((pos_left, pos_right))

    return bridged, unbridged


def _find_fragment_based_pairing(curr_eidx, nd_eidx, boundary_specs, enriched_entries):
    """
    Enhanced fragment bridge detection: walk into each domain from the boundary
    to find the nearest trace-mapped atoms and check if they come from the same
    trace fragment chain.

    Unlike _check_boundary_already_bridged which only checks exact boundary
    positions, this walks inward — mirroring the logic in repair_chains_with_fragments.

    Returns True if at least one boundary is confirmed bridged by fragment walk.
    """
    global shared_data
    ea = enriched_entries[curr_eidx]
    eb = enriched_entries[nd_eidx]
    did_a = ea['domain_id']

    for pos_left, pos_right, left_did, right_did in boundary_specs:
        if left_did == did_a:
            e_left, e_right = ea, eb
        else:
            e_left, e_right = eb, ea

        # Walk left from boundary into the left domain
        left_ri = e_left['residue_indices']
        left_trace = None
        if left_ri:
            for sp in range(pos_left, min(left_ri) - 1, -1):
                if sp not in left_ri:
                    continue
                lp = left_ri.index(sp)
                key = (e_left['source_domain_idx'], lp)
                if key in shared_data.domain_to_trace_map:
                    left_trace = shared_data.domain_to_trace_map[key][0]
                    break

        # Walk right from boundary into the right domain
        right_ri = e_right['residue_indices']
        right_trace = None
        if right_ri:
            for sp in range(pos_right, max(right_ri) + 1):
                if sp not in right_ri:
                    continue
                lp = right_ri.index(sp)
                key = (e_right['source_domain_idx'], lp)
                if key in shared_data.domain_to_trace_map:
                    right_trace = shared_data.domain_to_trace_map[key][0]
                    break

        if left_trace is not None and right_trace is not None:
            if left_trace[0] == right_trace[0]:  # same fragment chain
                return True

    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Pre-Pass-1: Cross-fasta homolog swap
#
# 同源互换 (放在 Pass 1 之前):
#   1. 跨 fasta 用 TM-align 对所有域实体两两比对, TM-score > 0.85 即并入同一同源池。
#      所谓同源 = 结构相似, 与 fasta 编号无关, 因此 7XML_3 的域2/域3、7MXL_1 的域2 与
#      7XML_3 的域2 均可同属一个池。
#   2. 对每个 A (按 cc_mask 降序处理):
#        - 按 3D 距离 (A 末端 ↔ C 末端) 从近到远遍历 A 在同 fasta 序列上的邻居 C
#        - 在同源池中找 B = 其末端离 C 末端最近且未超硬阈值的实体
#        - 限制: (B.fasta, B.dom) ≠ (A.fasta, A.dom) (B == A 是允许的, 表示 A 已最优)
#        - 若 B == A: 视为 A 已最优, 移出池
#        - 否则: 互换 A↔B 的实体, A 移出池 (B 仍可继续参与后续轮)
#        - 一旦做出决定 (swap 或 A 已最优), 立即处理下一个 A, 不再枚举更多 C
#   3. 互换方法 (与原 Pre-Pass-2 一致, 但移到 Pass1 之前):
#        snapshot A、B 当前坐标
#        place(A → B.snapshot): TM-align 得 pos1, 再 domainfit 得 pos2, 取 cc_mask 更高者
#        place(B → A.snapshot): 同上
# ═══════════════════════════════════════════════════════════════════════════════
def _homolog_swap_cross_fasta(enriched_entries):
    """
    Cross-fasta homolog swap, run BEFORE per-fasta Pass-1.

    Modifies *enriched_entries* in place (and propagates fitted_coords / cc_mask
    / rotation / translation back to shared_data.filtered_domain_list).
    """
    global shared_data
    start_time = time.time()
    print(f"  [Pre-Pass-1] Cross-fasta homolog swap")

    n_all = len(enriched_entries)
    if n_all < 2:
        print(f"        Only {n_all} entries – nothing to swap.")
        return

    # ── Helper: get domain AA sequence ───────────────────────────────────────
    def _hs_get_domain_seq(fn, dom_id, res_indices):
        sobj = shared_data.fastas.get(fn)
        if sobj is None:
            return 'A' * len(res_indices)
        full = sobj.sequence.upper()
        return ''.join(
            full[i] if 0 <= i < len(full) else 'A'
            for i in res_indices
        )

    # ── HS-1.  Build homolog pools across ALL entries (cross-fasta) ──────────
    # Union-Find: any pair with TM-score > 0.85 in either direction → same pool.
    _hs_TM_THR = 0.85
    uf_pool = UnionFind()
    for e in range(n_all):
        uf_pool.make_set(e)

    print(f"        Building homolog pools over {n_all} entries (TM > {_hs_TM_THR}) …")
    n_pairs_tested = 0
    n_pairs_linked = 0
    for i in range(n_all):
        ea = enriched_entries[i]
        ca = ea['fitted_coords']
        if len(ca) < 3:
            continue
        sa = _hs_get_domain_seq(ea['fasta_name'], ea['domain_id'],
                                 ea['residue_indices'])
        for j in range(i + 1, n_all):
            if uf_pool.connected(i, j):
                continue
            eb = enriched_entries[j]
            cb = eb['fitted_coords']
            if len(cb) < 3:
                continue
            sb = _hs_get_domain_seq(eb['fasta_name'], eb['domain_id'],
                                     eb['residue_indices'])
            try:
                tm_res = _hs_tmalign(ca, cb, sa, sb)
                tm_sc = max(tm_res['tm_score1'], tm_res['tm_score2'])
                n_pairs_tested += 1
                if tm_sc >= _hs_TM_THR:
                    uf_pool.union(i, j)
                    n_pairs_linked += 1
            except Exception:
                pass

    # Materialise pools
    _hs_groups = {}
    for e in range(n_all):
        r = uf_pool.find(e)
        _hs_groups.setdefault(r, set()).add(e)

    _hs_active_pools = {r: set(m) for r, m in _hs_groups.items() if len(m) >= 2}
    _hs_eidx2pool = {}
    for r, ms in _hs_active_pools.items():
        for e in ms:
            _hs_eidx2pool[e] = r

    n_pool_members = sum(len(m) for m in _hs_active_pools.values())
    print(f"        TM-align pairs tested : {n_pairs_tested}")
    print(f"        Homolog links (≥0.85) : {n_pairs_linked}")
    print(f"        Active pools           : {len(_hs_active_pools)} "
          f"({n_pool_members} members)")

    if not _hs_active_pools:
        print("        No homolog pool with ≥2 members – swap skipped.")
        return

    # Diagnostic: dump pool composition
    for r, ms in _hs_active_pools.items():
        members = [
            f"eidx{e}({enriched_entries[e]['fasta_name']}/dom{enriched_entries[e]['domain_id']})"
            for e in sorted(ms)
        ]
        print(f"          Pool {r}: " + ", ".join(members))

    # ── HS-2.  Per-fasta neighbor relationships ──────────────────────────────
    # Sequence neighbours only exist within the same fasta. We pre-compute
    # them per fasta so the inner loop can look them up cheaply.
    _hs_fasta_groups = {}
    for eidx, e in enumerate(enriched_entries):
        _hs_fasta_groups.setdefault(e['fasta_name'], []).append(eidx)

    _hs_per_fasta_neighbor_specs = {}
    _hs_per_fasta_all_neighbors  = {}
    _hs_per_fasta_by_domain      = {}
    for fn, ei_list in _hs_fasta_groups.items():
        bd = {}
        for ei in ei_list:
            did = enriched_entries[ei]['domain_id']
            bd.setdefault(did, []).append(ei)
        _hs_per_fasta_by_domain[fn] = bd
        nspecs, allnb = _find_domain_sequence_neighbors(bd, enriched_entries)
        _hs_per_fasta_neighbor_specs[fn] = nspecs
        _hs_per_fasta_all_neighbors[fn]  = allnb

    # ── HS-3.  Swap helper: place entity at snapshot_target position ─────────
    # 1) TM-align src → snapshot_target  → pos1, cc1
    # 2) On top of pos1, run _do_single_domain_fit → pos2, cc2
    # Return whichever has higher cc_mask.
    def _hs_place_entity(eidx_src, snapshot_target):
        e_src = enriched_entries[eidx_src]
        coords_src = np.asarray(e_src['fitted_coords'], dtype=np.float64)
        snap_tgt   = np.asarray(snapshot_target, dtype=np.float64)
        if len(coords_src) < 3 or len(snap_tgt) < 3:
            return coords_src, e_src['cc_mask']

        seq_src = _hs_get_domain_seq(
            e_src['fasta_name'], e_src['domain_id'],
            e_src['residue_indices'])
        seq_tgt = 'A' * len(snap_tgt)

        # Position 1: TM-aligned
        try:
            tm_res = _hs_tmalign(coords_src, snap_tgt, seq_src, seq_tgt)
            coords_tm = (np.dot(coords_src, tm_res['rotation'].T)
                         + tm_res['translation'])
        except Exception:
            # Fallback: refuse to move
            return coords_src, e_src['cc_mask']
        cc_tm = _get_cc_mask_for_coords(coords_tm)

        # Position 2: density-fitted on top of TM-aligned pose
        cc_fit = -1.0
        coords_fit = coords_tm
        try:
            em  = getattr(shared_data.config, 'em_path', None)
            res = float(getattr(shared_data.config, 'resolution', 4.0))
            con = getattr(shared_data.config, 'contour', 0.01)
            if em is not None:
                fit = _hs_domainfit(
                    coords_src,
                    tm_res['rotation'], tm_res['translation'],
                    em, res, con)
                coords_fit = np.asarray(fit['fitted_coords'], dtype=np.float64)
                cc_fit = _get_cc_mask_for_coords(coords_fit)
        except Exception:
            pass

        if cc_fit > cc_tm:
            return coords_fit, cc_fit
        return coords_tm, cc_tm

    # ── HS-4.  Main swap loop ────────────────────────────────────────────────
    # Process A's in cc_mask descending order (high-confidence first).
    _hs_DIST_THR = 60.0   # Å, hard threshold for B↔C distance
    _hs_sorted_eidxes = sorted(range(n_all),
                                key=lambda e: enriched_entries[e]['cc_mask'],
                                reverse=True)

    n_swaps   = 0
    n_optimal = 0
    n_noop    = 0
    for _hs_A in _hs_sorted_eidxes:
        if _hs_A not in _hs_eidx2pool:
            continue   # not in any pool (singleton) or already removed
        _hs_pool_root = _hs_eidx2pool[_hs_A]
        if _hs_pool_root not in _hs_active_pools:
            continue

        _hs_eA  = enriched_entries[_hs_A]
        _hs_riA = _hs_eA['residue_indices']
        if not _hs_riA:
            _hs_active_pools.get(_hs_pool_root, set()).discard(_hs_A)
            _hs_eidx2pool.pop(_hs_A, None)
            continue

        _hs_cmA = {int(sp): _hs_eA['fitted_coords'][lp]
                   for lp, sp in enumerate(_hs_riA)}
        _hs_minA = min(_hs_riA)
        _hs_maxA = max(_hs_riA)

        _hs_fasta_A = _hs_eA['fasta_name']
        _hs_did_A   = _hs_eA['domain_id']

        # Per-fasta lookups (A's fasta)
        _hs_nspecs = _hs_per_fasta_neighbor_specs.get(_hs_fasta_A, {})
        _hs_allnb  = _hs_per_fasta_all_neighbors.get(_hs_fasta_A, {})
        _hs_bd     = _hs_per_fasta_by_domain.get(_hs_fasta_A, {})

        # ── HS-4a. Enumerate all sequence-neighbour instances C of A ───────
        _hs_nbr_dist = []   # (dAC, C_eidx, nd)
        for _hs_nd in _hs_allnb.get(_hs_did_A, set()):
            _hs_bspec = (_hs_nspecs.get((_hs_did_A, _hs_nd))
                         or _hs_nspecs.get((_hs_nd, _hs_did_A)))
            for _hs_C in _hs_bd.get(_hs_nd, []):
                _hs_eC  = enriched_entries[_hs_C]
                _hs_riC = _hs_eC['residue_indices']
                if not _hs_riC:
                    continue
                _hs_cmC = {int(sp): _hs_eC['fitted_coords'][lp]
                           for lp, sp in enumerate(_hs_riC)}
                if _hs_bspec:
                    _hs_dAC = _compute_boundary_distance(
                        _hs_A, _hs_C, _hs_bspec, enriched_entries)
                else:
                    _hs_minC = min(_hs_riC)
                    _hs_maxC = max(_hs_riC)
                    _ac1 = _hs_cmA.get(_hs_maxA)
                    _cc1 = _hs_cmC.get(_hs_minC)
                    _ac2 = _hs_cmA.get(_hs_minA)
                    _cc2 = _hs_cmC.get(_hs_maxC)
                    _d1 = (float(np.linalg.norm(
                        np.asarray(_ac1) - np.asarray(_cc1)))
                        if _ac1 is not None and _cc1 is not None
                        else float('inf'))
                    _d2 = (float(np.linalg.norm(
                        np.asarray(_ac2) - np.asarray(_cc2)))
                        if _ac2 is not None and _cc2 is not None
                        else float('inf'))
                    _hs_dAC = min(_d1, _d2)
                _hs_nbr_dist.append((_hs_dAC, _hs_C, _hs_nd))

        # Sort C from nearest to farthest (3D)
        _hs_nbr_dist.sort(key=lambda x: x[0])

        if not _hs_nbr_dist:
            # No sequence neighbours at all – cannot decide; remove A from
            # pool to guarantee monotonic progress (no infinite loop).
            n_noop += 1
            print(f"          [HomologSwap] eidx={_hs_A} "
                  f"({_hs_eA['fasta_name']}/dom{_hs_did_A}): no sequence "
                  f"neighbour found – removed from pool")
            _hs_active_pools.get(_hs_pool_root, set()).discard(_hs_A)
            _hs_eidx2pool.pop(_hs_A, None)
            if not _hs_active_pools.get(_hs_pool_root, set()):
                _hs_active_pools.pop(_hs_pool_root, None)
            continue

        # ── HS-4b.  Iterate neighbours C ────────────────────────────────────
        _hs_processed_A = False
        for _hs_dAC, _hs_C, _hs_nd in _hs_nbr_dist:
            _hs_eC  = enriched_entries[_hs_C]
            _hs_riC = _hs_eC['residue_indices']
            if not _hs_riC:
                continue
            _hs_cmC = {int(sp): _hs_eC['fitted_coords'][lp]
                       for lp, sp in enumerate(_hs_riC)}
            _hs_minC = min(_hs_riC)
            _hs_maxC = max(_hs_riC)

            # Determine C's end-position that faces A (via boundary spec)
            _hs_bspec = (_hs_nspecs.get((_hs_did_A, _hs_nd))
                         or _hs_nspecs.get((_hs_nd, _hs_did_A)))
            _hs_c_face_sp = None
            if _hs_bspec:
                for pl, pr, ld, rd in _hs_bspec:
                    if ld == _hs_did_A and rd == _hs_nd:
                        _hs_c_face_sp = pr
                    elif ld == _hs_nd and rd == _hs_did_A:
                        _hs_c_face_sp = pl
                    if _hs_c_face_sp is not None:
                        break
            if _hs_c_face_sp is None:
                _hs_c_face_sp = (
                    _hs_minC
                    if abs(_hs_maxA - _hs_minC) < abs(_hs_minA - _hs_maxC)
                    else _hs_maxC)
            _hs_c_face_coord = _hs_cmC.get(_hs_c_face_sp)
            if _hs_c_face_coord is None:
                continue

            # ── HS-4c.  Find best B in A's pool (closest to C-face) ─────────
            _hs_pool_now = _hs_active_pools.get(_hs_pool_root, set())
            _hs_best_B   = None
            _hs_best_dBC = float('inf')

            for _hs_B in _hs_pool_now:
                _hs_eB  = enriched_entries[_hs_B]
                _hs_riB = _hs_eB['residue_indices']
                if not _hs_riB:
                    continue

                # Rule 3: (fasta_name, domain_id) must differ.
                # Note: B == A is intentionally allowed (signals "A is optimal").
                if (_hs_eB['fasta_name'] == _hs_eA['fasta_name']
                        and _hs_eB['domain_id'] == _hs_eA['domain_id']
                        and _hs_B != _hs_A):
                    continue

                _hs_cmB   = {int(sp): _hs_eB['fitted_coords'][lp]
                             for lp, sp in enumerate(_hs_riB)}
                _hs_minB  = min(_hs_riB)
                _hs_maxB  = max(_hs_riB)
                _hs_bmin  = _hs_cmB.get(_hs_minB)
                _hs_bmax  = _hs_cmB.get(_hs_maxB)
                _hs_cfc   = np.asarray(_hs_c_face_coord, dtype=np.float64)
                _d_vals = []
                if _hs_bmin is not None:
                    _d_vals.append(float(np.linalg.norm(
                        np.asarray(_hs_bmin) - _hs_cfc)))
                if _hs_bmax is not None:
                    _d_vals.append(float(np.linalg.norm(
                        np.asarray(_hs_bmax) - _hs_cfc)))
                if not _d_vals:
                    continue
                _hs_dBC = min(_d_vals)
                if _hs_dBC > _hs_DIST_THR:
                    continue   # exceeds hard threshold
                if _hs_dBC < _hs_best_dBC:
                    _hs_best_dBC = _hs_dBC
                    _hs_best_B   = _hs_B

            if _hs_best_B is None:
                continue   # this C yielded no valid B; try next C

            # ── HS-4d.  Decision: A optimal or do A↔B swap ──────────────────
            if _hs_best_B == _hs_A:
                # A is already optimal at its current location
                print(f"          [HomologSwap] eidx={_hs_A} "
                      f"({_hs_eA['fasta_name']}/dom{_hs_did_A}) already "
                      f"optimal near C=eidx{_hs_C}(dom{_hs_nd}, "
                      f"dAC={_hs_dAC:.2f}Å) – removed from pool")
                _hs_active_pools[_hs_pool_root].discard(_hs_A)
                _hs_eidx2pool.pop(_hs_A, None)
                if not _hs_active_pools[_hs_pool_root]:
                    del _hs_active_pools[_hs_pool_root]
                _hs_processed_A = True
                n_optimal += 1
                break

            # Real swap A ↔ best_B
            _hs_eB_best = enriched_entries[_hs_best_B]
            _hs_snap_A = np.asarray(
                _hs_eA['fitted_coords'], dtype=np.float64).copy()
            _hs_snap_B = np.asarray(
                _hs_eB_best['fitted_coords'], dtype=np.float64).copy()

            # Place A at B's snapshot location, and B at A's snapshot location.
            _hs_new_A_coords, _hs_new_A_cc = _hs_place_entity(
                _hs_A, _hs_snap_B)
            _hs_new_B_coords, _hs_new_B_cc = _hs_place_entity(
                _hs_best_B, _hs_snap_A)

            print(f"          [HomologSwap] Swap "
                  f"A=eidx{_hs_A}({_hs_eA['fasta_name']}/dom{_hs_did_A}) ↔ "
                  f"B=eidx{_hs_best_B}({_hs_eB_best['fasta_name']}"
                  f"/dom{_hs_eB_best['domain_id']}) "
                  f"via C=eidx{_hs_C}(dom{_hs_nd}, "
                  f"dBC={_hs_best_dBC:.2f}Å) "
                  f"cc_A:{_hs_eA['cc_mask']:.4f}→{_hs_new_A_cc:.4f} "
                  f"cc_B:{_hs_eB_best['cc_mask']:.4f}→{_hs_new_B_cc:.4f}")

            # Commit to enriched_entries
            _hs_eA['fitted_coords']      = _hs_new_A_coords
            _hs_eA['cc_mask']            = _hs_new_A_cc
            _hs_eB_best['fitted_coords'] = _hs_new_B_coords
            _hs_eB_best['cc_mask']       = _hs_new_B_cc

            # Recompute R/T for both
            _hs_R_A, _hs_T_A = _recompute_RT_for_domain(
                _hs_eA['fasta_name'], _hs_eA['domain_id'],
                _hs_new_A_coords, _hs_eA['residue_indices'])
            _hs_R_B, _hs_T_B = _recompute_RT_for_domain(
                _hs_eB_best['fasta_name'], _hs_eB_best['domain_id'],
                _hs_new_B_coords, _hs_eB_best['residue_indices'])
            _hs_eA['rotation']         = _hs_R_A
            _hs_eA['translation']      = _hs_T_A
            _hs_eB_best['rotation']    = _hs_R_B
            _hs_eB_best['translation'] = _hs_T_B

            # Propagate to filtered_domain_list
            _hs_idx_A = _hs_eA['source_domain_idx']
            _hs_idx_B = _hs_eB_best['source_domain_idx']
            for _k, _v in [('fitted_coords', _hs_new_A_coords),
                           ('cc_mask', _hs_new_A_cc),
                           ('rotation', _hs_R_A),
                           ('translation', _hs_T_A)]:
                shared_data.filtered_domain_list[_hs_idx_A][_k] = _v
            for _k, _v in [('fitted_coords', _hs_new_B_coords),
                           ('cc_mask', _hs_new_B_cc),
                           ('rotation', _hs_R_B),
                           ('translation', _hs_T_B)]:
                shared_data.filtered_domain_list[_hs_idx_B][_k] = _v

            # Remove A from pool; B may still take part in later iterations.
            _hs_active_pools[_hs_pool_root].discard(_hs_A)
            _hs_eidx2pool.pop(_hs_A, None)
            if not _hs_active_pools[_hs_pool_root]:
                del _hs_active_pools[_hs_pool_root]
            _hs_processed_A = True
            n_swaps += 1
            break   # done with this A

        if not _hs_processed_A:
            # All neighbours tried, no valid B – remove A from pool to
            # guarantee monotonic progress (avoids infinite-loop behaviour
            # if the caller wraps this in a while-pool-nonempty loop).
            n_noop += 1
            print(f"          [HomologSwap] eidx={_hs_A} "
                  f"({_hs_eA['fasta_name']}/dom{_hs_did_A}): no valid swap "
                  f"found across {len(_hs_nbr_dist)} neighbours – removed "
                  f"from pool")
            _hs_active_pools.get(_hs_pool_root, set()).discard(_hs_A)
            _hs_eidx2pool.pop(_hs_A, None)
            if not _hs_active_pools.get(_hs_pool_root, set()):
                _hs_active_pools.pop(_hs_pool_root, None)

    elapsed = record_time('_homolog_swap_cross_fasta', start_time)
    print(f"        Swaps: {n_swaps}, already-optimal: {n_optimal}, "
          f"no-op: {n_noop}")
    print(f"        Cross-fasta homolog swap done in {elapsed:.2f}s")


def build_chain_models():
    """
    Part 2: Build chain models using Union-Find based domain assembly.

    Algorithm:
    1. Enrich domain entries with residue_indices.
    2. Group by fasta_name, discover sequence neighbors.
    3. Sort domains by cc_mask descending.
    4. Process highest cc_mask domain first (seed). Then for each subsequent
       domain (in cc_mask order):
       a. Find all its sequence neighbors.
       b. If a neighbor already has a connected instance via fragment bridge,
          just verify remaining boundary endpoints are connected (direct link
          if needed), skip distance comparison.
       c. Otherwise compute boundary distance to all known instances of each
          neighbor domain, pick the closest one to connect.
       d. Union the connected domains into the same chain set.
    5. Assemble position maps and chain entries.
    """
    global shared_data
    start_time = time.time()
    print(f"  [2/3] Building chain models (cc_mask priority + Union-Find)...")

    enriched_entries = _enrich_domain_entries()

    # ── Pre-Pass-1: Cross-fasta homolog swap ─────────────────────────────────
    # Runs BEFORE Pass-1 (fragment-based connections) so that swapped positions
    # are visible to Pass-1.  Cross-fasta: any pair with TM-score > 0.85 joins
    # the same homolog pool, regardless of which fasta each entity belongs to.
    _homolog_swap_cross_fasta(enriched_entries)

    # Group by fasta
    fasta_groups = {}
    for eidx, e in enumerate(enriched_entries):
        fasta_groups.setdefault(e['fasta_name'], []).append(eidx)

    shared_data.init_connect_result = []

    for fasta_name, all_eidx in fasta_groups.items():

        # Group by domain_id
        by_domain = {}
        for eidx in all_eidx:
            did = enriched_entries[eidx]['domain_id']
            by_domain.setdefault(did, []).append(eidx)

        # Build neighbor relationships
        neighbor_specs, all_neighbors = _find_domain_sequence_neighbors(
            by_domain, enriched_entries)
        n_neighbor_pairs = len(set(frozenset(k) for k in neighbor_specs))
        print(f"\n      ── {fasta_name}: {len(all_eidx)} entries, "
              f"{len(by_domain)} domains, {n_neighbor_pairs} neighbor pairs ──")

        # Sort domains by max cc_mask descending
        def _domain_max_cc(did):
            return max(enriched_entries[e]['cc_mask'] for e in by_domain[did])

        sorted_dids = sorted(by_domain.keys(), key=_domain_max_cc, reverse=True)
        print(f"        Processing order: "
              + ", ".join(f"domain{d}(cc={_domain_max_cc(d):.3f})"
                          for d in sorted_dids))

        # ── Union-Find over eidx ──
        uf = UnionFind()
        for eidx in all_eidx:
            uf.make_set(eidx)

        # Track which eidx has been "assigned" (connected to some partner)
        # eidx -> the eidx it was connected through (for bookkeeping)
        processed_dids = set()

        # Seed: the highest cc_mask domain — each instance is an independent seed
        seed_did = sorted_dids[0]
        processed_dids.add(seed_did)
        print(f"        Seed domain{seed_did}: {len(by_domain[seed_did])} instances")

        # ── Two-pass connection strategy ──
        # Pass 1: Fragment-based connections (walk into domains to find
        #         trace-mapped atoms, connect instances sharing the same
        #         fragment chain).  This prevents distance-based greedy
        #         matching from making wrong pairings.
        # Pass 2: Distance-based connections for remaining unconnected pairs.

        # Global tracking: prevent two curr_eidx from grabbing the same
        # nd_eidx within a single neighbor domain (persists across passes).
        global_used_per_nd_did = {}  # (curr_did, nd) -> set of used nd_eidx

        # ── Pass 1: Fragment-based connections ──
        print(f"        --- Pass 1: Fragment-based connections ---")
        processed_dids_p1 = {seed_did}

        for curr_did in sorted_dids[1:]:
            curr_list = by_domain[curr_did]
            nb_processed = [nd for nd in all_neighbors.get(curr_did, set())
                            if nd in processed_dids_p1]

            if not nb_processed:
                processed_dids_p1.add(curr_did)
                continue

            for nd in nb_processed:
                global_used_per_nd_did.setdefault((curr_did, nd), set())

            for curr_eidx in sorted(curr_list,
                                     key=lambda e: enriched_entries[e]['cc_mask'],
                                     reverse=True):
                for nd in nb_processed:
                    bspecs = neighbor_specs.get((curr_did, nd))
                    if not bspecs:
                        continue

                    already_in_group = any(
                        uf.connected(curr_eidx, nd_eidx)
                        for nd_eidx in by_domain[nd]
                    )
                    if already_in_group:
                        continue

                    used_set = global_used_per_nd_did[(curr_did, nd)]

                    # Enhanced fragment bridge: walk into domains (like
                    # repair_chains_with_fragments) to find trace evidence
                    bridged_eidx = None
                    for nd_eidx in by_domain[nd]:
                        if nd_eidx in used_set:
                            continue
                        # First try the original boundary check
                        bridged, unbridged = _check_boundary_already_bridged(
                            curr_eidx, nd_eidx, bspecs, enriched_entries)
                        if bridged:
                            bridged_eidx = nd_eidx
                            break
                        # Then try the enhanced walk-based check
                        if _find_fragment_based_pairing(
                                curr_eidx, nd_eidx, bspecs, enriched_entries):
                            bridged_eidx = nd_eidx
                            break

                    if bridged_eidx is not None:
                        uf.union(curr_eidx, bridged_eidx)
                        used_set.add(bridged_eidx)
                        print(f"          [Fragment] domain{curr_did} "
                              f"eidx={curr_eidx} → eidx={bridged_eidx} "
                              f"(domain{nd})")

            processed_dids_p1.add(curr_did)


        # ── Pass 2: Hungarian algorithm with cluster-aware threshold ─────────
        #
        # Pass-1 UF groups (域连接体 / domain clusters) are treated as
        # atomic units.  The cost matrix is built at cluster granularity
        # (one row per slot, one column per unassigned cluster) and solved
        # exactly by scipy's linear_sum_assignment (Jonker-Volgenant O(N³)).
        # A physics-derived hard threshold rejects any cluster–slot pairing
        # whose boundary distance exceeds what is geometrically possible given
        # the AF2 spans of any intermediate missing domains plus minimum
        # peptide-bond lengths.
        #
        # threshold(did_a, did_b) =
        #     Σ span_AF2(D_inter)  +  3.8 × (n_inter + 1)  +  20  [Å]
        # ────────────────────────────────────────────────────────────────────
        print(f"        --- Pass 2: Hungarian cluster assignment ---")

        from collections import deque as _deque
        try:
            from scipy.optimize import linear_sum_assignment as _hungarian_solve
            _has_scipy_p2 = True
        except ImportError:
            _has_scipy_p2 = False
            print("        WARNING: scipy unavailable, using greedy fallback")

        num_slots = _num_chains_for_fasta(fasta_name)
        _seq_obj_slots = shared_data.fastas.get(fasta_name)
        _num_domains_expected = int(getattr(_seq_obj_slots, 'num_domains', len(by_domain))) \
            if _seq_obj_slots else len(by_domain)
        print(f"        Chain slots (homologous copies): {num_slots}, "
              f"expected domains per chain: {_num_domains_expected}, "
              f"domains in filtered_list: {len(by_domain)}")

        # ── 2a. Enumerate Pass-1 UF clusters (域连接体) ──────────────────────
        # Each UF root represents one atomic cluster: a set of eidx that Pass 1
        # connected via fragment bridge evidence.  Singleton clusters (one eidx)
        # are also valid — they simply have no Pass-1 evidence and will be
        # matched purely by distance in 2k below.
        p1_cluster_map = {}   # uf_root -> set of eidx
        for _eidx_p2 in all_eidx:
            _root_p2 = uf.find(_eidx_p2)
            p1_cluster_map.setdefault(_root_p2, set()).add(_eidx_p2)

        # ── 2b. Per-cluster metadata ──────────────────────────────────────────
        # cluster_meta_p2[root] = {
        #   'domain_ids': set of domain_ids covered by the cluster,
        #   'coords':     {seq_pos -> coord}  (union of all member residues),
        #   'cc_max':     float  (best cc_mask among member eidx),
        # }
        cluster_meta_p2 = {}
        for _root_p2, _eset_p2 in p1_cluster_map.items():
            _dids_p2   = set()
            _coords_p2 = {}
            _cc_p2     = 0.0
            for _ei_p2 in _eset_p2:
                _e_p2 = enriched_entries[_ei_p2]
                _dids_p2.add(_e_p2['domain_id'])
                for _lp_p2, _sp_p2 in enumerate(_e_p2['residue_indices']):
                    _coords_p2[int(_sp_p2)] = _e_p2['fitted_coords'][_lp_p2]
                _cc_p2 = max(_cc_p2, _e_p2['cc_mask'])
            cluster_meta_p2[_root_p2] = {
                'domain_ids': _dids_p2,
                'coords':     _coords_p2,
                'cc_max':     _cc_p2,
            }

        # ── 2c. Sequence-position → domain_id lookup ──────────────────────────
        # Built from filtered_domain_list first; absent domains are added below
        # after domain_pos_set_p2 is supplemented from seq_obj.
        _sp2did_p2 = {}
        for _ei_p2 in all_eidx:
            _e_p2 = enriched_entries[_ei_p2]
            for _sp_p2 in _e_p2['residue_indices']:
                _sp2did_p2[int(_sp_p2)] = _e_p2['domain_id']

        # ── 2d. Domain → union of sequence positions (all instances) ──────────
        # Primary: domains present in filtered_domain_list.
        domain_pos_set_p2 = {}
        for _did_p2, _elist_p2 in by_domain.items():
            _ps_p2 = set()
            for _ei_p2 in _elist_p2:
                _ps_p2.update(int(_s) for _s in enriched_entries[_ei_p2]['residue_indices'])
            domain_pos_set_p2[_did_p2] = _ps_p2

        # Supplement: domains that seq_obj knows about (via num_domains) but
        # are absent from filtered_domain_list.  This ensures _p2_hard_threshold_pos
        # correctly counts and spans intermediate domains even when they were
        # never placed, and _sp2did_p2 covers the full sequence.
        _seq_obj_full_p2 = shared_data.fastas.get(fasta_name)
        if _seq_obj_full_p2 is not None:
            _full_nd_p2 = int(getattr(_seq_obj_full_p2, 'num_domains', 0))
            for _did_abs in range(1, _full_nd_p2 + 1):
                if _did_abs not in domain_pos_set_p2:
                    _abs_indices = _get_domain_residue_indices(fasta_name, _did_abs)
                    if _abs_indices:
                        domain_pos_set_p2[_did_abs] = set(_abs_indices)
                        for _sp_abs in _abs_indices:
                            _sp2did_p2.setdefault(int(_sp_abs), _did_abs)
                        print(f"          [Pass2-2d] seq_obj 补充缺失域 "
                              f"domain{_did_abs} ({len(_abs_indices)} pos) "
                              f"至 domain_pos_set_p2/_sp2did_p2")

        # ── 2e. AF2 terminal-CA span per domain (legacy; kept for diagnostics) ─
        # span(D) = 3D distance between the first and last CA of D in AF2 model.
        # 多段域 (multi-segment) 时,该字典会把所有段当成一个整体取首末包络,
        # 不能代表任一具体中间段的几何障碍长度。所以阈值的真正计算改在
        # _p2_hard_threshold_pos 内基于位置区间逐段重做,本字典保留以备别处兼容。
        domain_af2_span_p2 = {}
        _af2_arr_p2 = None  # 提升到外层,供 _p2_hard_threshold_pos 闭包引用
        try:
            _seq_obj_p2 = shared_data.fastas.get(fasta_name)
            if _seq_obj_p2 is not None:
                _af2_arr_p2 = np.asarray(_seq_obj_p2.AF2_struct, dtype=np.float64)
                for _did_s, _ps_s in domain_pos_set_p2.items():
                    _pss = sorted(_ps_s)
                    if len(_pss) >= 2:
                        _p0, _p1 = _pss[0], _pss[-1]
                        if _p0 < len(_af2_arr_p2) and _p1 < len(_af2_arr_p2):
                            domain_af2_span_p2[_did_s] = float(np.linalg.norm(
                                _af2_arr_p2[_p0] - _af2_arr_p2[_p1]))
                        else:
                            domain_af2_span_p2[_did_s] = (_p1 - _p0) * 3.8
                    else:
                        domain_af2_span_p2[_did_s] = 0.0
        except Exception as _e_af2_p2:
            print(f"        [AF2 span warning: {_e_af2_p2}]")

        # ── 2f. Hard threshold function (position-based, multi-segment safe) ──
        def _p2_hard_threshold_pos(sp_a, sp_b):
            """
            两个序列位置之间允许的最大几何距离(同链约束)。

                threshold(sp_a, sp_b) =
                    Σ_seg span_AF2(seg)  +  3.8 × (n_seg + 1)  +  20 Å

            其中 seg 跑遍严格夹在 (sp_a, sp_b) 之间的每一段"连续 + 同域"
            位置块,span_AF2(seg) 是该段在 AF2 模板里首末 CA 的 3D 距离,
            n_seg 是这种段的个数,(n_seg + 1) 给出穿越该 gap 所需的肽键数。

            为何按段而不按域:多段域 (multi-segment) 时,同一个域可能
            在 (sp_a, sp_b) 之间只出现某一段,而它的总跨度并不代表那一段
            的几何障碍。按"位置连续 + 同域"切段后,每段独立估其 AF2 跨度,
            才能既正确判定单段域、又正确处理多段域。
            """
            if sp_a == sp_b:
                return 20.0  # 退化情形(实际不应发生)
            lo, hi = (sp_a, sp_b) if sp_a < sp_b else (sp_b, sp_a)
            # 严格夹在 (lo, hi) 之间且属于某个已知域的位置
            inter_ps = sorted(p for p in _sp2did_p2 if lo < p < hi)
            if not inter_ps:
                # 两位置直接相邻(中间没夹任何域 / 空位):一根肽键 + 余量
                return 3.8 + 20.0

            span_sum = 0.0
            n_seg    = 0
            seg_start = inter_ps[0]
            seg_did   = _sp2did_p2.get(seg_start)
            seg_end   = seg_start

            def _flush(s, e):
                nonlocal span_sum, n_seg
                if (_af2_arr_p2 is not None
                        and 0 <= s < len(_af2_arr_p2)
                        and 0 <= e < len(_af2_arr_p2)):
                    span_sum += float(np.linalg.norm(
                        _af2_arr_p2[s] - _af2_arr_p2[e]))
                else:
                    span_sum += (e - s) * 3.8
                n_seg += 1

            for p in inter_ps[1:]:
                d = _sp2did_p2.get(p)
                if p == seg_end + 1 and d == seg_did:
                    seg_end = p
                    continue
                _flush(seg_start, seg_end)
                seg_start, seg_end, seg_did = p, p, d
            _flush(seg_start, seg_end)

            return span_sum + 3.8 * (n_seg + 1) + 20.0

        # ── 2g. Cluster-to-slot assignment cost ───────────────────────────────
        def _cluster_slot_dist_p2(cluster_root, slot_eidx_set):
            """
            Returns the boundary distance (assignment cost) between a Pass-1
            cluster and the current contents of a slot.

            0.0   — slot is empty; any cluster may enter at zero cost.
            _INF  — domain conflict, no valid interface, or distance exceeds
                    the hard threshold for at least one interface.

            统一段扫描 (segment scan) 算法:
              把 cluster 和 slot 各自所有 seq 位置打源标签 ('C' / 'S'),
              按位置升序合并;遍历相邻对,只要源标签发生切换,就构成一个
              cluster↔slot 跨界接口,逐个用 _one_iface 检查距离并对照
              _p2_hard_threshold_pos 的位置版阈值。同源相邻 (内部 gap) 跳过
              ── 它们的几何由各自 cluster/slot 的刚体拟合姿态担保。

            该扫描天然覆盖所有几何形态:
              cluster 全在 slot 左/右            → 1 个接口
              cluster 嵌入 slot 缺口             → 2 个接口
              slot 嵌入 cluster 缺口(多段域)    → 2 个接口
              多段交错 (multi-segment 双向插入) → N 个接口

            每个接口内部:相邻域优先用 neighbor_specs +
            _compute_boundary_distance,否则退回末端 CA 欧氏距离。
            阈值采用 _p2_hard_threshold_pos(sp_l, sp_r),按位置区间
            逐段累加 AF2 跨度,对多段域正确。

            最终代价 = 所有接口距离的均值;任一接口超阈短路 _INF。
            """
            cm_p2          = cluster_meta_p2[cluster_root]
            clus_dids      = cm_p2['domain_ids']
            clus_coords    = cm_p2['coords']

            if not slot_eidx_set:
                return 0.0

            # Domain conflict check
            slot_dids = {enriched_entries[_e]['domain_id'] for _e in slot_eidx_set}
            if clus_dids & slot_dids:
                return _INF

            if not clus_coords:
                return _INF

            slot_coords = {}
            for _e2 in slot_eidx_set:
                _ee = enriched_entries[_e2]
                for _lp2, _sp2 in enumerate(_ee['residue_indices']):
                    slot_coords[int(_sp2)] = _ee['fitted_coords'][_lp2]

            if not slot_coords:
                return _INF

            def _one_iface(left_sp, right_sp,
                           left_cmap, right_cmap,
                           left_src, right_src):
                """
                Compute and threshold-check one (left_sp → right_sp) interface.
                Returns (dist: float, within_threshold: bool).
                """
                l_did = _sp2did_p2.get(left_sp)
                r_did = _sp2did_p2.get(right_sp)
                dist_val = _INF

                # Prefer neighbor-spec-based distance when domains are adjacent
                if l_did is not None and r_did is not None:
                    _bspec = neighbor_specs.get((l_did, r_did))
                    if _bspec:
                        _l_ei = next(
                            (e for e in left_src
                             if enriched_entries[e]['domain_id'] == l_did), None)
                        _r_ei = next(
                            (e for e in right_src
                             if enriched_entries[e]['domain_id'] == r_did), None)
                        if _l_ei is not None and _r_ei is not None:
                            _bd = _compute_boundary_distance(
                                _l_ei, _r_ei, _bspec, enriched_entries)
                            if _bd < _INF:
                                dist_val = _bd

                # Fall back to terminal-CA Euclidean distance
                if dist_val >= _INF:
                    _lc = left_cmap.get(left_sp)
                    _rc = right_cmap.get(right_sp)
                    if _lc is None or _rc is None:
                        return _INF, False
                    dist_val = float(np.linalg.norm(
                        np.asarray(_lc, dtype=np.float64) -
                        np.asarray(_rc, dtype=np.float64)))

                # Position-based hard threshold (multi-segment safe)
                _thr = _p2_hard_threshold_pos(left_sp, right_sp)
                if dist_val > _thr:
                    return dist_val, False
                return dist_val, True

            # ── 统一段扫描:枚举所有 cluster↔slot 跨界接口 ───────────────
            combined = sorted(
                [(sp, 'C') for sp in clus_coords] +
                [(sp, 'S') for sp in slot_coords],
                key=lambda x: x[0]
            )

            total_d = 0.0
            n_iface = 0
            for _i_sc in range(len(combined) - 1):
                sp_l, src_l = combined[_i_sc]
                sp_r, src_r = combined[_i_sc + 1]
                if src_l == src_r:
                    continue  # 内部 gap (同 cluster 或同 slot) — 跳过
                # 根据源标签挑选对应的坐标映射和 eidx 集合
                if src_l == 'C':
                    l_cmap, l_src = clus_coords, p1_cluster_map[cluster_root]
                    r_cmap, r_src = slot_coords, slot_eidx_set
                else:
                    l_cmap, l_src = slot_coords, slot_eidx_set
                    r_cmap, r_src = clus_coords, p1_cluster_map[cluster_root]
                d_i, ok_i = _one_iface(sp_l, sp_r,
                                       l_cmap, r_cmap,
                                       l_src, r_src)
                if not ok_i:
                    return _INF
                total_d += d_i
                n_iface += 1

            if n_iface == 0:
                # 理论上不应出现:既然 clus_coords 和 slot_coords 都非空,
                # combined 至少存在一处 src 切换。兜底返 _INF 防御。
                return _INF
            return total_d / n_iface

        # ── 2h. Greedy bijective fallback (used when scipy is unavailable) ────
        def _greedy_bijective_p2(cost_mat):
            """O(N² log N) greedy assignment; bijective (each row/col at most once)."""
            _nr, _nc = cost_mat.shape
            _cands = sorted(
                [(cost_mat[_si, _ji], _si, _ji)
                 for _si in range(_nr) for _ji in range(_nc)],
                key=lambda x: x[0])
            _used_r, _used_c = set(), set()
            _rows, _cols = [], []
            for _, _si, _ji in _cands:
                if _si not in _used_r and _ji not in _used_c:
                    _rows.append(_si); _cols.append(_ji)
                    _used_r.add(_si); _used_c.add(_ji)
            return _rows, _cols

        # ── 2i. BFS ordering from seed domain ────────────────────────────────
        _bfs_vis_p2 = {seed_did}
        _bfs_q_p2   = _deque([seed_did])
        bfs_order_p2 = []
        while _bfs_q_p2:
            _d_p2 = _bfs_q_p2.popleft()
            bfs_order_p2.append(_d_p2)
            for _nd_p2 in sorted(
                    all_neighbors.get(_d_p2, set()),
                    key=lambda x: -max(enriched_entries[e]['cc_mask']
                                       for e in by_domain[x])):
                if _nd_p2 not in _bfs_vis_p2:
                    _bfs_vis_p2.add(_nd_p2)
                    _bfs_q_p2.append(_nd_p2)
        for _d_p2 in sorted_dids:
            if _d_p2 not in _bfs_vis_p2:
                bfs_order_p2.append(_d_p2)
        # Append domain IDs known from seq_obj.num_domains but absent from
        # filtered_domain_list (no instances → will be logged as [Missing—no instances])
        for _did_bfs_abs in range(1, _num_domains_expected + 1):
            if _did_bfs_abs not in _bfs_vis_p2:
                bfs_order_p2.append(_did_bfs_abs)
                _bfs_vis_p2.add(_did_bfs_abs)

        # ── 2j. Slot initialisation from seed domain's clusters ───────────────
        # Each slot receives one seed cluster (including any domains that were
        # already fragment-bridged to the seed in Pass 1).
        slot_contents    = [set() for _ in range(num_slots)]  # k -> set of eidx
        slot_for_cluster = {}   # cluster_root -> slot index

        _seed_roots_p2 = sorted(
            [r for r, cm in cluster_meta_p2.items()
             if seed_did in cm['domain_ids']],
            key=lambda r: cluster_meta_p2[r]['cc_max'], reverse=True)

        for k in range(min(num_slots, len(_seed_roots_p2))):
            _sr = _seed_roots_p2[k]
            slot_contents[k] = set(p1_cluster_map[_sr])
            slot_for_cluster[_sr] = k
            print(f"        Seed cluster → slot {k} "
                  f"(domains={cluster_meta_p2[_sr]['domain_ids']}, "
                  f"cc={cluster_meta_p2[_sr]['cc_max']:.3f})")

        # ── 2k. Assign remaining clusters via Hungarian ───────────────────────
        # For each domain in BFS order, collect all unassigned clusters that
        # contain that domain, find which slots still need it, build an
        # (n_slots_needing × n_unassigned_clusters) cost matrix, and solve
        # with Hungarian (or greedy fallback).  Because entire clusters are
        # placed atomically, Pass-1 connections are always preserved.
        _INF      = float('inf')  # hard-inf sentinel used inside cost helpers
        _LARGE_P2 = 1e9           # sentinel for above-threshold / invalid pairings

        for curr_did in bfs_order_p2[1:]:
            # Clusters containing curr_did not yet placed in any slot
            _unass_p2 = [
                r for r, cm in cluster_meta_p2.items()
                if curr_did in cm['domain_ids'] and r not in slot_for_cluster
            ]
            if not _unass_p2:
                # curr_did may be a domain known from seq_obj.num_domains but
                # absent from filtered_domain_list (no instances at all).
                if curr_did not in by_domain:
                    print(f"          [Missing—no instances] domain{curr_did}: "
                          f"known from seq_obj.num_domains but absent from "
                          f"filtered_domain_list — cannot place")
                continue

            # Slots that don't yet contain curr_did
            _need_p2 = [
                k for k in range(num_slots)
                if not any(enriched_entries[e]['domain_id'] == curr_did
                           for e in slot_contents[k])
            ]
            if not _need_p2:
                for r in _unass_p2:
                    print(f"          [Orphan] domain{curr_did} cluster {r}: "
                          f"all slots already contain this domain")
                continue

            _n_sl_p2 = len(_need_p2)
            _n_un_p2 = len(_unass_p2)

            # Cost matrix [n_sl × n_un]
            _cost_p2 = np.full((_n_sl_p2, _n_un_p2), _LARGE_P2)
            for _si_p2, _k_p2 in enumerate(_need_p2):
                for _ji_p2, _r_p2 in enumerate(_unass_p2):
                    _d_p2 = _cluster_slot_dist_p2(_r_p2, slot_contents[_k_p2])
                    if _d_p2 < _INF:
                        _cost_p2[_si_p2, _ji_p2] = _d_p2

            # Solve
            if _has_scipy_p2:
                try:
                    _row_p2, _col_p2 = _hungarian_solve(_cost_p2)
                except Exception as _he_p2:
                    print(f"          [Hungarian error: {_he_p2}] → greedy")
                    _row_p2, _col_p2 = _greedy_bijective_p2(_cost_p2)
            else:
                _row_p2, _col_p2 = _greedy_bijective_p2(_cost_p2)

            for _si_p2, _ji_p2 in zip(_row_p2, _col_p2):
                if _cost_p2[_si_p2, _ji_p2] >= _LARGE_P2:
                    print(f"          [Missing] domain{curr_did} "
                          f"slot {_need_p2[_si_p2]}: distance exceeds hard "
                          f"threshold — domain absent for this slot")
                    continue
                _k_p2 = _need_p2[_si_p2]
                _r_p2 = _unass_p2[_ji_p2]
                # Place entire cluster atomically into the slot
                slot_contents[_k_p2].update(p1_cluster_map[_r_p2])
                slot_for_cluster[_r_p2] = _k_p2
                # Maintain UF connectivity for Part 3
                _exist_p2 = next(
                    (e for e in slot_contents[_k_p2]
                     if e not in p1_cluster_map[_r_p2]), None)
                if _exist_p2 is not None:
                    for _eu_p2 in p1_cluster_map[_r_p2]:
                        uf.union(_eu_p2, _exist_p2)
                print(f"          [Hungarian] domain{curr_did} "
                      f"cluster {_r_p2} → slot {_k_p2}, "
                      f"dist={_cost_p2[_si_p2, _ji_p2]:.2f}")

        # Fallback: clusters not reachable from seed via BFS → empty slots
        for _r_p2 in sorted(
                p1_cluster_map.keys(),
                key=lambda r: cluster_meta_p2[r]['cc_max'], reverse=True):
            if _r_p2 in slot_for_cluster:
                continue
            _empty_k = next(
                (k for k in range(num_slots) if not slot_contents[k]), None)
            if _empty_k is not None:
                slot_contents[_empty_k].update(p1_cluster_map[_r_p2])
                slot_for_cluster[_r_p2] = _empty_k
                _exist_e = next(
                    (e for e in slot_contents[_empty_k]
                     if e not in p1_cluster_map[_r_p2]), None)
                if _exist_e is not None:
                    for _eu_p2 in p1_cluster_map[_r_p2]:
                        uf.union(_eu_p2, _exist_e)
                print(f"          [EmptySlot] cluster {_r_p2} → slot {_empty_k} "
                      f"(domains={cluster_meta_p2[_r_p2]['domain_ids']})")
            else:
                print(f"          [Orphan] cluster {_r_p2}: no empty slot "
                      f"(domains={cluster_meta_p2[_r_p2]['domain_ids']})")

        # ── Assemble chains from slot_contents ────────────────────────────────
        inst_idx = 0
        for k, _slot_set in enumerate(slot_contents):
            if not _slot_set:
                continue

            eidx_list = list(_slot_set)

            # Build position map (higher cc_mask wins on overlap)
            pos_map = {}
            for eidx in eidx_list:
                e       = enriched_entries[eidx]
                coords  = e['fitted_coords']
                res_idx = e['residue_indices']
                cc      = e['cc_mask']
                for local_pos in range(len(res_idx)):
                    sp = int(res_idx[local_pos])
                    pos_map[sp] = (coords[local_pos].copy(), eidx, local_pos, cc)

            if not pos_map:
                continue

            covered = sorted(pos_map.keys())
            print(f"        Chain inst {inst_idx}: {len(covered)} positions "
                  f"[{covered[0]}..{covered[-1]}], {len(eidx_list)} domains")

            # Diagnostics: show domain segments
            for eidx in eidx_list:
                e  = enriched_entries[eidx]
                ri = e['residue_indices']
                if not ri:
                    continue
                segs, s0, prev = [], ri[0], ri[0]
                for p in ri[1:]:
                    if p != prev + 1:
                        segs.append((s0, prev)); s0 = p
                    prev = p
                segs.append((s0, prev))
                print(f"          Domain[{e['source_domain_idx']}] "
                      f"id={e['domain_id']}: "
                      f"{', '.join(f'[{s}..{en}]' for s, en in segs)} "
                      f"({len(ri)} res, cc={e['cc_mask']:.4f})")

            # ── 收集每域的序列范围和 R/T 变换 ──────────────────────────────────
            domain_seq_ranges      = {}
            rotation_dict_entry    = {}
            translation_dict_entry = {}

            for eidx in eidx_list:
                e   = enriched_entries[eidx]
                did = e['domain_id']
                ri  = e['residue_indices']

                if ri:
                    lo, hi = int(min(ri)), int(max(ri))
                    if did not in domain_seq_ranges:
                        domain_seq_ranges[did] = (lo, hi)
                    else:
                        prev_lo, prev_hi = domain_seq_ranges[did]
                        domain_seq_ranges[did] = (min(prev_lo, lo),
                                                   max(prev_hi, hi))

                if did in rotation_dict_entry:
                    continue

                R_stored = e.get('rotation', None)
                T_stored = e.get('translation', None)
                if R_stored is None: R_stored = np.eye(3)
                if T_stored is None: T_stored = np.zeros(3)

                is_fallback = (np.allclose(R_stored, np.eye(3), atol=1e-6) and
                               np.allclose(T_stored, np.zeros(3), atol=1e-6))
                if is_fallback and len(ri) >= 3:
                    print(f"          [RT] domain{did} R/T 为单位变换，尝试重新计算...")
                    R_stored, T_stored = _recompute_RT_for_domain(
                        fasta_name, did,
                        np.asarray(e['fitted_coords'], dtype=np.float64),
                        ri)

                rotation_dict_entry[did]    = R_stored
                translation_dict_entry[did] = T_stored

            # Assemble final entry
            final_positions = sorted(pos_map.keys())
            final_coords    = np.array([pos_map[p][0] for p in final_positions],
                                       dtype=np.float64)

            domain_ids = sorted(set(
                enriched_entries[eidx]['domain_id']
                for eidx in eidx_list
                if enriched_entries[eidx]['domain_id'] is not None
            ))
            source_dom_indices = [enriched_entries[eidx]['source_domain_idx']
                                  for eidx in eidx_list]

            result_entry = {
                'fasta_name':            fasta_name,
                'chain_instance_idx':    inst_idx,
                'domain_ids':            domain_ids,
                'fitted_coords':         final_coords,
                'residue_indices':       final_positions,
                'cc_mask':               max(enriched_entries[eidx]['cc_mask']
                                            for eidx in eidx_list),
                'source_domain_indices': source_dom_indices,
                'is_merged':             len(domain_ids) > 1,
                'num_domains':           len(domain_ids),
                'enriched_eidx_list':    list(eidx_list),
                'domain_seq_ranges':     domain_seq_ranges,
                'rotation_dict':         rotation_dict_entry,
                'translation_dict':      translation_dict_entry,
            }
            shared_data.init_connect_result.append(result_entry)
            print(f"        ✓ Instance {inst_idx}: "
                  f"{len(final_positions)} residues, "
                  f"domains={domain_ids}\n")
            inst_idx += 1

    # Store enriched entries for Part 3
    shared_data._enriched_entries = enriched_entries

    elapsed = record_time('build_chain_models', start_time)
    print(f"      Completed in {elapsed:.2f}s")
    print(f"      Final entries: {len(shared_data.init_connect_result)}")

    # Export Pass-1/2 snapshot (before template patch)
    try:
        _out = getattr(shared_data.config, 'output_dir', '.')
        export_init_connect_result_to_pdb(
            _out, filename_override='init_modeling_after_chain_build.pdb'
        )
    except Exception as _e:
        print(f"      [chain-build PDB export failed: {_e}]")

    # ── Pass 3: template-guided last-effort patch ─────────────────────────────
    _pass3_template_patch()

    # Export post-patch snapshot
    try:
        _out = getattr(shared_data.config, 'output_dir', '.')
        export_init_connect_result_to_pdb(
            _out, filename_override='init_modeling_after_pass3_patch.pdb'
        )
    except Exception as _e:
        print(f"      [Pass3 PDB export failed: {_e}]")
#   Guided by AF2 structure, reorder / reconnect trace atoms within each
#   fragment chain before the domain-junction repair step.
#
#   Pipeline (per chain):
#     Step 2  – Find "matched" trace atoms: for every AF2 atom B that any
#               trace atom A maps to (<2Å), keep only the CLOSEST A from this
#               chain as the unique match.
#     Step 3  – Build reconstruction edges: for each pair (Ax, Ay) whose AF2
#               counterparts are sequence-adjacent, add a recon edge and remove
#               all fragment edges spanning between them.
#     Step 4  – Identify islands (recon-edge connected components, ≥3 atoms).
#       4-1   – When a fragment edge connects to an island's interior atom,
#               reroute to the nearer terminal; split if bond distance outside
#               [3.1, 4.5] Å.  Abandoned bridge-only chains are discarded.
#       4-2   – Validate bridge residue count between adjacent islands; split
#               (duplicating the bridge) if count mismatches or cross-FASTA.
# ═══════════════════════════════════════════════════════════════════════════════

def _pass3_template_patch():
    """
    Pass 3 — Template-guided last-effort patching for chain instances that are
    still missing domains after Pass 2.

    For each missing (chain_instance, domain_id) pair, the algorithm applies
    the AF2 template's relative pose from an already-placed sequence-adjacent
    neighbour, refines with DomainFit, and validates via density overlap and
    clash checks.  Multiple rounds handle chains whose neighbours are also
    missing (they may be placed later in the same or subsequent round).

    Deadlock guard: if a round ends with zero new placements for a chain,
    all remaining pending items for that chain are purged.
    """
    global shared_data
    print("\n" + "=" * 62)
    print("[Pass 3] Template-patch: last-effort for missing domains")
    print("=" * 62)

    if not hasattr(shared_data, 'init_connect_result'):
        print("[Pass 3] init_connect_result not found – skipping.")
        return

    # ── internal helpers ─────────────────────────────────────────────────────

    def _af2_for_domain(fn, did):
        """Return (af2_coords, seq_indices) or (None, None)."""
        indices = _get_domain_residue_indices(fn, did)
        if not indices:
            return None, None
        sobj = shared_data.fastas.get(fn)
        if sobj is None:
            return None, None
        try:
            af2 = np.asarray(sobj.AF2_struct, dtype=np.float64)
        except Exception:
            return None, None
        valid = [i for i in indices if i < len(af2)]
        if not valid:
            return None, None
        return af2[valid], valid

    def _kabsch(P, Q):
        """Kabsch: R, t  so that  P @ R.T + t ≈ Q."""
        P, Q = np.asarray(P, np.float64), np.asarray(Q, np.float64)
        Pc, Qc = P.mean(0), Q.mean(0)
        H = (P - Pc).T @ (Q - Qc)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        return R, Qc - Pc @ R.T

    def _placed_coords_snapshot():
        """All currently placed CA coords across every entry (or None)."""
        parts = [np.asarray(ce['fitted_coords'], dtype=np.float64)
                 for ce in shared_data.init_connect_result
                 if len(ce.get('fitted_coords', [])) > 0]
        return np.vstack(parts) if parts else None

    def _check_valid(new_coords, occ_snap, skip_overlap=False):
        """
        Return (ok, reason_str).
        Fails if clash ratio ≥ 10 % or (unless skip_overlap) density overlap ratio < 30 %.

        skip_overlap=True mirrors complement_and_fix behaviour for local_fit candidates:
        DomainFit starts from an already-placed pose so overlap is trusted; only clash
        is checked.
        """
        new_coords = np.asarray(new_coords, np.float64)
        n = len(new_coords)
        if n == 0:
            return False, "empty"

        # Clash check against all currently placed atoms
        if occ_snap is not None and len(occ_snap) > 0:
            from scipy.spatial import cKDTree as _KD
            dists, _ = _KD(occ_snap).query(new_coords, k=1)
            clash_r = float(np.sum(dists < 3.0)) / n
            if clash_r >= 0.10:
                return False, f"clash {clash_r*100:.0f}%≥10%"

        # Density overlap check (shared_data._overlap_tgt_points set by new_alignment)
        # Skipped for DomainFit-refined positions (skip_overlap=True), consistent with
        # complement_and_fix which exempts local_fit candidates from this check.
        if not skip_overlap:
            tgt_pts = getattr(shared_data, '_overlap_tgt_points', None)
            if tgt_pts is not None:
                try:
                    from POINTFIT.experiments.point_vec.demo_mask import compute_overlap
                    radius = getattr(shared_data, '_overlap_search_radius', 1.5)
                    _, _, corr = compute_overlap(tgt_pts, new_coords, radius)
                    if corr is None or corr.size == 0:
                        overlap = 0.0
                    else:
                        overlap = float(corr.shape[1]) / n
                    if overlap < 0.30:
                        return False, f"overlap {overlap*100:.0f}%<30%"
                except Exception:
                    pass   # overlap check unavailable → skip

        return True, "ok"

    def _seq_neighbors_placed(fn, did, placed_dids):
        """
        Domain IDs that are sequence-adjacent to *did* (boundary gap ≤ 1 residue)
        AND already placed in this chain instance.
        """
        idx_t = _get_domain_residue_indices(fn, did)
        if not idx_t:
            return []
        lo_t, hi_t = min(idx_t), max(idx_t)
        result = []
        for ndid in placed_dids:
            if ndid == did:
                continue
            idx_n = _get_domain_residue_indices(fn, ndid)
            if not idx_n:
                continue
            lo_n, hi_n = min(idx_n), max(idx_n)
            # left-neighbour: n is left of t   | right-neighbour: n is right of t
            if abs(hi_n + 1 - lo_t) <= 1 or abs(hi_t + 1 - lo_n) <= 1:
                result.append(ndid)
        return result

    def _update_entry(entry, fn, did, new_coords, seq_indices, cc_new, R_new, t_new):
        """Merge a newly placed domain into an existing result entry in place."""
        covered = set(entry['residue_indices'])
        new_sp, new_xyz = [], []
        for sp, xyz in zip(seq_indices, new_coords):
            if sp not in covered:
                new_sp.append(sp)
                new_xyz.append(xyz)
                covered.add(sp)
        if not new_sp:
            return

        all_sp  = list(entry['residue_indices']) + new_sp
        all_xyz = list(entry['fitted_coords'])   + new_xyz
        order   = np.argsort(all_sp)
        entry['residue_indices'] = [all_sp[i]  for i in order]
        entry['fitted_coords']   = np.array([all_xyz[i] for i in order], dtype=np.float64)

        if did not in entry['domain_ids']:
            entry['domain_ids'].append(did)
            entry['domain_ids'].sort()
        entry['domain_seq_ranges'][did] = (min(seq_indices), max(seq_indices))
        entry['rotation_dict'][did]     = R_new
        entry['translation_dict'][did]  = t_new
        entry['num_domains']            = len(entry['domain_ids'])
        entry['is_merged']              = entry['num_domains'] > 1
        entry['cc_mask']                = max(entry['cc_mask'], cc_new)

    # ── build initial missing-work list ──────────────────────────────────────
    missing_work = []
    for ce in shared_data.init_connect_result:
        fn   = ce['fasta_name']
        sobj = shared_data.fastas.get(fn)
        if sobj is None:
            continue
        nd = int(getattr(sobj, 'num_domains', 0))
        if nd <= 0:
            continue
        placed = set(ce['domain_ids'])
        for did in range(1, nd + 1):
            if did not in placed:
                missing_work.append({
                    'entry':      ce,
                    'fasta_name': fn,
                    'did':        did,
                    'tried_nids': set(),   # neighbour domain IDs already attempted
                })

    if not missing_work:
        print("[Pass 3] All chain instances complete – nothing to patch.")
        return

    print(f"[Pass 3] {len(missing_work)} missing domain slot(s):")
    for mw in missing_work:
        print(f"  {mw['fasta_name']} / domain{mw['did']} "
              f"(chain inst {mw['entry']['chain_instance_idx']})")

    em_path    = getattr(shared_data.config, 'em_path',     None)
    resolution = float(getattr(shared_data.config, 'resolution', 4.0))
    contour    = getattr(shared_data.config, 'contour',     0.01)

    MAX_ROUNDS = len(missing_work) + 2

    # ── multi-round patching loop ─────────────────────────────────────────────
    for round_idx in range(1, MAX_ROUNDS + 1):
        if not missing_work:
            break

        print(f"\n[Pass 3] ── Round {round_idx} ── {len(missing_work)} pending ──")

        # per-entry progress flag for deadlock detection
        active_eids    = {id(mw['entry']) for mw in missing_work}
        entry_progress = {eid: False for eid in active_eids}

        # longest domain first within the round
        work_sorted = sorted(
            missing_work,
            key=lambda mw: len(_get_domain_residue_indices(mw['fasta_name'], mw['did'])),
            reverse=True,
        )

        newly_placed = set()   # id(mw)
        to_remove    = set()   # id(mw) – permanently failed

        for mw in work_sorted:
            if id(mw) in newly_placed or id(mw) in to_remove:
                continue

            entry      = mw['entry']
            fn         = mw['fasta_name']
            did        = mw['did']
            tried_nids = mw['tried_nids']

            placed_dids = set(entry['domain_ids'])   # grows as same round progresses

            # Sequence-adjacent neighbours that are already placed in this entry
            all_adj = _seq_neighbors_placed(fn, did, placed_dids)
            untried = [n for n in all_adj if n not in tried_nids]

            # Neighbours that are themselves still pending (might be placed this round)
            pending_nids = {
                mw2['did']
                for mw2 in missing_work
                if mw2['entry'] is entry and mw2['did'] != did
            }

            if not all_adj:
                print(f"  {fn}/domain{did}: no sequence neighbours → permanently failed")
                to_remove.add(id(mw))
                continue

            if not untried:
                # All placed neighbours have been tried and failed.
                # Stay if any neighbour is still pending (may become available).
                hopeful = [n for n in all_adj if n in pending_nids]
                if hopeful:
                    print(f"  {fn}/domain{did}: no untried placed neighbours; "
                          f"waiting for pending {hopeful}")
                else:
                    print(f"  {fn}/domain{did}: all neighbours tried+failed, "
                          f"none pending → permanently failed")
                    to_remove.add(id(mw))
                continue

            # Fetch AF2 coords for the target domain
            af2_t, seq_indices = _af2_for_domain(fn, did)
            if af2_t is None:
                print(f"  {fn}/domain{did}: no AF2 template coords → permanently failed")
                to_remove.add(id(mw))
                continue

            # Take a snapshot of all placed coords before trying any neighbour
            occ_snap = _placed_coords_snapshot()

            best_coords = None
            best_cc     = -1.0
            best_R      = None
            best_t      = None

            for nid in untried:
                tried_nids.add(nid)

                R_n = entry['rotation_dict'].get(nid)
                t_n = entry['translation_dict'].get(nid)
                if R_n is None or t_n is None:
                    print(f"    domain{did} ← neighbour{nid}: no R/T stored → skip")
                    continue
                R_n = np.asarray(R_n, np.float64)
                t_n = np.asarray(t_n, np.float64)

                # pos1: apply neighbour's rigid transform to AF2 template coords
                # Treated as global_fit equivalent → full check (clash + overlap)
                try:
                    pos1 = np.dot(af2_t, R_n.T) + t_n
                    cc1  = _get_cc_mask_for_coords(pos1)
                except Exception as _e1:
                    print(f"    domain{did} ← neighbour{nid}: pos1 error ({_e1}) → skip")
                    continue

                # pos2: DomainFit refinement starting from neighbour's pose
                # Treated as local_fit equivalent → clash-only check (no overlap),
                # consistent with complement_and_fix which exempts local_fit from overlap.
                pos2 = pos1
                cc2  = -1.0
                if em_path is not None:
                    try:
                        fit  = _hs_domainfit(
                            domain_coords     = af2_t,
                            rotation          = R_n,
                            translation       = t_n,
                            em_path           = em_path,
                            resolution        = resolution,
                            contour           = contour,
                        )
                        pos2 = np.asarray(fit['fitted_coords'], np.float64)
                        cc2  = _get_cc_mask_for_coords(pos2)
                    except Exception as _e2:
                        print(f"    domain{did} ← neighbour{nid}: DomainFit error ({_e2})")

                print(f"    domain{did} ← neighbour{nid}: "
                      f"cc1={cc1:.4f}  cc2={cc2:.4f}")

                # Validate each position independently, then pick best valid CC.
                # pos1 (template pose): full check — clash + overlap (≥30%).
                # pos2 (DomainFit):     clash-only — mirrors complement_and_fix local_fit.
                valid1, reason1 = _check_valid(pos1, occ_snap, skip_overlap=False)
                valid2, reason2 = _check_valid(pos2, occ_snap, skip_overlap=True)

                valid_candidates = []
                if valid1:
                    valid_candidates.append((cc1, pos1, "pos1(template)"))
                if valid2:
                    valid_candidates.append((cc2, pos2, "pos2(DomainFit)"))

                if not valid_candidates:
                    print(f"      ✗ pos1:{reason1}  pos2:{reason2} → rejected")
                    continue

                chosen_cc, chosen, chosen_label = max(valid_candidates, key=lambda x: x[0])
                print(f"      → {chosen_label}  cc={chosen_cc:.4f}")

                if chosen_cc > best_cc:
                    best_cc     = chosen_cc
                    best_coords = chosen
                    best_R, best_t = _kabsch(af2_t, chosen)
                    print(f"      ✓ new best  (cc={best_cc:.4f})")

            if best_coords is not None:
                _update_entry(entry, fn, did, best_coords, seq_indices,
                              best_cc, best_R, best_t)
                entry_progress[id(entry)] = True
                newly_placed.add(id(mw))
                print(f"  ✓ {fn}/domain{did} placed, cc={best_cc:.4f}")

        # Remove placed and permanently-failed items
        missing_work = [mw for mw in missing_work
                        if id(mw) not in newly_placed and id(mw) not in to_remove]

        # Deadlock detection: entries with zero progress this round → purge
        stuck_eids = {eid for eid in active_eids if not entry_progress[eid]}
        if stuck_eids:
            purge = [mw for mw in missing_work if id(mw['entry']) in stuck_eids]
            for mw in purge:
                print(f"  [Deadlock] {mw['fasta_name']}/domain{mw['did']} "
                      f"(chain inst {mw['entry']['chain_instance_idx']}): "
                      f"no progress this round → purged")
            missing_work = [mw for mw in missing_work
                            if id(mw['entry']) not in stuck_eids]

        n_placed = sum(1 for p in entry_progress.values() if p)
        print(f"  Round {round_idx}: {n_placed} domain(s) placed, "
              f"{len(missing_work)} still pending")

    # ── summary ──────────────────────────────────────────────────────────────
    if missing_work:
        print(f"\n[Pass 3] {len(missing_work)} domain(s) could not be patched:")
        for mw in missing_work:
            print(f"  {mw['fasta_name']}/domain{mw['did']} "
                  f"(chain inst {mw['entry']['chain_instance_idx']})")
    else:
        print("\n[Pass 3] All missing domains resolved.")


def reconstruct_fragments_with_af2():
    """
    AF2-guided intra-fragment reconstruction.  Operates on shared_data.final_traces
    in place; must run after find_corresponding_CA() (which populates
    shared_data.trace_to_domain_map) and before repair_chains_with_fragments().
    """
    global shared_data
    start_time = time.time()
    print(f"  [2.5/3] Reconstructing fragments with AF2 guidance...")

    MIN_ISLAND_SIZE = 3
    BOND_MIN = 3.1
    BOND_MAX = 4.5

    # ── Pre-build AF2 sequence-position table ─────────────────────────────────
    # (domain_idx, local_pos) -> (fasta_name, chain_instance_idx, global_seq_pos)
    # 使用 domain_idx → instance_idx 直接映射（来自 source_domain_indices），
    # 避免同 fasta 多实例时 (fasta, sp) → inst 键碰撞导致实例混淆
    _dom_to_inst_rec = {}
    for _ce in shared_data.init_connect_result:
        _inst = _ce['chain_instance_idx']
        for _di in _ce.get('source_domain_indices', []):
            _dom_to_inst_rec[_di] = _inst

    af2_seqpos = {}
    for domain_idx, entry in enumerate(shared_data.filtered_domain_list):
        fasta_name = entry.get('fasta_name', '')
        coords = entry.get('fitted_coords', [])
        if not fasta_name or len(coords) == 0:
            continue
        _inst = _dom_to_inst_rec.get(domain_idx)
        if _inst is None:
            continue
        res_indices = _estimate_residue_indices_from_coords(entry, fasta_name)
        for lp, sp in enumerate(res_indices):
            af2_seqpos[(domain_idx, lp)] = (fasta_name, _inst, int(sp))

    num_chains_original = shared_data.final_traces['num_chains']
    chains_modified = 0
    extra_chains = []
    # Track every ca_idx that belongs to a valid island across all chains.
    # Used at the end to discard chains that contain no island atoms at all.
    all_island_ca_indices = set()

    for chain_idx in range(num_chains_original):
        chain_info = shared_data.final_traces['chains'][chain_idx]
        ca_indices  = chain_info['ca_indices']
        ss_types    = chain_info.get('ss_types', None)
        N = len(ca_indices)

        if N < MIN_ISLAND_SIZE:
            continue

        # ── Local helpers (capture chain-specific arrays) ─────────────────────
        def get_coord(pos):
            return np.asarray(shared_data.ca_pos[ca_indices[pos]], dtype=np.float64)

        def bond_ok(p1, p2):
            return BOND_MIN <= float(np.linalg.norm(get_coord(p1) - get_coord(p2))) <= BOND_MAX

        # ── Step 2: find matched atoms ────────────────────────────────────────
        # domain_key -> (atom_pos, dist)  — closest trace atom from this chain
        domain_best = {}
        for pos in range(N):
            dk = shared_data.trace_to_domain_map.get((chain_idx, pos))
            if dk is None or dk not in af2_seqpos:
                continue
            dom_idx, dom_lp = dk
            dom_coords = shared_data.filtered_domain_list[dom_idx].get('fitted_coords', [])
            if dom_lp >= len(dom_coords):
                continue
            dist = float(np.linalg.norm(
                get_coord(pos) - np.asarray(dom_coords[dom_lp], dtype=np.float64)))
            if dk not in domain_best or dist < domain_best[dk][1]:
                domain_best[dk] = (pos, dist)

        # pos -> domain_key  (only for uniquely matched positions)
        pos_to_dk = {pos: dk for dk, (pos, _) in domain_best.items()}

        if len(pos_to_dk) < 2:
            continue

        # ── Step 3: reconstruction edges ──────────────────────────────────────
        matched_pos = sorted(pos_to_dk.keys())

        def af2info(pos):
            """(fasta_name, instance_idx, seq_pos) or None."""
            return af2_seqpos.get(pos_to_dk.get(pos))

        # 第一阶段：建候选重构边（同fasta + 同链实例 + seq相邻）
        recon_adj = {p: set() for p in matched_pos}
        matched_by_af2 = sorted(
            [(af2info(p), p) for p in matched_pos if af2info(p) is not None],
            key=lambda x: (x[0][0], x[0][1], x[0][2])
        )
        for idx in range(len(matched_by_af2) - 1):
            (fname_i, inst_i, sp_i), pi = matched_by_af2[idx]
            (fname_j, inst_j, sp_j), pj = matched_by_af2[idx + 1]
            if fname_i == fname_j and inst_i == inst_j and sp_j - sp_i == 1:
                recon_adj[pi].add(pj)
                recon_adj[pj].add(pi)

        # 第二阶段：过滤孤立重构边（连通块<MIN_ISLAND_SIZE则不纳入有效节点，但保留边结构）
        # 注意：不清除小连通块的 recon_adj，因为：
        #   - Phase 3 仅遍历 valid_recon_nodes，小连通块的边不会触发片段边删除
        #   - comb_nbs 仅对 pos_to_island 成员使用重构边，小连通块的边不会被误用
        # 原来清空 recon_adj 会导致 early-continue 在"全为小连通块"时误判
        # 整条链无重构边而直接跳过。
        vis_tmp = set()
        valid_recon_nodes = set()
        for seed in matched_pos:
            if seed in vis_tmp:
                continue
            if not recon_adj.get(seed):
                vis_tmp.add(seed); continue
            comp = set(); q = [seed]
            while q:
                c = q.pop()
                if c in comp: continue
                comp.add(c)
                q.extend(nb for nb in recon_adj[c] if nb not in comp)
            vis_tmp.update(comp)
            if len(comp) >= MIN_ISLAND_SIZE:
                valid_recon_nodes.update(comp)

        # 第三阶段：只对有效节点间的重构边删除对应片段边
        frag_removed = set()
        for pi in valid_recon_nodes:
            for pj in recon_adj[pi]:
                lo, hi = min(pi, pj), max(pi, pj)
                for k in range(lo, hi):
                    frag_removed.add((k, k + 1))

        if not valid_recon_nodes:
            continue     # no valid large-island recon edges — nothing to do

        # remaining fragment adjacency after edge removal
        rem_frag = {}   # pos -> [neighbor, ...]
        for k in range(N - 1):
            if (k, k + 1) not in frag_removed:
                rem_frag.setdefault(k,     []).append(k + 1)
                rem_frag.setdefault(k + 1, []).append(k)

        # ── Step 4: islands ───────────────────────────────────────────────────
        visited_recon = set()
        island_paths  = []          # list of linearised position lists
        pos_to_island = {}          # pos -> island_id (int)

        for seed in matched_pos:
            if seed in visited_recon:
                continue
            # BFS in recon graph
            comp = set()
            q = [seed]
            while q:
                cur = q.pop()
                if cur in comp:
                    continue
                comp.add(cur)
                q.extend(nb for nb in recon_adj.get(cur, set()) if nb not in comp)
            visited_recon.update(comp)

            if len(comp) < MIN_ISLAND_SIZE:
                continue

            # Sort atoms by AF2 sequence position — path[0] = min-sp (seq start),
            # path[-1] = max-sp (seq end).  Recon edges are built from sp-adjacent
            # pairs (sp diff == 1) so sorting by sp is equivalent to the correct
            # linear order, and is far simpler than graph traversal from an endpoint.
            def _sp_key(p):
                info = af2info(p)
                return info[2] if info is not None else float('inf')

            path = sorted(comp, key=_sp_key)

            iid = len(island_paths)
            island_paths.append(path)
            for p in path:
                pos_to_island[p] = iid

        if not island_paths:
            # No valid islands on this chain — leave it unchanged for now;
            # it will be filtered out at the end by the island_ca check.
            continue

        # Record all ca_indices that belong to valid islands on this chain.
        # This set is used for the final island-presence filter across ALL chains
        # (including extra_chains produced by splits), regardless of chain ordering.
        for ipath in island_paths:
            for pos in ipath:
                all_island_ca_indices.add(ca_indices[pos])

        # Orphaned: not in any island and no remaining fragment connections
        orphaned = set()
        for pos in range(N):
            if pos not in pos_to_island and not rem_frag.get(pos):
                orphaned.add(pos)

        # Combined neighbour function (frag + recon, excluding orphans)
        # Recon edges only apply to island members; small components (<MIN_ISLAND_SIZE)
        # that did not form islands must not pollute the combined graph.
        def comb_nbs(pos):
            nbs = set()
            for nb in rem_frag.get(pos, []):
                if nb not in orphaned:
                    nbs.add(nb)
            if pos in pos_to_island:
                for nb in recon_adj.get(pos, set()):
                    if nb not in orphaned:
                        nbs.add(nb)
            return nbs

        # Connected components in combined graph
        all_pos_set  = set(range(N)) - orphaned
        visited_comb = set()
        components   = []
        for pos in sorted(all_pos_set):
            if pos in visited_comb:
                continue
            comp = set()
            q    = [pos]
            while q:
                cur = q.pop()
                if cur in comp:
                    continue
                comp.add(cur)
                q.extend(nb for nb in comb_nbs(cur) if nb not in comp)
            visited_comb.update(comp)
            components.append(comp)

        # ── Step 4-1: walk each component, apply distance-split logic ────────
        all_result_chains = []   # list of atom-position lists

        for comp in components:
            def comp_deg(p):
                return sum(1 for nb in comb_nbs(p) if nb in comp)

            eps_c = [p for p in comp if comp_deg(p) <= 1]
            walk_start = min(eps_c) if eps_c else min(comp)

            result_chains_loc = []
            cur_chain         = []
            proc_islands      = set()
            visited_walk      = set()

            def flush():
                if cur_chain:
                    result_chains_loc.append(list(cur_chain))
                    cur_chain.clear()

            def ordered_island(iid, entry_pos, prev_pos=None):
                """
                Return (ordered_path, near_terminal, far_terminal) for island iid.

                Since island_paths are linearised from min-sp to max-sp, path[0]
                is always the sequence-start terminal and path[-1] is the end.

                Orientation (which terminal is 'near') is determined by:
                  1. If prev_pos is given: spatial distance from prev_pos to each
                     terminal — near_t is the spatially closer terminal.
                     This is the most reliable signal: the bridge atom that just
                     preceded the island naturally sits close to the terminal it
                     should connect to, regardless of trace-position ordering.
                  2. Otherwise (island is the walk-start): path-index distance from
                     entry_pos — near_t is the terminal closer in path order.
                  3. Fallback (entry_pos not in path): spatial distance from
                     entry_pos itself to each terminal.
                """
                path = island_paths[iid]
                t0, t1 = path[0], path[-1]

                if prev_pos is not None:
                    d0 = float(np.linalg.norm(get_coord(prev_pos) - get_coord(t0)))
                    d1 = float(np.linalg.norm(get_coord(prev_pos) - get_coord(t1)))
                elif entry_pos in path:
                    idx = path.index(entry_pos)
                    d0 = float(idx)
                    d1 = float(len(path) - 1 - idx)
                else:
                    d0 = float(np.linalg.norm(get_coord(entry_pos) - get_coord(t0)))
                    d1 = float(np.linalg.norm(get_coord(entry_pos) - get_coord(t1)))

                if d0 <= d1:
                    return path, t0, t1
                else:
                    return list(reversed(path)), t1, t0

            # Linear walk through the component
            pos  = walk_start
            prev = None

            while pos is not None:
                if pos in visited_walk:
                    break      # safety guard against unexpected cycles

                iid = pos_to_island.get(pos)

                if iid is not None and iid not in proc_islands:
                    # ── Island block ──────────────────────────────────────────
                    opath, near_t, far_t = ordered_island(iid, pos, prev_pos=prev)

                    # Step 4-1: reroute check
                    if near_t != pos and prev is not None:
                        if not bond_ok(prev, near_t):
                            flush()     # split before this island
                        # (if bond ok, rerouting is implicit: we just output
                        #  the island starting from near_t instead of pos)

                    cur_chain.extend(opath)
                    proc_islands.add(iid)
                    for ip in island_paths[iid]:
                        visited_walk.add(ip)

                    # Find continuation(s) via remaining fragment edges from
                    # any island atom, to an unvisited non-orphan atom
                    right_conts = [
                        (nb, src)
                        for src in island_paths[iid]
                        for nb  in rem_frag.get(src, [])
                        if nb not in visited_walk and nb not in orphaned
                    ]

                    if not right_conts:
                        pos  = None
                        prev = None
                        continue

                    # Prefer continuation from far terminal; else take first
                    from_far = [(nb, src) for nb, src in right_conts if src == far_t]
                    next_pos, _ = from_far[0] if from_far else right_conts[0]

                    # Exit bond check (step 4-1, right side)
                    if not bond_ok(far_t, next_pos):
                        flush()
                        prev = None
                    else:
                        prev = far_t
                    pos = next_pos

                elif iid is not None and iid in proc_islands:
                    # Already-processed island atom reached again — stop
                    visited_walk.add(pos)
                    pos  = None
                    prev = None

                else:
                    # ── Non-island atom ───────────────────────────────────────
                    visited_walk.add(pos)
                    cur_chain.append(pos)

                    # Advance via remaining fragment edges (excluding back-edge)
                    next_cands = [
                        nb for nb in rem_frag.get(pos, [])
                        if nb != prev
                        and nb not in visited_walk
                        and nb not in orphaned
                    ]
                    if next_cands:
                        prev = pos
                        pos  = next_cands[0]
                    else:
                        pos  = None
                        prev = None

            flush()
            all_result_chains.extend(result_chains_loc)

        # Drop pure-bridge chains that were abandoned by islands on BOTH sides.
        # Terminal bridges (one end has no island neighbour) must be preserved.
        def is_abandoned_bridge(c):
            if any(p in pos_to_island for p in c):
                return False  # contains island atoms → always keep
            left_adj_island  = any(nb in pos_to_island
                                   for nb in rem_frag.get(c[0],  []))
            right_adj_island = any(nb in pos_to_island
                                   for nb in rem_frag.get(c[-1], []))
            return left_adj_island and right_adj_island

        filtered_chains = [c for c in all_result_chains
                           if not is_abandoned_bridge(c)]

        if not filtered_chains:
            continue

        # ── Step 4-2: validate inter-island bridge residue counts ─────────────
        def island_af2_range(iid):
            """Return (fasta_name, instance_idx, min_seqpos, max_seqpos) for island iid,
            or (None, None, None, None) on failure.
            af2info 返回 (fasta_name, instance_idx, seq_pos)，必须同时区分实例。
            """
            infos = [af2info(p) for p in island_paths[iid]]
            infos = [x for x in infos if x is not None]
            if not infos:
                return None, None, None, None
            # 同一个岛内必须 fasta 和 instance 全部一致
            fnames   = {x[0] for x in infos}
            inst_ids = {x[1] for x in infos}
            if len(fnames) > 1 or len(inst_ids) > 1:
                return None, None, None, None
            sps = [x[2] for x in infos]   # x[2] = seq_pos（修复 Bug C：原来错用 x[1]）
            return infos[0][0], infos[0][1], min(sps), max(sps)

        final_chains = []

        for cpos in filtered_chains:
            # Segment chain into island / bridge runs
            segs = []          # list of (island_id_or_None, [positions])
            cur_iid = pos_to_island.get(cpos[0])
            cur_seg = [cpos[0]]
            for p in cpos[1:]:
                this_iid = pos_to_island.get(p)
                if this_iid == cur_iid:
                    cur_seg.append(p)
                else:
                    segs.append((cur_iid, cur_seg))
                    cur_iid, cur_seg = this_iid, [p]
            segs.append((cur_iid, cur_seg))

            # Identify inter-island bridges that need splitting
            split_at = set()   # indices into segs

            for si, (seg_iid, seg_pos) in enumerate(segs):
                if seg_iid is not None:
                    continue   # island segment — skip
                # Bridge: check only when flanked by islands on both sides
                left_iid  = segs[si - 1][0] if si > 0           else None
                right_iid = segs[si + 1][0] if si < len(segs)-1 else None
                if left_iid is None or right_iid is None:
                    continue   # terminal bridge — leave intact

                lf, l_inst, lmin, lmax = island_af2_range(left_iid)
                rf, r_inst, rmin, rmax = island_af2_range(right_iid)

                # 必须同 fasta 且同链实例，才视为同一条链（修复 Bug D）
                if lf is None or rf is None or lf != rf or l_inst != r_inst:
                    split_at.add(si)
                    continue

                # Expected residue gap between the two islands
                if lmax < rmin:
                    gap = rmin - lmax - 1
                elif rmax < lmin:
                    gap = lmin - rmax - 1
                else:
                    split_at.add(si)   # overlapping — be conservative
                    continue

                tol = 2 if gap > 10 else 1
                if abs(len(seg_pos) - gap) > tol:
                    split_at.add(si)

            if not split_at:
                final_chains.append(cpos)
                continue

            # Split: each flagged bridge is kept on BOTH sides (appended to left
            # sub-chain AND prepended to right sub-chain).  This is the correct
            # behaviour per spec: each resulting fragment keeps the bridge atoms
            # at its terminal so downstream steps can use them as context.
            cur_sub = []
            for si, (seg_iid, seg_pos) in enumerate(segs):
                if si in split_at:
                    cur_sub.extend(seg_pos)
                    final_chains.append(list(cur_sub))
                    cur_sub = list(seg_pos)   # bridge kept at start of next sub
                else:
                    cur_sub.extend(seg_pos)
            if cur_sub:
                final_chains.append(cur_sub)

        # ── Build new chain_info entries and update shared_data.final_traces ──
        new_entries = []
        for cpos in final_chains:
            if not cpos:
                continue
            entry = {k: v for k, v in chain_info.items()
                     if k not in ('ca_indices', 'ss_types')}
            entry['ca_indices'] = [ca_indices[p] for p in cpos]
            if ss_types is not None:
                entry['ss_types'] = [ss_types[p] for p in cpos]
            new_entries.append(entry)

        if not new_entries:
            continue

        # Check whether anything actually changed
        if (len(new_entries) == 1
                and new_entries[0]['ca_indices'] == list(ca_indices)):
            continue

        shared_data.final_traces['chains'][chain_idx] = new_entries[0]
        chains_modified += 1
        for extra in new_entries[1:]:
            extra_chains.append(extra)

    # Append any extra chains produced by splitting
    for entry in extra_chains:
        shared_data.final_traces['chains'].append(entry)

    # Discard chains that contain no island atoms.
    # This correctly handles:
    #   - Original chains with no valid islands at all
    #   - Sub-chains produced by splitting where the sub-chain is a pure bridge
    # Because all_island_ca_indices was populated from island_paths across all
    # original chains (before splitting), the check works regardless of chain
    # ordering or whether atoms came from original or extra_chains entries.
    n_before = len(shared_data.final_traces['chains'])

    def _has_island(chain_entry):
        return any(ca in all_island_ca_indices
                   for ca in chain_entry['ca_indices'])

    kept = [c for c in shared_data.final_traces['chains'] if _has_island(c)]
    n_discarded = n_before - len(kept)
    shared_data.final_traces['chains'] = kept
    if n_discarded:
        print(f"      Discarded {n_discarded} island-less fragment chains")
    shared_data.final_traces['num_chains'] = len(shared_data.final_traces['chains'])

    elapsed = record_time('reconstruct_fragments_with_af2', start_time)
    print(f"      Completed in {elapsed:.2f}s")
    print(f"      Chains modified : {chains_modified}")
    print(f"      Extra chains    : {len(extra_chains)}")
    print(f"      Total chains    : {shared_data.final_traces['num_chains']}")

    # Export reconstructed fragments to PDB
    try:
        from utils.EMtools import chainID_list
        _out  = getattr(shared_data.config, 'output_dir', '.')
        os.makedirs(_out, exist_ok=True)
        _frag_pdb = os.path.join(_out, 'reconstructed_fragments.pdb')
        with open(_frag_pdb, 'w') as _pf:
            _pf.write("REMARK   reconstructed fragments after AF2-guided reordering\n")
            _pf.write(f"REMARK   Total fragment chains: "
                      f"{shared_data.final_traces['num_chains']}\n")
            _pf.write("REMARK\n")
            _serial = 1
            for _ci in range(shared_data.final_traces['num_chains']):
                _cinfo  = shared_data.final_traces['chains'][_ci]
                _ca_idx = _cinfo['ca_indices']
                _chain_id = chainID_list[_ci % len(chainID_list)]
                for _res_num, _ca in enumerate(_ca_idx, start=1):
                    _x, _y, _z = shared_data.ca_pos[_ca]
                    _pf.write(
                        f"ATOM  {_serial:5d}  CA  ALA {_chain_id}"
                        f"{_res_num:4d}    "
                        f"{float(_x):8.3f}{float(_y):8.3f}{float(_z):8.3f}"
                        f"  1.00  0.00           C\n"
                    )
                    _serial += 1
                _pf.write(
                    f"TER   {_serial:5d}      ALA {_chain_id}"
                    f"{len(_ca_idx):4d}\n"
                )
                _serial += 1
            _pf.write("END\n")
        print(f"      Fragments exported to: {_frag_pdb}")
    except Exception as _e:
        print(f"      [fragment PDB export failed: {_e}]")


# ── Module-level helpers for repair_chains_with_fragments ────────────────────

def _repair_cc_worker(coords_input):
    """
    Top-level multiprocessing worker: compute CC_mask for a coordinate list.
    Accesses shared_data (module global) for config; receives only coords.
    """
    global shared_data
    import numpy as _np
    from POINTFIT.experiments.local_fit.fit_xxy_cc_map import (
        calculate_cc_mask_for_structure,
        create_temp_directory,
    )

    coords = _np.asarray(coords_input, dtype=_np.float64)
    if len(coords) == 0:
        return 0.0

    em_path    = getattr(shared_data.config, 'em_path', None)
    resolution = float(getattr(shared_data.config, 'resolution', 4.0))
    out_dir    = getattr(shared_data.config, 'output_dir', '.')

    if em_path is None:
        return 0.0

    class _S:
        def __init__(self, c):
            self.format = 'pdb'
            self.atoms = [
                {'record': 'ATOM', 'serial': i + 1, 'name': 'CA',
                 'altLoc': ' ', 'resName': 'ALA', 'chainID': 'A',
                 'resSeq': i + 1, 'iCode': ' ',
                 'x': float(v[0]), 'y': float(v[1]), 'z': float(v[2]),
                 'occupancy': 1.0, 'tempFactor': 0.0,
                 'element': 'C', 'charge': ''}
                for i, v in enumerate(c)
            ]

        def get_coordinates(self):
            return _np.array([[a['x'], a['y'], a['z']] for a in self.atoms],
                             dtype=_np.float64)

        def write_structure(self, fn):
            with open(fn, 'w') as f:
                for a in self.atoms:
                    f.write(
                        f"ATOM  {a['serial']:5d}  CA  ALA {a['chainID']}"
                        f"{a['resSeq']:4d}    "
                        f"{a['x']:8.3f}{a['y']:8.3f}{a['z']:8.3f}"
                        f"  1.00  1.00           C\n"
                    )
                f.write("END\n")

    try:
        tmp = create_temp_directory(out_dir)
        val = calculate_cc_mask_for_structure(_S(coords), em_path, resolution, tmp)
        return float(val) if val is not None else 0.0
    except Exception:
        return 0.0


# _kabsch_align 已废除，repair_chains_with_fragments 内使用 _tm_fit 局部函数


# ═══════════════════════════════════════════════════════════════════════════════
# Part 3.  repair_chains_with_fragments
#   For each chain, scan domain junctions and attempt fragment-based repair.
# ═══════════════════════════════════════════════════════════════════════════════

def _get_cc_mask_for_coords(coords):
    """Compute cc_mask for a set of CA coordinates using the alignment module."""
    global shared_data
    from POINTFIT.experiments.local_fit.fit_xxy_cc_map import (
        calculate_cc_mask_for_structure,
        create_temp_directory,
    )

    coords = np.asarray(coords, dtype=np.float64)
    if coords.size == 0:
        return 0.0

    class _TempStructureMerged:
        def __init__(self, coords):
            self.format = 'pdb'
            self.atoms = []
            for i, coord in enumerate(coords):
                self.atoms.append({
                    'record': 'ATOM', 'serial': i + 1, 'name': 'CA',
                    'altLoc': ' ', 'resName': 'ALA', 'chainID': 'A',
                    'resSeq': i + 1, 'iCode': ' ',
                    'x': coord[0], 'y': coord[1], 'z': coord[2],
                    'occupancy': 1.0, 'tempFactor': 0.0,
                    'element': 'C', 'charge': ''
                })

        def get_coordinates(self):
            return np.array([[a['x'], a['y'], a['z']] for a in self.atoms],
                            dtype=np.float64)

        def write_structure(self, filename):
            self._write_pdb(filename)

        def _write_pdb(self, filename):
            with open(filename, 'w') as f:
                for atom in self.atoms:
                    line = (
                        f"ATOM  {atom['serial']:5d}  {atom['name']:<3s} "
                        f"{atom['resName']:>3s} {atom['chainID']}"
                        f"{atom['resSeq']:4d}    "
                        f"{atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}"
                        f"{atom['occupancy']:6.2f}{atom['tempFactor']:6.2f}"
                        f"          {atom['element']:>2s}\n"
                    )
                    f.write(line)
                f.write("END\n")

    cc_mask_score = 0.0
    em_path = shared_data.config.em_path
    resolution = getattr(shared_data.config, 'resolution', 4.0)
    aligned_structure = _TempStructureMerged(coords)
    temp_dir = create_temp_directory(shared_data.config.output_dir)

    try:
        cc_mask_score = calculate_cc_mask_for_structure(
            aligned_structure, em_path, resolution, temp_dir
        )
    except Exception as e:
        print(f"      WARNING: CC_mask computation failed: {e}")
    return cc_mask_score


def _is_all_coil_between(fasta_name, seq_start, seq_end):
    """
    Check that all residues in [seq_start, seq_end] (inclusive) are coil
    in the AF2 secondary structure prediction.
    """
    for sp in range(seq_start, seq_end + 1):
        ss = _get_af2_ss_at_seq_pos(fasta_name, sp)
        if ss is not None and ss != COIL_SS:
            return False
    return True


def _is_all_coil_in_fragment(chain_idx, frag_start_pos, frag_end_pos):
    """
    Check that all atoms in the fragment between positions frag_start_pos
    and frag_end_pos (inclusive) are coil.

    Uses the SS types stored on the trace chain.
    """
    global shared_data
    chain_info = shared_data.final_traces['chains'][chain_idx]
    ss_types = chain_info.get('ss_types', None)
    if ss_types is None:
        return True  # Cannot verify, assume ok
    for pos in range(frag_start_pos, frag_end_pos + 1):
        if pos < 0 or pos >= len(ss_types):
            continue
        if int(ss_types[pos]) != COIL_SS:
            return False
    return True


def repair_chains_with_fragments():
    """
    Part 3: AF2-guided fragment repair of init_connect_result.

    Pre-step: Build a fresh ca_idx -> (fn, inst, sp) map by re-querying the
              domain KD-tree against every trace atom.  This is necessary
              because reconstruct_fragments_with_af2() reorders/splits chains,
              making the pre-reconstruction trace_to_domain_map stale.

              Then build the global island-CA map:
                (fn, inst, sp) -> coord   (one atom per seq-position)
              Conflicts (two fragments claim the same seq-position) are
              resolved by keeping the atom closer to the current modeling_2
              coord at that position.  modeling_2 coords come from domain
              fitting and are density-informed; this is more robust than
              distance to AF2 reference.

              Per-fragment jurisdiction (first-island-sp .. last-island-sp)
              is added to covered_seqpos so Step 3 skips those positions.

    Step 1  Connected-block island replacement.
            Island positions are grouped into contiguous runs on each chain
            entry.  Each run is evaluated independently (parallel CC) and
            applied if CC improves.  Runs are disjoint, so they are safe to
            merge.

    Step 2  Bridge replacement (within jurisdiction).
            For every bridge (non-island atoms between two islands in the
            same trace fragment): bond-check both ends, skip if any position
            is already an island position, evaluate CC, apply position-wise
            best.

    Step 3  Gap filling (positions outside all jurisdictions).
            For each contiguous uncovered gap: try TM-align fits using left
            and right anchor regions and any available tail atoms.  Best CC
            candidate wins (parallel evaluation).

    Updates shared_data.init_connect_result in place.
    """
    global shared_data
    import copy
    from multiprocessing import Pool
    from collections import defaultdict

    start_time = time.time()
    print("  [3/3] Repairing chains with fragments (AF2-guided)...")

    if not hasattr(shared_data, 'init_connect_result') or not shared_data.init_connect_result:
        print("      No chains to repair.")
        return

    num_procs  = int(getattr(shared_data.config, 'mul_proc_num', 4))
    BOND_MIN   = 3.1
    BOND_MAX   = 4.5
    MIN_ISLAND = 3
    MIN_ISLAND_REPLACE = 3
    ANCHOR_N   = 4
    MAX_TAIL   = 10
    DIST_THR   = DIST_THRESHOLD   # module-level constant (2.0 Å)

    def _tm_fit(P, Q):
        """TM-align: returns (u, t) such that P @ u.T + t ≈ Q.  N >= 2."""
        from tmtools import tm_align as _tma
        P = np.asarray(P, dtype=np.float64)
        Q = np.asarray(Q, dtype=np.float64)
        res = _tma(P, Q, 'A' * len(P), 'A' * len(Q))
        return res.u, res.t

    def _bond_ok(a, b):
        return BOND_MIN <= float(np.linalg.norm(
            np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
        )) <= BOND_MAX

    # ── 0a. Build domain instance map ────────────────────────────────────────
    dom_to_inst = {}
    for ce in shared_data.init_connect_result:
        inst = ce['chain_instance_idx']
        for di in ce.get('source_domain_indices', []):
            dom_to_inst[di] = inst

    # ── 0b. Build domain KD-tree and flat af2info list ────────────────────────
    # all_dom_coords[i]  : 3D coord of the i-th domain atom (fitted)
    # all_dom_af2info[i] : (fasta_name, instance_idx, global_seq_pos)
    all_dom_coords  = []
    all_dom_af2info = []   # (fn, inst, sp)

    for di, de in enumerate(shared_data.filtered_domain_list):
        fn = de.get('fasta_name', '')
        if not fn:
            continue
        inst = dom_to_inst.get(di)
        if inst is None:
            continue
        dom_coords = de.get('fitted_coords', [])
        if dom_coords is None or len(dom_coords) == 0:
            continue
        res_idx = _estimate_residue_indices_from_coords(de, fn)
        for lp, sp in enumerate(res_idx):
            if lp < len(dom_coords):
                all_dom_coords.append(np.asarray(dom_coords[lp], dtype=np.float64))
                all_dom_af2info.append((fn, inst, int(sp)))

    if not all_dom_coords:
        print("      WARNING: no domain atoms available for KD-tree.  Skipping repair.")
        return

    dom_tree = cKDTree(np.array(all_dom_coords, dtype=np.float64))
    print(f"      Domain KD-tree: {len(all_dom_coords)} atoms")

    # ── 0c. Map every trace atom to its domain af2info  ──────────────────────
    # Strategy: for each domain position (flat idx), keep only the CLOSEST
    # trace atom from all current trace chains.  Then invert: ca_idx -> af2info.
    #
    # This is a clean re-match that is immune to chain reordering because we
    # use physical coordinates (not stale chain-index/position pairs).

    flat_best = {}   # flat_domain_idx -> (ca_idx, dist)

    all_chains = shared_data.final_traces['chains']
    for chain_idx in range(shared_data.final_traces['num_chains']):
        ca_list = all_chains[chain_idx]['ca_indices']
        for pos, ca_idx in enumerate(ca_list):
            coord = np.asarray(shared_data.ca_pos[ca_idx], dtype=np.float64)
            dist, ni = dom_tree.query(coord, k=1)
            if dist < DIST_THR:
                if ni not in flat_best or dist < flat_best[ni][1]:
                    flat_best[ni] = (ca_idx, dist)

    ca_to_af2 = {}   # ca_idx -> (fn, inst, sp)
    for ni, (ca_idx, _) in flat_best.items():
        ca_to_af2[ca_idx] = all_dom_af2info[ni]

    print(f"      ca_to_af2 entries: {len(ca_to_af2)}")

    # ── 0d. modeling_1 (read-only reference) and modeling_2 (working copy) ───
    modeling_1 = shared_data.init_connect_result
    modeling_2 = copy.deepcopy(modeling_1)
    for e in modeling_2:
        e['fitted_coords'] = [np.asarray(c, dtype=np.float64)
                               for c in e['fitted_coords']]

    # Lookup: (fn, inst, sp) -> (entry_idx, local_idx)  in modeling_2
    def _build_m2_lookup(modeling):
        lkup = {}
        for ei, e in enumerate(modeling):
            fn   = e['fasta_name']
            inst = e['chain_instance_idx']
            for li, sp in enumerate(e['residue_indices']):
                k = (fn, inst, int(sp))
                if k not in lkup:
                    lkup[k] = (ei, li)
        return lkup

    m2_lookup = _build_m2_lookup(modeling_2)

    # ── 0e. Baseline CC for each chain entry ─────────────────────────────────
    print("      Computing baseline CC scores...")
    with Pool(num_procs) as pool:
        baseline_cc = list(pool.map(
            _repair_cc_worker,
            [e['fitted_coords'] for e in modeling_2]
        ))
    print(f"      Baselines: {len(baseline_cc)} entries")

    # ═════════════════════════════════════════════════════════════════════════
    # _build_fragment_data
    # Returns:
    #   island_list  : list of dicts {fasta_name, instance_idx, seq_positions,
    #                                 trace_coords, fragment_id}
    #   bridge_list  : same schema
    #   covered_seqpos : set of (fn, inst, sp) participating in steps 1 + 2
    #   frag_tail    : dict (eidx, side='L'/'R', boundary_sp) -> [coords]
    # ═════════════════════════════════════════════════════════════════════════
    def _build_fragment_data():
        islands = []
        bridges = []
        covered = set()
        ftail   = {}

        for tr_idx in range(shared_data.final_traces['num_chains']):
            cinfo   = shared_data.final_traces['chains'][tr_idx]
            ca_list = cinfo['ca_indices']
            N       = len(ca_list)
            if N < MIN_ISLAND:
                continue

            def tc(pos):
                return np.asarray(shared_data.ca_pos[ca_list[pos]], dtype=np.float64)

            # Per-position AF2 info, using fresh ca_to_af2 map
            pos_af2 = {}   # pos -> (fn, inst, sp)
            for pos in range(N):
                info = ca_to_af2.get(ca_list[pos])
                if info:
                    pos_af2[pos] = info

            if len(pos_af2) < 2:
                continue

            # ── Recon edges: sequence-adjacent matched positions ───────────────
            matched_sorted = sorted(
                [(pos_af2[p], p) for p in pos_af2],
                key=lambda x: (x[0][0], x[0][1], x[0][2])
            )
            recon_adj = {p: set() for p in pos_af2}
            for mi in range(len(matched_sorted) - 1):
                (fn_i, inst_i, sp_i), pi = matched_sorted[mi]
                (fn_j, inst_j, sp_j), pj = matched_sorted[mi + 1]
                if fn_i == fn_j and inst_i == inst_j and sp_j - sp_i == 1:
                    recon_adj[pi].add(pj)
                    recon_adj[pj].add(pi)

            if not any(recon_adj[p] for p in pos_af2):
                continue

            # ── Islands: recon-graph connected components >= MIN_ISLAND ──────
            # IMPORTANT: only atoms belonging to valid islands (size >= MIN_ISLAND)
            # should have their fragment edges removed.  Small spurious components
            # (size < MIN_ISLAND) must NOT cut the fragment graph — doing so would
            # split the bridge path between two real islands, making each half
            # appear to have only one adjacent island and thus getting dropped.
            visited = set()
            ipaths  = []
            p2i     = {}   # pos -> island_id
            valid_island_nodes = set()   # only these atoms cut frag edges
            for seed in sorted(pos_af2):
                if seed in visited:
                    continue
                comp = set()
                q = [seed]
                while q:
                    c = q.pop()
                    if c in comp:
                        continue
                    comp.add(c)
                    q.extend(nb for nb in recon_adj.get(c, set()) if nb not in comp)
                visited.update(comp)
                if len(comp) < MIN_ISLAND:
                    # Small component: clear its recon edges so it doesn't pollute
                    # the recon graph, and do NOT add to valid_island_nodes
                    for p in comp:
                        recon_adj[p] = set()
                    continue
                valid_island_nodes.update(comp)
                # Linearise along recon graph
                subadj = {p: [nb for nb in recon_adj[p] if nb in comp] for p in comp}
                eps    = [p for p in comp if len(subadj[p]) <= 1]
                cp     = min(eps) if eps else min(comp)
                path   = [cp]
                pp     = None
                while True:
                    nx = [nb for nb in subadj[cp] if nb != pp]
                    if not nx:
                        break
                    pp, cp = cp, nx[0]
                    path.append(cp)
                iid = len(ipaths)
                ipaths.append(path)
                for p in path:
                    p2i[p] = iid

            if not ipaths:
                continue

            # ── Remaining fragment adjacency ──────────────────────────────────
            # Only remove fragment edges between atoms that are BOTH in valid islands.
            frag_removed = set()
            for pi in valid_island_nodes:
                for pj in recon_adj[pi]:
                    if pj in valid_island_nodes:
                        lo, hi = min(pi, pj), max(pi, pj)
                        for k in range(lo, hi):
                            frag_removed.add((k, k + 1))

            rfrag = {}
            for k in range(N - 1):
                if (k, k + 1) not in frag_removed:
                    rfrag.setdefault(k,     []).append(k + 1)
                    rfrag.setdefault(k + 1, []).append(k)

            orphaned = set(
                p for p in range(N) if p not in p2i and not rfrag.get(p)
            )

            # ── Store islands ─────────────────────────────────────────────────
            for path in ipaths:
                fn_inst_set = set()
                sp_list = []
                tc_list = []
                for p in path:
                    info = pos_af2.get(p)
                    if info:
                        fn_inst_set.add((info[0], info[1]))
                        sp_list.append(info[2])
                    else:
                        sp_list.append(None)
                    tc_list.append(tc(p))
                if len(fn_inst_set) != 1:
                    continue
                fn, inst = fn_inst_set.pop()
                for sp in sp_list:
                    if sp is not None:
                        covered.add((fn, inst, sp))
                islands.append({
                    'fasta_name':    fn,
                    'instance_idx':  inst,
                    'seq_positions': sp_list,
                    'trace_coords':  tc_list,
                    'fragment_id':   tr_idx,
                })

            # ── Bridges between islands ───────────────────────────────────────
            bvis = set()
            for bp in range(N):
                if bp in p2i or bp in orphaned or bp in bvis:
                    continue
                # BFS over remaining-fragment edges, not crossing islands
                bcomp = set()
                q = [bp]
                while q:
                    c = q.pop()
                    if c in bcomp:
                        continue
                    bcomp.add(c)
                    for nb in rfrag.get(c, []):
                        if nb not in p2i and nb not in orphaned:
                            q.append(nb)
                bvis.update(bcomp)

                bsorted = sorted(bcomp)

                # Collect adjacent island AF2 ranges AND track which bridge atom
                # touches each island — this is the true boundary atom on that side.
                adj_info        = []   # (fn, inst, min_sp, max_sp)
                adj_bridge_atom = {}   # entry -> bridge atom pos (rfrag-adjacent to island)

                for p in bcomp:
                    for nb in rfrag.get(p, []):
                        if nb not in p2i:
                            continue
                        iid   = p2i[nb]
                        sps   = [pos_af2[q][2] for q in ipaths[iid] if q in pos_af2]
                        fn_n  = next((pos_af2[q][0] for q in ipaths[iid] if q in pos_af2), None)
                        ins_n = next((pos_af2[q][1] for q in ipaths[iid] if q in pos_af2), None)
                        if not sps or fn_n is None:
                            continue
                        entry = (fn_n, ins_n, min(sps), max(sps))
                        if entry not in adj_info:
                            adj_info.append(entry)
                        adj_bridge_atom[entry] = p  # last writer fine; bridge is linear

                bfn = binst = None
                bsp = []
                bcoords = None   # will be set below in correct order

                if len(adj_info) >= 2:
                    all_fn  = {e[0] for e in adj_info}
                    all_ins = {e[1] for e in adj_info}
                    if len(all_fn) == 1 and len(all_ins) == 1:
                        left  = sorted(adj_info, key=lambda x: x[3])[0]   # smallest max_sp
                        right = sorted(adj_info, key=lambda x: x[2])[-1]  # largest min_sp
                        if left[3] < right[2]:
                            lmax, rmin = left[3], right[2]
                            bfn, binst = left[0], left[1]
                            es, ee     = lmax + 1, rmin - 1
                            if ee >= es:
                                # Bridge length check deliberately removed.
                                # The original condition required (ee-es+1)==len(bsorted),
                                # which blocked bridges whose trace atom count differs from
                                # the sequence gap size (e.g. one isolated matched atom
                                # inside the bridge adds an extra trace atom).
                                # Step 2 is designed to handle such mismatches via its
                                # mix-splice tasks; enforcing equality here prevents Step 2
                                # from ever seeing the bridge.
                                # When counts differ we trim bcoords symmetrically so that
                                # sp_list and tc_list stay the same length — a requirement
                                # for Step 2's bond-check and CC evaluation logic.
                                bsp = list(range(es, ee + 1))
                                for sp in bsp:
                                    covered.add((bfn, binst, sp))

                                # Order bridge coords left→right in sequence space.
                                left_bridge_end  = adj_bridge_atom.get(left)
                                right_bridge_end = adj_bridge_atom.get(right)

                                if (left_bridge_end is not None
                                        and right_bridge_end is not None
                                        and left_bridge_end != right_bridge_end):
                                    bridge_path = [left_bridge_end]
                                    prev_b, cur_b = None, left_bridge_end
                                    for _ in range(len(bcomp) + 1):
                                        nexts = [nb for nb in rfrag.get(cur_b, [])
                                                 if nb in bcomp and nb != prev_b]
                                        if not nexts:
                                            break
                                        prev_b, cur_b = cur_b, nexts[0]
                                        bridge_path.append(cur_b)
                                    if len(bridge_path) == len(bcomp):
                                        bcoords = [tc(p) for p in bridge_path]

                                # Trim bcoords to match len(bsp) if counts differ.
                                # Drop symmetrically from both ends so the remaining
                                # atoms are the most central (best-connected) ones.
                                if bcoords is not None and len(bcoords) != len(bsp):
                                    n_need = len(bsp)
                                    if len(bcoords) > n_need:
                                        excess  = len(bcoords) - n_need
                                        trim_l  = excess // 2
                                        trim_r  = excess - trim_l
                                        bcoords = bcoords[trim_l: len(bcoords) - trim_r
                                                          if trim_r else len(bcoords)]
                                    else:
                                        # Fewer atoms than positions — cannot fill;
                                        # clear bsp so the bridge is not added.
                                        bsp = []

                if bcoords is None:
                    bcoords = [tc(p) for p in bsorted]

                if bsp:
                    bridges.append({
                        'fasta_name':    bfn,
                        'instance_idx':  binst,
                        'seq_positions': bsp,
                        'trace_coords':  bcoords,
                        'fragment_id':   tr_idx,
                    })

            # ── Tail atoms (adjacent unmatched atoms for Step-3 gap fills) ────
            # anchored = only atoms belonging to valid islands (p2i).
            # Using pos_af2 here would include small-component atoms that are
            # not real islands, causing tail BFS to stop prematurely at those
            # spurious "anchors" and producing truncated (or empty) tails.
            anchored = set(
                p for p in range(N)
                if p not in orphaned and p in p2i
            )
            run_vis = set()
            for ap in sorted(anchored):
                if ap in run_vis:
                    continue
                # BFS of anchored run
                run = set()
                q   = [ap]
                while q:
                    c = q.pop()
                    if c in run:
                        continue
                    run.add(c)
                    for nb in rfrag.get(c, []):
                        if nb in anchored:
                            q.append(nb)
                run_vis.update(run)

                # AF2 info for this run
                sp_run = [pos_af2[p][2] for p in run if p in pos_af2]
                fn_run = next((pos_af2[p][0] for p in run if p in pos_af2), None)
                ins_run = next((pos_af2[p][1] for p in run if p in pos_af2), None)
                if not sp_run or fn_run is None:
                    continue
                min_sp_run, max_sp_run = min(sp_run), max(sp_run)
                run_lo = min(run)
                run_hi = max(run)

                # Right tail: unmatched atoms to the RIGHT of run_hi
                r_tail = []
                cur = run_hi
                for _ in range(MAX_TAIL):
                    nxts = [nb for nb in rfrag.get(cur, [])
                            if nb > cur and nb not in anchored and nb not in orphaned]
                    if not nxts:
                        break
                    cur = nxts[0]
                    r_tail.append(tc(cur))

                # Left tail: unmatched atoms to the LEFT of run_lo
                l_tail = []
                cur = run_lo
                for _ in range(MAX_TAIL):
                    nxts = [nb for nb in rfrag.get(cur, [])
                            if nb < cur and nb not in anchored and nb not in orphaned]
                    if not nxts:
                        break
                    cur = nxts[0]
                    l_tail.insert(0, tc(cur))

                k_r = (fn_run, ins_run, max_sp_run)
                if k_r in m2_lookup:
                    eidx_r = m2_lookup[k_r][0]
                    tid = (eidx_r, 'R', max_sp_run)
                    if tid not in ftail or len(r_tail) > len(ftail[tid]):
                        ftail[tid] = r_tail

                k_l = (fn_run, ins_run, min_sp_run)
                if k_l in m2_lookup:
                    eidx_l = m2_lookup[k_l][0]
                    tid = (eidx_l, 'L', min_sp_run)
                    if tid not in ftail or len(l_tail) > len(ftail[tid]):
                        ftail[tid] = l_tail

        return islands, bridges, covered, ftail

    print("      Extracting island/bridge data from trace chains...")
    island_list, bridge_list, covered_seqpos, frag_tail = _build_fragment_data()
    print(f"      Islands: {len(island_list)}, bridges: {len(bridge_list)}, "
          f"covered: {len(covered_seqpos)}")

    # ═════════════════════════════════════════════════════════════════════════
    # PRE-STEP: Global island-CA map + jurisdiction
    # ═════════════════════════════════════════════════════════════════════════
    print("      Building global island-CA map (conflict → closer to modeling_2)...")

    # island_global_map[(fn, inst, sp)] = coord
    # Conflict resolution: keep the atom physically closer to the current
    # modeling_2 coord at that (fn, inst, sp) position.
    island_global_map  = {}   # (fn, inst, sp) -> np.ndarray coord
    island_global_dist = {}   # (fn, inst, sp) -> float  (lower = better)

    for isl in island_list:
        fn      = isl['fasta_name']
        inst    = isl['instance_idx']
        sp_list = isl['seq_positions']
        tc_list = isl['trace_coords']
        for sp, coord in zip(sp_list, tc_list):
            if sp is None:
                continue
            key    = (fn, inst, sp)
            coord  = np.asarray(coord, dtype=np.float64)
            # Reference: current modeling_2 coord at this sequence position
            if key in m2_lookup:
                ei_r, li_r = m2_lookup[key]
                ref  = modeling_2[ei_r]['fitted_coords'][li_r]
                dist = float(np.linalg.norm(coord - ref))
            else:
                dist = float('inf')
            if key not in island_global_map or dist < island_global_dist[key]:
                island_global_map[key]  = coord
                island_global_dist[key] = dist

    island_positions = set(island_global_map.keys())
    print(f"      Island positions in global map: {len(island_positions)}")

    # Jurisdiction: for each fragment, the span from its first island sp to
    # its last island sp (inclusive) is fully covered → Step 3 skips it.
    frag_sp_range = {}   # (fn, inst, frag_id) -> (min_sp, max_sp)
    for isl in island_list:
        fn   = isl['fasta_name']
        inst = isl['instance_idx']
        fid  = isl['fragment_id']
        sps  = [sp for sp in isl['seq_positions'] if sp is not None]
        if not sps:
            continue
        key = (fn, inst, fid)
        cur = frag_sp_range.get(key, (float('inf'), float('-inf')))
        frag_sp_range[key] = (min(cur[0], min(sps)), max(cur[1], max(sps)))

    juris_count = 0
    for (fn, inst, _), (mn, mx) in frag_sp_range.items():
        for sp in range(int(mn), int(mx) + 1):
            covered_seqpos.add((fn, inst, sp))
            juris_count += 1
    print(f"      Jurisdiction adds {juris_count} positions "
          f"from {len(frag_sp_range)} fragment(s)")

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 1: Connected-block island replacement
    # ═════════════════════════════════════════════════════════════════════════
    print("      Step 1: connected-block island replacement...")

    # Group global island positions by (eidx, local_idx)
    eidx_island_locs = defaultdict(dict)   # eidx -> {li: coord}
    for key, coord in island_global_map.items():
        if key in m2_lookup:
            ei, li = m2_lookup[key]
            eidx_island_locs[ei][li] = coord

    # Form contiguous blocks per chain entry, build one CC task per block
    s1_tasks = []   # (eidx, block_lis, full_cand)
    for ei, li_coord in eidx_island_locs.items():
        lis = sorted(li_coord)
        if not lis:
            continue
        # Split into contiguous runs
        blocks = []
        blk = [lis[0]]
        for k in range(1, len(lis)):
            if lis[k] == lis[k - 1] + 1:
                blk.append(lis[k])
            else:
                blocks.append(blk)
                blk = [lis[k]]
        blocks.append(blk)

        for blk in blocks:
            cand = [c.copy() for c in modeling_2[ei]['fitted_coords']]
            for li in blk:
                cand[li] = li_coord[li]
            s1_tasks.append((ei, blk, cand))

    if s1_tasks:
        with Pool(num_procs) as pool:
            s1_cc = list(pool.map(_repair_cc_worker,
                                  [c for _, _, c in s1_tasks]))

        # Collect improving blocks per entry; blocks within one entry are
        # disjoint so we can apply them all safely.
        s1_improvements = defaultdict(list)   # eidx -> [(cc, blk, cand)]
        for (ei, blk, cand), cc in zip(s1_tasks, s1_cc):
            if cc > baseline_cc[ei]:
                s1_improvements[ei].append((cc, blk, cand))

        for ei, imp_list in s1_improvements.items():
            for _, blk, cand in imp_list:
                for li in blk:
                    modeling_2[ei]['fitted_coords'][li] = cand[li]
            best = max(cc for cc, _, _ in imp_list)
            baseline_cc[ei] = best
            print(f"        Step1 chain {ei}: "
                  f"{len(imp_list)} block(s) applied, best cc={best:.4f}")

    # Export step-1 snapshot for inspection
    shared_data.init_connect_result = modeling_2
    try:
        _out = getattr(shared_data.config, 'output_dir', '.')
        export_init_connect_result_to_pdb(
            _out, filename_override='init_modeling_after_step1.pdb'
        )
    except Exception as _e:
        print(f"      [Step1 PDB export failed: {_e}]")
    shared_data.init_connect_result = modeling_1   # restore; modeling_2 continues

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 2: Bridge replacement (bond check + island protection)
    # ═════════════════════════════════════════════════════════════════════════
    print("      Step 2: bridge replacement (bond check + island protection)...")

    _out_dir = getattr(shared_data.config, 'output_dir', '.')
    _log_path = os.path.join(_out_dir, 'step2_bridge_log.txt')
    _log_lines = []

    def _log(msg):
        _log_lines.append(msg)

    _log("=" * 100)
    _log(f"STEP 2 BRIDGE REPLACEMENT LOG")
    _log(f"Total bridges in bridge_list: {len(bridge_list)}")
    _log("=" * 100)

    s2_tasks = []      # (ei, rep_list[(li, coord)], full_cand)
    s2_task_meta = []  # (br_idx, ei) for logging

    for br_idx, br in enumerate(bridge_list):
        fn      = br['fasta_name']
        inst    = br['instance_idx']
        sp_list = br['seq_positions']
        tc_list = br['trace_coords']
        fid     = br.get('fragment_id', '?')

        _log(f"\n--- Bridge #{br_idx} | fragment_id={fid} | fn={fn} inst={inst} ---")
        _log(f"    seq_positions : {sp_list}  (E res {[s+1 for s in sp_list] if sp_list else []})")
        _log(f"    trace_coords  : {len(tc_list)} atoms")

        if not fn or inst is None or not sp_list or len(sp_list) != len(tc_list):
            _log(f"    SKIP: invalid bridge metadata "
                 f"(fn={fn!r}, inst={inst}, sp_list={sp_list}, len_match={len(sp_list)==len(tc_list)})")
            continue

        # Collect (li, coord) pairs for each chain entry
        reps = defaultdict(list)
        for sp, coord in zip(sp_list, tc_list):
            k = (fn, inst, sp)
            if k in m2_lookup:
                ei, li = m2_lookup[k]
                reps[ei].append((li, np.asarray(coord, dtype=np.float64)))
            else:
                _log(f"    WARN: sp={sp} (E res {sp+1}) key={k} not found in m2_lookup")

        if not reps:
            _log(f"    SKIP: no positions found in m2_lookup for any sp")
            continue

        for ei, rep_list in reps.items():
            rep_list.sort(key=lambda x: x[0])
            ri      = modeling_2[ei]['residue_indices']
            fn_ei   = modeling_2[ei]['fasta_name']
            ins_ei  = modeling_2[ei]['chain_instance_idx']
            m2c     = modeling_2[ei]['fitted_coords']
            n_res   = len(m2c)
            min_li  = rep_list[0][0]
            max_li  = rep_list[-1][0]
            seq_sps = [int(ri[li]) for li, _ in rep_list]

            _log(f"    → chain entry ei={ei} (fn={fn_ei}, inst={ins_ei})")
            _log(f"      local_indices : {[li for li, _ in rep_list]}")
            _log(f"      seq_positions : {seq_sps}  (res {[s+1 for s in seq_sps]})")

            # Island protection check
            island_hits = [(fn_ei, ins_ei, int(ri[li])) for li, _ in rep_list
                           if (fn_ei, ins_ei, int(ri[li])) in island_positions]
            if island_hits:
                _log(f"      SKIP (island protection): positions {island_hits} are island CAs")
                continue

            # Bond check setup
            l_sp = int(ri[min_li - 1]) if min_li > 0 else None
            r_sp = int(ri[max_li + 1]) if max_li < n_res - 1 else None
            l_island_coord = island_global_map.get((fn_ei, ins_ei, l_sp)) if l_sp is not None else None
            r_island_coord = island_global_map.get((fn_ei, ins_ei, r_sp)) if r_sp is not None else None

            l_anchor_src = "island_map" if l_island_coord is not None else "m2c"
            r_anchor_src = "island_map" if r_island_coord is not None else "m2c"
            _log(f"      left anchor : sp={l_sp} (res {l_sp+1 if l_sp is not None else '?'}), "
                 f"src={l_anchor_src}")
            _log(f"      right anchor: sp={r_sp} (res {r_sp+1 if r_sp is not None else '?'}), "
                 f"src={r_anchor_src}")

            # Left bond check
            if min_li > 0:
                l_anchor = l_island_coord if l_island_coord is not None else m2c[min_li - 1]
                l_d = float(np.linalg.norm(
                    np.asarray(l_anchor) - np.asarray(rep_list[0][1])))
                l_ok = BOND_MIN <= l_d <= BOND_MAX
                _log(f"      left bond : anchor→bridge[0] = {l_d:.2f}Å "
                     f"[{BOND_MIN},{BOND_MAX}] → {'OK' if l_ok else 'FAIL'}")
                if not l_ok:
                    _log(f"      SKIP (left bond check failed: {l_d:.2f}Å)")
                    continue
            else:
                _log(f"      left bond : skipped (bridge is at chain start)")

            # Right bond check
            if max_li < n_res - 1:
                r_anchor = r_island_coord if r_island_coord is not None else m2c[max_li + 1]
                r_d = float(np.linalg.norm(
                    np.asarray(rep_list[-1][1]) - np.asarray(r_anchor)))
                r_ok = BOND_MIN <= r_d <= BOND_MAX
                _log(f"      right bond: bridge[-1]→anchor = {r_d:.2f}Å "
                     f"[{BOND_MIN},{BOND_MAX}] → {'OK' if r_ok else 'FAIL'}")
                if not r_ok:
                    _log(f"      SKIP (right bond check failed: {r_d:.2f}Å)")
                    continue
            else:
                _log(f"      right bond: skipped (bridge is at chain end)")

            _log(f"      → QUEUED for CC evaluation (task idx={len(s2_tasks)})")
            cand = [c.copy() for c in m2c]
            for li, coord in rep_list:
                cand[li] = coord
            s2_tasks.append((ei, rep_list, cand))
            s2_task_meta.append((br_idx, ei))

    _log(f"\n{'='*100}")
    _log(f"Total tasks queued for CC evaluation: {len(s2_tasks)}")

    if s2_tasks:
        with Pool(num_procs) as pool:
            s2_cc = list(pool.map(_repair_cc_worker,
                                  [c for _, _, c in s2_tasks]))

        _log(f"\n--- CC Results ---")
        s2_best = {}
        for tidx, ((ei, rep_list, _), cc) in enumerate(zip(s2_tasks, s2_cc)):
            br_idx, _ei = s2_task_meta[tidx]
            seq_sps = [int(modeling_2[ei]['residue_indices'][li]) for li, _ in rep_list]
            improved = cc > baseline_cc[ei]
            _log(f"  task {tidx} | bridge#{br_idx} | ei={ei} | "
                 f"sp={seq_sps} (res {[s+1 for s in seq_sps]}) | "
                 f"cc={cc:.4f} vs baseline={baseline_cc[ei]:.4f} | "
                 f"{'IMPROVE' if improved else 'NO_IMPROVE'}")
            if improved:
                for li, coord in rep_list:
                    k = (ei, li)
                    if k not in s2_best or cc > s2_best[k][0]:
                        s2_best[k] = (cc, coord)

        applied_ei = set()
        for (ei, li), (cc, coord) in s2_best.items():
            modeling_2[ei]['fitted_coords'][li] = coord
            baseline_cc[ei] = max(baseline_cc[ei], cc)
            applied_ei.add(ei)
        for ei in sorted(applied_ei):
            print(f"        Step2 chain {ei} improved")
            _log(f"\n  APPLIED: chain entry ei={ei}, new_cc={baseline_cc[ei]:.4f}")

        if not applied_ei:
            _log(f"\n  No improvements applied.")

    else:
        _log(f"\n  No tasks to evaluate.")

    # Write log
    try:
        os.makedirs(_out_dir, exist_ok=True)
        with open(_log_path, 'w', encoding='utf-8') as _lf:
            _lf.write('\n'.join(_log_lines) + '\n')
        print(f"      Step 2 log written to: {_log_path}")
    except Exception as _le:
        print(f"      [Step2 log write failed: {_le}]")

    # Export step-2 snapshot for inspection
    shared_data.init_connect_result = modeling_2
    try:
        export_init_connect_result_to_pdb(
            _out_dir, filename_override='init_modeling_after_step2.pdb'
        )
    except Exception as _e:
        print(f"      [Step2 PDB export failed: {_e}]")
    shared_data.init_connect_result = modeling_1   # restore; modeling_2 continues

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 3: Cross-fragment gap filling (outside all jurisdictions)
    # ═════════════════════════════════════════════════════════════════════════
    print("      Step 3: cross-fragment gap filling...")

    s3_tasks   = []
    s3_groups  = {}   # (eidx, gid) -> [task_indices]
    s3_gap_gis = {}   # (eidx, gid) -> [local_indices]
    gid_ctr    = [0]

    for eidx, entry in enumerate(modeling_2):
        fn      = entry['fasta_name']
        res_idx = entry['residue_indices']
        n_res   = len(res_idx)
        if n_res == 0:
            continue

        inst = entry['chain_instance_idx']

        # Local indices not covered by steps 1/2
        covered_local = set(
            i for i, sp in enumerate(res_idx)
            if (fn, inst, sp) in covered_seqpos
        )

        # Find contiguous gap runs
        gaps = []
        i = 0
        while i < n_res:
            if i not in covered_local:
                j = i
                while j < n_res and j not in covered_local:
                    j += 1
                gaps.append((i, j - 1))
                i = j
            else:
                i += 1
        if not gaps:
            continue

        m2c = [c.copy() for c in modeling_2[eidx]['fitted_coords']]
        m1c = [np.asarray(modeling_1[eidx]['fitted_coords'][i], dtype=np.float64)
               for i in range(len(modeling_1[eidx]['fitted_coords']))]

        def _add_s3(gid, label, gi_list, gap_coords):
            if len(gap_coords) != len(gi_list):
                return
            full = [c.copy() for c in m2c]
            for ii, gi in enumerate(gi_list):
                full[gi] = np.asarray(gap_coords[ii], dtype=np.float64)
            tidx = len(s3_tasks)
            s3_tasks.append((eidx, gid, label, full))
            s3_groups.setdefault((eidx, gid), []).append(tidx)

        for gs, ge in gaps:
            gi_list = list(range(gs, ge + 1))
            n_gap   = len(gi_list)
            gid     = gid_ctr[0]
            gid_ctr[0] += 1
            s3_gap_gis[(eidx, gid)] = gi_list

            L_li = gs - 1
            R_li = ge + 1
            L_sp = res_idx[L_li] if L_li >= 0 else None
            R_sp = res_idx[R_li] if R_li < n_res else None

            # Task 0: current state (baseline candidate)
            _add_s3(gid, 'original', gi_list, [m2c[gi] for gi in gi_list])

            # ── Left-anchor tasks ────────────────────────────────────────────
            if L_li >= 0:
                la_ii  = list(range(max(0, L_li - ANCHOR_N + 1), L_li + 1))
                m1_la  = np.array([m1c[i] for i in la_ii])
                m2_la  = np.array([m2c[i] for i in la_ii])
                m1_gap = np.array([m1c[gi] for gi in gi_list])
                l_anc  = np.asarray(m2c[L_li], dtype=np.float64)

                # left_fit_a: TM-align m1 left-anchor → m2 left-anchor
                if len(la_ii) >= 2:
                    try:
                        u, t = _tm_fit(m1_la, m2_la)
                        cand = list(m1_gap @ u.T + t)
                        if (_bond_ok(l_anc, cand[0]) and
                                (R_li >= n_res or _bond_ok(cand[-1], m2c[R_li]))):
                            _add_s3(gid, 'left_fit_a', gi_list, cand)
                    except Exception:
                        pass

                l_tail = frag_tail.get((eidx, 'R', L_sp), [])

                # left_fit_b: TM-align m1 gap-head → left tail
                k_tail = min(len(l_tail), n_gap)
                if k_tail >= 2:
                    try:
                        m1_gk  = np.array([m1c[gi_list[ii]] for ii in range(k_tail)])
                        u, t   = _tm_fit(m1_gk, np.array(l_tail[:k_tail]))
                        cand   = list(m1_gap @ u.T + t)
                        if (_bond_ok(l_anc, cand[0]) and
                                (R_li >= n_res or _bond_ok(cand[-1], m2c[R_li]))):
                            _add_s3(gid, 'left_fit_b', gi_list, cand)
                    except Exception:
                        pass

                # left_mix_n: splice n tail atoms into gap start
                for i_t in range(min(len(l_tail), MAX_TAIL)):
                    n_ins    = i_t + 1
                    tail_seg = l_tail[:n_ins]
                    rest_m2  = [m2c[gi_list[j]] for j in range(n_ins, n_gap)]
                    cand     = list(tail_seg) + rest_m2
                    if len(cand) != n_gap:
                        continue
                    if not _bond_ok(l_anc, tail_seg[0]):
                        continue
                    if n_ins >= n_gap:
                        if R_li < n_res and not _bond_ok(tail_seg[-1], m2c[R_li]):
                            continue
                    elif R_sp is None:
                        if rest_m2 and not _bond_ok(tail_seg[-1], rest_m2[0]):
                            continue
                    else:
                        if not _bond_ok(tail_seg[-1], m2c[gi_list[n_ins]]):
                            continue
                    _add_s3(gid, f'left_mix_{n_ins}', gi_list, cand)

            # ── Right-anchor tasks ───────────────────────────────────────────
            if R_li < n_res:
                ra_ii  = list(range(R_li, min(n_res, R_li + ANCHOR_N)))
                m1_ra  = np.array([m1c[i] for i in ra_ii])
                m2_ra  = np.array([m2c[i] for i in ra_ii])
                m1_gap = np.array([m1c[gi] for gi in gi_list])
                r_anc  = np.asarray(m2c[R_li], dtype=np.float64)

                # right_fit_a: TM-align m1 right-anchor → m2 right-anchor
                if len(ra_ii) >= 2:
                    try:
                        u, t = _tm_fit(m1_ra, m2_ra)
                        cand = list(m1_gap @ u.T + t)
                        if (_bond_ok(cand[-1], r_anc) and
                                (L_li < 0 or _bond_ok(m2c[L_li], cand[0]))):
                            _add_s3(gid, 'right_fit_a', gi_list, cand)
                    except Exception:
                        pass

                r_tail = frag_tail.get((eidx, 'L', R_sp), [])

                # right_fit_b: TM-align m1 gap-tail → right tail
                k_tail = min(len(r_tail), n_gap)
                if k_tail >= 2:
                    try:
                        m1_gk = np.array([m1c[gi_list[n_gap - k_tail + ii]]
                                          for ii in range(k_tail)])
                        u, t  = _tm_fit(m1_gk, np.array(r_tail[-k_tail:]))
                        cand  = list(m1_gap @ u.T + t)
                        if (_bond_ok(cand[-1], r_anc) and
                                (L_li < 0 or _bond_ok(m2c[L_li], cand[0]))):
                            _add_s3(gid, 'right_fit_b', gi_list, cand)
                    except Exception:
                        pass

                # right_mix_n: splice n tail atoms into gap end
                for j_t in range(min(len(r_tail), MAX_TAIL)):
                    n_ins    = j_t + 1
                    tail_seg = r_tail[-n_ins:]
                    front_m2 = [m2c[gi_list[ii]] for ii in range(n_gap - n_ins)]
                    cand     = front_m2 + list(tail_seg)
                    if len(cand) != n_gap:
                        continue
                    if not _bond_ok(tail_seg[-1], r_anc):
                        continue
                    if n_gap - n_ins <= 0:
                        if L_li >= 0 and not _bond_ok(m2c[L_li], tail_seg[0]):
                            continue
                    elif L_sp is None:
                        if front_m2 and not _bond_ok(front_m2[-1], tail_seg[0]):
                            continue
                    else:
                        prev = front_m2[-1] if front_m2 else m2c[gi_list[0]]
                        if not _bond_ok(prev, tail_seg[0]):
                            continue
                    _add_s3(gid, f'right_mix_{n_ins}', gi_list, cand)

    if s3_tasks:
        print(f"      Step 3: evaluating {len(s3_tasks)} candidates...")
        with Pool(num_procs) as pool:
            s3_cc = list(pool.map(_repair_cc_worker,
                                  [full for _, _, _, full in s3_tasks]))

        applied = 0
        for (eidx, gid), tidxs in s3_groups.items():
            if not tidxs:
                continue
            best_tidx = max(tidxs, key=lambda ti: s3_cc[ti])
            best_cc   = s3_cc[best_tidx]
            if best_cc > baseline_cc[eidx]:
                _, _, label, best_full = s3_tasks[best_tidx]
                for gi in s3_gap_gis.get((eidx, gid), []):
                    modeling_2[eidx]['fitted_coords'][gi] = best_full[gi]
                baseline_cc[eidx] = best_cc
                applied += 1
                print(f"        Step3 chain {eidx} gap {gid}: "
                      f"cc={best_cc:.4f} ({label})")
        print(f"      Step 3 applied {applied} improvements.")

    # ── Finalise ──────────────────────────────────────────────────────────────
    for e in modeling_2:
        e['fitted_coords'] = [
            c.tolist() if hasattr(c, 'tolist') else list(c)
            for c in e['fitted_coords']
        ]

    shared_data.init_connect_result = modeling_2

    elapsed = record_time('repair_chains_with_fragments', start_time)
    print(f"      Completed in {elapsed:.2f}s")
    print(f"      Final chain entries: {len(modeling_2)}")

# ═══════════════════════════════════════════════════════════════════════════════
# PDB export
# ═══════════════════════════════════════════════════════════════════════════════

def export_init_connect_result_to_pdb(output_path, filename_override=None):
    global shared_data
    from utils.EMtools import chainID_list

    if not hasattr(shared_data, 'init_connect_result') or not shared_data.init_connect_result:
        print("Error: init_connect_result is empty or not found.")
        return False

    os.makedirs(output_path, exist_ok=True)
    fname = filename_override if filename_override else 'init_modeling_connected.pdb'
    pdb_file_path = os.path.join(output_path, fname)

    try:
        with open(pdb_file_path, 'w') as pdb_file:
            pdb_file.write("REMARK   init_modeling connected backbone\n")
            pdb_file.write(f"REMARK   Total chain entries: "
                           f"{len(shared_data.init_connect_result)}\n")
            pdb_file.write("REMARK\n")

            for entry_idx, entry in enumerate(shared_data.init_connect_result):
                fasta_name   = entry['fasta_name']
                domain_ids   = entry.get('domain_ids', [])
                n_residues   = len(entry['fitted_coords'])
                cc_mask      = entry.get('cc_mask', 0.0)
                is_merged    = entry.get('is_merged', False)
                inst_idx     = entry.get('chain_instance_idx', entry_idx)
                chain_id     = chainID_list[entry_idx % len(chainID_list)]

                res_indices = entry.get('residue_indices', [])
                res_range = f"[{res_indices[0]}..{res_indices[-1]}]" if res_indices else "[?]"

                pdb_file.write(
                    f"REMARK   Chain {chain_id}: {fasta_name} inst={inst_idx}, "
                    f"domains={domain_ids}, "
                    f"residues={n_residues} {res_range}, "
                    f"cc_mask={cc_mask:.4f}, "
                    f"merged={is_merged}\n"
                )

            pdb_file.write("REMARK\n")

            atom_serial = 1

            for entry_idx, entry in enumerate(shared_data.init_connect_result):
                fasta_name    = entry['fasta_name']
                fitted_coords = entry['fitted_coords']
                res_indices   = entry.get('residue_indices', [])
                cc_mask       = entry.get('cc_mask', 0.0)
                chain_id      = chainID_list[entry_idx % len(chainID_list)]

                fasta_sequence = None
                if fasta_name in shared_data.fastas:
                    fasta_sequence = shared_data.fastas[fasta_name].sequence.upper()

                last_aa_name = 'ALA'
                res_num_pdb  = 1

                for local_res_num in range(len(fitted_coords)):
                    coord = fitted_coords[local_res_num]
                    x, y, z = float(coord[0]), float(coord[1]), float(coord[2])

                    aa_name = 'ALA'
                    if (fasta_sequence is not None and
                            local_res_num < len(res_indices)):
                        seq_idx = res_indices[local_res_num]
                        if 0 <= seq_idx < len(fasta_sequence):
                            aa_letter = fasta_sequence[seq_idx]
                            aa_name = abb2AA.get(aa_letter, 'ALA')

                    if local_res_num < len(res_indices):
                        res_num = res_indices[local_res_num] + 1
                    else:
                        res_num = local_res_num + 1

                    res_num_pdb = ((res_num - 1) % 9999) + 1

                    pdb_file.write(
                        f"ATOM  {atom_serial:5d}  CA  "
                        f"{aa_name:>3s} {chain_id:1s}"
                        f"{res_num_pdb:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}"
                        f"{1.00:6.2f}{cc_mask:6.2f}"
                        f"           C\n"
                    )
                    last_aa_name = aa_name
                    atom_serial += 1

                pdb_file.write(
                    f"TER   {atom_serial:5d}      "
                    f"{last_aa_name:>3s} {chain_id:1s}"
                    f"{res_num_pdb:4d}\n"
                )
                atom_serial += 1

            pdb_file.write("END\n")

        num_entries   = len(shared_data.init_connect_result)
        total_atoms   = sum(len(e['fitted_coords']) for e in shared_data.init_connect_result)
        chain_lengths = [len(e['fitted_coords']) for e in shared_data.init_connect_result]

        print(f"\n  ✓ Exported init_connect_result to PDB")
        print(f"    Output file : {pdb_file_path}")
        print(f"    Chain entries: {num_entries}")
        print(f"    Total atoms  : {total_atoms}")
        if chain_lengths:
            print(f"    Shortest chain: {min(chain_lengths)} residues")
            print(f"    Longest chain : {max(chain_lengths)} residues")
            print(f"    Average length: {np.mean(chain_lengths):.1f} residues")

        return True

    except Exception as e:
        print(f"Error exporting init_connect_result to PDB: {e}")
        import traceback
        traceback.print_exc()
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def Backbone_modeling():
    global shared_data

    print("=" * 60)
    print("[Backbone_modeling] Starting backbone modelling pipeline")
    print("=" * 60)

    overall_start = time.time()

    if not hasattr(shared_data, 'final_traces') or shared_data.final_traces is None:
        raise RuntimeError("shared_data.final_traces is not set.")
    if not hasattr(shared_data, 'ca_pos') or shared_data.ca_pos is None:
        raise RuntimeError("shared_data.ca_pos is not set.")
    if not hasattr(shared_data, 'filtered_domain_list') or \
       not shared_data.filtered_domain_list:
        raise RuntimeError("shared_data.filtered_domain_list is empty.")

    num_traces  = shared_data.final_traces['num_chains']
    num_domains = len(shared_data.filtered_domain_list)
    print(f"  Input: {num_traces} trace chains, {num_domains} filtered domains")

    # Part 1: Build trace ↔ domain correspondences
    find_corresponding_CA()

    # Part 2: Build chain models (Union-Find, cc_mask priority)
    build_chain_models()

    # Part 2.5: AF2-guided intra-fragment reconstruction
    reconstruct_fragments_with_af2()

    # Part 3: Fragment-based repair at domain junctions
    repair_chains_with_fragments()

    # Export
    output_dir = getattr(shared_data.config, 'output_dir', '.')
    export_init_connect_result_to_pdb(output_dir)

    overall_elapsed = time.time() - overall_start
    print(f"\n[Backbone_modeling] Finished in {overall_elapsed:.2f}s")
    print(f"  Output: {len(shared_data.init_connect_result)} chain entries")
    print("=" * 60)
