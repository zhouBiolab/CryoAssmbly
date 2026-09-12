#!/usr/bin/env python3
"""
结构域分析工具模块（修正版 - 支持PDB和CIF格式）
包含：
1. 相邻结构域之间的距离能量计算函数（支持片段断开）
2. 链-链之间结构域相似度判断函数
3. 跨文件夹TM-score批量计算函数
4. 距离阈值评分函数

支持格式：PDB (.pdb) 和 mmCIF (.cif)
"""

import os
import subprocess
import re
import glob
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from collections import defaultdict
from itertools import combinations
from protassem.assembly.refine.tr_rmsd import calculate_and_align_with_sequence


@dataclass
class Atom:
    """原子数据结构"""
    atom_id: int
    atom_name: str
    res_name: str
    chain_id: str
    res_id: int
    x: float
    y: float
    z: float
    occupancy: float = 1.0
    b_factor: float = 0.0
    element: str = ""
    charge: str = ""
    record_type: str = "ATOM"

    @property
    def coord(self) -> np.ndarray:
        """获取坐标向量"""
        return np.array([self.x, self.y, self.z])


@dataclass
class Residue:
    """残基数据结构"""
    res_id: int
    res_name: str
    chain_id: str
    atoms: List[Atom]

    @property
    def ca_coord(self) -> Optional[np.ndarray]:
        """获取CA原子坐标"""
        for atom in self.atoms:
            if atom.atom_name.strip() == "CA":
                return atom.coord
        return None

    @property
    def has_ca(self) -> bool:
        """判断是否有CA原子"""
        return self.ca_coord is not None


@dataclass
class DomainSegment:
    """结构域片段信息"""
    start_res: int
    end_res: int
    start_ca: np.ndarray
    end_ca: np.ndarray
    start_res_name: str
    end_res_name: str


def read_structure_file(structure_file: str) -> Tuple[List[Atom], List[Residue]]:
    """
    读取结构文件（PDB或CIF），返回原子列表和残基列表

    参数:
        structure_file: 结构文件路径（.pdb或.cif）

    返回:
        (all_atoms, residues): 原子列表和残基列表的元组
    """
    if not os.path.exists(structure_file):
        raise FileNotFoundError(f"结构文件不存在: {structure_file}")
    
    # 根据文件扩展名判断格式
    file_ext = os.path.splitext(structure_file)[1].lower()
    
    if file_ext == '.pdb':
        return _read_pdb_file(structure_file)
    elif file_ext == '.cif':
        return _read_cif_file(structure_file)
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}，仅支持.pdb和.cif")


def _read_pdb_file(pdb_file: str) -> Tuple[List[Atom], List[Residue]]:
    """
    读取PDB文件，返回原子列表和残基列表

    参数:
        pdb_file: PDB文件路径

    返回:
        (all_atoms, residues): 原子列表和残基列表的元组
    """
    all_atoms = []
    residue_dict = {}

    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                atom = _parse_atom_line(line)
                if atom:
                    all_atoms.append(atom)
                    if atom.res_id not in residue_dict:
                        residue_dict[atom.res_id] = Residue(
                            res_id=atom.res_id,
                            res_name=atom.res_name,
                            chain_id=atom.chain_id,
                            atoms=[]
                        )
                    residue_dict[atom.res_id].atoms.append(atom)

    residues = list(residue_dict.values())
    residues.sort(key=lambda r: r.res_id)

    return all_atoms, residues


def _read_cif_file(cif_file: str) -> Tuple[List[Atom], List[Residue]]:
    """
    读取CIF文件，返回原子列表和残基列表

    参数:
        cif_file: CIF文件路径

    返回:
        (all_atoms, residues): 原子列表和残基列表的元组
    """
    try:
        from Bio.PDB import MMCIFParser
    except ImportError:
        raise ImportError("需要安装BioPython来读取CIF文件: pip install biopython")
    
    all_atoms = []
    residue_dict = {}
    
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('structure', cif_file)
    
    atom_id = 1
    for model in structure:
        for chain in model:
            chain_id = chain.id
            for residue in chain:
                # 跳过HETATM（水分子等）
                if residue.id[0] != ' ':
                    continue
                    
                res_id = residue.id[1]
                res_name = residue.resname
                
                for atom_obj in residue:
                    atom_name = atom_obj.name
                    coord = atom_obj.coord
                    occupancy = atom_obj.occupancy
                    b_factor = atom_obj.bfactor
                    element = atom_obj.element
                    
                    atom = Atom(
                        atom_id=atom_id,
                        atom_name=atom_name,
                        res_name=res_name,
                        chain_id=chain_id,
                        res_id=res_id,
                        x=coord[0],
                        y=coord[1],
                        z=coord[2],
                        occupancy=occupancy,
                        b_factor=b_factor,
                        element=element,
                        charge="",
                        record_type="ATOM"
                    )
                    
                    all_atoms.append(atom)
                    
                    if res_id not in residue_dict:
                        residue_dict[res_id] = Residue(
                            res_id=res_id,
                            res_name=res_name,
                            chain_id=chain_id,
                            atoms=[]
                        )
                    residue_dict[res_id].atoms.append(atom)
                    
                    atom_id += 1
    
    residues = list(residue_dict.values())
    residues.sort(key=lambda r: r.res_id)
    
    return all_atoms, residues


def read_pdb_structure(pdb_file: str) -> Tuple[List[Atom], List[Residue]]:
    """
    读取PDB文件，返回原子列表和残基列表
    （保持向后兼容，实际调用read_structure_file）

    参数:
        pdb_file: PDB文件路径（也支持CIF文件）

    返回:
        (all_atoms, residues): 原子列表和残基列表的元组
    """
    return read_structure_file(pdb_file)


def _parse_atom_line(line: str) -> Optional[Atom]:
    """解析PDB ATOM/HETATM行"""
    try:
        record_type = line[0:6].strip()
        atom_id = int(line[6:11].strip())
        atom_name = line[12:16].strip()
        res_name = line[17:20].strip()
        chain_id = line[21:22].strip() if len(line) > 21 else 'A'
        res_id = int(line[22:26].strip())
        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())
        occupancy = float(line[54:60].strip()) if len(line) > 54 and line[54:60].strip() else 1.0
        b_factor = float(line[60:66].strip()) if len(line) > 60 and line[60:66].strip() else 0.0
        element = line[76:78].strip() if len(line) > 76 else ""
        charge = line[78:80].strip() if len(line) > 78 else ""

        return Atom(
            atom_id=atom_id, atom_name=atom_name, res_name=res_name, chain_id=chain_id,
            res_id=res_id, x=x, y=y, z=z, occupancy=occupancy, b_factor=b_factor,
            element=element, charge=charge, record_type=record_type
        )
    except:
        return None


def get_domain_segments(structure_file: str) -> List[DomainSegment]:
    """
    获取结构域的所有片段信息（处理断开的结构域）
    支持PDB和CIF格式

    参数:
        structure_file: 结构文件路径（.pdb或.cif）

    返回:
        DomainSegment列表，包含每个片段的起始/结束残基号和CA坐标
    """
    _, residues = read_structure_file(structure_file)

    # 只保留有CA原子的残基
    ca_residues = [r for r in residues if r.has_ca]
    if not ca_residues:
        print(f"警告: {structure_file} 中没有找到CA原子")
        return []

    # 按残基号排序
    ca_residues.sort(key=lambda r: r.res_id)

    # 识别连续的片段
    segments = []
    segment_residues = [ca_residues[0]]

    for i in range(1, len(ca_residues)):
        if ca_residues[i].res_id == segment_residues[-1].res_id + 1:
            # 连续，添加到当前片段
            segment_residues.append(ca_residues[i])
        else:
            # 不连续，保存当前片段并开始新片段
            if segment_residues:
                segments.append(DomainSegment(
                    start_res=segment_residues[0].res_id,
                    end_res=segment_residues[-1].res_id,
                    start_ca=segment_residues[0].ca_coord,
                    end_ca=segment_residues[-1].ca_coord,
                    start_res_name=segment_residues[0].res_name,
                    end_res_name=segment_residues[-1].res_name
                ))
            segment_residues = [ca_residues[i]]

    # 保存最后一个片段
    if segment_residues:
        segments.append(DomainSegment(
            start_res=segment_residues[0].res_id,
            end_res=segment_residues[-1].res_id,
            start_ca=segment_residues[0].ca_coord,
            end_ca=segment_residues[-1].ca_coord,
            start_res_name=segment_residues[0].res_name,
            end_res_name=segment_residues[-1].res_name
        ))

    return segments


# ==================== 函数1: 相邻结构域距离能量计算（修正版）====================

def calculate_segment_connection_energy(
        seg1: DomainSegment,
        seg2: DomainSegment,
        ideal_bond_distance: float = 3.8
) -> float:
    """
    计算两个片段之间的连接能量

    参数:
        seg1: 前一个片段
        seg2: 后一个片段
        ideal_bond_distance: 理想连接距离

    返回:
        连接能量
    """
    # 计算seg1的末端CA和seg2的起始CA之间的距离
    distance = np.linalg.norm(seg1.end_ca - seg2.start_ca)
    energy = (distance - ideal_bond_distance) ** 2

    print(f"    片段连接: 残基{seg1.end_res}({seg1.end_res_name}) -> 残基{seg2.start_res}({seg2.start_res_name})")
    print(f"    CA-CA距离: {distance:.3f} Å, 能量: {energy:.6f}")

    return energy


def _find_structure_files(directory: str, pattern: str = None) -> List[str]:
    """
    查找目录中的所有结构文件（PDB和CIF）

    参数:
        directory: 目录路径
        pattern: 文件名匹配模式（可选）

    返回:
        结构文件路径列表
    """
    structure_files = []
    
    # 查找PDB文件
    pdb_files = glob.glob(os.path.join(directory, "*.pdb"))
    structure_files.extend(pdb_files)
    
    # 查找CIF文件
    cif_files = glob.glob(os.path.join(directory, "*.cif"))
    structure_files.extend(cif_files)
    
    # 如果提供了匹配模式，进行过滤
    if pattern:
        structure_files = [f for f in structure_files if re.search(pattern, os.path.basename(f))]
    
    return structure_files


def calculate_chain_connection_energy_from_dir(
        structure_dir: str,
        ideal_bond_distance: float = 3.8,
        pattern: str = None
) -> float:
    """
    从目录中自动读取所有结构文件（PDB/CIF）并计算整条链的连接能量

    参数:
        structure_dir: 包含结构文件的目录路径
        ideal_bond_distance: 理想的连接距离（埃），默认3.8Å
        pattern: 文件名匹配模式（可选），例如 "component.*_d_\d+\.(pdb|cif)"

    返回:
        总连接能量

    示例:
        >>> total_energy = calculate_chain_connection_energy_from_dir("./domains/")
        >>> print(f"链的总能量: {total_energy:.4f}")
    """
    if not os.path.exists(structure_dir):
        raise FileNotFoundError(f"目录不存在: {structure_dir}")

    if not os.path.isdir(structure_dir):
        raise NotADirectoryError(f"路径不是目录: {structure_dir}")

    # 获取目录下所有结构文件
    structure_files = _find_structure_files(structure_dir, pattern)

    if not structure_files:
        print(f"警告: 目录 {structure_dir} 中没有找到结构文件（PDB或CIF）")
        return 0.0

    # 按文件名排序（确保一致的顺序）
    structure_files.sort()

    print(f"在目录 {structure_dir} 中找到 {len(structure_files)} 个结构文件:")
    for i, structure_file in enumerate(structure_files, 1):
        print(f"  {i}. {os.path.basename(structure_file)}")
    print()

    if len(structure_files) <= 1:
        print("警告: 结构文件数量不足2个，无需计算连接能量")
        return 0.0

    # 调用原有的计算函数
    return calculate_chain_connection_energy(structure_files, ideal_bond_distance)


def calculate_chain_connection_energy(
        structure_files: List[str],
        ideal_bond_distance: float = 3.8
) -> float:
    """
    计算整条链的连接能量（支持PDB和CIF格式）

    参数:
        structure_files: 结构文件路径列表（或目录路径）
        ideal_bond_distance: 理想的连接距离（埃），默认3.8Å

    返回:
        总连接能量
    """
    # 如果传入的是字符串且是目录，自动调用目录版本
    if isinstance(structure_files, str):
        if os.path.isdir(structure_files):
            return calculate_chain_connection_energy_from_dir(structure_files, ideal_bond_distance)
        else:
            structure_files = [structure_files]

    if len(structure_files) <= 1:
        return 0.0

    print("=" * 80)
    print("计算整条链的连接能量")
    print("=" * 80)

    # 收集所有结构域的所有片段，并记录来源
    all_segments = []

    for domain_idx, structure_file in enumerate(structure_files):
        segments = get_domain_segments(structure_file)
        if not segments:
            print(f"警告: 跳过文件 {structure_file} (无有效片段)")
            continue

        for seg in segments:
            all_segments.append({
                'domain_idx': domain_idx,
                'domain_file': structure_file,
                'segment': seg
            })

    if len(all_segments) < 2:
        print("警告: 可用片段不足2个")
        return 0.0

    # 按残基起始位置排序所有片段
    all_segments.sort(key=lambda x: x['segment'].start_res)

    print(f"\n总共 {len(all_segments)} 个片段（来自 {len(structure_files)} 个结构域）:")
    for i, seg_info in enumerate(all_segments):
        seg = seg_info['segment']
        domain_file = os.path.basename(seg_info['domain_file'])
        print(f"  片段{i + 1}: 残基 {seg.start_res:4d}-{seg.end_res:4d} (来自 {domain_file})")

    # 计算所有相邻片段对的能量
    print(f"\n计算相邻片段对的连接能量:")
    total_energy = 0.0
    connection_count = 0

    for i in range(len(all_segments) - 1):
        seg1_info = all_segments[i]
        seg2_info = all_segments[i + 1]

        seg1 = seg1_info['segment']
        seg2 = seg2_info['segment']

        # 计算gap
        gap = seg2.start_res - seg1.end_res

        # 只有当gap在合理范围内时才计算能量
        if gap <= 10:  # 允许最多10个残基的gap
            connection_count += 1
            print(f"\n连接 {connection_count}:")
            print(f"  从: {os.path.basename(seg1_info['domain_file'])} 片段 {seg1.start_res}-{seg1.end_res}")
            print(f"  到: {os.path.basename(seg2_info['domain_file'])} 片段 {seg2.start_res}-{seg2.end_res}")
            print(f"  Gap: {gap} 个残基")

            energy = calculate_segment_connection_energy(seg1, seg2, ideal_bond_distance)
            total_energy += energy
        else:
            print(f"\n跳过连接 (gap={gap} 太大):")
            print(f"  片段 {seg1.start_res}-{seg1.end_res} -> {seg2.start_res}-{seg2.end_res}")

    print(f"\n" + "=" * 80)
    print(f"计算了 {connection_count} 个连接")
    print(f"链的总连接能量: {total_energy:.6f}")
    print("=" * 80)

    return total_energy


# ==================== 函数2: 距离阈值评分函数 ====================

def calculate_chain_connection_score(
        structure_files: List[str],
        ideal_distance: float = 3.8,
        tight_tolerance: float = 1.5,
        loose_min: float = 5.3,
        loose_max: float = 20.0
) -> Tuple[float, Dict]:
    """
    计算整条链的连接评分（基于距离阈值）
    支持PDB和CIF格式

    参数:
        structure_files: 结构文件路径列表（或目录路径）
        ideal_distance: 理想连接距离（埃），默认3.8Å
        tight_tolerance: 紧密连接的容差范围，默认±1.5Å
        loose_min: 松散连接的最小距离，默认5.3Å
        loose_max: 松散连接的最大距离，默认20.0Å

    返回:
        (total_score, details): 总评分和详细信息字典

    评分规则:
        - 距离在 [ideal - tight_tolerance, ideal + tight_tolerance] 范围内: +1.0分
        - 距离在 (loose_min, loose_max] 范围内: +0.1分
        - 其他情况: +0分

    示例:
        >>> score, details = calculate_chain_connection_score("./domains/")
        >>> print(f"链的总评分: {score:.2f}")
        >>> print(f"紧密连接数: {details['tight_connections']}")
        >>> print(f"松散连接数: {details['loose_connections']}")
    """
    # 如果传入的是字符串且是目录，转换为文件列表
    if isinstance(structure_files, str):
        if os.path.isdir(structure_files):
            structure_dir = structure_files
            structure_files = _find_structure_files(structure_dir)
            structure_files.sort()
        else:
            structure_files = [structure_files]

    if len(structure_files) <= 1:
        return 0.0, {
            'tight_connections': 0,
            'loose_connections': 0,
            'no_score_connections': 0,
            'total_connections': 0
        }

    print("=" * 80)
    print("计算整条链的连接评分（距离阈值法）")
    print("=" * 80)
    print(f"评分规则:")
    print(f"  - 紧密连接 [{ideal_distance - tight_tolerance:.1f}, {ideal_distance + tight_tolerance:.1f}] Å: +1.0分")
    print(f"  - 松散连接 ({loose_min:.1f}, {loose_max:.1f}] Å: +0.1分")
    print(f"  - 其他距离: +0分")
    print()

    # 收集所有结构域的所有片段
    all_segments = []

    for domain_idx, structure_file in enumerate(structure_files):
        segments = get_domain_segments(structure_file)
        if not segments:
            print(f"警告: 跳过文件 {structure_file} (无有效片段)")
            continue

        for seg in segments:
            all_segments.append({
                'domain_idx': domain_idx,
                'domain_file': structure_file,
                'segment': seg
            })

    if len(all_segments) < 2:
        print("警告: 可用片段不足2个")
        return 0.0, {
            'tight_connections': 0,
            'loose_connections': 0,
            'no_score_connections': 0,
            'total_connections': 0
        }

    # 按残基起始位置排序所有片段
    all_segments.sort(key=lambda x: x['segment'].start_res)

    print(f"总共 {len(all_segments)} 个片段（来自 {len(structure_files)} 个结构域）:")
    for i, seg_info in enumerate(all_segments):
        seg = seg_info['segment']
        domain_file = os.path.basename(seg_info['domain_file'])
        print(f"  片段{i + 1}: 残基 {seg.start_res:4d}-{seg.end_res:4d} (来自 {domain_file})")

    # 计算所有相邻片段对的评分
    print(f"\n计算相邻片段对的连接评分:")
    total_score = 0.0
    connection_count = 0
    tight_count = 0
    loose_count = 0
    no_score_count = 0

    connection_details = []

    for i in range(len(all_segments) - 1):
        seg1_info = all_segments[i]
        seg2_info = all_segments[i + 1]

        seg1 = seg1_info['segment']
        seg2 = seg2_info['segment']

        # 计算gap
        gap = seg2.start_res - seg1.end_res

        # 只有当gap在合理范围内时才计算评分
        if gap <= loose_max:
            connection_count += 1

            # 计算CA-CA距离
            distance = np.linalg.norm(seg1.end_ca - seg2.start_ca)

            # 根据距离判断得分
            tight_min = ideal_distance - tight_tolerance
            tight_max = ideal_distance + tight_tolerance

            if tight_min <= distance <= tight_max:
                score = 1.0
                connection_type = "紧密连接"
                tight_count += 1
            elif loose_min < distance <= loose_max:
                score = 0.1
                connection_type = "松散连接"
                loose_count += 1
            else:
                score = 0.0
                connection_type = "无效连接"
                no_score_count += 1

            total_score += score

            print(f"\n连接 {connection_count}:")
            print(f"  从: {os.path.basename(seg1_info['domain_file'])} 片段 {seg1.start_res}-{seg1.end_res}")
            print(f"  到: {os.path.basename(seg2_info['domain_file'])} 片段 {seg2.start_res}-{seg2.end_res}")
            print(f"  Gap: {gap} 个残基")
            print(f"  CA-CA距离: {distance:.3f} Å")
            print(f"  类型: {connection_type}, 得分: {score:.1f}")

            connection_details.append({
                'from_file': os.path.basename(seg1_info['domain_file']),
                'from_res': f"{seg1.start_res}-{seg1.end_res}",
                'to_file': os.path.basename(seg2_info['domain_file']),
                'to_res': f"{seg2.start_res}-{seg2.end_res}",
                'gap': gap,
                'distance': distance,
                'type': connection_type,
                'score': score
            })
        else:
            print(f"\n跳过连接 (gap={gap} 太大):")
            print(f"  片段 {seg1.start_res}-{seg1.end_res} -> {seg2.start_res}-{seg2.end_res}")

    print(f"\n" + "=" * 80)
    print(f"统计结果:")
    print(f"  计算了 {connection_count} 个连接")
    print(f"  紧密连接 ({tight_min:.1f}-{tight_max:.1f} Å): {tight_count} 个")
    print(f"  松散连接 ({loose_min:.1f}-{loose_max:.1f} Å): {loose_count} 个")
    print(f"  无效连接: {no_score_count} 个")
    print(f"链的总评分: {total_score:.2f}")
    print("=" * 80)

    details = {
        'tight_connections': tight_count,
        'loose_connections': loose_count,
        'no_score_connections': no_score_count,
        'total_connections': connection_count,
        'connection_details': connection_details
    }

    return total_score, details


def calculate_chain_connection_score_from_dir(
        structure_dir: str,
        ideal_distance: float = 3.8,
        tight_tolerance: float = 1.5,
        loose_min: float = 5.3,
        loose_max: float = 12.0,
        pattern: str = None
) -> Tuple[float, Dict]:
    """
    从目录中自动读取所有结构文件（PDB/CIF）并计算整条链的连接评分

    参数:
        structure_dir: 包含结构文件的目录路径
        ideal_distance: 理想连接距离（埃），默认3.8Å
        tight_tolerance: 紧密连接的容差范围，默认±1.5Å
        loose_min: 松散连接的最小距离，默认5.3Å
        loose_max: 松散连接的最大距离，默认10.0Å
        pattern: 文件名匹配模式（可选）

    返回:
        (total_score, details): 总评分和详细信息字典

    示例:
        >>> score, details = calculate_chain_connection_score_from_dir("./domains/")
        >>> print(f"链的总评分: {score:.2f}")
    """
    if not os.path.exists(structure_dir):
        raise FileNotFoundError(f"目录不存在: {structure_dir}")

    if not os.path.isdir(structure_dir):
        raise NotADirectoryError(f"路径不是目录: {structure_dir}")

    # 获取目录下所有结构文件
    structure_files = _find_structure_files(structure_dir, pattern)

    if not structure_files:
        print(f"警告: 目录 {structure_dir} 中没有找到结构文件（PDB或CIF）")
        return 0.0, {
            'tight_connections': 0,
            'loose_connections': 0,
            'no_score_connections': 0,
            'total_connections': 0
        }

    # 按文件名排序
    structure_files.sort()

    print(f"在目录 {structure_dir} 中找到 {len(structure_files)} 个结构文件:")
    for i, structure_file in enumerate(structure_files, 1):
        print(f"  {i}. {os.path.basename(structure_file)}")
    print()

    if len(structure_files) <= 1:
        print("警告: 结构文件数量不足2个，无需计算连接评分")
        return 0.0, {
            'tight_connections': 0,
            'loose_connections': 0,
            'no_score_connections': 0,
            'total_connections': 0
        }

    # 调用主计算函数
    return calculate_chain_connection_score(
        structure_files,
        ideal_distance,
        tight_tolerance,
        loose_min,
        loose_max
    )


# ==================== 函数3: 跨文件夹TM-score批量计算 ====================

def _is_valid_structure_file(filename: str) -> bool:
    """判断是否是有效的结构文件"""
    # 支持.pdb和.cif两种格式
    if not (filename.endswith('.pdb') or filename.endswith('.cif')):
        return False
    
    # 去掉扩展名
    basename = filename.rsplit('.', 1)[0]
    
    # 检查是否符合命名规则
    if re.search(r'_d_\d+$', basename):
        return True
    if re.search(r'_\d+$', basename):
        return False
    return True


def _find_pdb_files(folder_path: str) -> List[str]:
    """
    查找文件夹中的所有有效结构文件（PDB和CIF）
    
    参数:
        folder_path: 文件夹路径
        
    返回:
        结构文件路径列表
    """
    structure_files = []
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if _is_valid_structure_file(file):
                structure_files.append(os.path.join(folder_path, file))
    return structure_files


def _extract_tm_scores(output: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """从USalign输出中提取TM-score"""
    rmsd_match = re.search(r"RMSD=\s*([\d.]+)", output)
    tm1_match = re.search(r"TM-score=\s*([\d.]+) \(normalized by length of Structure_1", output)
    tm2_match = re.search(r"TM-score=\s*([\d.]+) \(normalized by length of Structure_2", output)

    if rmsd_match and tm1_match and tm2_match:
        return float(rmsd_match.group(1)), float(tm1_match.group(1)), float(tm2_match.group(1))
    return None, None, None


def compare_all_pdbs_in_dir(
        root_dir: str,
        usalign_path: str = "USalign",
        threshold: float = 0.5
):
    """
    在一个目录下（包含多个子文件夹）递归搜索所有结构文件（PDB和CIF），
    并对这些文件进行两两 TM-score 比较。
    仅输出高于阈值的文件对（无输出文件，仅控制台打印）。

    参数:
        root_dir: 根目录路径
        usalign_path: USalign 可执行文件路径（默认在 PATH 中）
        threshold: TM-score 阈值（默认 0.5）
    """
    print("=" * 100)
    print(f"🔍 扫描目录: {root_dir}")
    print("=" * 100)

    # 获取所有结构文件路径（PDB和CIF）
    all_structure_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".pdb") or file.endswith(".cif"):
                all_structure_files.append(os.path.join(dirpath, file))

    all_structure_files.sort()
    n_files = len(all_structure_files)
    if n_files < 2:
        print(f"❌ 文件数量不足（找到 {n_files} 个结构文件）")
        return

    print(f"✅ 共找到 {n_files} 个结构文件（PDB/CIF），开始两两比较...\n")

    total_comparisons = n_files * (n_files - 1) // 2
    counter = 0
    passed_pairs = []  # 保存高于阈值的结果

    for i in range(n_files):
        for j in range(i + 1, n_files):
            counter += 1
            f1 = all_structure_files[i]
            f2 = all_structure_files[j]
            name1 = os.path.basename(f1)
            name2 = os.path.basename(f2)

            print(f"[{counter}/{total_comparisons}] 比较: {name1} vs {name2}")

            cmd = [usalign_path, f1, f2, "-TMscore", "7", "-ter", "0"]
            try:
                output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=60)
                rmsd, tm1, tm2 = _extract_tm_scores(output)
                if tm1 is None or tm2 is None:
                    print("⚠️ 无法解析 TM-score\n")
                    continue

                max_tm = max(tm1, tm2)
                if max_tm >= threshold:
                    passed_pairs.append((name1, name2, tm1, tm2, rmsd, max_tm))
                    print(f"   ✅ 高相似度匹配: TM1={tm1:.3f}  TM2={tm2:.3f}  RMSD={rmsd:.3f}  Max={max_tm:.3f}\n")
                else:
                    print(f"   TM1={tm1:.3f}  TM2={tm2:.3f}  RMSD={rmsd:.3f}  Max={max_tm:.3f}\n")

            except subprocess.TimeoutExpired:
                print(f"⏰ 超时: {name1} vs {name2}\n")
            except subprocess.CalledProcessError as e:
                print(f"❌ 比对错误: {e}\n")
            except Exception as e:
                print(f"❗ 异常: {e}\n")

    print("=" * 100)
    print(f"✅ 全部完成，共比较 {counter} 对结构。")
    print(f"🎯 满足阈值 (TM ≥ {threshold}) 的文件对数量: {len(passed_pairs)}")
    print("=" * 100)

    if passed_pairs:
        print("\n📄 高相似度文件对列表：")
        for (n1, n2, tm1, tm2, rmsd, max_tm) in passed_pairs:
            print(f" - {n1} vs {n2} | TM1={tm1:.3f} TM2={tm2:.3f} RMSD={rmsd:.3f} Max={max_tm:.3f}")
    else:
        print("⚪ 没有任何文件对超过设定阈值。")

    print("=" * 100)


def _calculate_tm_score(structure1: str, structure2: str, usalign_path: str) -> Tuple[
    Optional[float], Optional[float], Optional[float]]:
    """计算两个结构之间的TM-score"""
    cmd = [usalign_path, structure1, structure2, "-TMscore", "7", "-ter", "0"]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return _extract_tm_scores(output)
    except subprocess.CalledProcessError:
        return None, None, None


def cross_folder_tm_score_analysis(
        root_dir: str,
        threshold: float = 0.5,
        usalign_path: str = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "core", "USalign")
) -> List[Dict]:
    """
    跨文件夹TM-score批量计算，找出高于阈值的文件对
    支持PDB和CIF格式

    参数:
        root_dir: 包含多个文件夹的根目录
        threshold: TM-score阈值，默认0.5
        usalign_path: USalign可执行文件路径

    返回:
        高于阈值的结果列表，每个元素包含:
            - folder1, folder2: 文件夹名
            - pdb1, pdb2: 结构文件名
            - max_tm_score, rmsd, tm_score_1, tm_score_2
    """
    folder_info_map = {}

    if not os.path.exists(root_dir):
        print(f"目录不存在: {root_dir}")
        return []

    for folder_name in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            structure_files = _find_pdb_files(folder_path)
            if structure_files:
                folder_info_map[folder_name] = structure_files

    if len(folder_info_map) < 2:
        print("需要至少2个包含有效结构文件的文件夹")
        return []

    above_threshold_results = []

    for folder1_name, folder2_name in combinations(folder_info_map.keys(), 2):
        structure_files1 = folder_info_map[folder1_name]
        structure_files2 = folder_info_map[folder2_name]

        for structure1 in structure_files1:
            for structure2 in structure_files2:
                rmsd, tm1, tm2 = _calculate_tm_score(structure1, structure2, usalign_path)

                if rmsd is not None and tm1 is not None and tm2 is not None:
                    max_tm_score = min(tm1, tm2)

                    if max_tm_score >= threshold:
                        above_threshold_results.append({
                            'folder1': folder1_name,
                            'folder2': folder2_name,
                            'pdb1': os.path.basename(structure1),
                            'pdb2': os.path.basename(structure2),
                            'max_tm_score': max_tm_score,
                            'rmsd': rmsd,
                            'tm_score_1': tm1,
                            'tm_score_2': tm2
                        })

    if above_threshold_results:
        print(f"\n找到 {len(above_threshold_results)} 对高于阈值({threshold})的结构")
        above_threshold_results.sort(key=lambda x: x['max_tm_score'], reverse=True)
        for result in above_threshold_results:
            print(
                f"  {result['folder1']}/{result['pdb1']} <-> {result['folder2']}/{result['pdb2']}: TM={result['max_tm_score']:.3f}")
    else:
        print(f"没有找到高于阈值({threshold})的结构对")

    return above_threshold_results


# ==================== 使用示例 ====================

def example_usage(case_dir):
    """使用示例：case_dir 为含已拟合结构域的目录。"""

    # 示例1: 计算连接能量
    print("\n" + "=" * 80)
    print("示例1: 计算连接能量（支持PDB和CIF）")
    print("-" * 80)

    test_dir = case_dir
    if os.path.exists(test_dir):
        total_energy = calculate_chain_connection_energy(test_dir)
        print(f"\n总能量: {total_energy:.6f}")
    else:
        print("示例目录不存在，跳过...")

    # 示例2: 计算连接评分（距离阈值法）
    print("\n" + "=" * 80)
    print("示例2: 计算连接评分（距离阈值法，支持PDB和CIF）")
    print("-" * 80)

    if os.path.exists(test_dir):
        score, details = calculate_chain_connection_score_from_dir(test_dir)
        print(f"\n总评分: {score:.2f}")
        print(f"紧密连接数: {details['tight_connections']}")
        print(f"松散连接数: {details['loose_connections']}")
        print(f"无效连接数: {details['no_score_connections']}")
        print(f"总连接数: {details['total_connections']}")
    else:
        print("示例目录不存在，跳过...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="连接能量/连接评分示例")
    parser.add_argument("case_dir", help="含已拟合结构域的目录")
    example_usage(parser.parse_args().case_dir)
