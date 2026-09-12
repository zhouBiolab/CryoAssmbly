#!/usr/bin/env python3
"""
结构域TXT文件分割器（优化版 - 统一坐标系）
输入: .txt文件（采样点）和 .pdb文件
输出: 按结构域分割的多个.txt文件

主要优化:
1. 先对完整PDB文件计算统一的网格参数（原点、盒子尺寸、体素大小）
2. 所有结构域都使用相同的网格参数，确保在同一坐标系下
3. 避免重复计算网格参数，提高效率
4. 修复单域问题：确保即使只有一个结构域时也生成对应的PDB文件
5. 修正文件命名：PDB文件使用正确的 _d_数字 格式（如 protein_d_1.pdb）

使用流程:
1. 读取完整PDB文件，计算统一的网格参数
2. 调用DomainParser分解PDB为多个结构域
3. 为每个结构域生成掩膜（使用统一网格参数）
4. 根据掩膜过滤.txt中的点
5. 保存每个结构域的.txt文件和对应的.pdb文件
"""

import argparse
import numpy as np
import os
import sys
import subprocess
import shutil
from Bio.PDB import PDBParser, MMCIFParser
from numba.typed import List
import tempfile
import glob
import re

# Add project root for protassem imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from protassem.core.constants import VDW_RADII
from protassem.core.numba_kernels import add_sphere_mask as _njit_add_sphere_mask
from protassem.core.io import read_structure
from protassem.core import points_txt

# VDW_RADII imported from protassem.core.constants
# _njit_add_sphere_mask imported from protassem.core.numba_kernels


def load_sample_points_with_info(file_path):
    """读取点云 TXT 并保留原始行信息（兼容历史 6 元组返回值）。

    Returns:
        (points, vectors, densities, original_lines, line_mapping, header_info)；
        line_mapping[i] = (坐标行下标, 向量行下标)，用于按点过滤后原样写回。
    """
    cloud = points_txt.read_point_cloud(file_path)
    with open(file_path, encoding="utf-8") as handle:
        original_lines = handle.readlines()
    line_mapping = points_txt.line_mapping(cloud)
    header_info = {"sample": cloud.sample,
                   "origin": [float(v) for v in cloud.origin],
                   "line1": cloud.header_lines[1],
                   "line2": cloud.header_lines[2],
                   "origin_line": cloud.header_lines[3],
                   "line4": cloud.header_lines[4]}
    return (cloud.points, cloud.vectors, cloud.densities,
            original_lines, line_mapping, header_info)


def save_domain_txt(original_lines, kept_indices, line_mapping, header_info, output_path):
    """按 kept_indices 写回原始行对，头部原样复制（保留文本精度）。

    header_info 仅为兼容历史签名保留；头部直接从 original_lines 复制。
    """
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for line in original_lines[:points_txt.HEADER_LINES]:
            handle.write(line)
        for point_idx in kept_indices:
            coord_line_idx, vector_line_idx = line_mapping[point_idx]
            handle.write(original_lines[coord_line_idx])
            handle.write(original_lines[vector_line_idx])
    print("成功保存结构域TXT文件: %s" % output_path)
    print("保留了 %d 个点" % len(kept_indices))


def copy_pdb_with_new_name(source_pdb, target_pdb):
    """复制 PDB 并改名；失败时直接抛出，由调用方处理。"""
    shutil.copy2(source_pdb, target_pdb)
    print("成功复制PDB文件: %s -> %s" % (source_pdb, target_pdb))


def get_atom_list(pdb_file, backbone_only=False):
    """Wrapper around core.io.read_structure for backward compat."""
    return read_structure(pdb_file, backbone_only=backbone_only)



def calculate_unified_grid_parameters(sample_points, complete_pdb_atoms, resolution=6.0, voxel_size=None):
    """
    基于完整PDB文件和采样点计算统一的网格参数
    这些参数将被所有结构域共用，确保在同一坐标系下
    
    返回: origin, box_size, voxel_size_array
    """
    import math
    
    print("计算统一网格参数...")
    
    # 如果没有指定voxel_size，使用resolution/3
    if voxel_size is None:
        grid_spacing = resolution / 3.0
        voxel_size_array = np.array([grid_spacing, grid_spacing, grid_spacing])
    else:
        voxel_size_array = np.array([voxel_size, voxel_size, voxel_size])
    
    # 合并所有坐标点来计算边界
    all_coords = np.vstack([sample_points, complete_pdb_atoms])
    max_x, max_y, max_z = all_coords.max(axis=0)
    min_x, min_y, min_z = all_coords.min(axis=0)
    
    # 使用3倍分辨率作为padding
    pad = 3.0 * resolution
    
    # 计算网格尺寸
    x_size = math.ceil((max_x - min_x + 2 * pad) / voxel_size_array[0])
    y_size = math.ceil((max_y - min_y + 2 * pad) / voxel_size_array[1])
    z_size = math.ceil((max_z - min_z + 2 * pad) / voxel_size_array[2])
    
    # 原点计算
    x_origin = min_x - pad
    y_origin = min_y - pad
    z_origin = min_z - pad
    origin = np.array([x_origin, y_origin, z_origin])
    
    # 注意：返回的是(z_size, y_size, x_size)
    box_size = (z_size, y_size, x_size)
    
    print(f"统一网格参数:")
    print(f"  原点: {origin}")
    print(f"  盒子大小: {box_size}")
    print(f"  体素大小: {voxel_size_array}")
    print(f"  总体素数: {np.prod(box_size):,}")
    
    return origin, box_size, voxel_size_array

def make_atomic_mask_optimized(origin, voxel_size, box_size, atom_list, atom_type_list, solvent_radius=1.1):
    """生成原子掩膜（Numba加速版）- 使用统一网格参数"""
    mask = np.zeros(box_size, dtype=np.bool_)
    nz, ny, nx = mask.shape
    
    for coord, atom_type in zip(atom_list, atom_type_list):
        r_vdw = VDW_RADII.get(atom_type, 1.70)
        radius = r_vdw + solvent_radius
        radius_sq = radius * radius
        
        i_center = (coord[0] - origin[0]) / voxel_size[0]
        j_center = (coord[1] - origin[1]) / voxel_size[1]
        k_center = (coord[2] - origin[2]) / voxel_size[2]
        
        radius_voxels = np.array([
            radius / voxel_size[0],
            radius / voxel_size[1],
            radius / voxel_size[2]
        ])
        
        i0 = max(0, int(np.floor(i_center - radius_voxels[0])))
        i1 = min(nx, int(np.ceil(i_center + radius_voxels[0])) + 1)
        j0 = max(0, int(np.floor(j_center - radius_voxels[1])))
        j1 = min(ny, int(np.ceil(j_center + radius_voxels[1])) + 1)
        k0 = max(0, int(np.floor(k_center - radius_voxels[2])))
        k1 = min(nz, int(np.ceil(k_center + radius_voxels[2])) + 1)
        
        if i0 >= i1 or j0 >= j1 or k0 >= k1:
            continue
        
        _njit_add_sphere_mask(
            mask, i_center, j_center, k_center,
            voxel_size[0], voxel_size[1], voxel_size[2],
            radius_sq, i0, i1, j0, j1, k0, k1
        )
    
    return mask

def filter_points_by_mask_keep_inside(sample_points, mask, origin, voxel_size):
    """
    根据掩膜过滤采样点
    保留掩膜内的点，移除掩膜外的点
    """
    nz, ny, nx = mask.shape
    kept_indices = []
    removed_indices = []
    
    for i, point in enumerate(sample_points):
        voxel_coords = (point - origin) / voxel_size
        vi = int(np.round(voxel_coords[0]))
        vj = int(np.round(voxel_coords[1]))
        vk = int(np.round(voxel_coords[2]))
        
        if 0 <= vi < nx and 0 <= vj < ny and 0 <= vk < nz:
            # 保留掩膜内的点
            if mask[vk, vj, vi]:
                kept_indices.append(i)
            else:
                removed_indices.append(i)
        else:
            # 超出边界的点移除
            removed_indices.append(i)
    
    return kept_indices, removed_indices

def extract_domain_number(filename):
    """
    从DomainParser生成的文件名中提取结构域编号
    例如: chain_a_1_d_6.pdb -> 6
    """
    match = re.search(r'_d_(\d+)\.pdb$', filename)
    if match:
        return int(match.group(1))
    else:
        return 0  # 如果没有找到数字，返回0

def call_domain_parser(pdb_file, domain_parser_path):
    """
    调用DomainParser分解PDB文件
    返回: 结构域文件字典（编号->文件名）和domain信息
    """
    print(f"调用DomainParser分解PDB文件: {pdb_file}")
    
    # 检查DomainParser是否存在
    if not os.path.exists(domain_parser_path):
        raise FileNotFoundError(f"DomainParser不存在: {domain_parser_path}")
    
    # 获取当前工作目录
    current_dir = os.getcwd()
    
    try:
        # 调用DomainParser
        cmd = [sys.executable, domain_parser_path, pdb_file]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=current_dir)
        
        if result.returncode != 0:
            print(f"DomainParser执行失败: {result.stderr}")
            return {}, ""
        
        # 查找生成的结构域文件
        base_name = os.path.splitext(os.path.basename(pdb_file))[0]
        domain_files = glob.glob(f"{base_name}*_d_*.pdb")
        
        # 创建编号到文件名的字典
        domain_dict = {}
        for domain_file in domain_files:
            domain_num = extract_domain_number(domain_file)
            if domain_num > 0:
                domain_dict[domain_num] = domain_file
        
        print(f"找到 {len(domain_dict)} 个结构域文件: {list(domain_dict.values())}")
        
        return domain_dict, result.stdout.strip()
        
    except Exception as e:
        print(f"调用DomainParser时出错: {e}")
        return {}, ""

def handle_single_domain_case(input_txt, input_pdb, points, original_lines, line_mapping, header_info):
    """
    处理单结构域情况：生成对应的TXT和PDB文件
    """
    print("未找到多个结构域文件，可能PDB文件只有一个结构域")
    print("生成单结构域的TXT和PDB文件...")
    
    base_name_txt = os.path.splitext(input_txt)[0]
    base_name_pdb = os.path.splitext(input_pdb)[0]
    
    # 生成输出文件名 - 修正命名格式
    output_txt = f"{base_name_txt}_domain1.txt"
    output_pdb = f"{base_name_pdb}_d_1.pdb"  # 使用正确的 _d_数字 格式
    
    # 复制原始TXT文件（保留所有点）
    shutil.copy2(input_txt, output_txt)
    print(f"复制原始TXT文件为: {output_txt}")
    print(f"保留了所有 {len(points)} 个点")
    
    # 复制原始PDB文件
    copy_pdb_with_new_name(input_pdb, output_pdb)
    
    # 返回结果统计
    return [{
        'domain_num': 1,
        'domain_file': output_pdb,
        'txt_file': output_txt,
        'total_points': len(points),
        'kept_points': len(points),
        'removed_points': 0,
        'retention_rate': 100.0
    }]

def split_txt_by_domains_unified_grid(input_txt, input_pdb, domain_parser_path, 
                                     resolution=6.0, voxel_size=None, solvent_radius=1.1, backbone_only=False):
    """
    主函数：按结构域分割TXT文件（优化版 - 统一坐标系）
    
    主要优化:
    1. 先对完整PDB文件计算统一的网格参数
    2. 所有结构域都使用相同的网格参数
    3. 修复单域问题：确保即使只有一个结构域时也生成对应的PDB文件
    4. 修正文件命名：使用正确的 _d_数字 格式
    """
    print("=" * 80)
    print("结构域TXT文件分割器（优化版 - 统一坐标系 + 单域修复 + 修正命名）")
    print(f"输入TXT文件: {input_txt}")
    print(f"输入PDB文件: {input_pdb}")
    print(f"DomainParser路径: {domain_parser_path}")
    print(f"分辨率: {resolution} Å")
    if voxel_size:
        print(f"体素大小: {voxel_size} Å")
    else:
        print(f"体素大小: {resolution/3.0:.2f} Å (resolution/3)")
    print(f"溶剂半径: {solvent_radius} Å")
    print(f"仅使用骨架原子: {'是' if backbone_only else '否'}")
    print("=" * 80)
    
    # 检查输入文件
    if not os.path.exists(input_txt):
        raise FileNotFoundError("输入TXT文件不存在: %s" % input_txt)
        
    if not os.path.exists(input_pdb):
        raise FileNotFoundError("输入PDB文件不存在: %s" % input_pdb)
    
    # 1. 读取TXT文件
    print("\n1. 读取TXT文件...")
    points, vectors, densities, original_lines, line_mapping, header_info = load_sample_points_with_info(input_txt)
    print(f"成功读取 {len(points)} 个采样点")
    
    # 2. 调用DomainParser分解PDB
    print("\n2. 调用DomainParser分解PDB文件...")
    domain_dict, domain_info = call_domain_parser(input_pdb, domain_parser_path)
    
    # 3. 处理单结构域情况（修复的关键部分）
    if not domain_dict:
        results = handle_single_domain_case(input_txt, input_pdb, points, original_lines, line_mapping, header_info)
        
        # 输出单结构域总结
        print("\n" + "=" * 80)
        print("单结构域处理结果总结:")
        result = results[0]
        print(f"结构域 {result['domain_num']}:")
        print(f"  PDB文件: {result['domain_file']}")
        print(f"  TXT文件: {result['txt_file']}")
        print(f"  保留点数: {result['kept_points']}/{result['total_points']} ({result['retention_rate']:.1f}%)")
        print(f"\n已生成对应的PDB和TXT文件")
        print("=" * 80)
        print("单结构域处理完成!")
        return
    
    # 4. 多结构域处理 - 读取完整PDB文件，计算统一网格参数
    print("\n3. 读取完整PDB文件，计算统一网格参数...")
    complete_atoms, complete_atom_types = get_atom_list(input_pdb, backbone_only=backbone_only)
    print(f"完整PDB文件包含 {len(complete_atoms)} 个原子 ({'骨架' if backbone_only else '全部'})")
    
    # 计算统一的网格参数（所有结构域将使用相同参数）
    unified_origin, unified_box_size, unified_voxel_size = calculate_unified_grid_parameters(
        points, complete_atoms, resolution=resolution, voxel_size=voxel_size
    )
    
    # 5. 为每个结构域生成掩膜并过滤TXT文件（使用统一网格参数）
    print("\n4. 为每个结构域生成掩膜并过滤TXT文件...")
    print("   ※ 所有结构域使用统一的网格参数，确保在同一坐标系下")
    
    base_name_txt = os.path.splitext(input_txt)[0]
    base_name_pdb = os.path.splitext(input_pdb)[0]
    results = []
    
    # 按照结构域编号顺序处理
    for domain_num in sorted(domain_dict.keys()):
        domain_file = domain_dict[domain_num]
        print(f"\n处理结构域 {domain_num}: {domain_file}")
        
        try:
            # 读取结构域PDB文件
            print(f"  读取结构域PDB文件...")
            domain_atoms, domain_atom_types = get_atom_list(domain_file, backbone_only=backbone_only)
            print(f"  结构域包含 {len(domain_atoms)} 个原子 ({'骨架' if backbone_only else '全部'})")
            
            # 使用统一的网格参数（不重新计算）
            print(f"  使用统一网格参数生成掩膜...")
            print(f"    网格原点: {unified_origin}")
            print(f"    网格大小: {unified_box_size}")
            print(f"    体素大小: {unified_voxel_size}")
            
            # 生成原子掩膜（使用统一网格参数）
            mask = make_atomic_mask_optimized(
                unified_origin, unified_voxel_size, unified_box_size, 
                domain_atoms, domain_atom_types, solvent_radius=solvent_radius
            )
            print(f"  掩膜体素总数: {mask.size:,}")
            print(f"  掩膜内体素数: {np.sum(mask):,}")
            print(f"  掩膜覆盖率: {100.0 * np.sum(mask) / mask.size:.2f}%")
            
            # 根据掩膜过滤点（保留掩膜内的点）
            print(f"  根据掩膜过滤采样点...")
            kept_indices, removed_indices = filter_points_by_mask_keep_inside(
                points, mask, unified_origin, unified_voxel_size
            )
            
            # 保存结构域TXT文件，使用与PDB文件相同的编号
            domain_txt = f"{base_name_txt}_domain{domain_num}.txt"
            save_domain_txt(original_lines, kept_indices, line_mapping, header_info, domain_txt)
            
            # 注意：这里domain_file是DomainParser生成的文件，我们需要重命名它
            # 修正PDB文件命名格式：使用 _d_数字 格式
            target_domain_pdb = f"{base_name_pdb}_d_{domain_num}.pdb"
            if os.path.exists(domain_file) and domain_file != target_domain_pdb:
                shutil.move(domain_file, target_domain_pdb)
                print(f"重命名结构域PDB文件: {domain_file} -> {target_domain_pdb}")
            elif not os.path.exists(target_domain_pdb):
                print(f"警告：结构域PDB文件不存在: {domain_file}")
                target_domain_pdb = domain_file  # 使用原始文件名
            
            results.append({
                'domain_num': domain_num,
                'domain_file': target_domain_pdb,
                'txt_file': domain_txt,
                'total_points': len(points),
                'kept_points': len(kept_indices),
                'removed_points': len(removed_indices),
                'retention_rate': 100.0 * len(kept_indices) / len(points)
            })
            
        except Exception as e:
            print(f"处理结构域 {domain_file} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 6. 输出总结
    print("\n" + "=" * 80)
    print("分割结果总结:")
    print(f"统一网格参数:")
    print(f"  网格原点: {unified_origin}")
    print(f"  网格大小: {unified_box_size}")
    print(f"  体素大小: {unified_voxel_size}")
    print(f"  总体素数: {np.prod(unified_box_size):,}")
    print()
    
    for result in results:
        print(f"结构域 {result['domain_num']}:")
        print(f"  PDB文件: {result['domain_file']}")
        print(f"  TXT文件: {result['txt_file']}")
        print(f"  保留点数: {result['kept_points']}/{result['total_points']} ({result['retention_rate']:.1f}%)")
    
    print(f"\n总共生成了 {len(results)} 个结构域TXT和PDB文件")
    print("=" * 80)
    print("分割完成!")
    print("\n主要优化：")
    print("✓ 所有结构域使用统一的网格参数，确保在同一坐标系下")
    print("✓ 避免重复计算网格参数，提高处理效率")
    print("✓ 基于完整PDB文件计算网格边界，避免结构域边界不一致问题")
    print("✓ 每个结构域TXT文件只包含该结构域掩膜内的采样点")
    print("✓ 修复单结构域问题：确保即使只有一个结构域时也生成对应的PDB文件")
    print("✓ 修正文件命名：PDB文件使用正确的 _d_数字 格式（如 protein_d_1.pdb）")

def main():
    parser = argparse.ArgumentParser(
        description="结构域TXT文件分割器（优化版 - 统一坐标系 + 单域修复 + 修正命名）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python optimized_domain_splitter.py input.txt protein.pdb
  python optimized_domain_splitter.py input.txt protein.pdb --domain_parser /path/to/DomainParser.py
  python optimized_domain_splitter.py input.txt protein.pdb --resolution 4.0 --voxel_size 1.5 --backbone_only

主要优化:
  ✓ 统一坐标系：所有结构域使用相同的网格参数（原点、盒子尺寸、体素大小）
  ✓ 提高效率：避免重复计算网格参数，先对完整PDB计算一次，然后应用到所有结构域
  ✓ 确保一致性：基于完整PDB文件计算网格边界，避免各结构域网格不一致的问题
  ✓ 保持精度：每个结构域的掩膜仍基于各自的原子生成，只是使用统一的网格坐标系
  ✓ 单域修复：修复只有一个结构域时缺少PDB文件的问题，确保总是生成配对的TXT和PDB文件
  ✓ 修正命名：PDB文件使用正确的 _d_数字 格式，与DomainParser的输出格式保持一致

工作流程:
  1. 读取完整PDB文件，基于所有原子和采样点计算统一网格参数
  2. 调用DomainParser分解PDB为多个结构域文件
  3. 如果只有一个结构域：直接复制TXT和PDB文件，重命名为 _d_1 格式
  4. 如果有多个结构域：对每个结构域使用统一网格参数生成该结构域的原子掩膜
  5. 根据各结构域掩膜过滤采样点，生成对应的TXT文件，并重命名PDB文件为 _d_数字 格式

输出:
  - 结构域PDB文件: 原始蛋白质结构按结构域分割 (如: protein_d_1.pdb, protein_d_2.pdb, ...)
  - 结构域TXT文件: 每个结构域掩膜内的采样点数据 (如: input_domain1.txt, input_domain2.txt, ...)
  - 所有文件都基于统一的坐标系，可直接用于后续分析和可视化
  - 确保每个TXT文件都有对应的PDB文件，解决单结构域时缺少PDB文件的问题
  - 文件命名格式与DomainParser保持一致，便于后续处理
        """
    )
    parser.add_argument("input_txt", help="输入的TXT文件路径")
    parser.add_argument("input_pdb", help="输入的PDB文件路径")
    parser.add_argument("--domain_parser", 
                       default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "DomainParser.py"),
                       help="DomainParser.py的路径")
    parser.add_argument("--resolution", type=float, default=6.0, help="分辨率，用于网格计算（默认6.0 Å）")
    parser.add_argument("--voxel_size", type=float, default=None, help="体素大小（默认为resolution/3）")
    parser.add_argument("--solvent_radius", type=float, default=1.1, help="溶剂半径（默认1.1 Å）")
    parser.add_argument("--backbone_only", action="store_true", help="只使用骨架原子生成掩码")
    
    args = parser.parse_args()
    
    split_txt_by_domains_unified_grid(
        input_txt=args.input_txt,
        input_pdb=args.input_pdb,
        domain_parser_path=args.domain_parser,
        resolution=args.resolution,
        voxel_size=args.voxel_size,
        solvent_radius=args.solvent_radius,
        backbone_only=args.backbone_only
    )

if __name__ == "__main__":
    main()