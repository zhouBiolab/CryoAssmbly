#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDB文件间重叠检测脚本
检测目录下所有PDB文件之间是否存在大量重叠
"""

import os
import sys
from pathlib import Path
import numpy as np

# 尝试导入torch，如果没有则使用numpy fallback
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch未安装，将使用NumPy进行计算（速度较慢）")


def extract_residue_locations(pdb_file):
    """
    提取PDB文件中的残基位置（CA原子用于蛋白质，P原子用于DNA/RNA）
    
    Args:
        pdb_file: PDB文件路径
    
    Returns:
        numpy.array: 残基位置数组
    """
    residue_locations = []
    try:
        with open(pdb_file, 'r') as f:
            for line in f:
                # CA for protein, P for DNA/RNA
                if line.startswith("ATOM") and (line[12:16].strip() == "CA" or line[12:16].strip() == "P"):
                    residue_locations.append(
                        np.array(
                            (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                        )
                    )
    except Exception as e:
        print(f"警告: 读取文件 {pdb_file} 时出错: {e}")
        return np.array([])
    
    return np.array(residue_locations)


def calculate_overlap_ratio_torch(pdb_file1, pdb_file2, clash_distance=3.0):
    """
    使用PyTorch计算两个PDB文件之间的重叠比例（GPU加速）
    
    Args:
        pdb_file1: 第一个PDB文件路径
        pdb_file2: 第二个PDB文件路径
        clash_distance: 判定为重叠的距离阈值（埃）
    
    Returns:
        float: 重叠比例（0-1之间）
    """
    # 提取残基位置
    locations1 = extract_residue_locations(pdb_file1)
    locations2 = extract_residue_locations(pdb_file2)
    
    if len(locations1) == 0 or len(locations2) == 0:
        return 0.0
    
    # 转换为torch tensor并移到GPU
    locations1_torch = torch.from_numpy(locations1).float().cuda()
    locations2_torch = torch.from_numpy(locations2).float().cuda()
    
    # 计算距离矩阵
    distance_array = torch.cdist(locations1_torch, locations2_torch, p=2)
    
    # 对于file1中的每个原子，找到与file2中最近的距离
    min_distances = torch.amin(distance_array, dim=1)
    
    # 计算重叠比例
    ratio_close = (min_distances < clash_distance).sum().item() / len(locations1)
    
    return ratio_close


def calculate_overlap_ratio_numpy(pdb_file1, pdb_file2, clash_distance=3.0):
    """
    使用NumPy计算两个PDB文件之间的重叠比例
    
    Args:
        pdb_file1: 第一个PDB文件路径
        pdb_file2: 第二个PDB文件路径
        clash_distance: 判定为重叠的距离阈值（埃）
    
    Returns:
        float: 重叠比例（0-1之间）
    """
    # 提取残基位置
    locations1 = extract_residue_locations(pdb_file1)
    locations2 = extract_residue_locations(pdb_file2)
    
    if len(locations1) == 0 or len(locations2) == 0:
        return 0.0
    
    # 计算距离矩阵
    # 使用broadcasting计算欧氏距离
    diff = locations1[:, np.newaxis, :] - locations2[np.newaxis, :, :]
    distance_array = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # 对于file1中的每个原子，找到与file2中最近的距离
    min_distances = np.min(distance_array, axis=1)
    
    # 计算重叠比例
    ratio_close = np.sum(min_distances < clash_distance) / len(locations1)
    
    return ratio_close


def calculate_overlap_ratio(pdb_file1, pdb_file2, clash_distance=3.0):
    """
    计算两个PDB文件之间的重叠比例（自动选择PyTorch或NumPy）
    
    Args:
        pdb_file1: 第一个PDB文件路径
        pdb_file2: 第二个PDB文件路径
        clash_distance: 判定为重叠的距离阈值（埃）
    
    Returns:
        float: 重叠比例（0-1之间）
    """
    if TORCH_AVAILABLE:
        try:
            return calculate_overlap_ratio_torch(pdb_file1, pdb_file2, clash_distance)
        except Exception as e:
            print(f"警告: PyTorch计算失败，回退到NumPy: {e}")
            return calculate_overlap_ratio_numpy(pdb_file1, pdb_file2, clash_distance)
    else:
        return calculate_overlap_ratio_numpy(pdb_file1, pdb_file2, clash_distance)


def scan_directory(root_dir, clash_distance=3.0, ratio_threshold=0.05):
    """
    扫描目录下所有PDB文件，检测文件之间的大量重叠
    
    Args:
        root_dir: 根目录路径
        clash_distance: 判定为碰撞的距离阈值（埃）
        ratio_threshold: 重叠比例阈值，超过此值视为大量重叠
    """
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"错误: 目录 '{root_dir}' 不存在")
        return
    
    if not root_path.is_dir():
        print(f"错误: '{root_dir}' 不是一个目录")
        return
    
    # 查找所有PDB文件
    pdb_files = list(root_path.rglob("*.pdb"))
    
    if not pdb_files:
        print(f"在目录 '{root_dir}' 中没有找到PDB文件")
        return
    
    if len(pdb_files) < 2:
        print(f"只找到 {len(pdb_files)} 个PDB文件，至少需要2个文件才能进行配对比较")
        return
    
    print(f"找到 {len(pdb_files)} 个PDB文件")
    print(f"碰撞距离阈值: {clash_distance} Å")
    print(f"重叠比例阈值: {ratio_threshold * 100}%")
    print(f"使用计算后端: {'PyTorch (GPU)' if TORCH_AVAILABLE else 'NumPy (CPU)'}")
    print("=" * 80)
    
    overlap_pairs = []
    total_comparisons = len(pdb_files) * (len(pdb_files) - 1) // 2
    current_comparison = 0
    
    # 两两比较所有文件
    for i in range(len(pdb_files)):
        for j in range(i + 1, len(pdb_files)):
            current_comparison += 1
            
            if current_comparison % 10 == 0 or current_comparison == total_comparisons:
                print(f"进度: {current_comparison}/{total_comparisons} ({current_comparison/total_comparisons*100:.1f}%)")
            
            file1 = pdb_files[i]
            file2 = pdb_files[j]
            
            # 计算重叠比例
            try:
                ratio = calculate_overlap_ratio(file1, file2, clash_distance)
                
                if ratio >= ratio_threshold:
                    overlap_pairs.append((file1, file2, ratio))
                    
            except Exception as e:
                print(f"警告: 比较 {file1.name} 和 {file2.name} 时出错: {e}")
    
    print("\n" + "=" * 80)
    print(f"扫描完成!")
    print(f"总共比较了 {total_comparisons} 对文件")
    print(f"发现 {len(overlap_pairs)} 对文件存在大量重叠\n")
    
    if overlap_pairs:
        # 按重叠比例排序
        overlap_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print("重叠文件对详情:")
        print("-" * 80)
        for file1, file2, ratio in overlap_pairs:
            print(f"\n文件1: {file1}")
            print(f"文件2: {file2}")
            print(f"重叠比例: {ratio*100:.2f}%")
    else:
        print("未发现重叠超过阈值的文件对")


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("PDB文件间重叠检测工具")
        print("\n使用方法:")
        print("  python check_pdb_overlap.py <目录路径> [碰撞距离] [重叠比例阈值]")
        print("\n参数说明:")
        print("  目录路径: 包含PDB文件的根目录（必需）")
        print("  碰撞距离: 判定为碰撞的距离阈值（单位：埃），默认3.0")
        print("  重叠比例阈值: 超过此比例视为大量重叠，默认0.05 (5%)")
        print("\n示例:")
        print("  python check_pdb_overlap.py ./pdb_files")
        print("  python check_pdb_overlap.py ./pdb_files 3.0")
        print("  python check_pdb_overlap.py ./pdb_files 3.0 0.1")
        print("\n说明:")
        print("  - 重叠比例表示文件1中有多少比例的残基与文件2的残基距离小于碰撞距离")
        print("  - 使用CA原子（蛋白质）或P原子（DNA/RNA）进行比较")
        print("  - 如果安装了PyTorch，将自动使用GPU加速")
        sys.exit(1)
    
    directory = sys.argv[1]
    clash_distance = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    ratio_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
    
    scan_directory(directory, clash_distance, ratio_threshold)


if __name__ == "__main__":
    main()