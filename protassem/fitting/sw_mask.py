"""
球形掩码生成器模块（优化版）


主要功能：
- 基于点云数据生成球形掩码
- 第一个掩码始终是完整的原始数据
- 支持从文件加载或直接使用numpy数组
- 灵活的参数配置
- 可选的文件保存功能

优化内容：
- 使用向量化操作替代循环
- 优化距离计算（避免重复计算）
- 使用KD树加速邻近点搜索
- 优化内存使用
- 并行化文件I/O操作

作者: Assistant
版本: 2.0 (优化版)
"""

import numpy as np
import os
import logging
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass
from scipy.spatial import cKDTree

from protassem.core.points_txt import (read_point_cloud_file,
                                       write_point_cloud)


@dataclass
class MaskResult:
    """掩码结果数据类"""
    id: int
    center: np.ndarray
    radius: float
    points_count: int
    point_cloud_data: np.ndarray
    mask: np.ndarray
    strategy: str = 'spherical'
    type: str = 'sphere'


class SphericalMaskGenerator:
    """球形掩码生成器类（优化版）"""

    def __init__(self, verbose: bool = True):
        """
        初始化掩码生成器

        Args:
            verbose: 是否打印详细信息
        """
        self.verbose = verbose
        self.setup_logging()

    def setup_logging(self):
        """设置日志"""
        if self.verbose:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.WARNING)

    @staticmethod
    def load_sample_points(file_path: str) -> np.ndarray:
        """读取点云文件，返回结构化数组（index/point/vector/density）。

        格式契约与解析规则见 protassem.core.points_txt（单一实现）。
        """
        return read_point_cloud_file(file_path)

    @staticmethod
    def calculate_gyration_radius(points: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        计算点云的回转半径和质心（优化版 - 使用向量化操作）

        Args:
            points: 点云坐标数组，形状为(N, 3)

        Returns:
            回转半径和质心坐标
        """
        centroid = np.mean(points, axis=0)
        # 向量化计算距离平方和
        squared_distances = np.sum((points - centroid) ** 2, axis=1)
        gyration_radius = np.sqrt(np.mean(squared_distances))
        return gyration_radius, centroid

    @staticmethod
    def calculate_max_radius(points: np.ndarray, centroid: np.ndarray) -> float:
        """
        计算点云相对于质心的最大半径（优化版）

        Args:
            points: 点云坐标数组
            centroid: 质心坐标

        Returns:
            最大半径
        """
        # 向量化计算所有距离，只计算一次平方根
        distances_squared = np.sum((points - centroid) ** 2, axis=1)
        return np.sqrt(np.max(distances_squared))

    @staticmethod
    def points_in_sphere(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
        """
        判断点是否在球内（优化版）

        Args:
            points: 点云坐标数组
            center: 球心坐标
            radius: 球半径

        Returns:
            布尔数组，True表示点在球内
        """
        # 避免不必要的平方根计算
        radius_squared = radius ** 2
        distances_squared = np.sum((points - center) ** 2, axis=1)
        return distances_squared <= radius_squared

    def select_initial_points(self, src_points: np.ndarray, ref_points: np.ndarray,
                              mask_radius: float, min_point_distance: float) -> np.ndarray:
        """
        基于参考点云选择初始掩码中心位置（优化版 - 使用KD树）

        Args:
            src_points: 源点云
            ref_points: 参考点云（目标点云）
            mask_radius: 掩码球的半径
            min_point_distance: 最小点间距

        Returns:
            掩码中心位置数组
        """
        # 计算参考点云的质心和最大半径
        ref_centroid = np.mean(ref_points, axis=0)
        distances_squared = np.sum((ref_points - ref_centroid) ** 2, axis=1)
        ref_max_radius = np.sqrt(np.max(distances_squared))

        # 搜索范围：确保掩码球能包含有效区域
        search_extent = min(max(0.85 * (ref_max_radius - mask_radius), 5.0), ref_max_radius)

        if self.verbose:
            print(f"参考点云最大半径: {ref_max_radius:.3f}")
            print(f"掩码球半径: {mask_radius:.3f}")
            print(f"搜索范围: {search_extent:.3f}")
            print(f"最小点间距: {min_point_distance:.3f}")

        # 预筛选：只保留在搜索范围内的点
        distances_to_centroid = np.sqrt(distances_squared)
        valid_mask = distances_to_centroid < search_extent
        candidate_points = ref_points[valid_mask]

        if len(candidate_points) == 0:
            return np.array([ref_centroid])

        # 存储已选择的掩码中心
        mask_centers = [ref_centroid]
        
        # 构建KD树用于快速邻近点查询
        # 初始时只有质心
        tree = cKDTree([ref_centroid])
        
        # 批量处理候选点
        for point in candidate_points:
            # 使用KD树快速查询最近邻距离
            dist, _ = tree.query(point, k=1)
            
            if dist >= min_point_distance:
                mask_centers.append(point.copy())
                # 重建KD树（对于小规模数据集，重建开销可接受）
                tree = cKDTree(mask_centers)

        return np.array(mask_centers)

    def select_initial_points_fps(self, src_points, ref_points,
                                  mask_radius, min_point_distance,
                                  max_centers=4000):
        """[备选/实验] 用最远点采样(FPS)在目标实际点上选掩码中心。

        与 select_initial_points 同签名，可直接替换调用来对比效果。

        相比原贪心版的改进：
        - 中心落在目标实际占据的点上（不是绕质心的球），贴合真实形状；
        - FPS 保证中心最大程度均匀铺开、不扎堆，且与点序无关（确定性）；
        - 掩掉一块密度后点变少 -> 覆盖区变小 -> 中心更少（符合直觉，不再爆炸）。

        停止：所有目标点都已在某中心 min_point_distance 内即停（与原语义一致，
        只是改为均匀覆盖而非贪心顺序塞）。max_centers 兜底防爆。
        """
        pts = np.asarray(ref_points, dtype=np.float64)
        if len(pts) == 0:
            return np.array([np.zeros(3)])

        centroid = pts.mean(axis=0)
        seed = int(np.argmin(np.sum((pts - centroid) ** 2, axis=1)))
        centers_idx = [seed]
        min_d2 = np.sum((pts - pts[seed]) ** 2, axis=1)
        thr2 = float(min_point_distance) ** 2

        while len(centers_idx) < max_centers:
            i = int(np.argmax(min_d2))
            if min_d2[i] < thr2:
                break
            centers_idx.append(i)
            d2 = np.sum((pts - pts[i]) ** 2, axis=1)
            min_d2 = np.minimum(min_d2, d2)

        if self.verbose:
            print("[FPS] centers=%d (min_point_distance=%.3f, target_pts=%d)"
                  % (len(centers_idx), min_point_distance, len(pts)))
        return pts[centers_idx]

    def select_initial_points_kpconv(self, src_points, ref_points,
                                     mask_radius, min_point_distance,
                                     n_iter=50, step_factor=0.3,
                                     attract=0.3, k_neighbors=13,
                                     seed=0, max_centers=4000):
        """[备选/实验] KPConv 力学式掩码中心放置（与 select_initial_points* 同签名）。

        借 KPConv 核点放置思路：中心当带电粒子两两排斥（斥力 ~ 1/d^2）使其均匀
        铺开；每步再软吸引回目标实际占据点（ref_points）最近点，把中心约束在
        密度上（等价 KPConv 的球内约束，这里换成“占据点集”约束）。收敛后中心
        均匀、不扎堆、不集中，对掩码鲁棒（点少 -> 中心少）。

        初值与数量 N：复用 select_initial_points_fps（均匀、确定、可控），再做
        n_iter 步力学松弛。确定性（无随机，seed 仅占位）。排斥用 k 近邻 O(n*k)。
        """
        np.random.seed(seed)
        pts = np.asarray(ref_points, dtype=np.float64)
        if len(pts) == 0:
            return np.array([np.zeros(3)])

        centers = np.asarray(
            self.select_initial_points_fps(src_points, pts, mask_radius,
                                           min_point_distance, max_centers),
            dtype=np.float64)
        if len(centers) <= 2:
            return centers

        ref_tree = cKDTree(pts)
        step = step_factor * float(min_point_distance)
        eps = 1e-6

        for _ in range(n_iter):
            ctree = cKDTree(centers)
            k = min(len(centers), k_neighbors)
            dists, nbrs = ctree.query(centers, k=k)
            move = np.zeros_like(centers)
            for col in range(1, k):                 # col 0 是自身
                d = dists[:, col][:, None] + eps
                move += (centers - centers[nbrs[:, col]]) / (d ** 2)
            mnorm = np.linalg.norm(move, axis=1, keepdims=True) + eps
            centers = centers + step * move / mnorm            # 排斥：均匀铺开
            _, idx = ref_tree.query(centers, k=1)
            centers = centers + attract * (pts[idx] - centers)  # 软约束回密度

        _, idx = ref_tree.query(centers, k=1)
        centers = np.unique(pts[idx], axis=0)                   # 吸附到实际点+去重

        if self.verbose:
            print("[KPConv] centers=%d (min_point_distance=%.3f, n_iter=%d)"
                  % (len(centers), min_point_distance, n_iter))
        return centers

    def generate_masks(self, src_points: np.ndarray, ref_points: np.ndarray,
                       ref_structured_data: np.ndarray,
                       mask_radius_factor: float = 1.0,
                       min_coverage: float = 0.3,
                       min_point_distance_factor: float = 0.8,
                       center_mode: str = "greedy") -> List[MaskResult]:
        """
        生成球形掩码（优化版 - 第一个掩码始终是完整的原始数据）

        Args:
            src_points: 源点云坐标数组
            ref_points: 参考点云坐标数组
            ref_structured_data: 参考点云的结构化数据
            mask_radius_factor: 掩码半径缩放因子
            min_coverage: 最小覆盖率（相对于源点云）
            min_point_distance_factor: 最小点间距因子（相对于掩码半径）

        Returns:
            掩码结果列表（第一个掩码是完整数据）
        """
        if self.verbose:
            print("=== 球形掩码生成策略 ===")

        all_masks = []
        
        # 首先创建完整数据的掩码（mask id = 0）
        centroid = np.mean(ref_points, axis=0)
        max_radius = self.calculate_max_radius(ref_points, centroid)
        full_mask = np.ones(len(ref_points), dtype=bool)
        
        full_mask_result = MaskResult(
            id=0,
            center=centroid,
            radius=max_radius,
            points_count=len(ref_points),
            point_cloud_data=ref_structured_data,
            mask=full_mask,
            strategy='full',
            type='full'
        )
        
        all_masks.append(full_mask_result)
        
        if self.verbose:
            print(f"已创建完整数据掩码 (ID=0): 包含 {len(ref_points)} 个点")
            print(f"质心: {centroid}")
            print(f"最大半径: {max_radius:.3f}")

        # 计算源点云的回转半径作为掩码半径
        gyration_radius, src_centroid = self.calculate_gyration_radius(src_points)
        mask_radius = gyration_radius * mask_radius_factor

        # 最小点间距（避免掩码中心过于密集）
        min_point_distance = mask_radius * min_point_distance_factor

        # 最小点数阈值
        min_points_threshold = int(len(src_points) * min_coverage)

        if self.verbose:
            print(f"\n=== 生成局部球形掩码 ===")
            print(f"源点云包含 {len(src_points)} 个点")
            print(f"源点云回转半径: {gyration_radius:.3f}")
            print(f"掩码球半径: {mask_radius:.3f}")
            print(f"最小点数阈值: {min_points_threshold}")

        # 选择初始点位置（掩码中心）
        if center_mode == "fps" and hasattr(self, "select_initial_points_fps"):
            mask_centers = self.select_initial_points_fps(src_points, ref_points, mask_radius, min_point_distance)
        elif center_mode == "kpconv" and hasattr(self, "select_initial_points_kpconv"):
            mask_centers = self.select_initial_points_kpconv(src_points, ref_points, mask_radius, min_point_distance)
        else:
            mask_centers = self.select_initial_points(src_points, ref_points, mask_radius, min_point_distance)

        if self.verbose:
            print(f"找到 {len(mask_centers)} 个潜在的掩码中心")

        # 批量生成球形掩码（从 ID=1 开始）
        mask_id = 1
        
        # 预计算平方半径以避免重复计算
        radius_squared = mask_radius ** 2
        
        # 向量化处理所有掩码中心
        for center in mask_centers:
            # 向量化计算距离平方（避免平方根）
            distances_squared = np.sum((ref_points - center) ** 2, axis=1)
            mask = distances_squared <= radius_squared
            points_count = np.sum(mask)

            # 检查点数是否满足最小阈值
            if points_count >= min_points_threshold:
                # 提取掩码区域内的点云数据
                masked_data = ref_structured_data[mask]

                mask_result = MaskResult(
                    id=mask_id,
                    center=center,
                    radius=mask_radius,
                    points_count=points_count,
                    point_cloud_data=masked_data,
                    mask=mask,
                    strategy='spherical',
                    type='sphere'
                )

                all_masks.append(mask_result)
                mask_id += 1

        if self.verbose:
            print(f"生成了 {len(all_masks) - 1} 个有效的球形掩码")
            print(f"总掩码数量: {len(all_masks)} (包含1个完整掩码)")

        return all_masks


def generate_spherical_masks(source_data: Union[str, np.ndarray],
                             target_data: Union[str, np.ndarray],
                             mask_radius_factor: float = 1.0,
                             min_coverage: float = 0.3,
                             min_point_distance_factor: float = 0.8,
                             verbose: bool = True,
                             center_mode: str = "greedy") -> List[MaskResult]:
    """
    主函数:生成球形掩码(简化接口)
    注意：第一个掩码(ID=0)始终是完整的原始数据

    Args:
        source_data: 源点云数据,可以是文件路径或numpy数组
        target_data: 目标点云数据,可以是文件路径或numpy数组
        mask_radius_factor: 掩码半径缩放因子(基于回转半径)
        min_coverage: 最小覆盖率
        min_point_distance_factor: 最小点间距因子
        verbose: 是否打印详细信息

    Returns:
        掩码结果列表（第一个掩码是完整数据）

    Examples:
        # 使用文件路径
        masks = generate_spherical_masks("source.txt", "target.txt")
        # masks[0] 是完整数据, masks[1:] 是局部球形掩码

        # 使用numpy数组
        masks = generate_spherical_masks(src_points, tgt_points)

        # 自定义参数
        masks = generate_spherical_masks(
            "source.txt", "target.txt",
            mask_radius_factor=1.3,
            min_coverage=0.25,
            verbose=False
        )
    """
    generator = SphericalMaskGenerator(verbose=verbose)

    # 处理输入数据
    if isinstance(source_data, str):
        src_structured = generator.load_sample_points(source_data)
        src_points = src_structured['point']
    else:
        src_points = source_data
        src_structured = None

    if isinstance(target_data, str):
        tgt_structured = generator.load_sample_points(target_data)
        ref_points = tgt_structured['point']
    else:
        # 如果是numpy数组,需要创建基本的结构化数据
        ref_points = target_data
        if target_data.shape[1] == 3:  # 只有坐标数据
            n_points = len(target_data)
            dtype = [('index', np.int32),
                     ('point', np.float32, (3,)),
                     ('vector', np.float32, (3,)),
                     ('density', np.float32)]
            tgt_structured = np.zeros(n_points, dtype=dtype)
            tgt_structured['index'] = np.arange(n_points)
            tgt_structured['point'] = target_data
            # 设置默认向量和密度值
            tgt_structured['vector'] = np.zeros((n_points, 3), dtype=np.float32)
            tgt_structured['density'] = np.ones(n_points, dtype=np.float32)
        else:
            raise ValueError("如果使用numpy数组,需要提供形状为(N, 3)的坐标数据")

    if verbose:
        print(f"源点云形状: {src_points.shape}")
        print(f"目标点云形状: {ref_points.shape}")

    # 生成掩码（第一个掩码始终是完整数据）
    masks = generator.generate_masks(
        src_points, ref_points, tgt_structured,
        mask_radius_factor=mask_radius_factor,
        min_coverage=min_coverage,
        min_point_distance_factor=min_point_distance_factor,
        center_mode=center_mode
    )

    return masks


# 保存功能（可选 - 优化版）
def save_masks(masks: List[MaskResult], output_dir: str,
               save_individual_files: bool = True) -> str:
    """
    保存掩码结果到文件（优化版）

    Args:
        masks: 掩码结果列表
        output_dir: 输出目录
        save_individual_files: 是否保存每个掩码的单独文件

    Returns:
        输出目录路径
    """
    os.makedirs(output_dir, exist_ok=True)

    # 保存所有掩码中心位置（向量化）
    all_centers = np.array([mask.center for mask in masks])
    centers_path = os.path.join(output_dir, "all_mask_centers.txt")
    np.savetxt(centers_path, all_centers,
               header="Mask centers coordinates (x y z)", fmt='%.6f')

    if save_individual_files:
        # 为每个掩码保存点云数据
        for mask in masks:
            mask_path = os.path.join(output_dir, f"masked_points_{mask.id:04d}.txt")
            _save_point_cloud_as_txt(mask.point_cloud_data, mask_path)

    # 生成总结报告（优化写入）
    report_path = os.path.join(output_dir, "masks_report.txt")
    
    # 预构建报告内容
    report_lines = [
        "# 球形掩码生成报告\n\n",
        f"总掩码数量: {len(masks)}\n",
        f"第一个掩码 (ID=0): 完整数据 ({masks[0].points_count} 个点)\n"
    ]
    
    if len(masks) > 1:
        report_lines.append(f"球形掩码半径: {masks[1].radius:.3f}\n\n")
    else:
        report_lines.append("\n")
    
    report_lines.append("ID\tType\tCenter_X\tCenter_Y\tCenter_Z\tPoints_Count\tRadius\n")
    
    for mask in masks:
        center = mask.center
        report_lines.append(
            f"{mask.id}\t{mask.type}\t{center[0]:.3f}\t{center[1]:.3f}\t"
            f"{center[2]:.3f}\t{mask.points_count}\t{mask.radius:.3f}\n"
        )
    
    # 一次性写入
    with open(report_path, 'w') as f:
        f.writelines(report_lines)

    print(f"保存了 {len(masks)} 个掩码到 {output_dir}")
    print(f"  - 1 个完整数据掩码")
    print(f"  - {len(masks) - 1} 个球形掩码")
    return output_dir


def _save_point_cloud_as_txt(structured_data: np.ndarray, output_path: str):
    """把点云结构化数组写出为 TXT（坐标按 Å 原样写出，sample=1.0）。"""
    write_point_cloud(output_path,
                      points=structured_data["point"],
                      vectors=structured_data["vector"],
                      densities=structured_data["density"],
                      indices=structured_data["index"])


# 使用示例：python sw_mask.py --source chain.txt --target map.txt [--out DIR]
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="球形掩码生成示例")
    parser.add_argument("--source", required=True, help="源点云 TXT（链模板）")
    parser.add_argument("--target", required=True, help="目标点云 TXT（密度图）")
    parser.add_argument("--out", default=None, help="掩码输出目录（缺省不保存）")
    args = parser.parse_args()

    masks = generate_spherical_masks(
        args.source, args.target,
        mask_radius_factor=1.45,
        min_coverage=0.3,
        min_point_distance_factor=0.45,
        verbose=True
    )

    print("\n生成的掩码信息:")
    print("masks[0] (完整数据): %d 个点, type=%s" % (masks[0].points_count, masks[0].type))
    for i in range(1, len(masks)):
        print("masks[%d] (球形掩码): %d 个点, radius=%.3f"
              % (i, masks[i].points_count, masks[i].radius))

    if args.out:
        save_masks(masks, args.out)
