import pdb
from functools import partial

import numpy as np
import torch

from pareconv.modules.ops import grid_subsample, radius_search
from pareconv.utils.torch import build_dataloader

from typing import Tuple, List

try:
    from pytorch3d.ops import sample_farthest_points

    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False
    print("Warning: pytorch3d not available, falling back to CPU FPS implementation")

# Stack mode utilities

# 定义所有的voxel_size配置
VOXEL_SIZE_CONFIGS = {
    0: [2, 3.6, 6.48, 11.664],
    1: [1, 1, 3.6, 6.48],
    2: [1, 1, 4, 12],
    3: [1, 2, 4, 11],
    4: [1, 3.6, 6.48, 11.664]
}

# 全局配置变量
CURRENT_CONFIG_ID = 0
CURRENT_SAMPLING_METHOD = 'voxel'

# 兼容性全局变量（用于不同的命名方式）
current_config_id = 0
current_sampling_method = 'voxel'


def farthest_point_sampling(points: torch.Tensor, lengths: torch.Tensor, target_ratio: float = 0.5) -> Tuple[
    torch.Tensor, torch.Tensor]:
    """
    最远点采样算法 - 根据原始点数比例进行采样

    Args:
        points: (N, D) 所有点的坐标，D通常是3或更多维度
        lengths: (B,) 每个批次中点的数量
        target_ratio: 采样比例，默认为0.5（减半）

    Returns:
        sampled_points: 采样后的点
        sampled_lengths: 采样后每个批次的点数
    """
    device = points.device
    batch_size = lengths.shape[0]

    # 只使用xyz坐标计算距离
    xyz_points = points[:, :3]

    sampled_indices_list = []
    sampled_lengths = []

    start_idx = 0
    for b in range(batch_size):
        end_idx = start_idx + lengths[b].item()
        batch_points = xyz_points[start_idx:end_idx]
        batch_size_points = batch_points.shape[0]

        # 根据原始点数计算该batch需要采样的点数
        batch_num_samples = max(1, int(batch_size_points * target_ratio))

        if batch_num_samples >= batch_size_points:
            # 如果需要采样的点数大于等于原始点数，直接返回所有点
            sampled_indices = torch.arange(start_idx, end_idx, device=device)
        else:
            # FPS采样
            sampled_indices = fps_single_batch(batch_points, batch_num_samples, start_idx)

        sampled_indices_list.append(sampled_indices)
        sampled_lengths.append(sampled_indices.shape[0])
        start_idx = end_idx

    # 合并所有采样的索引
    all_sampled_indices = torch.cat(sampled_indices_list, dim=0)
    sampled_points = points[all_sampled_indices]
    sampled_lengths = torch.tensor(sampled_lengths, dtype=torch.long, device=device)

    return sampled_points, sampled_lengths


def farthest_point_sampling_gpu(points: torch.Tensor, lengths: torch.Tensor, target_ratio: float = 0.25) -> Tuple[
    torch.Tensor, torch.Tensor]:
    """
    最远点采样算法 - 使用PyTorch3D的GPU加速版本（如果可用）

    Args:
        points: (N, D) 所有点的坐标，D通常是3或更多维度
        lengths: (B,) 每个批次中点的数量
        target_ratio: 采样比例，默认为0.25

    Returns:
        sampled_points: 采样后的点
        sampled_lengths: 采样后每个批次的点数
    """
    device = points.device
    batch_size = lengths.shape[0]

    sampled_points_list = []
    sampled_lengths = []

    start_idx = 0
    for b in range(batch_size):
        end_idx = start_idx + lengths[b].item()
        batch_points = points[start_idx:end_idx]
        batch_size_points = batch_points.shape[0]

        # 根据原始点数计算该batch需要采样的点数
        batch_num_samples = max(1, int(batch_size_points * target_ratio))

        if batch_num_samples >= batch_size_points:
            # 如果需要采样的点数大于等于原始点数，直接返回所有点
            sampled_points_list.append(batch_points)
            sampled_lengths.append(batch_size_points)
        else:
            if PYTORCH3D_AVAILABLE:
                # 使用PyTorch3D的GPU加速FPS
                # 需要将点云reshape为 (1, N, 3) 格式
                batch_points_3d = batch_points[:, :3].unsqueeze(0)  # 只使用xyz坐标

                # PyTorch3D的FPS返回采样后的点和索引
                sampled_pts, sampled_idx = sample_farthest_points(
                    batch_points_3d,
                    K=batch_num_samples,
                    random_start_point=True
                )

                # 获取完整的特征（不仅仅是xyz）
                sampled_idx = sampled_idx[0]  # 去掉batch维度
                sampled_full_features = batch_points[sampled_idx]

                sampled_points_list.append(sampled_full_features)
                sampled_lengths.append(batch_num_samples)
            else:
                # 使用CPU版本的FPS
                sampled_indices = fps_single_batch(batch_points[:, :3], batch_num_samples, 0)
                sampled_full_features = batch_points[sampled_indices]

                sampled_points_list.append(sampled_full_features)
                sampled_lengths.append(batch_num_samples)

        start_idx = end_idx

    # 合并所有采样的点
    sampled_points = torch.cat(sampled_points_list, dim=0)
    sampled_lengths = torch.tensor(sampled_lengths, dtype=torch.long, device=device)

    return sampled_points, sampled_lengths


def fps_single_batch(points: torch.Tensor, num_samples: int, offset: int = 0) -> torch.Tensor:
    """
    对单个batch进行FPS采样

    Args:
        points: (N, 3) 点云坐标
        num_samples: 采样点数
        offset: 全局索引偏移量（默认为0）

    Returns:
        sampled_indices: 采样点的索引
    """
    device = points.device
    N = points.shape[0]

    # 初始化距离矩阵，所有点到采样集的最小距离
    distances = torch.full((N,), float('inf'), device=device)

    # 随机选择第一个点
    farthest_idx = torch.randint(0, N, (1,), device=device).item()
    sampled_indices = [farthest_idx]

    for _ in range(num_samples - 1):
        # 计算当前最远点到所有点的距离
        current_point = points[farthest_idx:farthest_idx + 1]
        dist_to_current = torch.sum((points - current_point) ** 2, dim=1)

        # 更新每个点到采样集的最小距离
        distances = torch.minimum(distances, dist_to_current)

        # 选择距离采样集最远的点
        farthest_idx = torch.argmax(distances).item()
        sampled_indices.append(farthest_idx)

    # 转换为索引张量
    sampled_indices = torch.tensor(sampled_indices, dtype=torch.long, device=device)

    # 如果有偏移量，加上偏移量
    if offset > 0:
        sampled_indices = sampled_indices + offset

    return sampled_indices


def get_current_config():
    """获取当前配置，兼容不同的全局变量命名方式"""
    global CURRENT_CONFIG_ID, CURRENT_SAMPLING_METHOD, current_config_id, current_sampling_method

    # 优先使用大写的变量
    config_id = CURRENT_CONFIG_ID
    sampling_method = CURRENT_SAMPLING_METHOD

    # 如果大写变量没有被设置，使用小写变量
    if config_id == 0 and current_config_id != 0:
        config_id = current_config_id
    if sampling_method == 'voxel' and current_sampling_method != 'voxel':
        sampling_method = current_sampling_method

    return config_id, sampling_method


def precompute_subsample(points, lengths, num_stages, voxel_size_list, num_neighbors, subsample_ratio):
    """
    使用全局配置变量指定的配置和采样方式进行下采样

    Args:
        points: 输入点云
        lengths: 每个batch的点数
        num_stages: 下采样阶段数
        voxel_size_list: 体素大小列表（兼容性参数，实际使用全局配置）
        num_neighbors: 邻居数列表
        subsample_ratio: 下采样比例

    Returns:
        dict: 包含points和lengths列表的字典
    """
    config_id, sampling_method = get_current_config()

    assert num_stages == len(num_neighbors)

    # 获取对应的voxel_size配置
    current_voxel_size_list = VOXEL_SIZE_CONFIGS.get(config_id, VOXEL_SIZE_CONFIGS[0])

    points_list = []
    lengths_list = []

    current_points = points
    current_lengths = lengths

    for i in range(num_stages):
        if i == 0:
            # 第0层不下采样，直接添加原始点
            points_list.append(current_points)
            lengths_list.append(current_lengths)

        elif i < len(current_voxel_size_list)-1:

            # 使用对应配置的体素大小进行网格下采样
            current_voxel_size = current_voxel_size_list[i]
            current_points, current_lengths = grid_subsample(
                current_points, current_lengths, voxel_size=current_voxel_size
            )
            points_list.append(current_points)
            lengths_list.append(current_lengths)

        else:
            # 最后一层：根据sampling_method选择采样方式
            if sampling_method == 'voxel':
                # 使用最后一个voxel_size继续体素下采样

                last_voxel_size = current_voxel_size_list[-1]
                current_points, current_lengths = grid_subsample(
                    current_points, current_lengths, voxel_size=last_voxel_size
                )
            elif sampling_method == 'fps':
                # 使用最远点采样

                current_points, current_lengths = farthest_point_sampling_gpu(
                    current_points, current_lengths, target_ratio=0.25#0.25
                )
            else:
                # 默认使用体素采样

                last_voxel_size = current_voxel_size_list[-1]
                current_points, current_lengths = grid_subsample(
                    current_points, current_lengths, voxel_size=last_voxel_size
                )

            points_list.append(current_points)
            lengths_list.append(current_lengths)

    return {
        'points': points_list,
        'lengths': lengths_list,
    }


def precompute_neibors(points_list, lengths_list, num_stages, num_neighbors):
    """
    预计算邻居关系
    """
    neighbors_list = []
    subsampling_list = []
    upsampling_list = []

    # knn search
    for i in range(num_stages):
        cur_points = points_list[i][:, :3]
        cur_lengths = lengths_list[i]
        if i < num_stages:  # without adaptive sampling  i < num_stages, with: i < num_stages
            neighbors = radius_search(
                cur_points,
                cur_points,
                cur_lengths,
                cur_lengths,
                num_neighbors[i],
            )
            neighbors_list.append(neighbors)

        if i < num_stages - 1:
            sub_points = points_list[i + 1]
            sub_lengths = lengths_list[i + 1]

            subsampling = radius_search(
                sub_points,
                cur_points,
                sub_lengths,
                cur_lengths,
                num_neighbors[i],
            )
            subsampling_list.append(subsampling)

            if i > 0:
                upsampling = radius_search(
                    cur_points,
                    sub_points,
                    cur_lengths,
                    sub_lengths,
                    1,
                )
                upsampling_list.append(upsampling)
    return {
        'neighbors': neighbors_list,
        'subsampling': subsampling_list,
        'upsampling': upsampling_list,
    }


def registration_collate_fn_stack_mode(
        data_dicts, num_stages, voxel_size, num_neighbors, subsample_ratio, precompute_data=True
):
    """
    Collate function for registration in stack mode.

    Points are organized in the following order: [ref_1, ..., ref_B, src_1, ..., src_B].
    The correspondence indices are within each point cloud without accumulation.

    Args:
        data_dicts (List[Dict])
        num_stages (int)
        voxel_size (float): 兼容性参数，实际使用全局配置
        num_neighbors (List[int])
        subsample_ratio: 下采样比例
        precompute_data (bool)
    Returns:
        collated_dict (Dict)
    """
    batch_size = len(data_dicts)

    # merge data with the same key from different samples into a list
    collated_dict = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if key not in collated_dict:
                collated_dict[key] = []
            collated_dict[key].append(value)

    # handle special keys: [ref_feats, src_feats] -> feats, [ref_points, src_points] -> points, lengths
    feats = torch.cat(collated_dict.pop('ref_feats') + collated_dict.pop('src_feats'), dim=0)
    points_list = collated_dict.pop('ref_points') + collated_dict.pop('src_points')
    lengths = torch.LongTensor([points.shape[0] for points in points_list])
    points = torch.cat(points_list, dim=0)

    if batch_size == 1:
        # remove wrapping brackets if batch_size is 1
        for key, value in collated_dict.items():
            collated_dict[key] = value[0]

    collated_dict['features'] = feats
    if precompute_data:
        with torch.no_grad():
            # 使用原来的函数签名，配置信息通过全局变量传递
            input_dict = precompute_subsample(
                points, lengths, num_stages, voxel_size, num_neighbors, subsample_ratio
            )
            torch.cuda.empty_cache()

            collated_dict.update(input_dict)
    else:
        collated_dict['points'] = points
        collated_dict['lengths'] = lengths
    collated_dict['batch_size'] = batch_size

    return collated_dict


def calibrate_neighbors_stack_mode(
        dataset, collate_fn, num_stages, voxel_size, search_radius, keep_ratio=0.8, sample_threshold=2000
):
    # Compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (search_radius / voxel_size + 1) ** 2))
    neighbor_hists = np.zeros((num_stages, hist_n), dtype=np.int32)
    max_neighbor_limits = [hist_n] * num_stages

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i in range(len(dataset)):
        data_dict = collate_fn(
            [dataset[i]], num_stages, voxel_size, search_radius, max_neighbor_limits, precompute_data=True
        )

        # update histogram
        counts = [np.sum(neighbors.numpy() < neighbors.shape[0], axis=1) for neighbors in data_dict['neighbors']]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighbor_hists += np.vstack(hists)

        if np.min(np.sum(neighbor_hists, axis=1)) > sample_threshold:
            break

    cum_sum = np.cumsum(neighbor_hists.T, axis=0)
    neighbor_limits = np.sum(cum_sum < (keep_ratio * cum_sum[hist_n - 1, :]), axis=0)

    return neighbor_limits


def build_dataloader_stack_mode(
        dataset,
        collate_fn,
        num_stages,
        voxel_size,
        num_neighbors,
        subsample_ratio,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        drop_last=False,
        distributed=False,
        precompute_data=True,
):
    """
    构建数据加载器，配置信息通过全局变量传递
    """
    dataloader = build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=partial(
            collate_fn,
            num_stages=num_stages,
            voxel_size=voxel_size,
            num_neighbors=num_neighbors,
            subsample_ratio=subsample_ratio,
            precompute_data=precompute_data,
        ),
        drop_last=drop_last,
        distributed=distributed,
    )
    return dataloader