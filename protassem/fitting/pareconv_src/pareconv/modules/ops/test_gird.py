import pdb
from functools import partial

import numpy as np
import torch

from pareconv.modules.ops import grid_subsample_with_density, radius_search
from pareconv.utils.torch import build_dataloader

batch_size = 2
points_per_batch = 100
N = batch_size * points_per_batch

# 随机生成点云和密度
points = torch.rand(N, 3) * 10.0   # 均匀分布在 [0,10) 的 3D 点
densities = torch.rand(N) * 5.0     # 随机密度值
# lengths 表示每个 batch 的点数
lengths = torch.tensor([points_per_batch] * batch_size, dtype=torch.long)
voxel_size = 2.0

# 调用下采样
s_points, s_densities, s_lengths = grid_subsample_with_density(
    points, densities, lengths, voxel_size
)

# 输出结果信息
print(f"原始点总数: {N}")
print(f"下采样后点总数: {s_points.shape[0]}")
print(f"下采样后 batch lengths: {s_lengths.tolist()}")

# 打印前 5 个结果样本
print("采样后前 5 个点坐标和密度:")
for i in range(min(5, s_points.shape[0])):
    coord = s_points[i].tolist()
    dens = float(s_densities[i])
    print(f"Point {i}: {coord}, Density={dens:.3f}")