import torch
from grid_subsample import grid_subsample

# 创建带有密度的点云数据
# 方法1：直接创建带密度的点云
points_with_density = torch.tensor([
    [2, 2, 2, 0],  # x, y, z, density
    [2, 2, 2, 1],
    [2, 2, 2, 2],  # x, y, z, density
    [2, 2, 2, 1],
    # ...
], dtype=torch.float32)

# 方法2：添加密度到现有点云
#points = torch.randn(10, 3)  # 原始点云 [N, 3]
#density = torch.rand(10, 1)  # 点云密度 [N, 1]
#points_with_density = torch.cat([points, density], dim=1)  # [N, 4]

lengths = torch.tensor([points_with_density.shape[0]], dtype=torch.long)
print("lengths",lengths)
voxel_size = 1
print("points_with_density",points_with_density.shape)

# 下采样点云，同时计算平均密度
s_points, s_lengths = grid_subsample(points_with_density, lengths, voxel_size)

print("s_points",s_points.shape,s_lengths)
print("s_points",s_points)
# s_points: [M, 4]，其中第4列是平均密度值