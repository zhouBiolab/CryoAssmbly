"""T08：位姿假设评分分块（不改共享 pareconv，只在本仓库做包装）。

原实现一次性计算 (假设数 P × 验证点数 N) 的临时张量：

    aligned   = apply_transform(src_corr_points.unsqueeze(0), transforms)      # (P, N, 3)
    residuals = torch.linalg.norm(ref_corr_points.unsqueeze(0) - aligned, 2)   # (P, N)
    masks     = residuals < acceptance_radius                                  # (P, N)
    counts    = masks.sum(dim=1)                                               # (P,)

P 可达 2200（`fine_matching.num_hypotheses`），N 为验证对应数（可达数千）：
(P, N, 3) 的 float32 在 2200×2200 时约 58 MB，(P, N) 的中间量另占数十 MB。

本模块按**假设**分块：每个假设的计算彼此独立——同一 (3,3)×(3,N) 变换、同样的行内归约，
因此分块与整批**逐位一致**；argmax 用严格大于比较，保留"首个最大值"，
与 `counts.argmax()` 的 tie-break 相同。

`chunk_size <= 0` 或 `chunk_size >= P` 时走原整批路径（默认即原路径）。
只分块独立的假设维度：不减少假设数、不改评分、不改全局顺序。
"""

import torch
from pareconv.modules.ops import apply_transform


def select_best_hypothesis(ref_corr_points, src_corr_points, transforms,
                           acceptance_radius, chunk_size=0):
    """在假设集合里挑"内点最多"的那个，返回 (best_index, best_inlier_mask)。

    best_inlier_mask 是 (N,) 的 bool 掩码，等价于整批路径的 `masks[best_index]`。
    """
    num_hypotheses = int(transforms.shape[0])
    if num_hypotheses == 0:
        raise ValueError("transforms 为空：没有可评分的假设")

    if chunk_size <= 0 or chunk_size >= num_hypotheses:
        aligned = apply_transform(src_corr_points.unsqueeze(0), transforms)
        residuals = torch.linalg.norm(ref_corr_points.unsqueeze(0) - aligned, dim=2)
        masks = torch.lt(residuals, acceptance_radius)
        best_index = int(masks.sum(dim=1).argmax())
        return best_index, masks[best_index]

    best_index = 0
    best_count = None
    best_mask = None
    for start in range(0, num_hypotheses, chunk_size):
        stop = min(start + chunk_size, num_hypotheses)
        chunk = transforms[start:stop]
        aligned = apply_transform(src_corr_points.unsqueeze(0), chunk)
        residuals = torch.linalg.norm(ref_corr_points.unsqueeze(0) - aligned, dim=2)
        masks = torch.lt(residuals, acceptance_radius)
        counts = masks.sum(dim=1)
        local_index = int(counts.argmax())
        local_count = counts[local_index]
        # 严格大于才替换 → 与整批 argmax 一样取"首个最大值"
        if best_count is None or bool(local_count > best_count):
            best_count = local_count
            best_index = start + local_index
            best_mask = masks[local_index]
    return best_index, best_mask
