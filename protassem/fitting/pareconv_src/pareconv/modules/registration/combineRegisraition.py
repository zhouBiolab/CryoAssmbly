import pdb
from typing import Optional
import open3d as o3d
from typing import Union, Tuple, List, Dict
import numpy as np
import torch
import torch.nn as nn

from pareconv.modules.ops import apply_transform
from pareconv.modules.registration import WeightedProcrustes, solve_local_rotations
from pareconv.modules.registration.Correspondence_regenerate_v2 import Regenerator


def compute_overlap_torch(src: torch.Tensor,
                          tgt: torch.Tensor,
                          search_radius: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算两个点云之间的重叠区域（torch tensor版本），并返回：
      - has_corr_src: 源点云中有对应匹配的点（布尔张量）
      - has_corr_tgt: 目标点云中有对应匹配的点（布尔张量）
      - src_tgt_corr: 形状为 (2, n_corr) 的张量，其中每列 [i, j] 表示源点 i 与目标点 j 互为匹配

    Args:
        src: 源点云 (N, 3) 或 (B, N, 3)
        tgt: 目标点云 (M, 3) 或 (B, M, 3)
        search_radius: 搜索半径
    """
    try:
        # 处理batch维度
        if src.dim() == 3:
            # 假设batch size为1，去掉batch维度
            src = src.squeeze(0)
        if tgt.dim() == 3:
            tgt = tgt.squeeze(0)

        device = src.device
        src_size = src.shape[0]
        tgt_size = tgt.shape[0]

        # 计算距离矩阵
        # src: (N, 3), tgt: (M, 3)
        # distances: (N, M)
        distances = torch.cdist(src, tgt, p=2)

        # 找到每个目标点的最近源点
        tgt_corr = torch.full((tgt_size,), -1, dtype=torch.long, device=device)
        tgt_min_dist, tgt_min_idx = torch.min(distances, dim=0)  # (M,)
        valid_tgt = tgt_min_dist < search_radius
        tgt_corr[valid_tgt] = tgt_min_idx[valid_tgt]

        # 找到每个源点的最近目标点
        src_corr = torch.full((src_size,), -1, dtype=torch.long, device=device)
        src_min_dist, src_min_idx = torch.min(distances, dim=1)  # (N,)
        valid_src = src_min_dist < search_radius
        src_corr[valid_src] = src_min_idx[valid_src]

        # 找到互相匹配的点对
        src_indices = torch.arange(src_size, device=device)
        mutual = torch.zeros(src_size, dtype=torch.bool, device=device)

        # 检查互相匹配：对于每个有效的源点，检查其对应的目标点是否也指向它
        valid_src_mask = src_corr >= 0
        if valid_src_mask.any():
            # 获取有效源点的目标点对应关系
            valid_src_indices = src_indices[valid_src_mask]
            valid_tgt_indices = src_corr[valid_src_mask]

            # 检查这些目标点是否也指向相应的源点
            mutual_mask = tgt_corr[valid_tgt_indices] == valid_src_indices
            mutual[valid_src_mask] = mutual_mask

        # 构建对应关系矩阵
        mutual_indices = torch.nonzero(mutual, as_tuple=False).squeeze(-1)
        if mutual_indices.numel() > 0:
            src_tgt_corr = torch.stack([mutual_indices, src_corr[mutual_indices]], dim=0)
        else:
            src_tgt_corr = torch.empty((2, 0), dtype=torch.long, device=device)

        has_corr_src = src_corr >= 0
        has_corr_tgt = tgt_corr >= 0

        return has_corr_src, has_corr_tgt, src_tgt_corr

    except Exception as e:
        print(f"计算重叠率时出错: {e}")
        return None, None, None


def extract_mutual_correspondences(src: torch.Tensor,
                                   tgt: torch.Tensor,
                                   search_radius: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    提取两个点云之间的互相对应点对

    Args:
        src: 源点云 (N, 3) 或 (B, N, 3)
        tgt: 目标点云 (M, 3) 或 (B, M, 3)
        search_radius: 搜索半径

    Returns:
        src_corr_points: 源点云中的对应点 (K, 3)
        tgt_corr_points: 目标点云中的对应点 (K, 3)
    """
    has_corr_src, has_corr_tgt, src_tgt_corr = compute_overlap_torch(src, tgt, search_radius)

    if src_tgt_corr is None or src_tgt_corr.shape[1] == 0:
        # 没有找到对应关系，返回空张量
        device = src.device
        return torch.empty((0, 3), device=device), torch.empty((0, 3), device=device)

    # 处理batch维度
    if src.dim() == 3:
        src = src.squeeze(0)
    if tgt.dim() == 3:
        tgt = tgt.squeeze(0)

    # 提取对应点
    src_indices = src_tgt_corr[0]  # 源点索引
    tgt_indices = src_tgt_corr[1]  # 目标点索引

    src_corr_points = src[src_indices].unsqueeze(0)
    tgt_corr_points = tgt[tgt_indices].unsqueeze(0)

    return src_corr_points, tgt_corr_points
def get_rotation_from_transform(transform: torch.Tensor) -> torch.Tensor:
    """
    从刚性变换矩阵中提取旋转部分。

    Args:
        transform: (4,4) 或 (B,4,4) 的刚性变换矩阵

    Returns:
        rotation: (3,3) 或 (B,3,3) 的旋转矩阵
    """
    if transform.ndim == 2:
        return transform[:3, :3]
    elif transform.ndim == 3:
        return transform[:, :3, :3]
    else:
        raise ValueError(f"Unsupported transform shape {transform.shape}")

def rotate_equivariant_feats(
    re_feats: torch.Tensor,
    lgr_transform: torch.Tensor
) -> torch.Tensor:
    """
    只把 LGR 的旋转作用到等变特征上（去掉平移）。

    Args:
        re_feats: (B, K, D, 3)，等变特征的 xyz 分量在最后一维
        lgr_transform: (B, 4, 4) 或 (4, 4)，LGR 输出的刚性变换

    Returns:
        re_feats_rot: (B, K, D, 3)，旋转后的等变特征
    """
    # 提取旋转
    R = get_rotation_from_transform(lgr_transform)  # (3,3) or (B,3,3)

    B, K, D, _ = re_feats.shape
    # 拉平成 (B, K*D, 3)
    feats_flat = re_feats.view(B, -1, 3)  # (B, K*D, 3)

    # 对点（行向量）做右乘 R^T，即 feats @ R^T
    if R.ndim == 2:
        # 单一旋转矩阵
        feats_rot = torch.matmul(feats_flat, R.transpose(0, 1))
    else:
        # batch 每帧各自的旋转
        feats_rot = torch.matmul(feats_flat, R.transpose(-1, -2))

    # 恢复形状
    return feats_rot.view(B, K, D, 3)
class LocalGlobalRegistration(nn.Module):
    def __init__(
            self,
            k: int,
            acceptance_radius: float,
            mutual: bool = True,
            confidence_threshold: float = 0.05,
            use_dustbin: bool = False,
            use_global_score: bool = False,
            correspondence_threshold: int = 3,
            correspondence_limit: Optional[int] = None,
            num_refinement_steps: int = 5,
    ):
        r"""Point Matching with Local-to-Global Registration.

        Args:
            k (int): top-k selection for matching.
            acceptance_radius (float): acceptance radius for LGR.
            mutual (bool=True): mutual or non-mutual matching.
            confidence_threshold (float=0.05): ignore matches whose scores are below this threshold.
            use_dustbin (bool=False): whether dustbin row/column is used in the score matrix.
            use_global_score (bool=False): whether use patch correspondence scores.
            correspondence_threshold (int=3): minimal number of correspondences for each patch correspondence.
            correspondence_limit (optional[int]=None): maximal number of verification correspondences.
            num_refinement_steps (int=5): number of refinement steps.
        """
        super(LocalGlobalRegistration, self).__init__()
        self.k = k
        self.acceptance_radius = acceptance_radius
        self.mutual = mutual
        self.confidence_threshold = confidence_threshold
        self.use_dustbin = use_dustbin
        self.use_global_score = use_global_score
        self.correspondence_threshold = correspondence_threshold
        self.correspondence_limit = correspondence_limit
        self.num_refinement_steps = num_refinement_steps
        self.procrustes = WeightedProcrustes(return_transform=True)

    def compute_correspondence_matrix(self, score_mat, ref_knn_masks, src_knn_masks):
        r"""Compute matching matrix and score matrix for each patch correspondence."""
        mask_mat = torch.logical_and(ref_knn_masks.unsqueeze(2), src_knn_masks.unsqueeze(1))
        #print("mask_mat",mask_mat.shape) 256,64,64
        batch_size, ref_length, src_length = score_mat.shape
        batch_indices = torch.arange(batch_size).cuda()

        # correspondences from reference side
        ref_topk_scores, ref_topk_indices = score_mat.topk(k=self.k, dim=2)  # (B, N, K)
        ref_batch_indices = batch_indices.view(batch_size, 1, 1).expand(-1, ref_length, self.k)  # (B, N, K)
        ref_indices = torch.arange(ref_length).cuda().view(1, ref_length, 1).expand(batch_size, -1, self.k)  # (B, N, K)
        ref_score_mat = torch.zeros_like(score_mat)
        ref_score_mat[ref_batch_indices, ref_indices, ref_topk_indices] = ref_topk_scores
        ref_corr_mat = torch.gt(ref_score_mat, self.confidence_threshold)

        # correspondences from source side
        src_topk_scores, src_topk_indices = score_mat.topk(k=self.k, dim=1)  # (B, K, N)
        src_batch_indices = batch_indices.view(batch_size, 1, 1).expand(-1, self.k, src_length)  # (B, K, N)
        src_indices = torch.arange(src_length).cuda().view(1, 1, src_length).expand(batch_size, self.k, -1)  # (B, K, N)
        src_score_mat = torch.zeros_like(score_mat)
        src_score_mat[src_batch_indices, src_topk_indices, src_indices] = src_topk_scores
        src_corr_mat = torch.gt(src_score_mat, self.confidence_threshold)

        # merge results from two sides
        if self.mutual:
            corr_mat = torch.logical_and(ref_corr_mat, src_corr_mat)
        else:
            corr_mat = torch.logical_or(ref_corr_mat, src_corr_mat)

        if self.use_dustbin:
            corr_mat = corr_mat[:, :-1, :-1]

        corr_mat = torch.logical_and(corr_mat, mask_mat)

        return corr_mat

    @staticmethod
    def convert_to_batch(ref_corr_points, src_corr_points, corr_scores, chunks):
        r"""Convert stacked correspondences to batched points.

        The extracted dense correspondences from all patch correspondences are stacked. However, to compute the
        transformations from all patch correspondences in parallel, the dense correspondences need to be reorganized
        into a batch.

        Args:
            ref_corr_points (Tensor): (C, 3)
            src_corr_points (Tensor): (C, 3)
            corr_scores (Tensor): (C,)
            chunks (List[Tuple[int, int]]): the starting index and ending index of each patch correspondences.

        Returns:
            batch_ref_corr_points (Tensor): (B, K, 3), padded with zeros.
            batch_src_corr_points (Tensor): (B, K, 3), padded with zeros.
            batch_corr_scores (Tensor): (B, K), padded with zeros.
        """
        batch_size = len(chunks)
        indices = torch.cat([torch.arange(x, y) for x, y in chunks], dim=0).cuda()
        ref_corr_points = ref_corr_points[indices]  # (total, 3)
        src_corr_points = src_corr_points[indices]  # (total, 3)
        corr_scores = corr_scores[indices]  # (total,)

        max_corr = np.max([y - x for x, y in chunks])
        target_chunks = [(i * max_corr, i * max_corr + y - x) for i, (x, y) in enumerate(chunks)]
        indices = torch.cat([torch.arange(x, y) for x, y in target_chunks], dim=0).cuda()
        indices0 = indices.unsqueeze(1).expand(indices.shape[0], 3)  # (total,) -> (total, 3)
        indices1 = torch.arange(3).unsqueeze(0).expand(indices.shape[0], 3).cuda()  # (3,) -> (total, 3)

        batch_ref_corr_points = torch.zeros(batch_size * max_corr, 3).cuda()
        batch_ref_corr_points.index_put_([indices0, indices1], ref_corr_points)
        batch_ref_corr_points = batch_ref_corr_points.view(batch_size, max_corr, 3)

        batch_src_corr_points = torch.zeros(batch_size * max_corr, 3).cuda()
        batch_src_corr_points.index_put_([indices0, indices1], src_corr_points)
        batch_src_corr_points = batch_src_corr_points.view(batch_size, max_corr, 3)

        batch_corr_scores = torch.zeros(batch_size * max_corr).cuda()
        batch_corr_scores.index_put_([indices], corr_scores)
        batch_corr_scores = batch_corr_scores.view(batch_size, max_corr)

        return batch_ref_corr_points, batch_src_corr_points, batch_corr_scores

    def recompute_correspondence_scores(self, ref_corr_points, src_corr_points, corr_scores, estimated_transform):
        aligned_src_corr_points = apply_transform(src_corr_points, estimated_transform)
        corr_residuals = torch.linalg.norm(ref_corr_points - aligned_src_corr_points, dim=1)
        inlier_masks = torch.lt(corr_residuals, self.acceptance_radius)
        new_corr_scores = corr_scores * inlier_masks.float()
        return new_corr_scores

    def local_to_global_registration(self, ref_knn_points, src_knn_points, score_mat, corr_mat):
        # extract dense correspondences
        batch_indices, ref_indices, src_indices = torch.nonzero(corr_mat, as_tuple=True)
        #print("corr_mat",corr_mat.shape) 256,64,64
        global_ref_corr_points = ref_knn_points[batch_indices, ref_indices]
        #print("global_ref_corr_points",global_ref_corr_points.shape) k,3
        global_src_corr_points = src_knn_points[batch_indices, src_indices]
        global_corr_scores = score_mat[batch_indices, ref_indices, src_indices]

        # build verification set
        if self.correspondence_limit is not None and global_corr_scores.shape[0] > self.correspondence_limit:
            corr_scores, sel_indices = global_corr_scores.topk(k=self.correspondence_limit, largest=True)
            ref_corr_points = global_ref_corr_points[sel_indices]
            src_corr_points = global_src_corr_points[sel_indices]
        else:
            ref_corr_points = global_ref_corr_points
            src_corr_points = global_src_corr_points
            corr_scores = global_corr_scores

        # compute starting and ending index of each patch correspondence.
        # torch.nonzero is row-major, so the correspondences from the same patch correspondence are consecutive.
        # find the first occurrence of each batch index, then the chunk of this batch can be obtained.
        unique_masks = torch.ne(batch_indices[1:], batch_indices[:-1])
        unique_indices = torch.nonzero(unique_masks, as_tuple=True)[0] + 1
        unique_indices = unique_indices.detach().cpu().numpy().tolist()
        unique_indices = [0] + unique_indices + [batch_indices.shape[0]]
        chunks = [
            (x, y) for x, y in zip(unique_indices[:-1], unique_indices[1:]) if y - x >= self.correspondence_threshold
        ]

        batch_size = len(chunks)
        if batch_size > 0:
            # local registration
            batch_ref_corr_points, batch_src_corr_points, batch_corr_scores = self.convert_to_batch(
                global_ref_corr_points, global_src_corr_points, global_corr_scores, chunks
            )
            batch_transforms = self.procrustes(batch_src_corr_points, batch_ref_corr_points, batch_corr_scores)
            batch_aligned_src_corr_points = apply_transform(src_corr_points.unsqueeze(0), batch_transforms)
            batch_corr_residuals = torch.linalg.norm(
                ref_corr_points.unsqueeze(0) - batch_aligned_src_corr_points, dim=2
            )
            batch_inlier_masks = torch.lt(batch_corr_residuals, self.acceptance_radius)  # (P, N)
            best_index = batch_inlier_masks.sum(dim=1).argmax()
            cur_corr_scores = corr_scores * batch_inlier_masks[best_index].float()
        else:
            # degenerate: initialize transformation with all correspondences
            estimated_transform = self.procrustes(src_corr_points, ref_corr_points, corr_scores)
            cur_corr_scores = self.recompute_correspondence_scores(
                ref_corr_points, src_corr_points, corr_scores, estimated_transform
            )

        # global refinement
        estimated_transform = self.procrustes(src_corr_points, ref_corr_points, cur_corr_scores)
        for _ in range(self.num_refinement_steps - 1):
            cur_corr_scores = self.recompute_correspondence_scores(
                ref_corr_points, src_corr_points, corr_scores, estimated_transform
            )
            estimated_transform = self.procrustes(src_corr_points, ref_corr_points, cur_corr_scores)

        return global_ref_corr_points, global_src_corr_points, global_corr_scores, estimated_transform

    def forward(
            self,
            ref_knn_points,
            src_knn_points,
            ref_knn_masks,
            src_knn_masks,
            score_mat,
            global_scores,
    ):
        r"""Point Matching Module forward propagation with Local-to-Global registration.

        Args:
            ref_knn_points (Tensor): (B, K, 3)
            src_knn_points (Tensor): (B, K, 3)
            ref_knn_masks (BoolTensor): (B, K)
            src_knn_masks (BoolTensor): (B, K)
            score_mat (Tensor): (B, K, K) or (B, K + 1, K + 1), log likelihood
            global_scores (Tensor): (B,)

        Returns:
            ref_corr_points: torch.LongTensor (C, 3)
            src_corr_points: torch.LongTensor (C, 3)
            corr_scores: torch.Tensor (C,)
            estimated_transform: torch.Tensor (4, 4)
        """
        score_mat = torch.exp(score_mat)

        corr_mat = self.compute_correspondence_matrix(score_mat, ref_knn_masks, src_knn_masks)  # (B, K, K)

        if self.use_dustbin:
            score_mat = score_mat[:, :-1, :-1]
        if self.use_global_score:
            score_mat = score_mat * global_scores.view(-1, 1, 1)
        score_mat = score_mat * corr_mat.float()#？

        ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.local_to_global_registration(
            ref_knn_points, src_knn_points, score_mat, corr_mat
        )

        return ref_corr_points, src_corr_points, corr_scores, estimated_transform,corr_mat


class HypothesisProposer(nn.Module):
    def __init__(
            self,
            k: int,
            acceptance_radius: float,
            confidence_threshold: float = 0.025,
            num_hypotheses: int = 1000,
            num_refinement_steps: int = 5,
    ):
        r"""Point Matching with Local-to-Global Registration.

        Args:
            k (int): top-k selection for matching.
            acceptance_radius (float): acceptance radius for LGR.
            confidence_threshold (float=0.05): ignore matches whose scores are below this threshold.
            correspondence_limit (optional[int]=None): maximal number of verification correspondences.
            num_refinement_steps (int=5): number of refinement steps.
        """
        super(HypothesisProposer, self).__init__()
        self.k = k
        self.acceptance_radius = acceptance_radius
        self.confidence_threshold = confidence_threshold
        self.num_hypotheses = num_hypotheses
        self.num_refinement_steps = num_refinement_steps
        self.procrustes = WeightedProcrustes(return_transform=True)
        #print("num_hypotheses",num_hypotheses)
    def compute_correspondence_matrix(self, score_mat, ref_knn_masks, src_knn_masks):
        """Compute matching matrix and score matrix for each patch correspondence."""
        mask_mat = torch.logical_and(ref_knn_masks.unsqueeze(2), src_knn_masks.unsqueeze(1))

        batch_size, ref_length, src_length = score_mat.shape
        batch_indices = torch.arange(batch_size).cuda()

        # correspondences from reference side
        ref_topk_scores, ref_topk_indices = score_mat.topk(k=self.k, dim=2)  # (B, N, K)
        ref_batch_indices = batch_indices.view(batch_size, 1, 1).expand(-1, ref_length, self.k)  # (B, N, K)
        ref_indices = torch.arange(ref_length).cuda().view(1, ref_length, 1).expand(batch_size, -1, self.k)  # (B, N, K)
        ref_score_mat = torch.zeros_like(score_mat)
        ref_score_mat[ref_batch_indices, ref_indices, ref_topk_indices] = ref_topk_scores

        # correspondences from source side
        src_topk_scores, src_topk_indices = score_mat.topk(k=self.k, dim=1)  # (B, K, N)
        src_batch_indices = batch_indices.view(batch_size, 1, 1).expand(-1, self.k, src_length)  # (B, K, N)
        src_indices = torch.arange(src_length).cuda().view(1, 1, src_length).expand(batch_size, self.k, -1)  # (B, K, N)
        src_score_mat = torch.zeros_like(score_mat)
        src_score_mat[src_batch_indices, src_topk_indices, src_indices] = src_topk_scores
        # correspondences used to vote for hypotheses
        voter_corr_mat = torch.logical_or(torch.gt(ref_score_mat, self.confidence_threshold),
                                          torch.gt(src_score_mat, self.confidence_threshold))

        # top-k hypotheses used to generate hypotheses
        num_correspondences = min(self.num_hypotheses, mask_mat.sum())
        corr_scores, corr_indices = score_mat.reshape(-1).topk(k=num_correspondences, largest=True)
        batch_sel_indices = corr_indices // (score_mat.shape[1] * score_mat.shape[2])
        ref_sel_indices0 = corr_indices % (score_mat.shape[1] * score_mat.shape[2])
        ref_sel_indices = ref_sel_indices0 // (score_mat.shape[2])
        src_sel_indices = ref_sel_indices0 % score_mat.shape[1]
        corr_mat = torch.zeros_like(mask_mat, device=mask_mat.device)
        corr_mat[batch_sel_indices, ref_sel_indices, src_sel_indices] = True

        corr_mat = torch.logical_and(corr_mat, mask_mat)
        voter_corr_mat = torch.logical_and(voter_corr_mat, mask_mat)
        return corr_mat, voter_corr_mat

    def recompute_correspondence_scores(self, ref_corr_points, src_corr_points, corr_scores, estimated_transform):
        aligned_src_corr_points = apply_transform(src_corr_points, estimated_transform)
        corr_residuals = torch.linalg.norm(ref_corr_points - aligned_src_corr_points, dim=1)
        inlier_masks = torch.lt(corr_residuals, self.acceptance_radius)
        new_corr_scores = corr_scores * inlier_masks.float()
        return new_corr_scores

    def extract_fine_transforms(self, ref_corr_feats, src_corr_feats, ref_corr_points, src_corr_points):
        point_rotations = solve_local_rotations(src_corr_feats, ref_corr_feats)  # B 3 3
        aligned_src_points = torch.einsum('bmn, bn->bm', point_rotations, src_corr_points)
        t = ref_corr_points - aligned_src_points
        transforms = torch.eye(4, device=ref_corr_feats.device).unsqueeze(0).repeat(t.shape[0], 1, 1)
        transforms[:, :3, :3] = point_rotations
        transforms[:, :3, 3] = t
        return transforms

    def feature_based_hypothesis_proposer(self, ref_knn_points,
                                          src_knn_points,
                                          ref_knn_feats,
                                          src_knn_feats,
                                          score_mat,
                                          corr_mat,
                                          voter_corr_mat):
        # extract dense correspondences
        batch_indices, ref_indices, src_indices = torch.nonzero(corr_mat, as_tuple=True)
        global_ref_corr_points = ref_knn_points[batch_indices, ref_indices]
        #print("hp_global_ref_corr_points",global_ref_corr_points.shape) 2000,3
        global_src_corr_points = src_knn_points[batch_indices, src_indices]
        global_corr_scores = score_mat[batch_indices, ref_indices, src_indices]
        ref_corr_feats, src_corr_feats = ref_knn_feats[batch_indices, ref_indices], src_knn_feats[
            batch_indices, src_indices]
        #print("ref_corr_feats1111111",ref_corr_feats.shape)
        #print("2222",ref_knn_feats.shape)
        # build verification set
        batch_v_indices, ref_v_indices, src_v_indices = torch.nonzero(voter_corr_mat, as_tuple=True)
        ref_corr_points = ref_knn_points[batch_v_indices, ref_v_indices]
        src_corr_points = src_knn_points[batch_v_indices, src_v_indices]
        #print("src_corr_points",src_corr_points.shape)
        corr_scores = score_mat[batch_v_indices, ref_v_indices, src_v_indices]

        # generate hypotheses using rotation-equivarint features
        transformation_hypotheses = self.extract_fine_transforms(ref_corr_feats, src_corr_feats, global_ref_corr_points,
                                                                 global_src_corr_points)

        # select the hypothesis with the most supporter
        batch_aligned_src_corr_points = apply_transform(src_corr_points.unsqueeze(0), transformation_hypotheses)
        batch_corr_residuals = torch.linalg.norm(ref_corr_points.unsqueeze(0) - batch_aligned_src_corr_points, dim=2)
        batch_inlier_masks = torch.lt(batch_corr_residuals, self.acceptance_radius)  # (P, N)
        #ir = batch_inlier_masks.float().mean(dim=1)
        #best_index = ir.argmax()
        inlier_counts = batch_inlier_masks.long().sum(dim=1)  # 或 batch_inlier_masks.sum(dim=1)
        best_index = inlier_counts.argmax()
        cur_corr_scores = corr_scores * batch_inlier_masks[best_index].float()
        #print("src_corr_points",cur_corr_scores)
        # global refinement
        estimated_transform = self.procrustes(src_corr_points, ref_corr_points, cur_corr_scores)
        for _ in range(self.num_refinement_steps - 1):
            cur_corr_scores = self.recompute_correspondence_scores(
                ref_corr_points, src_corr_points, corr_scores, estimated_transform
            )
            estimated_transform = self.procrustes(src_corr_points, ref_corr_points, cur_corr_scores)
            #print("src_corr_points",src_corr_points.shape)

        return global_ref_corr_points, global_src_corr_points, global_corr_scores, estimated_transform, transformation_hypotheses, ref_corr_feats, src_corr_feats,


    def forward(
            self,
            ref_knn_points,
            src_knn_points,
            re_ref_knn_feats,
            re_src_knn_feats,
            ref_knn_masks,
            src_knn_masks,
            score_mat,
    ):
        r"""Point Matching Module forward propagation with Local-to-Global registration.

        Args:
            ref_knn_points (Tensor): (N, K, 3)
            src_knn_points (Tensor): (N, K, 3)
            re_ref_knn_feats (Tensor): (N, K, D, 3)
            re_src_knn_feats (Tensor): (N, K, D, 3)
            ref_knn_masks (BoolTensor): (N, K)
            src_knn_masks (BoolTensor): (N, K)
            score_mat (Tensor): (B, K, K)
        Returns:
            ref_corr_points: (Tensor) (C, 3)
            src_corr_points: (Tensor) (C, 3)
            corr_scores: (Tensor) (C,)
            estimated_transform: (Tensor) (4, 4)
            hypotheses: (Tensor) (N, 4, 4)
            ref_corr_feats: (Tensor) (N, D, 3)
            src_corr_feats: (Tensor) (N, D, 3)
        """

        corr_mat, voter_corr_mat = self.compute_correspondence_matrix(score_mat, ref_knn_masks,
                                                                      src_knn_masks)  # (B, K, K)

        ref_corr_points, src_corr_points, corr_scores, estimated_transform, hypotheses, ref_corr_feats, src_corr_feats, \
            = self.feature_based_hypothesis_proposer(
            ref_knn_points,
            src_knn_points,
            re_ref_knn_feats,
            re_src_knn_feats,
            score_mat,
            corr_mat,
            voter_corr_mat
        )
        return ref_corr_points, src_corr_points, corr_scores, estimated_transform, hypotheses, ref_corr_feats, src_corr_feats,


class combineRegisraition(nn.Module):
    def __init__(
            self,
            # LocalGlobalRegistration parameters
            lgr_k: int = 3,
            lgr_acceptance_radius: float = 0.1,
            lgr_mutual: bool = True,
            lgr_confidence_threshold: float = 0.05,
            lgr_use_dustbin: bool = False,
            lgr_use_global_score: bool = False,
            lgr_correspondence_threshold: int = 3,
            lgr_correspondence_limit: Optional[int] = None,
            lgr_num_refinement_steps: int = 5,
            # HypothesisProposer parameters
            hp_k: int = 3,
            hp_acceptance_radius: float = 0.05,
            hp_confidence_threshold: float = 0.025,
            hp_num_hypotheses: int = 1000,
            hp_num_refinement_steps: int = 5,
    ):
        """
        Combined registration pipeline that first applies LocalGlobalRegistration
        for initial rigid transformation, then applies HypothesisProposer for refinement.

        Args:
            lgr_* : Parameters for LocalGlobalRegistration
            hp_* : Parameters for HypothesisProposer
        """
        super(combineRegisraition, self).__init__()

        self.lgr = LocalGlobalRegistration(
            k=lgr_k,
            acceptance_radius=lgr_acceptance_radius,
            mutual=lgr_mutual,
            confidence_threshold=lgr_confidence_threshold,
            use_dustbin=lgr_use_dustbin,
            use_global_score=lgr_use_global_score,
            correspondence_threshold=lgr_correspondence_threshold,
            correspondence_limit=lgr_correspondence_limit,
            num_refinement_steps=lgr_num_refinement_steps,
        )

        self.hp = HypothesisProposer(
            k=hp_k,
            acceptance_radius=hp_acceptance_radius,
            confidence_threshold=hp_confidence_threshold,
            num_hypotheses=hp_num_hypotheses,
            num_refinement_steps=hp_num_refinement_steps,
        )
        

    def forward(
            self,
            ref_knn_points,
            src_knn_points,
            re_ref_knn_feats,
            re_src_knn_feats,
            ref_knn_masks,
            src_knn_masks,
            score_mat,
            global_scores,
            ref_feats,
            src_feats,
            ref_points_f,
            src_points_f,
    ):
        """
        Combined registration forward pass.

        Args:
            ref_knn_points (Tensor): (B, K, 3)
            src_knn_points (Tensor): (B, K, 3)
            re_ref_knn_feats (Tensor): (B, K, D, 3) - rotation-equivariant features
            re_src_knn_feats (Tensor): (B, K, D, 3) - rotation-equivariant features
            ref_knn_masks (BoolTensor): (B, K)
            src_knn_masks (BoolTensor): (B, K)
            score_mat (Tensor): (B, K, K) or (B, K + 1, K + 1), log likelihood
            global_scores (Tensor): (B,)
            ref_feats(Tensor),(ref_nums_f,63) 旋转不变性特征
            src_feats(Tensor),(src_nums_f,63)

        Returns:
            stage1_results: Dict containing LGR results
            stage2_results: Dict containing HypothesisProposer results
            final_transform: Final transformation matrix (4, 4)
        """

    


        # Stage 1: LocalGlobalRegistration for initial rigid transformation

        #num_true_per_row = ref_knn_masks.sum(dim=1)
        #print("True per row:", num_true_per_row)
        #print("ref_knn_point",ref_knn_points.shape)
        lgr_ref_corr_points, lgr_src_corr_points, lgr_corr_scores, lgr_transform,corr_mat = self.lgr(
            ref_knn_points=ref_knn_points,
            src_knn_points=src_knn_points,
            ref_knn_masks=ref_knn_masks,
            src_knn_masks=src_knn_masks,
            score_mat=score_mat,
            global_scores=global_scores,
        )
        #print("lgr_corr_scores",lgr_corr_scores.shape)
        transformed_src_knn_points = apply_transform(src_knn_points, lgr_transform)
        #print("lgr_ref_corr_points", lgr_ref_corr_points.shape)
        #print("lgr_ref_corr_points", lgr_src_corr_points.shape)
        re_src_knn_feats_1 = rotate_equivariant_feats(re_src_knn_feats, lgr_transform)

        hp_ref_corr_points, hp_src_corr_points, hp_corr_scores, hp_transform, hp_hypotheses, hp_ref_feats, hp_src_feats = self.hp(
            ref_knn_points=ref_knn_points,
            src_knn_points=transformed_src_knn_points,  # Use transformed points
            re_ref_knn_feats=re_ref_knn_feats,
            re_src_knn_feats=re_src_knn_feats_1,
            ref_knn_masks=ref_knn_masks,
            src_knn_masks=src_knn_masks,
            score_mat=torch.exp(score_mat),  # HypothesisProposer expects non-log scores

        )
        #print("hp_corr_scores", hp_corr_scores)#2000
        #batch_indices, ref_indices, src_indices = torch.nonzero(score_mat, as_tuple=True)
        #global_ref_corr_points = ref_knn_points[batch_indices, ref_indices]
        #print("ref_points",global_ref_corr_points.shape)

        #batch_indices, ref_indices, src_indices = torch.nonzero(torch.exp(score_mat), as_tuple=True)
        #global_ref_corr_points = ref_knn_points[batch_indices, ref_indices]
        #print("ref_points",global_ref_corr_points.shape)


        #print("hp_ref_corr_points",hp_ref_corr_points.shape)
        #print("hp_ref_corr_points", hp_src_corr_points.shape)

        # Apply initial transformation to source points
        # Transform the original source points using the LGR result

       #re_src_knn_feats = rotate_equivariant_feats(re_src_knn_feats, lgr_transform)

        # Stage 2: HypothesisProposer for refined registration
        # Use the transformed source points as input

        # Combine transformations: final_transform = hp_transform @ lgr_transform
        final_transform = torch.matmul(hp_transform, lgr_transform)
        #batch_indices, ref_indices, src_indices = torch.nonzero(corr_mat, as_tuple=True)
        #print("corr_mat",corr_mat.shape) 256,64,64

        #ref_feats,
        #src_feats,
        #ref_points_f,
        #src_points_f,
        #print("ref_knn_points",ref_knn_points.shape,re_ref_knn_feats.shape)
        global_ref_corr_points = ref_points_f.unsqueeze(0)

        global_src_corr_points = src_points_f.unsqueeze(0)
        aligned_src_points = apply_transform(global_src_corr_points, final_transform)
        '''
        #筛选出初始内点 但是如果我一开始估计就错误怎么办呢？
        #corr_residuals = torch.linalg.norm(global_ref_corr_points - aligned_src_points, dim=2) 这个不行，这个需要要求两个张量维度一样
        src_corr_points, tgt_corr_points = extract_mutual_correspondences(
            aligned_src_points, global_ref_corr_points, 4
        )
        #print("corr_residuals",src_corr_points.shape,tgt_corr_points.shape)

        #src_corr_feats = rotate_equivariant_feats(src_corr_feats, final_transform)
      
        希望这里调用regor方法两次渐进的
    
        #sel_ind = np.random.choice(src_corr_points.shape[1], src_corr_points.shape[1])

        #src_keypts_corr_final = src_corr_points[:, sel_ind, :]
        #tgt_keypts_corr_final = tgt_corr_points[:, sel_ind, :]
        #("src_keypts_corr_final",src_keypts_corr_final.shape)
        #print("tgt_keypts_corr_final", tgt_keypts_corr_final.shape)
   
   
        regenerator = Regenerator()



     
        src_keypts_corr_final, tgt_keypts_corr_final, pred_trans = regenerator.regenerate(
            src_corr_points,
            tgt_corr_points,
            aligned_src_points,
            global_ref_corr_points,
            src_feats.unsqueeze(0),
            ref_feats.unsqueeze(0),
            knn_num=40,
            sampling_num=src_corr_points.shape[1]
        )
        src_keypts_corr_final, tgt_keypts_corr_final, pred_trans_1 = regenerator.regenerate(
            src_keypts_corr_final,
            tgt_keypts_corr_final,
            aligned_src_points,
            global_ref_corr_points,
            src_feats.unsqueeze(0),
            ref_feats.unsqueeze(0),
            knn_num=40,
            sampling_num=src_corr_points.shape[1]
        )
        #print('trans',final_transform.shape,pred_trans.shape)
        if pred_trans.dim() == 3 and pred_trans.shape[0] == 1:
            pred_trans = pred_trans.squeeze(0)  # [1, 4, 4] -> [4, 4]
    
        #final_transform = torch.matmul(pred_trans, final_transform)
             '''

        return hp_ref_corr_points, hp_src_corr_points, hp_corr_scores,  final_transform , hp_hypotheses, hp_ref_feats, hp_src_feats,
