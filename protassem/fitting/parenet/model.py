import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
from dataclasses import dataclass
from typing import Optional

from pareconv.modules.ops import point_to_node_partition, index_select
from pareconv.modules.registration import get_node_correspondences
from pareconv.modules.sinkhorn import LearnableLogOptimalTransport
from pareconv.modules.dual_matching import PointDualMatching

from pareconv.modules.geotransformer import (
    GeometricTransformer,
    SuperPointMatching,
    SuperPointTargetGenerator,
    LocalGlobalRegistration,
)

from pareconv.modules.registration import HypothesisProposer, combineRegisraition

from protassem.fitting.chunked_registration import build_registration
from protassem.fitting.parenet.backbone import PAREConvFPN
from protassem.fitting.cloud_encoding import backbone_input, node_partition

# 推理链路（demo_mask.process_single_pair）只消费最终位姿与两侧点数；
# 训练/诊断路径传 output_fields=None，保持返回全部字段的旧行为。
INFERENCE_OUTPUT_FIELDS = ("estimated_transform", "ref_points", "src_points")


@dataclass
class EncodedCloud:
    """单侧编码结果（T06）：只含与本侧几何有关的张量，可独立缓存。

    字段含义与 `forward()` 里的同名中间量一一对应（ref=target、src=source）：
    `node_knn_points` 已含 padding 哨兵点；`node_masks`/`node_knn_*` 为节点分区。
    """

    points: torch.Tensor
    points_f: torch.Tensor
    points_c: torch.Tensor
    feats_f: torch.Tensor
    re_feats_f: torch.Tensor
    feats_c: torch.Tensor
    re_feats_c: torch.Tensor
    m_scores: torch.Tensor
    node_masks: torch.Tensor
    node_knn_indices: torch.Tensor
    node_knn_masks: torch.Tensor
    node_knn_points: torch.Tensor
    scale: torch.Tensor
    geometry_key: Optional[str] = None


def select_output_fields(output_dict, output_fields):
    """按白名单裁剪 forward 的返回值；output_fields=None 表示全部保留。

    白名单里出现 forward 未写入的字段时直接抛 KeyError：字段契约要显式失败，
    不允许静默兜底（漏字段会让调用方拿到不完整的输出）。
    """
    if output_fields is None:
        return output_dict
    missing = [name for name in output_fields if name not in output_dict]
    if missing:
        raise KeyError("output_fields 在 forward 输出中不存在: %s" % missing)
    return {name: output_dict[name] for name in output_fields}


def SM(corr, src_keypts, tgt_keypts, inlier_threshold=0.1, top_ratio=0.85):
    diff = corr - corr.permute(1, 0, 2)
    M = torch.sum(diff[:, :, 0:3] ** 2, dim=-1) ** 0.5 - torch.sum(diff[:, :, 3:6] ** 2, dim=-1) ** 0.5
    M = M[None, :, :]
    # pdb.set_trace()
    ## polynomial funcition
    sigma = inlier_threshold / 3
    M = torch.max(torch.zeros_like(M), 4.5 - M ** 2 / 2 / sigma ** 2)
    M[:, torch.arange(M.shape[1]), torch.arange(M.shape[1])] = 0  # 对角设置为0

    # calculate the principal eigenvector
    leading_eig = torch.ones_like(M[:, :, 0:1])
    for _ in range(10):
        leading_eig = torch.bmm(M, leading_eig)
        leading_eig = leading_eig / (torch.norm(leading_eig, dim=1,
                                                keepdim=True) + 1e-6)  # 迭代 幂迭代法 我们从一个随便的向量（比如全 1 向量）开始，不断把它乘上 M，每次都做归一化。理论上，迭代多次以后它就会**“收敛”**到最大特征值对应的特征向量。
    leading_eig = leading_eig.squeeze(-1)
    # pdb.set_trace()
    # select top-10% as the inliers
    top_10p = torch.argsort(leading_eig, dim=1, descending=True)[:, 0:int(leading_eig.shape[1] * top_ratio)]
    pred_labels = torch.zeros_like(leading_eig)
    pred_labels[0, top_10p[0]] = 1  # assert bs = 1

    # compute the transformation
    # pred_trans = rigid_transform_3d(src_keypts, tgt_keypts, leading_eig * pred_labels)
    return pred_labels
class PARE_Net(nn.Module):
    def __init__(self, cfg, hypothesis_chunk=0):
        super(PARE_Net, self).__init__()
        # T08：假设评分分块大小（0 = 原整批路径）。显式传参而非改 cfg ——
        # `make_cfg()` 返回模块级单例，改它会污染同进程内的其他模型。
        self.hypothesis_chunk = int(hypothesis_chunk)
        self.num_points_in_patch = cfg.model.num_points_in_patch
        self.matching_radius = cfg.model.ground_truth_matching_radius

        self.backbone = PAREConvFPN(
            cfg.backbone.init_dim,
            cfg.backbone.output_dim,
            cfg.backbone.kernel_size,
            cfg.backbone.share_nonlinearity,
            cfg.backbone.conv_way,
            cfg.backbone.use_xyz,
            cfg.fine_matching.use_encoder_re_feats
        )

        self.transformer = GeometricTransformer(
            cfg.geotransformer.input_dim,
            cfg.geotransformer.output_dim,
            cfg.geotransformer.hidden_dim,
            cfg.geotransformer.num_heads,
            cfg.geotransformer.blocks,
            cfg.geotransformer.sigma_d,
            cfg.geotransformer.sigma_a,
            cfg.geotransformer.angle_k,
            reduction_a=cfg.geotransformer.reduction_a,
        )

        self.coarse_target = SuperPointTargetGenerator(
            cfg.coarse_matching.num_targets, cfg.coarse_matching.overlap_threshold
        )

        self.coarse_matching = SuperPointMatching(
            cfg.coarse_matching.num_correspondences, cfg.coarse_matching.dual_normalization
        )

        self.fine_matching = HypothesisProposer(
            cfg.fine_matching.topk,
            cfg.fine_matching.acceptance_radius,
            confidence_threshold=cfg.fine_matching.confidence_threshold,
            num_hypotheses=cfg.fine_matching.num_hypotheses,
            num_refinement_steps=cfg.fine_matching.num_refinement_steps,
        )


        self.point_matching = PointDualMatching(dim=cfg.backbone.output_dim // 3 * 3)
        #self.optimal_transport = LearnableLogOptimalTransport(cfg.model.num_sinkhorn_iterations)  # 迭代100次 寻找组中的最优匹配点
        self.corr_num = cfg.coarse_matching.num_correspondences
        #self.proj1 = nn.Linear(cfg.backbone.output_dim // 3 * 3, cfg.backbone.output_dim // 3 * 3, True)

        
        # T08：hypothesis_chunk > 0 时用分块版替换 LGR/HP 的假设评分（0 = 原路径）
        self.combienrefistration=build_registration(
        hypothesis_chunk=self.hypothesis_chunk,
        # LocalGlobalRegistration 参数 (第一阶段)
        lgr_k=cfg.fine_matching_geo.topk,
        lgr_acceptance_radius=cfg.fine_matching_geo.acceptance_radius,
        lgr_mutual=cfg.fine_matching_geo.mutual,
        lgr_confidence_threshold=cfg.fine_matching_geo.confidence_threshold,
        lgr_use_dustbin=cfg.fine_matching_geo.use_dustbin,
        lgr_use_global_score=cfg.fine_matching_geo.use_global_score,
        lgr_correspondence_threshold=cfg.fine_matching_geo.correspondence_threshold,
        lgr_correspondence_limit=cfg.fine_matching_geo.correspondence_limit,
        lgr_num_refinement_steps=cfg.fine_matching_geo.num_refinement_steps,

        # HypothesisProposer 参数 (第二阶段)
        hp_k=cfg.fine_matching.topk,
        hp_acceptance_radius=cfg.fine_matching.acceptance_radius,
        hp_confidence_threshold=cfg.fine_matching.confidence_threshold,
        hp_num_hypotheses=cfg.fine_matching.num_hypotheses,
        hp_num_refinement_steps=cfg.fine_matching.num_refinement_steps,
    )


    def forward(self, data_dict, output_fields=None):
        """output_fields: 需要回传的字段名（None = 全部）。只影响返回内容，
        不改变任何计算；传入 INFERENCE_OUTPUT_FIELDS 时中间张量在返回后即可释放。
        """
        output_dict = {}
        # Downsample point clouds
        #print("data_dict keys:", data_dict.keys())
        feats = data_dict['features'].detach()
        transform = data_dict['transform'].detach()
        scale = data_dict['scale']
        device = next(self.parameters()).device
        data_dict['scale'] = torch.tensor(scale, device=device)
        #~print("scale",scale)
        #norm_ref = data_dict['ref_vectors'].detach()
        #norm_src = data_dict['src_vectors'].detach()

        ref_length_c = data_dict['lengths'][-1][0].item()
        ref_length_f = data_dict['lengths'][1][0].item()
        ref_length = data_dict['lengths'][0][0].item()
        points_c = data_dict['points'][-1][:, :3].detach()
        points_f = data_dict['points'][1][:, :3].detach()#一次采样以后的
        points = data_dict['points'][0][:, :3].detach()
        #print("points",points.shape,points_f.shape,points_c.shape)
        #print("data_dict['points'][0]",data_dict['points'][0].shape)
       #ref_vec = data_dict['ref_vectors']
        #src_vec = data_dict['src_vectors']
        #print("vec",ref_vec.shape,src_vec.shape)
        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]
        ref_points = points[:ref_length]
        src_points = points[ref_length:]
        #print("ref_points",ref_points.shape,ref_points_f.shape,ref_points_c.shape)
        #print("src_points",src_points.shape,src_points_f.shape,src_points_c.shape)

        output_dict['ref_points_c'] = ref_points_c
        output_dict['src_points_c'] = src_points_c
        output_dict['ref_points_f'] = ref_points_f
        output_dict['src_points_f'] = src_points_f
        output_dict['ref_points'] = ref_points
        output_dict['src_points'] = src_points

        # 1. Generate ground truth node correspondences
        _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
            ref_points_f, ref_points_c, self.num_points_in_patch
        )  # ref_N_c,  [ref_N_c, 64],  [ref_N_c, 64],
        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
            src_points_f, src_points_c, self.num_points_in_patch
        )
        '''
        # 1.1 Generate ground truth node correspondences
        _, ref_node_masks_shot, ref_node_knn_indices_shot, ref_node_knn_masks_shot = point_to_node_partition(
            ref_points, ref_points_c, 1
        )  # ref_N_c,  [ref_N_c, 64],  [ref_N_c, 64],
        _, src_node_masks_shot, src_node_knn_indices_shot, src_node_knn_masks_shot = point_to_node_partition(
            src_points, ref_points_c, 1
        )
        '''
        #print("ref_node_knn_indices_shot",ref_node_knn_indices_shot)
        #print("ref_points", ref_points[0])
        #print("ref_norm", norm_ref[0])
        #print("ref_points", ref_points[1])
        #print("ref_norm", norm_ref[1])
        #print("ref_points", ref_points[2])
        #print("ref_norm", norm_ref[2])
        #print("src_node_knn_indices_shot",src_node_knn_indices_shot.shape)

        #超点 来匹配原始点 带有密度向量的点

        output_dict['ref_node_knn_indices'] = ref_node_knn_indices
        output_dict['src_node_knn_indices'] = src_node_knn_indices

        ref_padded_points_f = torch.cat([ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0)# [ref_N_f + 1, 3]
        src_padded_points_f = torch.cat([src_points_f, torch.zeros_like(src_points_f[:1])], dim=0)
        ref_node_knn_points = index_select(ref_padded_points_f, ref_node_knn_indices, dim=0) #[ref_N_c, 64, 3]
        src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices, dim=0)

        # 1.2 审计结论（T03）：GT 对应只被下面的 `if self.training:` 分支（coarse_target）消费，
        # 推理时是纯浪费——它会构造 (M,N) 距离矩阵与 (B,K,K) 重叠矩阵。
        # get_node_correspondences 无随机、无副作用、不写任何全局状态，跳过不改变其余计算。
        if self.training:
            gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
                ref_points_c,
                src_points_c,
                ref_node_knn_points,
                src_node_knn_points,
                transform,
                self.matching_radius,
                ref_masks=ref_node_masks,
                src_masks=src_node_masks,
                ref_knn_masks=ref_node_knn_masks,
                src_knn_masks=src_node_knn_masks,
            )  # coarse correspondences  gt_node_corr_indices: [N, 2]  gt_node_corr_overlaps : N

            output_dict['gt_node_corr_indices'] = gt_node_corr_indices
            output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps

        # 2. PARE-Conv Encoder
        re_feats_f, feats_f, re_feats_c, feats_c, m_scores = self.backbone(data_dict)
        #print("re_feats_f", re_feats_f.shape, feats_f.shape, re_feats_c.shape, feats_c.shape)
        #points1 = data_dict['points'][0][:, :3].detach()
        #print("points1",points1)
        #print("re_feats_f",re_feats_f.shape)
        #print("re_feats_c", re_feats_c.shape)
        # 3. Conditional Transformer
        ref_feats_c = feats_c[:ref_length_c]
        src_feats_c = feats_c[ref_length_c:]

        ref_feats_c_re = re_feats_c[:ref_length_c]
        src_feats_c_re = re_feats_c[ref_length_c:]
        output_dict['ref_feats_c_re'] = ref_feats_c_re
        output_dict['src_feats_c_re'] = src_feats_c_re

        #all_points = torch.cat([ref_points_c, src_points_c], dim=0)
        #center = all_points.mean(dim=0, keepdim=True)
        #scale = all_points.std(dim=0, keepdim=True) + 1e-6  # 防止除0

        #ref_pc = (ref_points_c - center) / scale
        #src_pc = (src_points_c - center) / scale
 
        #ref_length_iss=data_dict['ref_length_iss']
        #src_length_iss=data_dict['src_length_iss']
        ref_pc = ref_points_c / scale
        src_pc = src_points_c / scale

        # Step 1: 计算质心
        #ref_center = ref_points_c.mean(dim=0, keepdim=True)  # (1, 3)
        #src_center = src_points_c.mean(dim=0, keepdim=True)  # (1, 3)

        # Step 2: 中心化点云
        #ref_points_centered = ref_points_c - ref_center  # (N, 3)
        #src_points_centered = src_points_c - ref_center  # (M, 3)

        # Step 3: 计算各自的归一化尺度（可选方案见下）
        #ref_scale = ref_points_centered.norm(dim=1).max()  # 用最大模长归一化
        #src_scale = src_points_centered.norm(dim=1).max()

        # Step 4: 归一化
        #ref_pc = ref_points_centered /scale
        #src_pc = src_points_centered /scale

        ref_feats_c, src_feats_c, scores_list = self.transformer(
            ref_pc.unsqueeze(0),
            src_pc.unsqueeze(0),
            ref_feats_c.unsqueeze(0),
            src_feats_c.unsqueeze(0),
        )
        #print("ref_feats_c",ref_feats_c.shape)
        #print("src_feats_c",src_feats_c.shape)
        #print("ref_feats_css", ref_feats_c.squeeze(0).shape)


        ref_feats_c_norm = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)
        src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)
        #print("ref_feats_c_norm",ref_feats_c_norm.shape)
        #print("111",torch.norm( ref_feats_c_norm, dim=1),torch.max(ref_feats_c_norm, dim=1)[0],torch.min(ref_feats_c_norm, dim=1)[0])
        #ref_shot_t = data_dict['ref_shot']   # [N_ref, 352]
        #src_shot_t = data_dict['src_shot']   # [N_src, 352]

        #ref_shot_norm = F.normalize(ref_shot_t, p=2, dim=1)  # [N_ref, 352]
        #src_shot_norm = F.normalize(src_shot_t, p=2, dim=1)  # [N_src, 352]

        #ref_feats_c_norm = torch.cat([ref_feats_c_norm,ref_shot_norm], dim=1)
        #src_feats_c_norm = torch.cat([src_feats_c_norm, src_shot_norm], dim=1)
        #print("ref_node_masks", ref_node_masks.shape)
        output_dict['ref_feats_c'] = ref_feats_c_norm
        output_dict['src_feats_c'] = src_feats_c_norm

        # 4. Head for fine level matching
        ref_feats_f = feats_f[:ref_length_f]
        src_feats_f = feats_f[ref_length_f:]
        m_ref_scores = m_scores[:ref_length_f]
        m_src_scores = m_scores[ref_length_f:]
        re_ref_feats_f = re_feats_f[:ref_length_f]
        re_src_feats_f = re_feats_f[ref_length_f:]


        output_dict['m_ref_scores'] = m_ref_scores
        output_dict['m_src_scores'] = m_src_scores
        output_dict['ref_feats_f'] = ref_feats_f
        output_dict['src_feats_f'] = src_feats_f
        output_dict['re_ref_feats_f'] = re_ref_feats_f
        output_dict['re_src_feats_f'] = re_src_feats_f

        # 5. Select topk nearest node correspondences
        # 5.1. Select topk nearest node correspondences

        '''
        dual_level=True
        with torch.no_grad():
            if dual_level == False:
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                    ref_feats_c_norm, src_feats_c_norm, ref_node_masks, src_node_masks
                )

                #output_dict['ref_node_corr_indices'] = ref_node_corr_indices
                #output_dict['src_node_corr_indices'] = src_node_corr_indices

            else:
                # pdb.set_trace()
                # salient point matching
                ref_node_corr_indices_iss, src_node_corr_indices_iss, node_corr_scores_iss = self.coarse_matching(
                    ref_feats_c_norm[:ref_length_iss], src_feats_c_norm[:src_length_iss],
                    ref_node_masks[:ref_length_iss], src_node_masks[:src_length_iss]
                )
                # non-salient point matching
                ref_node_corr_indices_sup, src_node_corr_indices_sup, node_corr_scores_sup = self.coarse_matching(
                    ref_feats_c_norm[ref_length_iss:], src_feats_c_norm[src_length_iss:],
                    ref_node_masks[ref_length_iss:], src_node_masks[src_length_iss:]
                )
                ref_node_corr_indices_sup = ref_node_corr_indices_sup + ref_length_iss
                src_node_corr_indices_sup = src_node_corr_indices_sup + src_length_iss

                iss_select = int(
                    (ref_length_iss + src_length_iss) / (len(ref_feats_c_norm) + len(src_feats_c_norm)) * self.corr_num)
                sup_select = self.corr_num - iss_select

                ref_keypt_sup = ref_points_c[ref_node_corr_indices_sup[:sup_select]].unsqueeze(0)
                src_keypt_sup = src_points_c[src_node_corr_indices_sup[:sup_select]].unsqueeze(0)
                corr = torch.cat([src_keypt_sup, ref_keypt_sup], dim=-1)
                # use SM on non-salient correspondences
                pred_labels = SM(corr, src_keypt_sup, ref_keypt_sup)
                if pred_labels.sum() == 0:
                    ref_node_corr_indices = ref_node_corr_indices_iss[:iss_select]
                    src_node_corr_indices = src_node_corr_indices_iss[:iss_select]
                    node_corr_scores = node_corr_scores_iss[:iss_select]

                else:
                    ref_node_corr_indices=torch.cat([ref_node_corr_indices_iss[:iss_select],ref_node_corr_indices_sup[:sup_select][pred_labels.bool().squeeze()]],dim=-1)
                    src_node_corr_indices=torch.cat([src_node_corr_indices_iss[:iss_select],src_node_corr_indices_sup[:sup_select][pred_labels.bool().squeeze()]],dim=-1)
                    node_corr_scores=torch.cat([node_corr_scores_iss[:iss_select],node_corr_scores_sup[:sup_select][pred_labels.bool().squeeze()]],dim=-1)

                # 7 Random select ground truth node correspondences during training
            '''

        with torch.no_grad():
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                ref_feats_c_norm, src_feats_c_norm, ref_node_masks, src_node_masks
            )

            output_dict['ref_node_corr_indices'] = ref_node_corr_indices
            output_dict['src_node_corr_indices'] = src_node_corr_indices



            if self.training:
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_target(
                    gt_node_corr_indices, gt_node_corr_overlaps
                )



        # 6 Generate batched node points & feats
        ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices]  # (P, K, 3)
        src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]  # (P, K, 3)

        ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)
        #print("ref_padded_feats_f", ref_padded_feats_f.shape)P,63
        src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)
        ref_node_corr_knn_feats = index_select(ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)  # (P, K, C)
        #print("ref_padded_feats_f", ref_node_corr_knn_feats.shape)P ,64, 63
        src_node_corr_knn_feats = index_select(src_padded_feats_f, src_node_corr_knn_indices, dim=0)  # (P, K, C)

        m_ref_padded_scores = torch.cat([m_ref_scores, torch.zeros_like(m_ref_scores[:1])], dim=0)
        m_src_padded_scores = torch.cat([m_src_scores, torch.zeros_like(m_src_scores[:1])], dim=0)
        ref_node_corr_knn_scores = index_select(m_ref_padded_scores, ref_node_corr_knn_indices, dim=0)  # (P, K, C)
        src_node_corr_knn_scores = index_select(m_src_padded_scores, src_node_corr_knn_indices, dim=0)  # (P, K, C)

        output_dict['ref_node_corr_knn_points'] = ref_node_corr_knn_points   # 256 64 3
        output_dict['src_node_corr_knn_points'] = src_node_corr_knn_points
        output_dict['ref_node_corr_knn_masks'] = ref_node_corr_knn_masks
        output_dict['src_node_corr_knn_masks'] = src_node_corr_knn_masks
        #print("output_dict['ref_node_corr_knn_points']",ref_node_corr_knn_points)
        #print("ref_node_corr_knn_points[0]:", ref_node_corr_knn_points[4].cpu().numpy().tolist())
        #mask = ref_node_corr_knn_points[3] > 0
        #count = mask.sum().item()

        #print("第 5 个点对应的 ref_node_corr_knn_points 中 > 0 的数量:", count)
        re_ref_padded_feats_f = torch.cat([re_ref_feats_f, torch.zeros_like(re_ref_feats_f[:1])], dim=0)
        re_src_padded_feats_f = torch.cat([re_src_feats_f, torch.zeros_like(re_src_feats_f[:1])], dim=0)
        re_ref_node_corr_knn_feats = index_select(re_ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)  # (P, K, C)
        re_src_node_corr_knn_feats = index_select(re_src_padded_feats_f, src_node_corr_knn_indices, dim=0)  # (P, K, C)

        output_dict['re_ref_node_corr_knn_feats'] = re_ref_node_corr_knn_feats   # 256 64 21 3
        output_dict['re_src_node_corr_knn_feats'] = re_src_node_corr_knn_feats
        #print("re_ref_node_corr_knn_feats", re_ref_node_corr_knn_feats.shape)

        # 7 Match batched points
        matching_scores = self.point_matching(ref_node_corr_knn_feats, src_node_corr_knn_feats, ref_node_corr_knn_scores, src_node_corr_knn_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks)
        #m_ref_feats, m_src_feats = self.proj1(ref_node_corr_knn_feats), self.proj1(src_node_corr_knn_feats)
        #matching_scores = torch.einsum('bnd,bmd->bnm', m_ref_feats, m_src_feats)  # (P, K, K)

        #matching_scores = matching_scores / feats_f.shape[1] ** 0.5
        #matching_scores = matching_scores * ref_node_corr_knn_scores[:, :, None] * src_node_corr_knn_scores[:, None, :]
        #matching_scores = self.optimal_transport(matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks)
        #matching_scores = matching_scores[:, :-1, :-1]#取掉最后一列
        #
        #print("matching_scores",matching_scores.shape)#matching_scores torch.Size([256, 129, 129])
        #print("matching_scores",matching_scores.shape)
        #num_positive = (matching_scores > 0).sum()
        #print("大于 0 的元素数量：", num_positive.item())
        output_dict['matching_scores'] = matching_scores   # 256 64 64
        output_dict['ref_node_corr_knn_scores'] = ref_node_corr_knn_scores
        output_dict['src_node_corr_knn_scores'] = src_node_corr_knn_scores

        # 8 Generate hypotheses and select the best one
        #ref_node_corr_knn_points = ref_node_corr_knn_points * scale
        #src_node_corr_knn_points = src_node_corr_knn_points * scale
        '''
        with torch.no_grad():
            ref_corr_points, src_corr_points, corr_scores, estimated_transform, hypotheses, re_ref_corr_feats, re_src_corr_feats, = self.fine_matching(
                ref_node_corr_knn_points,
                src_node_corr_knn_points,
                re_ref_node_corr_knn_feats,
                re_src_node_corr_knn_feats,
                ref_node_corr_knn_masks,
                src_node_corr_knn_masks,
                matching_scores,
                #data_dict['ref_vectors'],  # 添加ref_vectors参数
                #data_dict['src_vectors']  # 添加src_vectors参数
            )

      
        with torch.no_grad():
            ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.fine_matching_geo(
                ref_node_corr_knn_points,
                src_node_corr_knn_points,
                ref_node_corr_knn_masks,
                src_node_corr_knn_masks,
                matching_scores,
                node_corr_scores,
            )
       '''

     
        with torch.no_grad():
            ref_corr_points, src_corr_points, corr_scores, estimated_transform, hypotheses, re_ref_corr_feats, re_src_corr_feats, =  self.combienrefistration(
                ref_node_corr_knn_points,
                src_node_corr_knn_points,
                re_ref_node_corr_knn_feats,
                re_src_node_corr_knn_feats,
                ref_node_corr_knn_masks,
                src_node_corr_knn_masks,
                matching_scores,
                node_corr_scores,
                ref_feats_f,
                src_feats_f,
                ref_points_f,
                src_points_f,
            )


        output_dict['re_ref_corr_feats'] = re_ref_corr_feats
        output_dict['re_src_corr_feats'] = re_src_corr_feats
        output_dict['hypotheses'] = hypotheses
        output_dict['ref_corr_points'] = ref_corr_points
        output_dict['src_corr_points'] = src_corr_points
        output_dict['corr_scores'] = corr_scores
        output_dict['estimated_transform'] = estimated_transform
        output_dict['transform'] = transform
        return select_output_fields(output_dict, output_fields)

    # ==================================================================
    # T06：单侧编码 / 双侧配准拆分
    # ==================================================================
    # 上面 forward() 是**兼容对照实现**（联合布局、原样保留，训练入口仍走它）；
    # 下面两个方法把同一条推理链路拆成"与本侧几何有关"和"依赖两侧交互"两段。
    # 两者的数值等价性由 tools/check_encoding_split.py 逐层校验（仅开发服务器保留；统一验收容差
    # 特征 atol=1e-6 / rtol=1e-5；位姿与候选身份要求一致）。

    def encode_cloud(self, geometry, scale):
        """单侧编码：backbone + 节点分区，只依赖本侧几何与共享 scale。

        geometry: CloudGeometry（T04/T05，本侧局部索引）
        scale   : 两侧共享的归一化尺度（联合布局时的同一个标量）
        返回 EncodedCloud（可独立缓存；不含任何跨侧交互量）。
        """
        device = next(self.parameters()).device

        re_feats_f, feats_f, re_feats_c, feats_c, m_scores = self.backbone(
            backbone_input(geometry, scale, device=device))
        partition = geometry.node_partition
        if partition is None:
            partition = node_partition(geometry, self.num_points_in_patch)

        points = geometry.points[0][:, :3].detach()
        points_f = geometry.points[1][:, :3].detach()
        points_c = geometry.points[-1][:, :3].detach()
        # 细节点邻域点（含 padding 哨兵点，与 forward 里的 ref_padded_points_f 一致）
        padded_points_f = torch.cat([points_f, torch.zeros_like(points_f[:1])], dim=0)
        node_knn_points = index_select(padded_points_f, partition.knn_indices, dim=0)

        return EncodedCloud(
            points=points, points_f=points_f, points_c=points_c,
            feats_f=feats_f, re_feats_f=re_feats_f, feats_c=feats_c, re_feats_c=re_feats_c,
            m_scores=m_scores, node_masks=partition.masks,
            node_knn_indices=partition.knn_indices, node_knn_masks=partition.knn_masks,
            node_knn_points=node_knn_points,
            scale=torch.as_tensor(scale, device=device),
            geometry_key=geometry.fingerprint())

    def register_pair(self, target_encoded, source_encoded, output_fields=None):
        """双侧配准：cross-attention → 粗匹配 → 点匹配 → LGR 假设与选优。

        target_encoded / source_encoded: 两侧的 EncodedCloud（scale 必须一致）。
        返回字段与 forward() 同名（不含输入 transform：推理不使用参考位姿）。
        只支持推理（`self.training == False`）；训练入口请用 forward()。
        """
        if self.training:
            raise RuntimeError("register_pair 只用于推理；训练请用 forward()")
        output_dict = {}
        ref = target_encoded
        src = source_encoded
        scale = ref.scale
        if not torch.equal(scale, src.scale):
            raise ValueError("两侧编码的 scale 不一致：%s vs %s" % (scale, src.scale))

        ref_points_c, src_points_c = ref.points_c, src.points_c
        ref_points_f, src_points_f = ref.points_f, src.points_f
        ref_points, src_points = ref.points, src.points
        output_dict['ref_points_c'] = ref_points_c
        output_dict['src_points_c'] = src_points_c
        output_dict['ref_points_f'] = ref_points_f
        output_dict['src_points_f'] = src_points_f
        output_dict['ref_points'] = ref_points
        output_dict['src_points'] = src_points
        output_dict['ref_node_knn_indices'] = ref.node_knn_indices
        output_dict['src_node_knn_indices'] = src.node_knn_indices

        ref_feats_c = ref.feats_c
        src_feats_c = src.feats_c
        output_dict['ref_feats_c_re'] = ref.re_feats_c
        output_dict['src_feats_c_re'] = src.re_feats_c
        ref_pc = ref_points_c / scale
        src_pc = src_points_c / scale

        ref_feats_c, src_feats_c, scores_list = self.transformer(
            ref_pc.unsqueeze(0),
            src_pc.unsqueeze(0),
            ref_feats_c.unsqueeze(0),
            src_feats_c.unsqueeze(0),
        )
        ref_feats_c_norm = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)
        src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)
        output_dict['ref_feats_c'] = ref_feats_c_norm
        output_dict['src_feats_c'] = src_feats_c_norm

        ref_feats_f, src_feats_f = ref.feats_f, src.feats_f
        m_ref_scores, m_src_scores = ref.m_scores, src.m_scores
        re_ref_feats_f, re_src_feats_f = ref.re_feats_f, src.re_feats_f
        output_dict['m_ref_scores'] = m_ref_scores
        output_dict['m_src_scores'] = m_src_scores
        output_dict['ref_feats_f'] = ref_feats_f
        output_dict['src_feats_f'] = src_feats_f
        output_dict['re_ref_feats_f'] = re_ref_feats_f
        output_dict['re_src_feats_f'] = re_src_feats_f

        with torch.no_grad():
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                ref_feats_c_norm, src_feats_c_norm, ref.node_masks, src.node_masks
            )
            output_dict['ref_node_corr_indices'] = ref_node_corr_indices
            output_dict['src_node_corr_indices'] = src_node_corr_indices

        ref_node_corr_knn_indices = ref.node_knn_indices[ref_node_corr_indices]
        src_node_corr_knn_indices = src.node_knn_indices[src_node_corr_indices]
        ref_node_corr_knn_masks = ref.node_knn_masks[ref_node_corr_indices]
        src_node_corr_knn_masks = src.node_knn_masks[src_node_corr_indices]
        ref_node_corr_knn_points = ref.node_knn_points[ref_node_corr_indices]
        src_node_corr_knn_points = src.node_knn_points[src_node_corr_indices]

        ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)
        src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)
        ref_node_corr_knn_feats = index_select(ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)
        src_node_corr_knn_feats = index_select(src_padded_feats_f, src_node_corr_knn_indices, dim=0)

        m_ref_padded_scores = torch.cat([m_ref_scores, torch.zeros_like(m_ref_scores[:1])], dim=0)
        m_src_padded_scores = torch.cat([m_src_scores, torch.zeros_like(m_src_scores[:1])], dim=0)
        ref_node_corr_knn_scores = index_select(m_ref_padded_scores, ref_node_corr_knn_indices, dim=0)
        src_node_corr_knn_scores = index_select(m_src_padded_scores, src_node_corr_knn_indices, dim=0)

        output_dict['ref_node_corr_knn_points'] = ref_node_corr_knn_points
        output_dict['src_node_corr_knn_points'] = src_node_corr_knn_points
        output_dict['ref_node_corr_knn_masks'] = ref_node_corr_knn_masks
        output_dict['src_node_corr_knn_masks'] = src_node_corr_knn_masks

        re_ref_padded_feats_f = torch.cat([re_ref_feats_f, torch.zeros_like(re_ref_feats_f[:1])], dim=0)
        re_src_padded_feats_f = torch.cat([re_src_feats_f, torch.zeros_like(re_src_feats_f[:1])], dim=0)
        re_ref_node_corr_knn_feats = index_select(re_ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)
        re_src_node_corr_knn_feats = index_select(re_src_padded_feats_f, src_node_corr_knn_indices, dim=0)
        output_dict['re_ref_node_corr_knn_feats'] = re_ref_node_corr_knn_feats
        output_dict['re_src_node_corr_knn_feats'] = re_src_node_corr_knn_feats

        matching_scores = self.point_matching(
            ref_node_corr_knn_feats, src_node_corr_knn_feats,
            ref_node_corr_knn_scores, src_node_corr_knn_scores,
            ref_node_corr_knn_masks, src_node_corr_knn_masks)
        output_dict['matching_scores'] = matching_scores
        output_dict['ref_node_corr_knn_scores'] = ref_node_corr_knn_scores
        output_dict['src_node_corr_knn_scores'] = src_node_corr_knn_scores

        with torch.no_grad():
            (ref_corr_points, src_corr_points, corr_scores, estimated_transform,
             hypotheses, re_ref_corr_feats, re_src_corr_feats) = self.combienrefistration(
                ref_node_corr_knn_points,
                src_node_corr_knn_points,
                re_ref_node_corr_knn_feats,
                re_src_node_corr_knn_feats,
                ref_node_corr_knn_masks,
                src_node_corr_knn_masks,
                matching_scores,
                node_corr_scores,
                ref_feats_f,
                src_feats_f,
                ref_points_f,
                src_points_f,
            )
        output_dict['re_ref_corr_feats'] = re_ref_corr_feats
        output_dict['re_src_corr_feats'] = re_src_corr_feats
        output_dict['hypotheses'] = hypotheses
        output_dict['ref_corr_points'] = ref_corr_points
        output_dict['src_corr_points'] = src_corr_points
        output_dict['corr_scores'] = corr_scores
        output_dict['estimated_transform'] = estimated_transform
        return select_output_fields(output_dict, output_fields)



def _install_query_tiling(model, query_chunk):
    import sys
    import torch
    from protassem.fitting.parenet.backbone import PARE_Conv_Block, PARE_Conv_Resblock
    count = 0
    for module in model.backbone.modules():
        if isinstance(module, (PARE_Conv_Block, PARE_Conv_Resblock)):
            original = module.forward
            def tiled(q_pts, s_pts, s_feats, neighbor_indices,
                      _original=original, _module=module):
                if _module.training:
                    raise RuntimeError("Query tiling prototype is inference-only")
                if q_pts.shape[0] <= query_chunk:
                    return _original(q_pts, s_pts, s_feats, neighbor_indices)
                return torch.cat([
                    _original(q_pts[j:j+query_chunk], s_pts, s_feats,
                              neighbor_indices[j:j+query_chunk])
                    for j in range(0, q_pts.shape[0], query_chunk)
                ], dim=0)
            module.forward = tiled
            count += 1
    if count == 0:
        raise RuntimeError("No PAREConv modules patched")
    print("INFRA_QUERY_TILING chunk=%d modules=%d" % (query_chunk, count),
          file=sys.stderr, flush=True)

def create_model(config, hypothesis_chunk=0):
    model = PARE_Net(config, hypothesis_chunk=hypothesis_chunk)
    _install_query_tiling(model, 1024)
    return model


def model_fingerprint(model):
    """权重与结构的指纹（T07 编码缓存 key 的一部分）。

    覆盖 state_dict 的键名/形状/dtype 与全部权重字节：权重一变（重新训练、换了 checkpoint）
    缓存即失效。只看一次调用（模型加载后计算一次即可），约 0.1–0.2 s。
    """
    digest = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(state):
        tensor = state[name]
        digest.update(("%s;%s;%s;" % (name, tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def main():
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    #print(model.state_dict().keys())
    #print(model)


if __name__ == '__main__':
    main()
