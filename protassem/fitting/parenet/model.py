import torch
import torch.nn as nn
import torch.nn.functional as F
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

from protassem.fitting.parenet.backbone import PAREConvFPN
from protassem.runtime.cuda_timing import CudaStageRecorder, cuda_stage


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
    def __init__(self, cfg):
        super(PARE_Net, self).__init__()
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

        
        self.combienrefistration=combineRegisraition(
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


    def forward(self, data_dict, timing=None):
        """timing: 可选的 sink(name, seconds)；用于分阶段计时（T02）。"""
        recorder = CudaStageRecorder(timing) if timing is not None else None
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
        with cuda_stage(recorder, "model_backbone"):
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

        with cuda_stage(recorder, "model_transformer"):
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
            with cuda_stage(recorder, "model_lgr"):
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
        if recorder is not None:
            recorder.flush()   # 同步一次后回放各阶段耗时
        return output_dict


def create_model(config):
    model = PARE_Net(config)
    return model

def main():
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    #print(model.state_dict().keys())
    #print(model)


if __name__ == '__main__':
    main()
