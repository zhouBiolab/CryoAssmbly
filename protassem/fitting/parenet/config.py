import os
import os.path as osp
import argparse

from easydict import EasyDict as edict

from pareconv.utils.common import ensure_dir


_C = edict()

# common
_C.seed = 7351

# dirs
_C.working_dir = osp.dirname(osp.realpath(__file__))
_C.root_dir = osp.dirname(osp.dirname(_C.working_dir))
_C.exp_name = osp.basename(_C.working_dir)
_C.output_dir = osp.join(_C.root_dir, 'output', _C.exp_name)
_C.snapshot_dir = osp.join(_C.output_dir, 'snapshots')
_C.log_dir = osp.join(_C.output_dir, 'logs')
_C.event_dir = osp.join(_C.output_dir, 'wandb_events')
_C.feature_dir = osp.join(_C.output_dir, 'features')
_C.registration_dir = osp.join(_C.output_dir, 'registration')

# ensure_dir(_C.output_dir)  # disabled: not needed for inference
# ensure_dir(_C.snapshot_dir)  # disabled: not needed for inference
# ensure_dir(_C.log_dir)  # disabled: not needed for inference
# ensure_dir(_C.event_dir)  # disabled: not needed for inference
# ensure_dir(_C.feature_dir)  # disabled: not needed for inference
# ensure_dir(_C.registration_dir)  # disabled: not needed for inference

# data
_C.data = edict()
_C.data.dataset_root = '/xiangyux/PARENet-main/data/demo'
_C.data.metadata_root = osp.join(_C.root_dir, 'data', '3DMatch', 'metadata')

# train data
_C.train = edict()
_C.train.batch_size = 1
_C.train.num_workers = 1
_C.train.point_limit = 30000
_C.train.use_augmentation = True
_C.train.augmentation_noise =0.02
_C.train.augmentation_rotation = 1.0
_C.train.augmentation_crop = True
_C.train.point_keep_ratio = 0.7 #这是匹配率过高情况 进行crop \
_C.train.matching_radius = 2#3dmatch 0.1 init_voxel_size = 0.025 先试试2

# test data
_C.test = edict()
_C.test.batch_size = 1
_C.test.num_workers = 8
_C.test.point_limit = None

# evaluation
_C.eval = edict()
_C.eval.acceptance_overlap = 0.0
_C.eval.acceptance_radius = 1.0
_C.eval.inlier_ratio_threshold = 0.05
_C.eval.rmse_threshold = 1
_C.eval.rre_threshold = 10
_C.eval.rte_threshold = 2
_C.eval.feat_rre_threshold = 20.0

# ransac
_C.ransac = edict()
_C.ransac.distance_threshold = 0.05
_C.ransac.num_points = 3
_C.ransac.num_iterations = 1000

# optim
_C.optim = edict()
_C.optim.lr = 1e-4
_C.optim.lr_decay = 0.95
_C.optim.lr_decay_steps = 1
_C.optim.weight_decay = 1e-6
_C.optim.max_epoch = 40
_C.optim.grad_acc_steps = 1

# model - backbone
_C.backbone = edict()
_C.backbone.num_stages = 4
_C.backbone.num_neighbors = [35] * _C.backbone.num_stages  # we use constant neighbors
_C.backbone.init_voxel_size =1#初始2 4层相当于下采样三次# 3dmatch 0.025 0.006 2
_C.backbone.subsample_ratio =1.8#1.3 1.8 :0.998
_C.backbone.kernel_size = 4
_C.backbone.share_nonlinearity = False
_C.backbone.conv_way = 'edge_conv'  # 'edge_conv' or 'node_conv'
_C.backbone.use_xyz = True
_C.backbone.init_dim = 96#96
_C.backbone.output_dim = 64#96

# model - Global
_C.model = edict()
_C.model.ground_truth_matching_radius = 2#2# 3dmatch 0.5
_C.model.num_points_in_patch = 64
_C.model.num_sinkhorn_iterations = 100
# model - Coarse Matching
_C.coarse_matching = edict()
_C.coarse_matching.num_targets = 256
_C.coarse_matching.overlap_threshold = 0.05
_C.coarse_matching.num_correspondences = 256#256
_C.coarse_matching.dual_normalization = True

# model - GeoTransformer
_C.geotransformer = edict()
_C.geotransformer.input_dim =768

_C.geotransformer.hidden_dim = 72#72 原本96
_C.geotransformer.output_dim = 128#128
_C.geotransformer.num_heads = 4
#_C.geotransformer.blocks = ['self', 'cross', 'self', 'cross', 'self', 'cross']
_C.geotransformer.blocks = ['self', 'self', 'self']
_C.geotransformer.sigma_d = 0.2#
_C.geotransformer.sigma_a = 15
_C.geotransformer.angle_k = 3
_C.geotransformer.reduction_a = 'max'

# model - Fine Matching
_C.fine_matching = edict()
_C.fine_matching.topk = 3
_C.fine_matching.acceptance_radius =3.5#3.5#5.5#6#4#3  7.8
_C.fine_matching.confidence_threshold = 0.005#?
_C.fine_matching.num_hypotheses = 2200
_C.fine_matching.num_refinement_steps = 5
_C.fine_matching.use_encoder_re_feats = True

_C.fine_matching_geo = edict()
_C.fine_matching_geo.topk = 3
_C.fine_matching_geo.acceptance_radius = 3.5#4#3.5 4
_C.fine_matching_geo.mutual = True
_C.fine_matching_geo.confidence_threshold = 0.05
_C.fine_matching_geo.use_dustbin = False
_C.fine_matching_geo.use_global_score = False
_C.fine_matching_geo.correspondence_threshold = 3
_C.fine_matching_geo.correspondence_limit = None
_C.fine_matching_geo.num_refinement_steps = 5

# loss - Coarse level
_C.coarse_loss = edict()
_C.coarse_loss.positive_margin = 0.1
_C.coarse_loss.negative_margin = 1.4
_C.coarse_loss.positive_optimal = 0.1
_C.coarse_loss.negative_optimal = 1.4
_C.coarse_loss.log_scale = 24
_C.coarse_loss.positive_overlap = 0.1

# loss - Fine level
_C.fine_loss = edict()
_C.fine_loss.positive_radius = 3#2.0#太
_C.fine_loss.negative_radius = 5.5#6.0#???
_C.fine_loss.positive_margin = 0.1
_C.fine_loss.negative_margin = 1.4

# loss - Overall
_C.loss = edict()
_C.loss.weight_coarse_loss = 1.0
_C.loss.weight_fine_ri_loss = 1.0
_C.loss.weight_fine_re_loss = 1.0

def make_cfg():
    return _C


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--link_output', dest='link_output', action='store_true', help='link output dir')
    args = parser.parse_args()
    return args


def main():
    cfg = make_cfg()
    args = parse_args()
    if args.link_output:
        os.symlink(cfg.output_dir, 'output')


if __name__ == '__main__':
    main()
