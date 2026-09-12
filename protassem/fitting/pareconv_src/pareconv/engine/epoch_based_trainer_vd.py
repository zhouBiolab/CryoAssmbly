import os
import os.path as osp
import pdb
from typing import Tuple, Dict

import ipdb
import torch
import tqdm
import wandb

from pareconv.engine.base_trainer import BaseTrainer
from pareconv.utils.torch import to_cuda
from pareconv.utils.summary_board import SummaryBoard
from pareconv.utils.timer import Timer
from pareconv.utils.common import get_log_string
from pareconv.utils.data import precompute_neibors
from scipy.spatial import KDTree


def iss(data, gamma21, gamma32, KDTree_radius, NMS_radius, max_num=100):
    leaf_size = 32
    tree = KDTree(data, leaf_size)
    radius_neighbor = tree.query_ball_point(data, KDTree_radius)
    keypoints = []
    min_feature_value = []
    for index in range(len(radius_neighbor)):
        neighbor_idx = radius_neighbor[index]
        neighbor_idx.remove(index)
        if len(neighbor_idx) == 0:
            continue

        weight = np.linalg.norm(data[neighbor_idx] - data[index], axis=1)
        weight[weight == 0] = 0.001
        weight = 1 / weight

        cov = np.zeros((3, 3))
        tmp = (data[neighbor_idx] - data[index])[:, :, np.newaxis]
        for i in range(len(neighbor_idx)):
            cov += weight[i] * tmp[i].dot(tmp[i].transpose())
        cov /= np.sum(weight)

        '''
        tmp = (data[neighbor_idx] - data[index])[:, :, np.newaxis]  # N,3,1
        cov = np.sum(weight[:, np.newaxis, np.newaxis] *
                     (tmp @ tmp.transpose(0, 2, 1)), axis=0) / np.sum(weight)
        '''

        s = np.linalg.svd(cov, compute_uv=False)

        if (s[1] / (s[0] + 0.000001) < gamma21) and (s[2] / (s[1] + 0.000001) < gamma32):
            keypoints.append(data[index])
            min_feature_value.append(s[2])

    # NMS step
    keypoints_after_NMS = []
    leaf_size = 10
    nms_tree = KDTree(keypoints, leaf_size)
    index_all = [i for i in range(len(keypoints))]
    for iter in range(max_num):
        max_index = min_feature_value.index(max(min_feature_value))
        tmp_point = keypoints[max_index]
        del_indexs = nms_tree.query_ball_point(tmp_point, NMS_radius)
        for del_index in del_indexs:
            if del_index in index_all:
                del min_feature_value[index_all.index(del_index)]
                del keypoints[index_all.index(del_index)]
                del index_all[index_all.index(del_index)]
        keypoints_after_NMS.append(tmp_point)
        if len(keypoints) == 0:
            break

    return np.array(keypoints_after_NMS)

class EpochBasedTrainer(BaseTrainer):
    def __init__(
        self,
        cfg,
        max_epoch,
        parser=None,
        cudnn_deterministic=True,
        autograd_anomaly_detection=False,
        save_all_snapshots=True,
        run_grad_check=False,
        grad_acc_steps=1,
    ):
        super().__init__(
            cfg,
            parser=parser,
            cudnn_deterministic=cudnn_deterministic,
            autograd_anomaly_detection=autograd_anomaly_detection,
            save_all_snapshots=save_all_snapshots,
            run_grad_check=run_grad_check,
            grad_acc_steps=grad_acc_steps,
        )
        self.max_epoch = max_epoch

    def before_train_step(self, epoch, iteration, data_dict) -> None:
        pass

    def before_val_step(self, epoch, iteration, data_dict) -> None:
        pass

    def after_train_step(self, epoch, iteration, data_dict, output_dict, result_dict) -> None:
        pass

    def after_val_step(self, epoch, iteration, data_dict, output_dict, result_dict) -> None:
        pass

    def before_train_epoch(self, epoch) -> None:
        pass

    def before_val_epoch(self, epoch) -> None:
        pass

    def after_train_epoch(self, epoch) -> None:
        pass

    def after_val_epoch(self, epoch) -> None:
        pass

    def train_step(self, epoch, iteration, data_dict) -> Tuple[Dict, Dict]:
        pass

    def val_step(self, epoch, iteration, data_dict) -> Tuple[Dict, Dict]:
        pass

    def after_backward(self, epoch, iteration, data_dict, output_dict, result_dict) -> None:
        pass

    def check_gradients(self, epoch, iteration, data_dict, output_dict, result_dict):
        if not self.run_grad_check:
            return
        if not self.check_invalid_gradients():
            self.logger.error('Epoch: {}, iter: {}, invalid gradients.'.format(epoch, iteration))
            torch.save(data_dict, 'data.pth')
            torch.save(self.model, 'model.pth')
            self.logger.error('Data_dict and model snapshot saved.')
            # ipdb.set_trace()

    def train_epoch(self):
        if self.distributed:
            pass
            # self.train_loader.sampler.set_epoch(self.epoch)
        self.before_train_epoch(self.epoch)
        self.optimizer.zero_grad()
        total_iterations = len(self.train_loader)
        for iteration, data_dict in enumerate(self.train_loader):
            self.inner_iteration = iteration + 1
            self.iteration += 1
            sigmma =6
            data_dict = to_cuda(data_dict)
            data = precompute_neibors(data_dict['points'], data_dict['lengths'],
                                              self.cfg.backbone.num_stages,
                                              self.cfg.backbone.num_neighbors,
                                              )
            data_dict.update(data)
            ######4.22——xxy-混合点以及密度聚类-旋转不变性描述符

            ref_length_d = data_dict['lengths'][-2][0].item()#倒数第二层 作为输入进行聚类算法 看点数选择不同层数
            ref_length_c = data_dict['lengths'][-1][0].item()#超点 体素采样
            points_c_iput = data_dict['points'][-2]

            ref_points_d = points_c_iput[:ref_length_d].cpu().numpy()
            src_points_d = points_c_iput[ref_length_d:].cpu().numpy()

            ref_points_co = data_dict['points'][-1][:ref_length_c]#.cpu().numpy()
            src_points_co = data_dict['points'][-1][ref_length_c:]#.cpu().numpy() 体素采样的超点



            #keypoint_ref = mean_shift(ref_points_d, gamma21=0.6, gamma32=0.6, KDTree_radius=0.15, NMS_radius=0.15, max_num=5000)
            #keypoint_src = mean_shift(src_points_d, gamma21=0.6, gamma32=0.6, KDTree_radius=0.15, NMS_radius=0.15, max_num=5000)

            keypoint_ref = iss(ref_points_d, gamma21=0.6, gamma32=0.6, KDTree_radius=15, NMS_radius=15, max_num=len(ref_points_co)//2)
            keypoint_src = iss(src_points_d, gamma21=0.6, gamma32=0.6, KDTree_radius=15, NMS_radius=15, max_num=len(ref_points_co)//2)

            #keypoint_ref = DBSCAN(ref_points_d, gamma21=0.6, gamma32=0.6, KDTree_radius=0.15, NMS_radius=0.15, max_num=5000)
            #keypoint_src = DBSCAN(src_points_d, gamma21=0.6, gamma32=0.6, KDTree_radius=0.15, NMS_radius=0.15, max_num=5000)

            keypoint_ref = torch.from_numpy(keypoint_ref)
            keypoint_src = torch.from_numpy(keypoint_src)

            dist_matrix = torch.sqrt(pairwise_distance(ref_points_co,keypoint_ref))
            dist_c = torch.topk(dist_matrix, k=1, dim=-1, largest=False)[0]
            ref_mask_s = (dist_c > sigmma).view(-1)#3*voxel_size
            del dist_matrix, dist_c  # 显式删除不再需要的张量


            dist_matrix = torch.sqrt(pairwise_distance(src_points_co,keypoint_src))
            dist_c = torch.topk(dist_matrix, k=1, dim=-1, largest=False)[0]
            src_mask_s= (dist_c > sigmma).view(-1)#3*voxel_size
            del dist_matrix, dist_c  # 显式删除不再需要的张量

            ref_points_h = torch.cat((keypoint_ref, ref_points_co[ref_mask_s]), dim=0)#混合点ref
            src_points_h = torch.cat((keypoint_src, src_points_co[src_mask_s]), dim=0)#混合点src
            del ref_points_co, src_points_co, ref_mask_s, src_mask_s

            points_h = torch.cat((ref_points_h,src_points_h),dim = 0)
            old_points = data_dict['points'][-1]

            # 2. 覆盖写入新的混合点张量
            #    （确保 device/dtype 和原来一致）
            device = old_points.device
            dtype = old_points.dtype
            points_h = points_h.to(device=device, dtype=dtype)
            data_dict['points'][-1] = points_h

            #old_length = data_dict['lengths'][-1]
            #del old_length
            data_dict['lengths'][-1][0] = ref_points_h.size(0)
            data_dict['lengths'][-1][1] = src_points_h.size(0)

            ref_length_iss = keypoint_ref.shape[0]
            src_length_iss = keypoint_src.shape[0]
            data_dict.update({
                'ref_length_iss': ref_length_iss,
                'src_length_iss': src_length_iss,

            })
            del ref_points_h, src_points_h, points_h, keypoint_ref, keypoint_src









            self.before_train_step(self.epoch, self.inner_iteration, data_dict)
            self.timer.add_prepare_time()
            # forward
            output_dict, result_dict = self.train_step(self.epoch, self.inner_iteration, data_dict)
            # backward & optimization
            result_dict['loss'].backward()

            self.after_backward(self.epoch, self.inner_iteration, data_dict, output_dict, result_dict)
            self.check_gradients(self.epoch, self.inner_iteration, data_dict, output_dict, result_dict)
            self.optimizer_step(self.inner_iteration)
            # after training
            self.timer.add_process_time()
            self.after_train_step(self.epoch, self.inner_iteration, data_dict, output_dict, result_dict)
            result_dict = self.release_tensors(result_dict)
            self.summary_board.update_from_result_dict(result_dict)
            # logging
            if self.inner_iteration % self.log_steps == 0:
                summary_dict = self.summary_board.summary()
                message = get_log_string(
                    result_dict=summary_dict,
                    epoch=self.epoch,
                    max_epoch=self.max_epoch,
                    iteration=self.inner_iteration,
                    max_iteration=total_iterations,
                    lr=self.get_lr(),
                    timer=self.timer,
                )
                self.logger.info(message)
                self.write_event('train', summary_dict, self.iteration)
            torch.cuda.empty_cache()
        self.after_train_epoch(self.epoch)
        message = get_log_string(self.summary_board.summary(), epoch=self.epoch, timer=self.timer)
        self.logger.critical(message)
        # scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        # snapshot
        self.save_snapshot(f'epoch-{self.epoch}.pth.tar')
        if not self.save_all_snapshots:
            last_snapshot = f'epoch-{self.epoch - 1}.pth.tar'
            if osp.exists(last_snapshot):
                os.remove(last_snapshot)

    def inference_epoch(self):
        self.set_eval_mode()
        self.before_val_epoch(self.epoch)
        summary_board = SummaryBoard(adaptive=True)
        timer = Timer()
        total_iterations = len(self.val_loader)
        pbar = tqdm.tqdm(enumerate(self.val_loader), total=total_iterations, ncols=180)
        for iteration, data_dict in pbar:
            self.inner_iteration = iteration + 1
            data_dict = to_cuda(data_dict)
            data = precompute_neibors(data_dict['points'], data_dict['lengths'],
                                              self.cfg.backbone.num_stages,
                                              self.cfg.backbone.num_neighbors,
                                      )
            data_dict.update(data)
            self.before_val_step(self.epoch, self.inner_iteration, data_dict)
            timer.add_prepare_time()
            output_dict, result_dict = self.val_step(self.epoch, self.inner_iteration, data_dict)
            torch.cuda.synchronize()
            timer.add_process_time()
            self.after_val_step(self.epoch, self.inner_iteration, data_dict, output_dict, result_dict)
            result_dict = self.release_tensors(result_dict)
            summary_board.update_from_result_dict(result_dict)
            message = get_log_string(
                result_dict=summary_board.summary(),
                epoch=self.epoch,
                iteration=self.inner_iteration,
                max_iteration=total_iterations,
                timer=timer,
            )
            pbar.set_description(message)
            torch.cuda.empty_cache()
        self.after_val_epoch(self.epoch)
        summary_dict = summary_board.summary()
        message = '[Val] ' + get_log_string(summary_dict, epoch=self.epoch, timer=timer)
        self.logger.critical(message)
        self.write_event('val', summary_dict, self.epoch)
        self.set_train_mode()

    def run(self):
        assert self.train_loader is not None
        assert self.val_loader is not None

        if self.args.resume:
            self.load_snapshot(osp.join(self.snapshot_dir, 'snapshot.pth.tar'))
        elif self.args.snapshot is not None:
            self.load_snapshot(self.args.snapshot)
        self.set_train_mode()
        # self.inference_epoch()
        while self.epoch < self.max_epoch:
            self.epoch += 1
            self.train_epoch()
            self.inference_epoch()
        if not self.debug:
            wandb.finish()
