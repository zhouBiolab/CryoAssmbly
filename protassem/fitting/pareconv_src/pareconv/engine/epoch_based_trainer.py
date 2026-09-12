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
import numpy as np
from typing import Union, Tuple


"""
Implementation from scratch of the SHOT descriptor based on a careful reading of:
Samuele Salti, Federico Tombari, Luigi Di Stefano,
SHOT: Unique signatures of histograms for surface and texture description,
Computer Vision and Image Understanding,
"""

import warnings


import numpy.typing as npt
#from sklearn.neighbors import KDTree
import tqdm
from typing import Tuple

def get_local_rf(
    values: Tuple[np.ndarray, npt.NDArray[np.float64], float],
) -> npt.NDArray[np.float64]:
    """
    Extracts a local reference frame based on the eigendecomposition of the weighted covariance matrix.
    Arguments are given in a tuple to allow for multiprocessing using multiprocessing.Pool.
    """
    point, neighbors, radius = values
    if neighbors.shape[0] == 0:
        return np.eye(3)

    centered_points = neighbors - point

    # EVD of the weighted covariance matrix
    radius_minus_distances = radius - np.linalg.norm(centered_points, axis=1)
    weighted_cov_matrix = (
        centered_points.T
        @ (centered_points * radius_minus_distances[:, None])
        / radius_minus_distances.sum()
    )
    eigenvalues, eigenvectors = np.linalg.eigh(weighted_cov_matrix)

    # disambiguating the axes
    x_orient = (neighbors - point) @ eigenvectors[:, 2]
    if (x_orient < 0).sum() > (x_orient >= 0).sum():
        eigenvectors[:, 2] *= -1
    z_orient = (neighbors - point) @ eigenvectors[:, 0]
    if (z_orient < 0).sum() > (z_orient >= 0).sum():
        eigenvectors[:, 0] *= -1
    eigenvectors[:, 1] = np.cross(eigenvectors[:, 0], eigenvectors[:, 2])

    return np.flip(eigenvectors, axis=1)


def get_azimuth_idx(
    x: Union[float, npt.NDArray[np.float64]],
    y: Union[float, npt.NDArray[np.float64]],
) -> Union[int, npt.NDArray[np.int32]]:
    """
    Finds the bin index of the azimuth of a point in a division in 8 bins.
    Bins are indexed clockwise, and the first bin is between pi and 3 * pi / 4.
    """
    a = (y > 0) | ((y == 0) & (x < 0))
    return (
        4 * a
        + 2 * np.logical_xor((x > 0) | ((x == 0) & (y > 0)), a)
        + np.where(
            (x * y > 0) | (x == 0),
            np.abs(x) < np.abs(y),
            np.abs(x) > np.abs(y),
        )
    )


def interpolate_on_adjacent_husks(
    distance: Union[float, npt.NDArray[np.float64]],
    radius: float
) -> Tuple[
    Union[float, npt.NDArray[np.float64]],
    Union[float, npt.NDArray[np.float64]],
    Union[float, npt.NDArray[np.float64]]
]:
    """
    Interpolates on the adjacent husks.
    Assumes there are only two husks, centered around radius / 4 and 3 * radius / 4.
    """
    radial_bin_size = radius / 2
    inner_bin = (
        ((distance > radius / 2) & (distance < radius * 3 / 4))
        * (radius * 3 / 4 - distance)
        / radial_bin_size
    )
    outer_bin = (
        ((distance < radius / 2) & (distance > radius / 4))
        * (distance - radius / 4)
        / radial_bin_size
    )
    current_bin = (
        (distance < radius / 2)
        * (1 - np.abs(distance - radius / 4) / radial_bin_size)
    ) + (
        (distance > radius / 2)
        * (1 - np.abs(distance - radius * 3 / 4) / radial_bin_size)
    )

    return outer_bin, inner_bin, current_bin


def interpolate_vertical_volumes(
    phi: Union[float, npt.NDArray[np.float64]],
    z: Union[float, npt.NDArray[np.float64]]
) -> Tuple[
    Union[float, npt.NDArray[np.float64]],
    Union[float, npt.NDArray[np.float64]],
    Union[float, npt.NDArray[np.float64]],
]:
    """
    Interpolates on the adjacent vertical volumes.
    Assumes there are only two volumes, centered around pi / 4 and 3 * pi / 4.
    """
    phi_bin_size = np.pi / 2
    upper_volume = (
        (
            ((phi > np.pi / 2) | ((np.abs(phi - np.pi / 2) < 1e-10) & (z <= 0)))
            & (phi <= np.pi * 3 / 4)
        )
        * (np.pi * 3 / 4 - phi)
        / phi_bin_size
    )
    lower_volume = (
        (
            ((phi < np.pi / 2) & ((np.abs(phi - np.pi / 2) >= 1e-10) | (z > 0)))
            & (phi >= np.pi / 4)
        )
        * (phi - np.pi / 4)
        / phi_bin_size
    )
    current_volume = (
        (phi < np.pi / 2)
        * (1 - np.abs(phi - np.pi / 4) / phi_bin_size)
    ) + (
        (phi >= np.pi / 2)
        * (1 - np.abs(phi - np.pi * 3 / 4) / phi_bin_size)
    )

    return upper_volume, lower_volume, current_volume


def compute_single_shot_descriptor(
    values: Tuple[
        np.ndarray,
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        float,
        npt.NDArray[np.float64],
        bool,
        int,
    ]
) -> npt.NDArray[np.float64]:
    """
    Computes a single SHOT descriptor.
    """
    n_cosine_bins, n_azimuth_bins, n_elevation_bins, n_radial_bins = 11, 8, 2, 2

    descriptor = np.zeros((n_cosine_bins, n_azimuth_bins, n_elevation_bins, n_radial_bins))
    (point, neighbors, normals, radius, eigenvectors, normalize, min_neighborhood_size) = values
    rho = np.linalg.norm(neighbors - point, axis=1)
    if (rho > 0).sum() > min_neighborhood_size:
        neighbors = neighbors[rho > 0]
        local_coordinates = (neighbors - point) @ eigenvectors
        cosine = np.clip(normals[rho > 0] @ eigenvectors[:, 2].T, -1, 1)
        rho = rho[rho > 0]

        order = np.argsort(rho)
        rho = rho[order]
        local_coordinates = local_coordinates[order]
        cosine = cosine[order]

        theta = np.arctan2(local_coordinates[:, 1], local_coordinates[:, 0])
        phi = np.arccos(np.clip(local_coordinates[:, 2] / rho, -1, 1))

        cos_bin_pos = (cosine + 1.0) * n_cosine_bins / 2.0 - 0.5
        cos_bin_idx = np.rint(cos_bin_pos).astype(int)
        theta_bin_idx = get_azimuth_idx(local_coordinates[:, 0], local_coordinates[:, 1])
        phi_bin_idx = (local_coordinates[:, 2] > 0).astype(int)
        rho_bin_idx = (rho > radius / 2).astype(int)

        delta_cos = cos_bin_pos - cos_bin_idx
        delta_cos_sign = np.sign(delta_cos)
        abs_delta_cos = delta_cos_sign * delta_cos
        descriptor[(cos_bin_idx + delta_cos_sign).astype(int) % n_cosine_bins,
                   theta_bin_idx, phi_bin_idx, rho_bin_idx] += abs_delta_cos * \
                   ((cos_bin_idx > -0.5) & (cos_bin_idx < n_cosine_bins - 0.5))
        descriptor[cos_bin_idx, theta_bin_idx, phi_bin_idx, rho_bin_idx] += (1 - abs_delta_cos)

        outer_bin, inner_bin, current_bin = interpolate_on_adjacent_husks(rho, radius)
        descriptor[cos_bin_idx, theta_bin_idx, phi_bin_idx, 1] += outer_bin * (rho_bin_idx == 0)
        descriptor[cos_bin_idx, theta_bin_idx, phi_bin_idx, 0] += inner_bin * (rho_bin_idx == 1)
        descriptor[cos_bin_idx, theta_bin_idx, phi_bin_idx, rho_bin_idx] += current_bin

        upper_volume, lower_volume, current_volume = interpolate_vertical_volumes(phi, local_coordinates[:, 2])
        descriptor[cos_bin_idx, theta_bin_idx, 1, rho_bin_idx] += upper_volume * (phi_bin_idx == 0)
        descriptor[cos_bin_idx, theta_bin_idx, 0, rho_bin_idx] += lower_volume * (phi_bin_idx == 1)
        descriptor[cos_bin_idx, theta_bin_idx, phi_bin_idx, rho_bin_idx] += current_volume

        theta_bin_size = 2 * np.pi / n_azimuth_bins
        delta_theta = np.clip((theta - (-np.pi + theta_bin_idx * theta_bin_size)) / theta_bin_size - 0.5, -0.5, 0.5)
        delta_theta_sign = np.sign(delta_theta)
        abs_delta_theta = delta_theta_sign * delta_theta
        descriptor[cos_bin_idx,
                   (theta_bin_idx + delta_theta_sign).astype(int) % n_azimuth_bins,
                   phi_bin_idx, rho_bin_idx] += abs_delta_theta
        descriptor[cos_bin_idx, theta_bin_idx, phi_bin_idx, rho_bin_idx] += (1 - abs_delta_theta)

        if normalize and (norm := np.linalg.norm(descriptor)) > 0:
            return descriptor.ravel() / norm
        return descriptor.ravel()

    return np.zeros(n_cosine_bins * n_azimuth_bins * n_elevation_bins * n_radial_bins)


def compute_shot_descriptor(
    keypoints: npt.NDArray[np.float64],
    cloud_points: npt.NDArray[np.float64],
    normals: npt.NDArray[np.float64],
    radius: float,
    min_neighborhood_size: int = 10,
    n_cosine_bins: int = 11,
    n_azimuth_bins: int = 8,
    n_elevation_bins: int = 2,
    n_radial_bins: int = 2,
    debug_mode: bool = False,
    disable_progress_bars: bool = True,
) -> npt.NDArray[np.float64]:
    """
    Computes the SHOT descriptor on a point cloud. This function should not be used in practice and is only kept for
    debugging purposes. Use the methods from class ShotMultiprocessor instead.
    Normals are expected to be normalized to 1.
    Only the number of cosine bins can be changed as it is.
    See get_azimuth_idx for a description on how to change the number of azimuth bins.
    """
    assert (
        n_azimuth_bins == 8
    ), "Generic function for other than 8 azimuth divisions not implemented"
    assert (
        n_elevation_bins == 2
    ), "Generic function for other than 2 elevation divisions not implemented"
    assert (
        n_radial_bins == 2
    ), "Generic function for other than 2 radial divisions not implemented"
    from sklearn.neighbors import KDTree
    from tqdm import tqdm

    kdtree = KDTree(cloud_points)
    neighborhoods = kdtree.query_radius(keypoints, radius)

    all_descriptors = np.zeros(
        (
            keypoints.shape[0],
            n_cosine_bins * n_azimuth_bins * n_elevation_bins * n_radial_bins,
        )
    )

    for i, point in tqdm(
        enumerate(keypoints),
        desc="SHOT",
        total=len(keypoints),
        disable=disable_progress_bars,
    ):
        descriptor = np.zeros(
            (n_cosine_bins, n_azimuth_bins, n_elevation_bins, n_radial_bins)
        )
        neighbors = cloud_points[neighborhoods[i]]
        distances = np.linalg.norm(neighbors - point, axis=1)
        if (distances > 0).sum() > min_neighborhood_size:
            neighbors = neighbors[distances > 0]
            eigenvectors = get_local_rf((point, neighbors, radius))
            local_coordinates = (neighbors - point) @ eigenvectors
            cosine = np.clip(
                normals[neighborhoods[i]][distances > 0] @ eigenvectors[:, 2].T, -1, 1
            )
            distances = distances[distances > 0]

            order = np.argsort(distances)
            distances = distances[order]
            local_coordinates = local_coordinates[order]
            cosine = cosine[order]

            if debug_mode:
                assert distances.shape[0] <= neighbors.shape[0]
                assert local_coordinates.shape[0] == neighbors.shape[0]
                assert local_coordinates.shape[1] == 3
                assert cosine.shape[0] == neighbors.shape[0]

            # computing the spherical coordinates in the local coordinate system
            theta = np.arctan2(local_coordinates[:, 1], local_coordinates[:, 0])
            phi = np.arccos(np.clip(local_coordinates[:, 2] / distances, -1, 1))

            # computing the indices in the histograms
            bin_dist = (cosine + 1.0) * n_cosine_bins / 2.0 - 0.5
            bin_idx = np.rint(bin_dist).astype(int)
            azimuth_idx = get_azimuth_idx(
                local_coordinates[:, 0], local_coordinates[:, 1]
            )
            # the two arrays below have to be cast as ints, otherwise they will be treated as masks
            elevation_idx = (local_coordinates[:, 2] > 0).astype(int)
            radial_idx = (distances > radius / 2).astype(int)

            # interpolation on the local bins
            bin_dist -= bin_idx  # normalized distance with the neighbor bin
            dist_sign = np.sign(bin_dist)  # left-neighbor or right-neighbor
            abs_bin_dist = dist_sign * bin_dist  # probably faster than np.abs
            # noinspection PyRedundantParentheses
            descriptor[
                (bin_idx + dist_sign).astype(int) % n_cosine_bins,
                azimuth_idx,
                elevation_idx,
                radial_idx,
            ] += abs_bin_dist * (((bin_idx > -0.5) & (bin_idx < n_cosine_bins - 0.5)))
            descriptor[bin_idx, azimuth_idx, elevation_idx, radial_idx] += (
                1 - abs_bin_dist
            )

            # interpolation on the adjacent husks
            outer_bin, inner_bin, current_bin = interpolate_on_adjacent_husks(
                distances, radius
            )
            if debug_mode:
                if np.any((radial_idx == 1) & (np.abs(outer_bin) > 1e-4)):
                    warnings.warn(
                        "Nonzero value for outer volume although current volume is already the outer one."
                    )
                    warnings.warn(
                        str(outer_bin[radial_idx == 1 & (np.abs(outer_bin) > 1e-4)])
                    )
                if np.any((radial_idx == 0) & (np.abs(inner_bin) > 1e-4)):
                    warnings.warn(
                        "Nonzero value for inner volume although current volume is already the inner one."
                    )
                    warnings.warn(
                        str(inner_bin[radial_idx == 0 & (np.abs(inner_bin) > 1e-4)])
                    )
            descriptor[bin_idx, azimuth_idx, elevation_idx, 1] += outer_bin * (
                radial_idx == 0
            )
            descriptor[bin_idx, azimuth_idx, elevation_idx, 0] += inner_bin * (
                radial_idx == 1
            )
            descriptor[bin_idx, azimuth_idx, elevation_idx, radial_idx] += current_bin

            # interpolation between adjacent vertical volumes
            upper_volume, lower_volume, current_volume = interpolate_vertical_volumes(
                phi, local_coordinates[:, 2]
            )
            if debug_mode:
                if np.any((elevation_idx == 1) & (np.abs(upper_volume) > 1e-4)):
                    warnings.warn(
                        "Nonzero value for upper volume although current volume is already the upper one."
                    )
                    warnings.warn(
                        str(
                            upper_volume[
                                elevation_idx == 1 & (np.abs(upper_volume) > 1e-4)
                            ]
                        )
                    )
                if np.any((elevation_idx == 0) & (np.abs(lower_volume) > 1e-4)):
                    warnings.warn(
                        "Nonzero value for lower volume although current volume is already the lower one."
                    )
                    warnings.warn(
                        str(
                            lower_volume[
                                elevation_idx == 0 & (np.abs(lower_volume) > 1e-4)
                            ]
                        )
                    )
            descriptor[bin_idx, azimuth_idx, 1, radial_idx] += upper_volume * (
                elevation_idx == 0
            )
            descriptor[bin_idx, azimuth_idx, 0, radial_idx] += lower_volume * (
                elevation_idx == 1
            )
            descriptor[
                bin_idx, azimuth_idx, elevation_idx, radial_idx
            ] += current_volume

            # interpolation between adjacent horizontal volumes
            # local_coordinates[:, 0] * local_coordinates[:, 1] != 0
            azimuth_bin_size = 2 * np.pi / n_azimuth_bins
            azimuth_dist = np.clip(
                (theta - (-np.pi + azimuth_idx * azimuth_bin_size)) / azimuth_bin_size
                - 0.5,
                -0.5,
                0.5,
            )
            azimuth_dist_sign = np.sign(azimuth_dist)  # left-neighbor or right-neighbor
            azimuth_abs_dist = azimuth_dist_sign * azimuth_dist
            descriptor[
                bin_idx,
                (azimuth_idx + azimuth_dist_sign).astype(int) % n_azimuth_bins,
                elevation_idx,
                radial_idx,
            ] += azimuth_abs_dist
            descriptor[bin_idx, azimuth_idx, elevation_idx, radial_idx] += (
                1 - azimuth_abs_dist
            )

            # normalizing the descriptor to Euclidian norm 1
            if (descriptor_norm := np.linalg.norm(descriptor)) > 0:
                all_descriptors[i] = descriptor.ravel() / descriptor_norm

    return all_descriptors
def pairwise_distance(
    x: torch.Tensor, y: torch.Tensor, normalized: bool = False, channel_first: bool = False
) -> torch.Tensor:
    r"""Pairwise distance of two (batched) point clouds.

    Args:
        x (Tensor): (*, N, C) or (*, C, N)
        y (Tensor): (*, M, C) or (*, C, M)
        normalized (bool=False): if the points are normalized, we have "x2 + y2 = 1", so "d2 = 2 - 2xy".
        channel_first (bool=False): if True, the points shape is (*, C, N).

    Returns:
        dist: torch.Tensor (*, N, M)
    """
    if channel_first:
        channel_dim = -2
        xy = torch.matmul(x.transpose(-1, -2), y)  # [(*, C, N) -> (*, N, C)] x (*, C, M)
    else:
        channel_dim = -1
        xy = torch.matmul(x, y.transpose(-1, -2))  # (*, N, C) x [(*, M, C) -> (*, C, M)]
    if normalized:
        sq_distances = 2.0 - 2.0 * xy
    else:
        x2 = torch.sum(x ** 2, dim=channel_dim).unsqueeze(-1)  # (*, N, C) or (*, C, N) -> (*, N) -> (*, N, 1)
        y2 = torch.sum(y ** 2, dim=channel_dim).unsqueeze(-2)  # (*, M, C) or (*, C, M) -> (*, M) -> (*, 1, M)
        sq_distances = x2 - 2 * xy + y2
    sq_distances = sq_distances.clamp(min=0.0)
    return sq_distances
def iss(data, gamma21=0.6, gamma32=0.6, KDTree_radius=6, NMS_radius=6, max_num=100, min_neighbors=5):
    from scipy.spatial import KDTree
    leaf_size = 32
    tree = KDTree(data, leaf_size)
    radius_neighbor = tree.query_ball_point(data, KDTree_radius)

    keypoints = []
    min_feature_value = []

    for index in range(len(radius_neighbor)):
        neighbor_idx = radius_neighbor[index]
        if index in neighbor_idx:
            neighbor_idx.remove(index)
        if len(neighbor_idx) < min_neighbors:
            continue

        try:
            diff = data[neighbor_idx] - data[index]
            dists = np.linalg.norm(diff, axis=1)
            dists[dists == 0] = 1e-3
            weight = 1.0 / dists

            cov = np.zeros((3, 3))
            tmp = diff[:, :, np.newaxis]
            for i in range(len(neighbor_idx)):
                cov += weight[i] * (tmp[i] @ tmp[i].transpose())
            cov /= np.sum(weight)

            if np.isnan(cov).any() or np.isinf(cov).any():
                continue

            s = np.linalg.svd(cov, compute_uv=False)

            # 稳定性判断
            if np.any(np.isnan(s)) or np.any(np.isinf(s)):
                continue
            if s[0] < 1e-6 or s[1] < 1e-6:
                continue

            if (s[1] / (s[0] + 1e-6) < gamma21) and (s[2] / (s[1] + 1e-6) < gamma32):
                keypoints.append(data[index])
                min_feature_value.append(s[2])

        except Exception as e:
            warnings.warn(f"Covariance SVD failed at index {index}: {e}")
            continue

    if len(keypoints) == 0:
        return None

    # NMS
    from scipy.spatial import KDTree
    keypoints_after_NMS = []
    nms_tree = KDTree(keypoints, 10)
    index_all = list(range(len(keypoints)))

    for _ in range(max_num):
        if len(min_feature_value) == 0:
            break
        max_index = min_feature_value.index(max(min_feature_value))
        tmp_point = keypoints[max_index]
        del_indexs = nms_tree.query_ball_point(tmp_point, NMS_radius)
        for del_index in del_indexs:
            if del_index in index_all:
                i = index_all.index(del_index)
                del min_feature_value[i]
                del keypoints[i]
                del index_all[i]
        keypoints_after_NMS.append(tmp_point)

    return np.array(keypoints_after_NMS) if len(keypoints_after_NMS) > 0 else None

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
            sigmma = 4.5
            data_dict = to_cuda(data_dict)

            # 点数判断
            ref_length_d = data_dict['lengths'][-2][0].item()
            ref_length_c = data_dict['lengths'][-1][0].item()

            points_c_iput = data_dict['points'][-2]
            ref_points_d = points_c_iput[:ref_length_d][:, :3].cpu().numpy()
            src_points_d = points_c_iput[ref_length_d:][:, :3].cpu().numpy()

            ref_points_co = data_dict['points'][-1][:ref_length_c][:, :3]
            src_points_co = data_dict['points'][-1][ref_length_c:][:, :3]

            # 提取关键点（numpy）
            keypoint_ref = iss(ref_points_d, gamma21=0.6, gamma32=0.6, KDTree_radius=0.15, NMS_radius=0.15,
                               max_num=1)
            keypoint_src = iss(src_points_d, gamma21=0.6, gamma32=0.6, KDTree_radius=0.15, NMS_radius=0.15,
                               max_num=1)

            # device 和 dtype 一致性
            device = ref_points_co.device
            dtype = ref_points_co.dtype

            # 安全转换为 tensor
            has_ref = keypoint_ref is not None and len(keypoint_ref) > 0
            has_src = keypoint_src is not None and len(keypoint_src) > 0

            final_ref_pts = ref_points_co
            final_src_pts = src_points_co
            new_ref_len = ref_points_co.size(0)
            new_src_len = src_points_co.size(0)

            # 处理 ref 侧

            if has_ref:
                kp_ref_t = torch.from_numpy(keypoint_ref).to(device=device, dtype=dtype)
                assert kp_ref_t.shape[1] == ref_points_co.shape[1], "Ref dimension mismatch"
                if kp_ref_t.size(0) > 0:
                    dist_r = torch.sqrt(pairwise_distance(ref_points_co, kp_ref_t))  # [N, K]
                    ref_mask_s = (torch.topk(dist_r, k=1, dim=-1, largest=False)[0] > sigmma).view(-1)
                    final_ref_pts = torch.cat([kp_ref_t, ref_points_co[ref_mask_s]], dim=0)
                    new_ref_len = final_ref_pts.size(0)
                    del dist_r, ref_mask_s

            # 处理 src 侧
            if has_src:
                kp_src_t = torch.from_numpy(keypoint_src).to(device=device, dtype=dtype)
                assert kp_src_t.shape[1] == src_points_co.shape[1], "Src dimension mismatch"
                if kp_src_t.size(0) > 0:
                    dist_s = torch.sqrt(pairwise_distance(src_points_co, kp_src_t))  # [N, K]
                    src_mask_s = (torch.topk(dist_s, k=1, dim=-1, largest=False)[0] > sigmma).view(-1)
                    final_src_pts = torch.cat([kp_src_t, src_points_co[src_mask_s]], dim=0)
                    new_src_len = final_src_pts.size(0)
                    del dist_s, src_mask_s


            # 替换 data_dict['points'][-1]
            if final_ref_pts.size(0) > 0 and final_src_pts.size(0) > 0:
                merged = torch.cat([final_ref_pts, final_src_pts], dim=0)
            elif final_ref_pts.size(0) > 0:
                merged = final_ref_pts
            elif final_src_pts.size(0) > 0:
                merged = final_src_pts
            else:
                raise RuntimeError("关键点提取失败：ref/src 点均为空")

            # 保持原 device/dtype
            merged = merged.to(device=data_dict['points'][-1].device, dtype=data_dict['points'][-1].dtype)
            data_dict['points'][-1] = merged
            data_dict['lengths'][-1][0] = new_ref_len
            data_dict['lengths'][-1][1] = new_src_len
            del final_ref_pts, final_src_pts
            del merged
            del ref_points_d, src_points_d
            del ref_points_co, src_points_co
            del keypoint_ref, keypoint_src

            # 无论如何，都要记录 ref_length_iss、src_length_iss
            #data_dict['ref_length_iss'] = new_ref_len
            #data_dict['src_length_iss'] = new_src_len

            data = precompute_neibors(data_dict['points'], data_dict['lengths'],
                                                  self.cfg.backbone.num_stages,
                                                  self.cfg.backbone.num_neighbors,
                                                  )
            data_dict.update(data)
            del data, points_c_iput


            '''
            
            ######4.22——xxy-混合点以及密度聚类-旋转不变性描述符
            norm_ref = data_dict['ref_vectors']
            norm_src = data_dict['src_vectors']
            ref_length_c = data_dict['lengths'][-1][0].item()
            ref_length_f = data_dict['lengths'][1][0].item()
            ref_length = data_dict['lengths'][0][0].item()
            points = data_dict['points'][0][:, :3].detach()
            points_c = data_dict['points'][-1][:, :3].detach()
            ref_points = points[:ref_length]
            src_points = points[ref_length:]
            ref_points_c = points_c[:ref_length_c]
            src_points_c = points_c[ref_length_c:]
            ref_points_c = ref_points_c.detach().cpu().numpy()
            src_points_c = src_points_c.detach().cpu().numpy()
            ref_points = ref_points.detach().cpu().numpy()
            src_points = src_points.detach().cpu().numpy()
            norm_src = norm_src.detach().cpu().numpy()
            norm_ref = norm_ref.detach().cpu().numpy()
            ref_shot = compute_shot_descriptor(keypoints = ref_points_c, cloud_points = ref_points,normals=norm_ref,radius=33,debug_mode=False, disable_progress_bars=True)
            src_shot = compute_shot_descriptor(keypoints=src_points_c, cloud_points=src_points, normals=norm_src,
                                               radius=33, debug_mode = False, disable_progress_bars=True)

            ref_shot_t = torch.from_numpy(ref_shot).to(device=device, dtype=dtype)
            src_shot_t = torch.from_numpy(src_shot).to(device=device, dtype=dtype)

            # 3. 写入 data_dict
            data_dict['ref_shot'] = ref_shot_t  # [N_ref, 352]
            data_dict['src_shot'] = src_shot_t  # [N_src, 352]

            #print("train_shot_ref", src_shot)
            '''










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
            #del output_dict, data_dict ,result_dict
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
            sigmma = 4.5
            data_dict = to_cuda(data_dict)

            # 点数判断
            ref_length_d = data_dict['lengths'][-2][0].item()
            ref_length_c = data_dict['lengths'][-1][0].item()

            points_c_iput = data_dict['points'][-2]
            ref_points_d = points_c_iput[:ref_length_d][:, :3].cpu().numpy()
            src_points_d = points_c_iput[ref_length_d:][:, :3].cpu().numpy()

            ref_points_co = data_dict['points'][-1][:ref_length_c][:, :3]
            src_points_co = data_dict['points'][-1][ref_length_c:][:, :3]

            # 提取关键点（numpy）
            keypoint_ref = iss(ref_points_d, gamma21=0.6, gamma32=0.6, KDTree_radius=0.15, NMS_radius=0.15,
                               max_num=1)#6,6
            keypoint_src = iss(src_points_d, gamma21=0.6, gamma32=0.6, KDTree_radius=0.15, NMS_radius=0.15,
                               max_num=1)

            # device 和 dtype 一致性
            device = ref_points_co.device
            dtype = ref_points_co.dtype

            # 安全转换为 tensor
            has_ref = keypoint_ref is not None and len(keypoint_ref) > 0
            has_src = keypoint_src is not None and len(keypoint_src) > 0

            final_ref_pts = ref_points_co
            final_src_pts = src_points_co
            new_ref_len = ref_points_co.size(0)
            new_src_len = src_points_co.size(0)

            # 处理 ref 侧
            if has_ref:
                kp_ref_t = torch.from_numpy(keypoint_ref).to(device=device, dtype=dtype)
                assert kp_ref_t.shape[1] == ref_points_co.shape[1], "Ref dimension mismatch"
                if kp_ref_t.size(0) > 0:
                    dist_r = torch.sqrt(pairwise_distance(ref_points_co, kp_ref_t))  # [N, K]
                    ref_mask_s = (torch.topk(dist_r, k=1, dim=-1, largest=False)[0] > sigmma).view(-1)
                    final_ref_pts = torch.cat([kp_ref_t, ref_points_co[ref_mask_s]], dim=0)
                    new_ref_len = final_ref_pts.size(0)

                    del dist_r, ref_mask_s

            # 处理 src 侧

            if has_src:
                kp_src_t = torch.from_numpy(keypoint_src).to(device=device, dtype=dtype)
                assert kp_src_t.shape[1] == src_points_co.shape[1], "Src dimension mismatch"
                if kp_src_t.size(0) > 0:
                    dist_s = torch.sqrt(pairwise_distance(src_points_co, kp_src_t))  # [N, K]
                    src_mask_s = (torch.topk(dist_s, k=1, dim=-1, largest=False)[0] > sigmma).view(-1)
                    final_src_pts = torch.cat([kp_src_t, src_points_co[src_mask_s]], dim=0)
                    new_src_len = final_src_pts.size(0)
                    del dist_s,src_mask_s

            # 替换 data_dict['points'][-1]
            if final_ref_pts.size(0) > 0 and final_src_pts.size(0) > 0:
                merged = torch.cat([final_ref_pts, final_src_pts], dim=0)
            elif final_ref_pts.size(0) > 0:
                merged = final_ref_pts
            elif final_src_pts.size(0) > 0:
                merged = final_src_pts
            else:
                raise RuntimeError("关键点提取失败：ref/src 点均为空")

            # 保持原 device/dtype
            merged = merged.to(device=data_dict['points'][-1].device, dtype=data_dict['points'][-1].dtype)
            data_dict['points'][-1] = merged
            data_dict['lengths'][-1][0] = new_ref_len
            data_dict['lengths'][-1][1] = new_src_len
            del merged
            del final_ref_pts, final_src_pts
            del ref_points_d, src_points_d
            del ref_points_co, src_points_co
            del keypoint_ref, keypoint_src



            # 无论如何，都要记录 ref_length_iss、src_length_iss
            #data_dict['ref_length_iss'] = new_ref_len
            #data_dict['src_length_iss'] = new_src_len


            data = precompute_neibors(data_dict['points'], data_dict['lengths'],
                                      self.cfg.backbone.num_stages,
                                      self.cfg.backbone.num_neighbors,
                                      )
            data_dict.update(data)
            del data, points_c_iput
            '''
            
            norm_ref = data_dict['ref_vectors']
            norm_src = data_dict['src_vectors']
            ref_length_c = data_dict['lengths'][-1][0].item()
            ref_length_f = data_dict['lengths'][1][0].item()
            ref_length = data_dict['lengths'][0][0].item()
            points = data_dict['points'][0][:, :3].detach()
            points_c = data_dict['points'][-1][:, :3].detach()
            ref_points = points[:ref_length]
            src_points = points[ref_length:]
            ref_points_c = points_c[:ref_length_c]
            src_points_c = points_c[ref_length_c:]
            ref_points_c = ref_points_c.detach().cpu().numpy()
            src_points_c = src_points_c.detach().cpu().numpy()
            ref_points = ref_points.detach().cpu().numpy()
            norm_ref = norm_ref.detach().cpu().numpy()
            src_points = src_points.detach().cpu().numpy()
            norm_src = norm_src.detach().cpu().numpy()
            ref_shot = compute_shot_descriptor(keypoints = ref_points_c, cloud_points = ref_points,normals=norm_ref,radius=33,debug_mode=False, disable_progress_bars=True)
            src_shot = compute_shot_descriptor(keypoints=src_points_c, cloud_points=src_points, normals=norm_src,
                                               radius=33, debug_mode=False, disable_progress_bars=True)

            ref_shot_t = torch.from_numpy(ref_shot).to(device=device, dtype=dtype)
            src_shot_t = torch.from_numpy(src_shot).to(device=device, dtype=dtype)

            # 3. 写入 data_dict
            data_dict['ref_shot'] = ref_shot_t  # [N_ref, 352]
            data_dict['src_shot'] = src_shot_t  # [N_src, 352]
            '''
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
            #del output_dict, data_dict, result_dict
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
