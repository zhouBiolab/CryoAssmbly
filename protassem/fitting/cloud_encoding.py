"""单侧点云几何：源与目标各自独立构建多尺度点、邻居与上下采样索引（任务卡 T04/T05）。

设计要点
--------
1. `CloudGeometry` 只保存**单侧局部索引**；跨侧拼接由 `join_geometries()` 适配层加偏移完成，
   从而保持旧联合入口（pareconv `registration_collate_fn_stack_mode` + `precompute_neibors`）
   的语义与逐位结果不变。
2. 多尺度配置（各级体素尺寸、最后一层采样方式）**显式传参**，不再依赖 pareconv 的模块级
   全局变量；调用方（`demo_mask`）把当前真正生效的值读出来传入并记录。
3. pointops 的 k-NN 在邻居不足时用 0 填充尾部槽位（哨兵）。只有 `source_count >= k` 时才
   保证没有哨兵；出现哨兵时保持 0、**不加偏移**（见 `offset_indices`），否则会指向另一侧的
   真实点。
4. `upsampling` 按 **stage 对齐**（stage 0 为 None）；pareconv 的紧凑列表约定（元素 j 对应
   stage j+1）由适配层负责还原，避免"不同尺度偏移错"。
5. **几何缓存（T05）**：key 由输入点集/顺序、特征、生效采样配置与**邻居数**共同决定（构建前
   即可计算）；只缓存确定性采样（`DETERMINISTIC_SAMPLING`），含 RNG 的采样一律不缓存
   （见 `acquire_geometry` 的偏差处理）。缓存条目存**CPU** 张量并按字节计费，命中时搬回设备。

一行一个操作、参数显式传递、不新增隐式全局状态。
"""

import hashlib
import struct
import time
from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from pareconv.extensions.pointops.functions import pointops
from pareconv.modules.ops import grid_subsample, point_to_node_partition
from pareconv.utils.data_mask import farthest_point_sampling_gpu

from protassem.fitting.feature_cache import ByteLruCache

# 单侧几何结构版本：任何影响 points/neighbors/索引语义的改动都必须 +1（缓存 key 的一部分）
GEOMETRY_VERSION = 1
FPS_TARGET_RATIO = 0.25     # 与 pareconv precompute_subsample 的 fps 分支一致
# 只有确定性采样才能缓存：fps 走 pytorch3d 的 random_start_point，结果依赖全局 RNG 状态
DETERMINISTIC_SAMPLING = ("voxel",)
# 编码语义版本（T07）：改动 encode_cloud 的输出内容/含义时必须 +1（编码缓存 key 的一部分）
ENCODING_CACHE_VERSION = 1


class GeometryError(ValueError):
    """几何构建的输入不满足契约。"""


@dataclass
class NodePartition:
    """一个点云在粗阶段上的节点分区（把细节点分配给最近的粗节点）。"""

    masks: torch.Tensor          # (N_c,) bool
    knn_indices: torch.Tensor    # (N_c, K) long，局部索引
    knn_masks: torch.Tensor      # (N_c, K) bool


@dataclass
class CloudGeometry:
    """单侧多尺度几何。所有索引都是本侧局部索引。"""

    points: List[torch.Tensor]              # 每阶段 (N_i, D)；points[0] 是原始点
    lengths: List[torch.Tensor]             # 每阶段 (1,) 点数
    features: torch.Tensor                  # (N_0, C)
    voxel_sizes: Tuple[float, ...]          # 实际生效的体素尺寸
    sampling_method: str                    # 实际生效的最后一层采样方式（voxel/fps）
    centroid: Optional[np.ndarray] = None   # 中心化前的质心（原坐标系），仅记录用
    neighbors: List[torch.Tensor] = field(default_factory=list)     # (N_i, k) 指向 points[i]
    subsampling: List[torch.Tensor] = field(default_factory=list)   # (N_{i+1}, k) 指向 points[i]
    upsampling: List[Optional[torch.Tensor]] = field(default_factory=list)  # (N_i, 1) 指向 points[i+1]
    node_partition: Optional[NodePartition] = None
    # 指纹记忆（不参与相等性比较）：只覆盖不可变部分（点/特征/配置），
    # build_neighbors/attach_node_partition 的改动不影响它，因此缓存安全
    _fingerprint: Optional[str] = field(default=None, repr=False, compare=False)

    @property
    def device(self):
        return self.points[0].device

    @property
    def num_stages(self):
        return len(self.points)

    @property
    def stage_counts(self):
        return [int(item.shape[0]) for item in self.points]

    @property
    def point_count(self):
        return int(self.points[0].shape[0])

    def fingerprint(self):
        """几何指纹：实际点集与顺序、特征、生效采样配置、结构版本。

        供 T05 几何缓存与 T07 编码缓存做 key；不含质心（质心只影响写出，不影响网络输入）。
        首次计算后记忆（只覆盖不可变部分，见 `_fingerprint` 字段说明）。
        """
        if self._fingerprint is not None:
            return self._fingerprint
        digest = hashlib.sha256()
        digest.update(("geometry_version=%d;" % GEOMETRY_VERSION).encode())
        digest.update(("stages=%d;voxels=%s;sampling=%s;"
                       % (self.num_stages,
                          ",".join("%.6f" % value for value in self.voxel_sizes),
                          self.sampling_method)).encode())
        for index, item in enumerate(self.points):
            array = item.detach().cpu().numpy()
            digest.update(("stage=%d;dtype=%s;shape=%s;" % (index, array.dtype,
                                                            array.shape)).encode())
            digest.update(np.ascontiguousarray(array).tobytes())
        features = self.features.detach().cpu().numpy()
        digest.update(("features;dtype=%s;shape=%s;" % (features.dtype, features.shape)).encode())
        digest.update(np.ascontiguousarray(features).tobytes())
        self._fingerprint = digest.hexdigest()
        return self._fingerprint


def _check_points(points, features):
    if not isinstance(points, torch.Tensor) or points.dim() != 2:
        raise GeometryError("points 必须是二维张量 (N, D)")
    if points.shape[0] == 0:
        raise GeometryError("points 为空")
    if not isinstance(features, torch.Tensor) or features.dim() != 2:
        raise GeometryError("features 必须是二维张量 (N, C)")
    if features.shape[0] != points.shape[0]:
        raise GeometryError("features 行数 %d 与 points 点数 %d 不一致"
                            % (features.shape[0], points.shape[0]))
    if points.dim() < 2 or points.shape[1] < 3:
        raise GeometryError("points 至少要有 3 列坐标")


def _check_sampling(voxel_sizes, sampling_method):
    """校验采样配置并返回阶段数（阶段数 = len(voxel_sizes)）。"""
    num_stages = len(voxel_sizes)
    if num_stages < 2:
        raise GeometryError("阶段数 %d < 2：多尺度几何至少需要 2 个阶段" % num_stages)
    if sampling_method not in ("voxel", "fps"):
        raise GeometryError("未知的采样方式: %r（只支持 voxel/fps）" % (sampling_method,))
    return num_stages


def _check_neighbors(num_neighbors, num_stages):
    if len(num_neighbors) != num_stages:
        raise GeometryError("num_neighbors 长度 %d 与阶段数 %d 不一致"
                            % (len(num_neighbors), num_stages))


def _stage_voxel_size(voxel_sizes, stage, num_stages, sampling_method):
    """复刻 pareconv precompute_subsample 的体素选择规则；最后一层可能返回 None（走 fps）。"""
    if stage < len(voxel_sizes) - 1:
        return voxel_sizes[stage]
    if stage == num_stages - 1 and sampling_method == "fps":
        return None
    return voxel_sizes[-1]


def build_stage_points(points, features, voxel_sizes, sampling_method,
                       centroid=None, device="cuda"):
    """单侧多尺度点（对应 precompute_subsample 的单侧等价实现）；阶段数 = len(voxel_sizes)。

    下采样是 CPU 实现（pareconv.ext.grid_subsampling），因此输入必须是 CPU 张量；
    构建完成后点/特征搬到 `device`（默认 CUDA，与旧联合入口一致）。
    """
    _check_points(points, features)
    num_stages = _check_sampling(voxel_sizes, sampling_method)
    if points.is_cuda:
        raise GeometryError("build_stage_points 需要 CPU 点云（grid_subsampling 是 CPU 实现）")

    points_list = [points]
    cur_points = points
    for stage in range(1, num_stages):
        lengths = torch.tensor([cur_points.shape[0]], dtype=torch.long)
        voxel_size = _stage_voxel_size(voxel_sizes, stage, num_stages, sampling_method)
        if voxel_size is None:
            cur_points, _ = farthest_point_sampling_gpu(cur_points, lengths, FPS_TARGET_RATIO)
        else:
            cur_points, _ = grid_subsample(cur_points, lengths, voxel_size)
        points_list.append(cur_points)

    moved = [item.to(device) for item in points_list]
    lengths = [torch.tensor([item.shape[0]], dtype=torch.long, device=device)
               for item in moved]
    return CloudGeometry(points=moved, lengths=lengths, features=features.to(device),
                         voxel_sizes=tuple(float(value) for value in voxel_sizes),
                         sampling_method=sampling_method,
                         centroid=None if centroid is None else np.asarray(centroid),
                         upsampling=[None] * num_stages)


def build_neighbors(geometry, num_neighbors):
    """就地补齐单侧邻居/上下采样索引（pointops k-NN 只有 CUDA 实现）。

    邻居指向本阶段点集；subsampling 行是下一阶段点、值指向本阶段点；
    upsampling 行是本阶段点、值指向下一阶段点（按 stage 对齐，stage 0 为 None）。
    """
    num_stages = geometry.num_stages
    if _check_sampling(geometry.voxel_sizes, geometry.sampling_method) != num_stages:
        raise GeometryError("voxel_sizes 长度与几何阶段数 %d 不一致" % num_stages)
    _check_neighbors(num_neighbors, num_stages)
    if not geometry.points[0].is_cuda:
        raise GeometryError("build_neighbors 需要 CUDA 点云（pointops.knnquery_heap 只有 CUDA 实现）")

    neighbors: List[torch.Tensor] = []
    subsampling: List[torch.Tensor] = []
    upsampling: List[Optional[torch.Tensor]] = [None] * num_stages
    for stage in range(num_stages):
        cur = geometry.points[stage][:, :3].contiguous().unsqueeze(0)
        neighbors.append(pointops.knnquery_heap(num_neighbors[stage], cur, cur).squeeze(0))
        if stage < num_stages - 1:
            sub = geometry.points[stage + 1][:, :3].contiguous().unsqueeze(0)
            subsampling.append(pointops.knnquery_heap(num_neighbors[stage], cur, sub).squeeze(0))
            if stage > 0:
                upsampling[stage] = pointops.knnquery_heap(1, sub, cur).squeeze(0)
    geometry.neighbors = neighbors
    geometry.subsampling = subsampling
    geometry.upsampling = upsampling
    return geometry


def build_geometry(points, features, voxel_sizes, sampling_method, num_neighbors,
                   centroid=None, device="cuda"):
    """单侧几何（多尺度点 + 邻居）：等价于 build_stage_points + build_neighbors。"""
    geometry = build_stage_points(points, features, voxel_sizes, sampling_method,
                                  centroid=centroid, device=device)
    return build_neighbors(geometry, num_neighbors)


def node_partition(geometry, num_points_in_patch, fine_stage=1, coarse_stage=-1):
    """单侧节点分区（细节点 → 最近粗节点），与模型 `point_to_node_partition` 同源。

    注意：分区依赖 `num_points_in_patch`，属于**编码**层（T06 的 `EncodedCloud` 保存它）；
    几何缓存（T05）的 key 不含该项，因此生产路径不把分区放进几何、而是编码时现算。
    """
    if geometry.num_stages < 2:
        raise GeometryError("节点分区至少需要 2 个阶段")
    fine = geometry.points[fine_stage][:, :3].contiguous()
    coarse = geometry.points[coarse_stage][:, :3].contiguous()
    _, masks, knn_indices, knn_masks = point_to_node_partition(fine, coarse, num_points_in_patch)
    return NodePartition(masks=masks, knn_indices=knn_indices, knn_masks=knn_masks)


def attach_node_partition(geometry, num_points_in_patch, fine_stage=1, coarse_stage=-1):
    """就地补齐节点分区（T04 的几何记录能力；编码路径按需现算，见 `node_partition`）。"""
    geometry.node_partition = node_partition(geometry, num_points_in_patch,
                                             fine_stage=fine_stage, coarse_stage=coarse_stage)
    return geometry


def backbone_input(geometry, scale, device=None):
    """单侧几何 → backbone 需要的输入字典（局部索引，无需跨侧偏移）。

    `upsampling` 在 pareconv 里是紧凑列表（元素 j 对应 stage j+1），这里按该约定还原。
    """
    device = geometry.device if device is None else device
    return {
        "points": geometry.points,
        "neighbors": geometry.neighbors,
        "subsampling": geometry.subsampling,
        "upsampling": [item for item in geometry.upsampling if item is not None],
        "scale": torch.as_tensor(scale, device=device),
    }


def offset_indices(indices, offset, source_count):
    """局部索引 → 联合索引；尾部哨兵槽位保持 0，不参与偏移。

    pointops 在邻居不足时用 0 填充尾部槽位，槽位数 = k - source_count（source_count < k 时）。
    """
    if offset == 0:
        return indices
    if source_count >= indices.shape[-1]:
        return indices + offset
    slot = torch.arange(indices.shape[-1], device=indices.device)
    valid = (slot < source_count).unsqueeze(0).expand_as(indices)
    return torch.where(valid, indices + offset, torch.zeros_like(indices))


def join_geometries(ref, src, scale, transform):
    """把两段单侧几何拼成旧联合入口的 data_dict（跨侧索引加偏移）。"""
    if ref.num_stages != src.num_stages:
        raise GeometryError("两侧阶段数不同：%d vs %d" % (ref.num_stages, src.num_stages))
    if ref.device != src.device:
        raise GeometryError("两侧几何设备不同：%s vs %s" % (ref.device, src.device))
    for name in ("neighbors", "subsampling"):
        if len(getattr(ref, name)) != len(getattr(src, name)):
            raise GeometryError("%s 长度不同：%d vs %d"
                                % (name, len(getattr(ref, name)), len(getattr(src, name))))

    num_stages = ref.num_stages
    device = ref.device
    points, lengths = [], []
    neighbors, subsampling, upsampling = [], [], []
    for stage in range(num_stages):
        ref_points = ref.points[stage]
        src_points = src.points[stage]
        points.append(torch.cat([ref_points, src_points], dim=0))
        lengths.append(torch.tensor([ref_points.shape[0], src_points.shape[0]],
                                    dtype=torch.long, device=device))
        neighbors.append(torch.cat([
            ref.neighbors[stage],
            offset_indices(src.neighbors[stage], ref_points.shape[0], src_points.shape[0])],
            dim=0))
        if stage < num_stages - 1:
            ref_next = ref.points[stage + 1]
            src_next = src.points[stage + 1]
            subsampling.append(torch.cat([
                ref.subsampling[stage],
                offset_indices(src.subsampling[stage], ref_points.shape[0],
                               src_points.shape[0])], dim=0))
            if stage > 0:
                upsampling.append(torch.cat([
                    ref.upsampling[stage],
                    offset_indices(src.upsampling[stage], ref_next.shape[0],
                                   src_next.shape[0])], dim=0))

    return {
        "points": points,
        "lengths": lengths,
        "features": torch.cat([ref.features, src.features], dim=0),
        "neighbors": neighbors,
        "subsampling": subsampling,
        "upsampling": upsampling,
        "transform": transform.to(device) if isinstance(transform, torch.Tensor) else transform,
        "scale": scale,
        "batch_size": 1,
    }


# ======================================================================
# 几何缓存（T05）：有界 CPU 存储 + 命中搬回设备
# ======================================================================

def geometry_tensors(geometry):
    """几何里全部张量（用于字节计费与整体搬运）。"""
    tensors = list(geometry.points) + list(geometry.lengths) + [geometry.features]
    tensors += list(geometry.neighbors) + list(geometry.subsampling)
    tensors += [item for item in geometry.upsampling if item is not None]
    if geometry.node_partition is not None:
        tensors += [geometry.node_partition.masks, geometry.node_partition.knn_indices,
                    geometry.node_partition.knn_masks]
    return tensors


def geometry_bytes(geometry):
    """条目字节数 = 全部张量字节之和（缓存还会另加固定的条目开销）。"""
    total = 0
    for tensor in geometry_tensors(geometry):
        total += tensor.numel() * tensor.element_size()
    return total


def geometry_to(geometry, device, copy=False):
    """整体搬到设备，返回**新对象**（缓存条目本身不被修改）。

    `copy=True` 时进一步保证返回的张量与输入**存储无关**（`Tensor.to()` 在同设备时是空操作，
    不会复制；缓存条目必须独立存储，避免 view 把调用方的大数组一直留在内存里）。
    """
    def move(tensor):
        tensor = tensor.to(device)
        return tensor.clone() if copy else tensor

    partition = None
    if geometry.node_partition is not None:
        partition = NodePartition(
            masks=move(geometry.node_partition.masks),
            knn_indices=move(geometry.node_partition.knn_indices),
            knn_masks=move(geometry.node_partition.knn_masks))
    return replace(
        geometry,
        points=[move(item) for item in geometry.points],
        lengths=[move(item) for item in geometry.lengths],
        features=move(geometry.features),
        neighbors=[move(item) for item in geometry.neighbors],
        subsampling=[move(item) for item in geometry.subsampling],
        upsampling=[None if item is None else move(item) for item in geometry.upsampling],
        node_partition=partition)


def geometry_cache_key(points, features, voxel_sizes, sampling_method, num_neighbors,
                       centroid=None):
    """构建前即可计算的几何 key。

    覆盖：结构版本、实际点集**内容与顺序**（dtype/shape 一并计入）、特征、生效体素尺寸、
    最后一层采样方式、每阶段邻居数、质心（仅元信息，一并入 key 避免跨质心误命中）。
    """
    digest = hashlib.sha256()
    digest.update(("geometry_version=%d;sampling=%s;voxels=%s;neighbors=%s;"
                   % (GEOMETRY_VERSION, sampling_method,
                      ",".join("%.6f" % float(value) for value in voxel_sizes),
                      ",".join(str(int(value)) for value in num_neighbors))).encode())
    for name, tensor in (("points", points), ("features", features)):
        array = tensor.detach().cpu().numpy()
        digest.update(("%s;dtype=%s;shape=%s;" % (name, array.dtype, array.shape)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    if centroid is not None:
        digest.update(np.ascontiguousarray(np.asarray(centroid, dtype=np.float32)).tobytes())
    return digest.hexdigest()


@dataclass
class AcquireResult:
    """一次几何获取的结果。"""

    geometry: CloudGeometry
    hit: bool
    cacheable: bool


class GeometryCache:
    """源/目标单侧几何的有界 **CPU** 缓存（T05）。

    - 存储的是 CPU 张量（不占显存），按字节计费，容量为 0 时关闭；
    - 命中时把条目搬回 `device`（默认 CUDA）后返回，跳过多尺度下采样与 k-NN；
    - 只缓存确定性采样；`fps` 含 RNG，按任务卡偏差处理**保留采样、不缓存**；
    - 条目只读：调用方不得原地修改 `get()` 返回的几何张量（`join_geometries` 只读拼接）。
    """

    def __init__(self, capacity_bytes, device="cuda", name="geometry"):
        self.device = device
        self._cache = ByteLruCache(capacity_bytes, geometry_bytes, name=name)

    @property
    def enabled(self):
        return self._cache.enabled

    @property
    def capacity_bytes(self):
        return self._cache.capacity_bytes

    def get(self, key):
        return self._cache.get(key)

    def put(self, key, geometry):
        """存**独立 CPU 副本**（同设备 `.to()` 不复制，因此显式 copy）；返回是否真的存下。"""
        return self._cache.put(key, geometry_to(geometry, "cpu", copy=True))

    def snapshot(self):
        return self._cache.snapshot()


def acquire_geometry(cache, points, features, voxel_sizes, sampling_method, num_neighbors,
                     centroid=None, device="cuda"):
    """取得单侧几何（缓存开启与关闭走**同一份构建代码**）。

    cache=None 或采样方式不确定（`fps`）时不查也不存；命中时只做搬设备。
    """
    cacheable = sampling_method in DETERMINISTIC_SAMPLING
    key = None
    if cache is not None and cacheable:
        key = geometry_cache_key(points, features, voxel_sizes, sampling_method,
                                 num_neighbors, centroid)
        cached = cache.get(key)
        if cached is not None:
            return AcquireResult(geometry=geometry_to(cached, device),
                                 hit=True, cacheable=cacheable)

    geometry = build_stage_points(points, features, voxel_sizes, sampling_method,
                                  centroid=centroid, device=device)
    build_neighbors(geometry, num_neighbors)
    if key is not None:
        cache.put(key, geometry)

    return AcquireResult(geometry=geometry, hit=False, cacheable=cacheable)


# ======================================================================
# 源编码缓存（T07）：精确 scale，GPU 预算 256 MiB
# ======================================================================

def object_tensor_bytes(value):
    """通用字节计费：dataclass 实例（或张量/列表）里全部张量的字节数。

    编码结果（EncodedCloud）与几何一样是 dataclass，字段里有张量/可选张量；
    这里递归求和，避免遗漏新字段导致账面低估。
    """
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, (list, tuple)):
        return sum(object_tensor_bytes(item) for item in value)
    if hasattr(value, "__dataclass_fields__"):
        return sum(object_tensor_bytes(getattr(value, name))
                   for name in value.__dataclass_fields__)
    return 0


def encoding_cache_key(geometry_key, scale, model_fingerprint, dtype,
                       version=ENCODING_CACHE_VERSION):
    """编码缓存 key：几何指纹 + 精确 scale bits + 权重/配置指纹 + dtype + 编码版本。

    - 几何指纹覆盖"实际点集与顺序 + 特征 + 生效采样配置"（旋转/平移/换掩码都会变）；
    - scale 用**精确位模式**（`struct.pack(">f")`），不做分桶，避免近似尺度错命中；
    - 权重/配置指纹来自 `model_fingerprint()`，权重变化即失效。
    """
    digest = hashlib.sha256()
    digest.update(("encoding_version=%d;" % version).encode())
    digest.update(("geometry=%s;" % geometry_key).encode())
    digest.update(("scale_bits=%s;" % struct.pack(">f", float(scale)).hex()).encode())
    digest.update(("model=%s;" % model_fingerprint).encode())
    digest.update(("dtype=%s;" % dtype).encode())
    return digest.hexdigest()


class EncodingCache:
    """源编码的**有界 GPU** 缓存（T07）。

    - 只缓存**源**侧编码；目标编码只保留当前请求（用完即释放），符合任务卡"优先当前源"；
    - 预算以字节计（默认 256 MiB），0 = 关闭；单条超预算时正常使用但不缓存；
    - 不缓存 attention 大矩阵、逐层激活或 hypotheses（这些本来就不在 `EncodedCloud` 里）；
    - key 用精确 scale 位模式，无分桶、无磁盘持久化。
    """

    def __init__(self, capacity_bytes, model_fingerprint, name="encoding"):
        self.model_fingerprint = model_fingerprint
        self._cache = ByteLruCache(capacity_bytes, object_tensor_bytes, name=name)

    @property
    def enabled(self):
        return self._cache.enabled

    @property
    def capacity_bytes(self):
        return self._cache.capacity_bytes

    def key_for(self, geometry, scale, dtype="torch.float32"):
        return encoding_cache_key(geometry.fingerprint(), scale, self.model_fingerprint,
                                  dtype)

    def get(self, geometry, scale, dtype="torch.float32"):
        """命中返回编码（GPU 张量，只读）；未命中返回 None。"""
        return self._cache.get(self.key_for(geometry, scale, dtype))

    def put(self, geometry, scale, encoded, dtype="torch.float32"):
        """存入编码（同设备、只读约定）；返回是否真的存下。"""
        return self._cache.put(self.key_for(geometry, scale, dtype), encoded)

    def snapshot(self):
        return self._cache.snapshot()
