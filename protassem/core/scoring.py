"""CC_mask calculation — direct computation, no subprocess.

老卡收口 P4：把「密度侧输入」与「结构坐标」抽成可复用的上下文，并加**有界**缓存。

- `DensityMapContext` 保存**原 dtype** 的密度数组（只做一次阈值化：`>contour ? x : -1.0`）
  以及 origin / voxel_size / shape / contour / 指纹；**不改变评分路径的 dtype 与阈值顺序**
  （不在这里做类型转换优化）。
- `score_coords(context, coords, elements, resolution)` 是坐标数组入口；
  `calculate_cc_mask(...)` 保留原签名，退化为薄包装（读上下文 + 读结构 + 评分）。
- 缓存：`score_cache_mb` 默认 **128 MiB / 进程**，密度上下文与结构坐标**共享**该预算；
  0 = 关闭。缓存是**进程本地**的（worker 进程各自持有），多 worker 下总量按进程数放大。
- 失效：`(abspath, size, mtime_ns, contour, SCORING_VERSION)`；会被同名覆盖的动态密度必须
  显式给出 `density_version` 或调用 `invalidate_density()` —— mtime 不是版本契约。
- 只读约定：缓存里的数组只读，需要修改时用 `.copy()`。
- 配置变化（预算）会**清空并重建**缓存；`apply_score_cache()` 也写环境变量，
  这样 fork 出来的 worker 与 spawn 出来的 worker 都能拿到同一个预算。
"""

import os

import numpy as np
import mrcfile
from scipy.ndimage import distance_transform_edt, gaussian_filter
from numba import njit
from protassem.core.io import read_structure
from protassem.core.constants import atomic_number_dict, VDW_RADII
from protassem.core.numba_kernels import add_gaussian_to_grid, add_sphere_mask
from protassem.runtime.byte_cache import ByteLruCache, object_bytes
from protassem.runtime.config import DEFAULT_SCORE_CACHE_MB, SCORE_CACHE_ENV

SCORING_VERSION = 1
STRUCTURE_VERSION = 1


def read_mrc_full(filename):
    """Read MRC with axis reordering (handles mapc/mapr/maps)."""
    with mrcfile.open(filename, permissive=True) as mrc:
        data = mrc.data.copy()
        vx = float(mrc.voxel_size.x) or 1.0
        vy = float(mrc.voxel_size.y) or 1.0
        vz = float(mrc.voxel_size.z) or 1.0
        voxel_size = np.array([vx, vy, vz], dtype=np.float32)

        origin = np.array([
            float(mrc.header.origin.x),
            float(mrc.header.origin.y),
            float(mrc.header.origin.z)
        ], dtype=np.float32)

        nstart = np.array([
            int(mrc.header.nxstart),
            int(mrc.header.nystart),
            int(mrc.header.nzstart)
        ], dtype=np.float32)

        mapcrs = np.array([
            int(mrc.header.mapc),
            int(mrc.header.mapr),
            int(mrc.header.maps)
        ], dtype=int)

        sort = np.array([0, 1, 2], dtype=int)
        for i in range(3):
            sort[mapcrs[i] - 1] = i
        nstart = np.array([nstart[i] for i in sort], dtype=np.float32)
        data = np.transpose(data, axes=2 - sort[::-1])

        if np.all(origin == 0):
            origin = origin + nstart * voxel_size

    return data, voxel_size, origin, data.shape


@njit(fastmath=True)
def _pearson(x, y):
    n = len(x)
    if n == 0:
        return 0.0
    sx, sy = 0.0, 0.0
    for i in range(n):
        sx += x[i]
        sy += y[i]
    mx, my = sx / n, sy / n
    num, vx, vy = 0.0, 0.0, 0.0
    for i in range(n):
        dx = x[i] - mx
        dy = y[i] - my
        num += dx * dy
        vx += dx * dx
        vy += dy * dy
    denom = np.sqrt(vx * vy)
    if denom == 0.0:
        return 0.0
    return num / denom


def _make_sim_map(origin, voxel_size, box_size, coords, elements, resolution):
    sigma_factor = 1.0 / (np.pi * np.sqrt(2.0))
    sigma_real = resolution * sigma_factor
    sigma_vox = np.array([sigma_real / voxel_size[i] for i in range(3)])

    grid = np.zeros(box_size, dtype=np.float32)

    for coord, elem in zip(coords, elements):
        w = atomic_number_dict.get(elem, 1.0)
        atom_coord = np.array(coord, dtype=np.float64)
        add_gaussian_to_grid(grid, atom_coord, w, origin, voxel_size,
                             sigma_vox, 5.0)

    norm = np.power(2 * np.pi, -1.5) * np.power(sigma_real, -3)
    grid *= norm
    return grid


def _make_phenix_mask(origin, voxel_size, box_size, coords, elements,
                      solvent_radius=1.1, mask_radius=2.0, resolution=None):
    mask = np.zeros(box_size, dtype=np.bool_)
    nz, ny, nx = mask.shape

    for coord, elem in zip(coords, elements):
        r_vdw = VDW_RADII.get(elem, 1.70)
        radius = r_vdw + solvent_radius
        radius_sq = radius * radius
        ic = (coord[0] - origin[0]) / voxel_size[0]
        jc = (coord[1] - origin[1]) / voxel_size[1]
        kc = (coord[2] - origin[2]) / voxel_size[2]
        rv = np.array([radius / voxel_size[i] for i in range(3)])
        i0 = max(0, int(np.floor(ic - rv[0])))
        i1 = min(nx, int(np.ceil(ic + rv[0])) + 1)
        j0 = max(0, int(np.floor(jc - rv[1])))
        j1 = min(ny, int(np.ceil(jc + rv[1])) + 1)
        k0 = max(0, int(np.floor(kc - rv[2])))
        k1 = min(nz, int(np.ceil(kc + rv[2])) + 1)
        if i0 >= i1 or j0 >= j1 or k0 >= k1:
            continue
        add_sphere_mask(mask, ic, jc, kc,
                        voxel_size[0], voxel_size[1], voxel_size[2],
                        radius_sq, i0, i1, j0, j1, k0, k1)

    avg_vs = np.mean(voxel_size)
    dist = distance_transform_edt(~mask) * avg_vs
    expanded = (dist <= mask_radius) | mask

    sigma = (resolution / 4.0) if resolution else 1.0
    sigma_vox = np.array([sigma / voxel_size[i] for i in range(3)])
    soft = gaussian_filter(expanded.astype(np.float32), sigma=sigma_vox,
                           mode="constant", cval=0.0)
    if soft.max() > 0:
        soft /= soft.max()
    return soft > 0.5


class DensityMapContext:
    """一次评分所需的密度侧输入（阈值化只做一次；**不改变原 dtype**）。"""

    __slots__ = ("path", "data", "voxel_size", "origin", "shape", "contour",
                 "fingerprint")

    def __init__(self, path, data, voxel_size, origin, contour, fingerprint):
        self.path = path
        self.data = data
        self.voxel_size = voxel_size
        self.origin = origin
        self.shape = data.shape
        self.contour = contour
        self.fingerprint = fingerprint


class _ScoreCache:
    """进程本地缓存：密度上下文 + 结构坐标共享一个字节预算。"""

    def __init__(self, capacity_mb):
        self.capacity_mb = int(capacity_mb)
        capacity_bytes = max(0, self.capacity_mb) * 1024 * 1024
        self.density = ByteLruCache(capacity_bytes, self._density_bytes, "density")
        self.structure = ByteLruCache(capacity_bytes, self._structure_bytes,
                                      "structure")

    @staticmethod
    def _density_bytes(context):
        return object_bytes(context.data) + object_bytes(context.voxel_size) \
            + object_bytes(context.origin) + 512

    @staticmethod
    def _structure_bytes(value):
        coords, elements = value
        return object_bytes(coords) + object_bytes(elements) + 256

    @property
    def enabled(self):
        return self.density.enabled or self.structure.enabled

    def clear(self):
        self.density.clear()
        self.structure.clear()

    def snapshot(self):
        return {"score_cache_mb": self.capacity_mb,
                "density": self.density.snapshot(),
                "structure": self.structure.snapshot()}


def _env_capacity_mb():
    raw = os.environ.get(SCORE_CACHE_ENV)
    if raw in (None, ""):
        return DEFAULT_SCORE_CACHE_MB
    try:
        return max(0, int(float(raw)))
    except ValueError:
        return DEFAULT_SCORE_CACHE_MB


_CACHE = _ScoreCache(_env_capacity_mb())


def configure_score_cache(capacity_mb):
    """设置本进程的评分缓存预算（MiB，0 = 关闭）；预算变化时重建，否则清空。"""
    global _CACHE
    capacity_mb = max(0, int(capacity_mb))
    if capacity_mb != _CACHE.capacity_mb:
        _CACHE = _ScoreCache(capacity_mb)
    else:
        _CACHE.clear()
    return _CACHE.snapshot()


def apply_score_cache(capacity_mb):
    """父进程应用预算：写环境变量（worker 继承）+ 直接配置本进程。"""
    os.environ[SCORE_CACHE_ENV] = str(int(max(0, int(capacity_mb))))
    return configure_score_cache(capacity_mb)


def score_cache_snapshot():
    return _CACHE.snapshot()


def invalidate_density():
    """显式失效（返回清掉的条目数）。

    同名覆盖的动态密度有两种正确做法：给 `density_version`（精确、按 key 失效），
    或在这里整体失效。这里不做"按路径删除"—— 半吊子的部分失效比整体失效更危险。
    """
    entries = _CACHE.density.stats.entries
    _CACHE.density.clear()
    return entries


def density_fingerprint(density_mrc, contour, density_version=None):
    """内容指纹：版本 + 路径 + size + mtime_ns + contour。

    `density_version` 由调用方给出（例如掩码/写入轮次）：**同名覆盖的动态密度必须给**，
    否则只能依赖 mtime —— 那不是版本契约。
    """
    stat = os.stat(str(density_mrc))
    return "|".join([str(SCORING_VERSION), os.path.abspath(str(density_mrc)),
                     str(stat.st_size), str(stat.st_mtime_ns),
                     repr(float(contour)),
                     "-" if density_version is None else str(density_version)])


def load_density_context(density_mrc, contour, density_version=None):
    """读取 + 阈值化（无缓存路径；缓存走 `density_context`）。"""
    data, voxel_size, origin, _dims = read_mrc_full(density_mrc)
    data = np.where(data > contour, data, -1.0)
    return DensityMapContext(
        path=os.path.abspath(str(density_mrc)), data=data, voxel_size=voxel_size,
        origin=origin, contour=float(contour),
        fingerprint=density_fingerprint(density_mrc, contour, density_version))


def density_context(density_mrc, contour, density_version=None):
    """取（可能命中缓存的）密度上下文；返回的数组视为**只读**。"""
    key = density_fingerprint(density_mrc, contour, density_version)
    cached = _CACHE.density.get(key)
    if cached is not None:
        return cached
    context = load_density_context(density_mrc, contour, density_version)
    _CACHE.density.put(key, context)
    return context


def structure_coords(structure_file):
    """结构坐标（缓存 key 含 size + mtime_ns + 版本）。"""
    path = os.path.abspath(str(structure_file))
    try:
        stat = os.stat(path)
        key = "|".join([str(STRUCTURE_VERSION), path, str(stat.st_size),
                        str(stat.st_mtime_ns)])
    except OSError:
        key = None
    cached = _CACHE.structure.get(key)
    if cached is not None:
        return cached
    coords, elements = read_structure(str(structure_file))
    value = (list(coords), list(elements))
    _CACHE.structure.put(key, value)
    return value


def score_coords(context, coords, elements, resolution):
    """按坐标数组评分（原 `calculate_cc_mask` 的计算部分，逐行等价）。"""
    exp_map = context.data
    dims = context.shape
    origin, voxel_size = context.origin, context.voxel_size
    elem_list = list(elements)

    sim_map = _make_sim_map(origin, voxel_size, dims, coords, elem_list, resolution)
    mask = _make_phenix_mask(origin, voxel_size, dims, coords, elem_list,
                             resolution=resolution)

    sel = mask
    if np.count_nonzero(sel) == 0:
        return 0.0
    exp_vals = exp_map[sel].astype(np.float64)
    sim_vals = sim_map[sel].astype(np.float64)
    return float(_pearson(exp_vals, sim_vals))


def calculate_cc_mask(density_mrc, structure_file, resolution, contour,
                      density_version=None):
    """Compute CC_mask between experimental map and structure（薄包装）。"""
    context = density_context(density_mrc, contour, density_version)
    coords, elements = structure_coords(structure_file)
    return score_coords(context, coords, elements, resolution)
