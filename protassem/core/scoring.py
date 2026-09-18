"""CC_mask calculation — direct computation, no subprocess.

老卡收口 P4：把「密度侧输入」与「结构坐标」抽成可复用的上下文，并加**有界**缓存。

- `DensityMapContext` 保存**原 dtype** 的密度数组（只做一次阈值化：`>contour ? x : -1.0`）
  以及 origin / voxel_size / shape / contour / 指纹；**不改变评分路径的 dtype 与阈值顺序**
  （不在这里做类型转换优化）。
- `score_coords(context, coords, elements, resolution)` 是坐标数组入口；
  `calculate_cc_mask(...)` 保留原签名，退化为薄包装（读上下文 + 读结构 + 评分）。
- 缓存：`score_cache_mb` 默认 **1024 MiB / 进程**（装得下一张 400^3 float32 图），
  密度上下文与结构坐标**共享**该预算；0 = 关闭。缓存是**进程本地**的
  （worker 进程各自持有），多 worker 下总量按进程数放大。
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
    """原子球体掩码：球体 splat -> EDT 膨胀 -> 高斯软化 -> 0.5 阈值。

    只在**原子包围盒 ⊕ padding** 的子体积上做 EDT 与高斯卷积，再贴回全图：
    这两个全图操作占本函数耗时的 99%，而原子包围盒通常只占全图很小一部分
    （实测 400³ 图上某拟合位姿为 0.5% 体积）。padding 取
    ``mask_radius + 5σ + 2 voxel``（σ = resolution/4）：EDT 只需覆盖 mask_radius
    的膨胀范围，高斯核半宽为 4σ，5σ + 2 留余量。

    子体积之外 ``expanded`` 恒为 0（膨胀范围已被 padding 覆盖），与
    ``mode="constant", cval=0.0`` 的假设一致，故与全图实现逐点相同 ——
    等价性由 tests/test_phenix_mask_crop.py 用独立的全图参考实现锁定。

    没有任何原子落在网格内时返回全 False，``score_coords`` 据此返回 CC 0.0。
    （旧实现在该情形下会把 ``distance_transform_edt`` 对"无背景输入"的未定义
    输出当成地图用，在网格中部选出一片无意义区域。）
    """
    nz, ny, nx = box_size
    vs = [float(voxel_size[i]) for i in range(3)]
    grid = [nx, ny, nz]

    # 第一遍：每个原子球体的网格包围盒（公式与夹取规则同旧实现）
    boxes = []
    for coord, elem in zip(coords, elements):
        radius = VDW_RADII.get(elem, 1.70) + solvent_radius
        c = [(coord[i] - origin[i]) / vs[i] for i in range(3)]
        rv = [radius / vs[i] for i in range(3)]
        idx = [(max(0, int(np.floor(c[i] - rv[i]))),
                min(grid[i], int(np.ceil(c[i] + rv[i])) + 1)) for i in range(3)]
        if any(a >= b for a, b in idx):
            continue
        boxes.append((c, radius * radius, idx))

    if not boxes:
        return np.zeros(box_size, dtype=np.bool_)

    sigma = (resolution / 4.0) if resolution else 1.0
    sigma_vox = np.array([sigma / vs[i] for i in range(3)])
    pad = np.ceil(np.array([mask_radius / vs[i] for i in range(3)])
                  + 5.0 * sigma_vox + 2.0).astype(int)
    lo = np.maximum([min(b[2][i][0] for b in boxes) for i in range(3)] - pad, 0)
    hi = np.minimum([max(b[2][i][1] for b in boxes) for i in range(3)] + pad,
                    grid)

    # 第二遍：把球体 splat 进子体积（网格坐标整体平移 lo）
    sub = np.zeros((hi[2] - lo[2], hi[1] - lo[1], hi[0] - lo[0]), dtype=np.bool_)
    for c, radius_sq, idx in boxes:
        (i0, i1), (j0, j1), (k0, k1) = idx
        add_sphere_mask(sub, c[0] - lo[0], c[1] - lo[1], c[2] - lo[2],
                        vs[0], vs[1], vs[2], radius_sq,
                        i0 - lo[0], i1 - lo[0], j0 - lo[1], j1 - lo[1],
                        k0 - lo[2], k1 - lo[2])

    avg_vs = float(np.mean(voxel_size))
    dist = distance_transform_edt(~sub) * avg_vs
    expanded = (dist <= mask_radius) | sub
    soft = gaussian_filter(expanded.astype(np.float32), sigma=sigma_vox,
                           mode="constant", cval=0.0)
    if soft.max() > 0:
        soft /= soft.max()
    out = np.zeros(box_size, dtype=np.bool_)
    out[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]] = soft > 0.5
    return out


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


def _entry_bytes(value):
    """共享 LRU 的计费函数：密度上下文与结构坐标都走这里。"""
    if isinstance(value, DensityMapContext):
        return (object_bytes(value.data) + object_bytes(value.voxel_size)
                + object_bytes(value.origin) + 512)
    coords, elements = value
    return object_bytes(coords) + object_bytes(elements) + 256


class _ScoreCache:
    """进程本地缓存：密度上下文与结构坐标**共享同一个字节预算**。

    一个 LRU、两类键前缀（`d:` / `s:`）—— 两条独立 LRU 各自按上限计费的话，
    实际占用会到 2×预算（第 7 步实测到 113 MB + 44 MB > 128 MiB 才暴露）。
    命中/未命中按类别计数，占用与峰值按共享预算计。
    """

    KIND_PREFIX = {"density": "d:", "structure": "s:"}

    def __init__(self, capacity_mb):
        self.capacity_mb = int(capacity_mb)
        capacity_bytes = max(0, self.capacity_mb) * 1024 * 1024
        self.cache = ByteLruCache(capacity_bytes, _entry_bytes, "score")
        self.counters = {kind: {"hits": 0, "misses": 0, "puts": 0, "rejects": 0}
                         for kind in self.KIND_PREFIX}

    @property
    def enabled(self):
        return self.cache.enabled

    def get(self, kind, key):
        value = self.cache.get(self.KIND_PREFIX[kind] + key)
        counter = self.counters[kind]
        if value is None:
            counter["misses"] += 1
        else:
            counter["hits"] += 1
        return value

    def put(self, kind, key, value):
        stored = self.cache.put(self.KIND_PREFIX[kind] + key, value)
        self.counters[kind]["puts" if stored else "rejects"] += 1
        return stored

    def clear(self):
        """清空条目并重置统计（"配置变化即重设"的口径：统计跟着配置走）。"""
        self.cache.clear()
        for counter in self.counters.values():
            for key in counter:
                counter[key] = 0

    def snapshot(self):
        shared = self.cache.snapshot()
        result = {"score_cache_mb": self.capacity_mb,
                  "capacity_bytes": shared["capacity_bytes"],
                  "entries": shared["entries"], "bytes": shared["bytes"],
                  "peak_bytes": shared["peak_bytes"],
                  "evictions": shared["evictions"]}
        for kind, counter in self.counters.items():
            total = counter["hits"] + counter["misses"]
            result[kind] = dict(counter)
            result[kind]["hit_rate"] = round(float(counter["hits"]) / total, 6) \
                if total else None
            result[kind]["capacity_bytes"] = shared["capacity_bytes"]
        return result


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
    entries = _CACHE.snapshot()["entries"]
    _CACHE.clear()
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
    cached = _CACHE.get("density", key)
    if cached is not None:
        return cached
    context = load_density_context(density_mrc, contour, density_version)
    _CACHE.put("density", key, context)
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
    cached = _CACHE.get("structure", key) if key is not None else None
    if cached is not None:
        return cached
    coords, elements = read_structure(str(structure_file))
    value = (list(coords), list(elements))
    if key is not None:
        _CACHE.put("structure", key, value)
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
