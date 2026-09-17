"""密度图原点规范化：把 MRC header 化成不依赖读法约定的一种形式。

背景
----
项目里有两条原点口径，对 ``nstart`` 非零的图会给出**不同**的原点：

* 采样器 ``Sample``（预编译二进制，改不了）：``shifting = nstart*voxel + origin``，
  两者都计入。参考实现 ``sampling/extract_points/Sample_based_VoxEM.py``。
* 评分 / 掩膜 ``core.scoring.read_mrc_full``：只当 ``origin`` 全为 0 时才补
  ``nstart * voxel_size``；``origin`` 非零时 ``nstart`` 被整个丢弃。

对 ``nstart`` 全为 0 的规范图（EMDB 下载的图，以及本项目 ``pdb2vol`` /
``io.write_mrc`` 产出的图）两者同值，看不出差别。但对 ``nstart`` 与 ``origin``
都非零的图（典型来源：从父图裁出来的子体积），点云与密度图会整体错开
``nstart * voxel_size``，掩膜与 CC 落在错误区域，拟合必然失败。

做法
----
把 header 化成"约定无关"的形式：

    origin  +=  nstart * voxel_size
    nstart   =  0

此后采样端算 ``origin + 0``、评分端算 ``origin``（或零原点分支的 ``origin + 0``），
两条路径**必然同值**。

注意这**不改变采样器的输出**：Sample 自己做那个加法，得到的仍是
``origin + nstart*voxel``，点云逐点不变；变的只是评分/掩膜端从此与点云同框。
即本模块是"让评分端对齐采样端"，不是改变采样结果。

只改 header 的这 4 个字段，数组数据按字节搬运 —— 不读数组、不转置、不重采样，
也**不碰** ``mapc/mapr/maps``（``io.write_mrc`` 会把轴序重置成 1/2/3，非标准轴序
的图会被改变解释，所以这里不用它）。

残余风险【待验证】
----------------
``origin`` 与 ``nstart`` 同时非零时，"``origin`` 字段是否已经含 ``nstart`` 偏移"
**在文件内部无法判定**：若 ``origin`` 是"首个体素的绝对位置"（MRC2014 读法），
那么规范化后再加一次 ``nstart*voxel`` 会偏大这么多；若 ``origin`` 是"父图原点"，
规范化后即为真值。两种情况下拟合本身都能正常工作（采样端与评分端同框，输出
只是整体刚性平移），差别仅在输出坐标的绝对位置。要判定这一点，需要一张
"结构确实位于密度内部"的参考图。
"""

import collections
import os
import shutil

import mrcfile
import numpy as np

NormalizedMap = collections.namedtuple(
    "NormalizedMap", "path origin previous_origin nstart voxel_size")


def canonical_origin(origin, nstart, voxel_size):
    """采样器 ``Sample`` 使用的原点口径：``nstart`` 与 ``origin`` 都计入。"""
    return (np.asarray(origin, dtype=np.float64)
            + np.asarray(nstart, dtype=np.float64)
            * np.asarray(voxel_size, dtype=np.float64))


def normalize_density_map(source, dest):
    """写出 header 与读法约定无关的 MRC 副本。

    Args:
        source: 输入密度图路径。
        dest: 输出路径；**仅当需要修正时**才写出，输入已是规范形式则不建文件。

    Returns:
        NormalizedMap：
            path            下游应使用的路径（无需修正时即 ``source``）
            origin          修正后的原点
            previous_origin 修正前 ``origin`` 字段的读数
            nstart          读到的 ``nstart``
            voxel_size      读到的体素尺寸

    Raises:
        FileNotFoundError: ``source`` 不存在。
    """
    if not os.path.isfile(source):
        raise FileNotFoundError("density map not found: %s" % source)

    with mrcfile.open(source, permissive=True) as mrc:
        origin = np.array([mrc.header.origin.x, mrc.header.origin.y,
                           mrc.header.origin.z], dtype=np.float64)
        nstart = np.array([mrc.header.nxstart, mrc.header.nystart,
                           mrc.header.nzstart], dtype=np.float64)
        voxel_size = np.array([mrc.voxel_size.x, mrc.voxel_size.y,
                               mrc.voxel_size.z], dtype=np.float64)

    if not nstart.any():
        return NormalizedMap(source, origin, origin, nstart, voxel_size)

    fixed = canonical_origin(origin, nstart, voxel_size)
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    shutil.copyfile(source, dest)
    with mrcfile.open(dest, mode="r+", permissive=True) as mrc:
        mrc.header.origin.x = float(fixed[0])
        mrc.header.origin.y = float(fixed[1])
        mrc.header.origin.z = float(fixed[2])
        mrc.header.nxstart = 0
        mrc.header.nystart = 0
        mrc.header.nzstart = 0
        mrc.flush()
    return NormalizedMap(dest, fixed, origin, nstart, voxel_size)
