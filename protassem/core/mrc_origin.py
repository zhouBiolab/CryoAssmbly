"""密度图原点规范化：把 MRC header 化成不依赖读法约定的一种形式。

背景
----
项目里有两条原点口径：

* 采样器 ``Sample``（预编译二进制，改不了）：点云锚在 ``origin + nstart*voxel``。
* 评分 / 掩膜 ``core.scoring.read_mrc_full``：锚在 ``origin``，仅当 ``origin``
  全为 0 时才回退 ``nstart * voxel_size``。

按 origin / nstart 是否为零分三种情况：

    ===========  ===========  ====================  ===============  ========
    origin       nstart       Sample 锚点           read_mrc_full    冲突
    ===========  ===========  ====================  ===============  ========
    非零         0            origin                origin           否
    0            非零         nstart*voxel          nstart*voxel     否
    非零         非零         origin+nstart*voxel   origin           **是**
    ===========  ===========  ====================  ===============  ========

所以**只有 origin 与 nstart 都非零时**才错位 ``nstart*voxel``（典型来源：从父图
裁出来的子体积）。此时点云与密度图整体错开，掩膜与 CC 落在错误区域，拟合必然
失败。前两种情况下两条路径本来就同值，看不出差别，也不该改动文件。

Sample 锚点的实测证据
--------------------
``Sample`` 写进点云 TXT 的第 3 行（采样盒原点）满足

    line3 = (origin + nstart*voxel) + n*voxel/2 - box*sample/2

在 6 张图上逐位吻合（3 张按本模块公式构造的合成图 + EMD-8436 / EMD-29607 /
`current_density`）。最干净的判据是**差分检验**：固定 origin、把 nstart 从 0
改成 (5,7,9)，line3 恰好增加 (10,14,18) = nstart*voxel —— box 与居中项被完全
抵消。即 Sample 把 nstart 算了一遍，而评分端没算。

做法
----
把 header 化成"两种口径同值"的形式，**且不移动图的渲染位置**：

    origin := display_origin      # origin 非零则原样保留；origin 为零才取 nstart*voxel
    nstart := 0

此后采样端算 ``display_origin + 0``、评分端算 ``display_origin``，两条路径必然
同值；而外部软件（ChimeraX 等）读 MRC 的规则本来就是"origin 非零则用它，否则
退回 nstart*voxel"，所以它看到的渲染位置**与原图逐位相同**。

注意这里**不能**写成 ``origin += nstart*voxel``：那虽然也让两端同框（都变成
``origin + nstart*voxel``），但代价是把整张图在外部软件里平移了 ``nstart*voxel``
——用户在 ChimeraX 里会看到密度图整体跑掉。

只改 header 的这 4 个字段，数组数据按字节搬运 —— 不读数组、不转置、不重采样，
也**不碰** ``mapc/mapr/maps``（``io.write_mrc`` 会把轴序重置成 1/2/3，非标准轴序
的图会被改变解释，所以这里不用它）。

残余假设【待验证】
----------------
``origin`` 与 ``nstart`` 同时非零时，"``origin`` 是否已是首个体素的位置"在文件
内部无法判定。本模块按 MRC2014 读（origin 就是位置），因此会丢弃 ``nstart``。
若某个文件的 ``origin`` 实际是"父图原点"、真实偏移应由 nstart 提供，则此规则会
丢掉那部分偏移 —— 此时图不会移动（安全），但 Sample 的锚点会变。工具
``tools/normalize_mrc_sample_pdb.py``（仅开发服务器保留）的候选诊断表会把这种不一致暴露出来
（点云落点最佳的候选不等于 ``sample_anchor`` 时明确告警）。
"""

import collections
import os
import shutil

import mrcfile
import numpy as np

NormalizedMap = collections.namedtuple(
    "NormalizedMap", "path origin previous_origin nstart voxel_size sample_anchor")


def sample_anchor(origin, nstart, voxel_size):
    """采样器 ``Sample`` 使用的点云锚点：``nstart`` 与 ``origin`` 都计入（实测）。"""
    return (np.asarray(origin, dtype=np.float64)
            + np.asarray(nstart, dtype=np.float64)
            * np.asarray(voxel_size, dtype=np.float64))


def display_origin(origin, nstart, voxel_size):
    """MRC 标准 / 外部软件 / 评分端使用的原点。

    ``origin`` 非零则用它（这就是首个体素的位置）；``origin`` 全为 0 时才退回
    ``nstart * voxel_size`` —— 与 ``scoring.read_mrc_full`` 的规则一致。
    """
    origin = np.asarray(origin, dtype=np.float64)
    if np.any(np.abs(origin) > 1e-6):
        return origin
    return (np.asarray(nstart, dtype=np.float64)
            * np.asarray(voxel_size, dtype=np.float64))


def normalize_density_map(source, dest):
    """写出 header 与读法约定无关的 MRC 副本。

    Args:
        source: 输入密度图路径。
        dest: 输出路径；**仅当需要修正时**才写出，输入已是规范形式则不建文件。

    Returns:
        NormalizedMap：
            path            下游应使用的路径（无需修正时即 ``source``）
            origin          修正后的原点（即 ``display_origin``）
            previous_origin 修正前 ``origin`` 字段的读数
            nstart          读到的 ``nstart``
            voxel_size      读到的体素尺寸
            sample_anchor   修正前 Sample 使用的锚点（``origin + nstart*voxel``）

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

    previous_anchor = sample_anchor(origin, nstart, voxel_size)

    if not nstart.any():
        return NormalizedMap(source, origin, origin, nstart, voxel_size,
                             previous_anchor)

    fixed = display_origin(origin, nstart, voxel_size)
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
    return NormalizedMap(dest, fixed, origin, nstart, voxel_size,
                         previous_anchor)
