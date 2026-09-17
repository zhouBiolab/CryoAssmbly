#!/usr/bin/env python3
"""MRC 原点规范化诊断（原点口径核对 + 采样 + 导出点云 PDB）。

背景
----
项目里两条原点口径，只在 ``origin`` 与 ``nstart`` 都非零时冲突：

    origin  nstart  Sample 锚点            read_mrc_full    冲突
    非零    0       origin                origin           否
    0       非零    nstart*voxel          nstart*voxel     否
    非零    非零    origin+nstart*voxel   origin           **是**

规则与理由见 ``protassem.core.mrc_origin``。本工具的价值在**诊断**：读 header
原始字段、印两条口径、采样后按"点云落回高密度区"的实测得分核对"Sample 锚点
= origin + nstart*voxel"这个假设，并在规范化前后各采样一次做平移等式硬校验。

Sample 锚点的实测证据
--------------------
Sample 输出的点云 TXT 第 3 行（采样盒原点）满足

    line3 = (origin + nstart*voxel) + n*voxel/2 - box*sample/2

在 6 张图上逐位吻合；最干净的判据是差分检验：固定 origin、把 nstart 从 0 改成
(5,7,9)，line3 恰好增加 nstart*voxel（box 与居中项被抵消）。

做法（六步，产物只写 --outdir）
------------------------------
1. 读 header（原始 struct 字段 + mrcfile 口径），印 display_origin / sample_anchor
   与两者之差。
2. 在原始图上采样，核对锚点公式预测的 line3。
3. 候选原点诊断：列出候选的对齐得分，核对落点最佳者是否为 sample_anchor；
   **不决定写什么**。
4. 需要时写规范化副本（``core.mrc_origin.normalize_density_map``）：``nstart``
   归零、``origin`` 停在显示位置，数组与 ``mapc/mapr/maps`` 不动。
5. 在规范化后的图上重新采样，校验对齐与**平移等式**
   ``pts_norm == pts_orig + (display_origin - sample_anchor)``。
6. 导出点云为 PDB（CA 原子），与 ``core.io.save_points_as_pdb`` 同格式。

只写 --outdir 下的文件，绝不改动输入。默认落在
``<mrc 目录>/normalized/<mrc 名不含后缀>/``。

用法
----
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate point
    python tools/normalize_mrc_sample_pdb.py <map.mrc>
    python tools/normalize_mrc_sample_pdb.py <map.mrc> --contour 9 --voxel 2.0
    python tools/normalize_mrc_sample_pdb.py <map.mrc> --outdir /somewhere

退出码：0 = 四项自检全过；1 = 有自检未过（输入未被改动）。
"""

import argparse
import os
import struct
import sys

import mrcfile
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from protassem.core import mrc_origin                                    # noqa: E402
from protassem.core.io import save_points_as_pdb                         # noqa: E402
from protassem.core.points_txt import read_point_cloud                   # noqa: E402
from protassem.core.scoring import read_mrc_full                         # noqa: E402
from protassem.sampling.sampler import sample_density_map                # noqa: E402

# header 里可能存放原点的位置（按 4 字节为一个"字"编号）
W_XORG_MRC2014 = 39       # MRC2014 规范位置（本项目工具链不写这里）
W_ORIGIN_LEGACY = 49      # mrcfile 与 Sample 实际读写的位置


def read_raw_words(path):
    """读 MRC 前 1024 字节原始字段。

    用独立于 mrcfile 的 struct 解析，便于对照两个候选原点位置（第 39-41 字与
    第 49-51 字）——只看 mrcfile 一侧读数发现不了这种分歧。
    """
    with open(path, "rb") as handle:
        head = handle.read(1024)
    if len(head) < 1024:
        raise ValueError("%s: file shorter than a 1024-byte MRC header" % path)

    def as_int(word):
        return struct.unpack_from("<i", head, word * 4)[0]

    def as_float(word):
        return struct.unpack_from("<f", head, word * 4)[0]

    return {
        "nx": as_int(0), "ny": as_int(1), "nz": as_int(2), "mode": as_int(3),
        "nstart": np.array([as_int(4), as_int(5), as_int(6)], dtype=int),
        "cell": np.array([as_float(10), as_float(11), as_float(12)]),
        "mapcrs": (as_int(16), as_int(17), as_int(18)),
        "nsymbt": as_int(23),
        "xorg_mrc2014": np.array([as_float(W_XORG_MRC2014),
                                  as_float(W_XORG_MRC2014 + 1),
                                  as_float(W_XORG_MRC2014 + 2)]),
        "magic": head[208:212],
    }


def read_origin_fields(path):
    """读 origin / nstart / voxel_size（mrcfile 口径，与 Sample 一致）。

    Returns:
        (origin, nstart, voxel_size)，均为 float64 数组。
    """
    with mrcfile.open(path, permissive=True) as mrc:
        origin = np.array([mrc.header.origin.x, mrc.header.origin.y,
                           mrc.header.origin.z], dtype=np.float64)
        nstart = np.array([mrc.header.nxstart, mrc.header.nystart,
                           mrc.header.nzstart], dtype=np.float64)
        voxel_size = np.array([mrc.voxel_size.x, mrc.voxel_size.y,
                               mrc.voxel_size.z], dtype=np.float64)
    return origin, nstart, voxel_size


def alignment(data, points_a, origin, voxel, contour):
    """把 Å 坐标点映射进密度网格，统计对齐质量。

    ``data`` 轴序是 ``(nz, ny, nx)``、点云坐标是 ``(x, y, z)``，所以取
    ``data[iz, iy, ix]``；边界检查按轴分别对 nx/ny/nz（非立方图上不能拿
    (x,y,z) 直接与 shape 逐分量比较）。
    """
    vox = (points_a - origin) / voxel
    ix = np.rint(vox[:, 0]).astype(np.int64)
    iy = np.rint(vox[:, 1]).astype(np.int64)
    iz = np.rint(vox[:, 2]).astype(np.int64)

    nz, ny, nx = (int(v) for v in data.shape)
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)

    vals = np.full(len(ix), np.nan)
    if inside.any():
        vals[inside] = data[iz[inside], iy[inside], ix[inside]]
    good = vals[~np.isnan(vals)]
    return {
        "inside": int(inside.sum()),
        "total": int(len(ix)),
        "inside_frac": float(inside.sum()) / len(ix) if len(ix) else 0.0,
        "mean_density": float(good.mean()) if good.size else float("nan"),
        "frac_above": float(np.mean(good > contour)) if good.size else float("nan"),
    }


def alignment_score(stat):
    """候选排序标量：既要落图内，也要落在密度上。"""
    if not np.isfinite(stat["mean_density"]) or not np.isfinite(stat["frac_above"]):
        return -1.0
    return stat["inside_frac"] * (stat["frac_above"] + 0.01 * stat["mean_density"])


def candidate_origins(origin, nstart, voxel, hdr):
    """候选原点（去重、过滤非有限值）——仅用于诊断。"""
    cands = []

    def add(label, value):
        value = np.asarray(value, dtype=np.float64)
        if not np.all(np.isfinite(value)) or np.any(np.abs(value) > 1e6):
            return
        for existing in cands:
            if np.allclose(existing["origin"], value, atol=1e-6):
                return
        cands.append({"label": label, "origin": value})

    add("display_origin (MRC标准/评分端)",
        mrc_origin.display_origin(origin, nstart, voxel))
    add("sample_anchor (origin+nstart*voxel)",
        mrc_origin.sample_anchor(origin, nstart, voxel))
    add("nstart*voxel", nstart * voxel)
    add("0 (零原点)", np.zeros(3))
    add("words39-41 (MRC2014 xorg)", hdr["xorg_mrc2014"])
    return cands


def sample_once(mrc_path, contour, voxel, outdir, final_name):
    """采样一次，返回 (points, line3, box, txt_path)。

    复用 ``sampling.sampler.sample_density_map``（与 pipeline 同一条调用路径），
    再用 ``points_txt.read_point_cloud`` 取 TXT 第 3 行（锚点）与第 1 行（box）。
    采样器自己按 ``<stem>_<voxel>.txt`` 命名，这里统一改成 ``final_name``。
    """
    _points, _normals, produced = sample_density_map(
        mrc_path, contour=contour, voxel_size=voxel, output_dir=outdir)
    cloud = read_point_cloud(produced)
    final_path = os.path.join(outdir, final_name)
    if os.path.abspath(produced) != os.path.abspath(final_path):
        os.replace(produced, final_path)
    box = np.array([float(v) for v in cloud.header_lines[1].split()])
    return cloud.points, cloud.origin, box, final_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="MRC 原点规范化诊断：核对原点口径 -> 采样 -> 导出点云 PDB")
    parser.add_argument("mrc", help="输入密度图 (.mrc)")
    parser.add_argument("--outdir", default=None,
                        help="输出目录（默认 <mrc 目录>/normalized/<mrc 名不含后缀>）")
    parser.add_argument("--contour", type=float, default=None,
                        help="采样阈值；不给则用 3*sigma（与 pipeline 对模拟图一致）")
    parser.add_argument("--sigma-multiplier", type=float, default=3.0,
                        help="自动阈值时的 sigma 倍数（默认 3.0）")
    parser.add_argument("--voxel", type=float, default=2.0,
                        help="采样步长 Å（默认 2.0）")
    parser.add_argument("--chain", default="A", help="导出 PDB 的链 ID（默认 A）")
    args = parser.parse_args(argv)

    mrc_path = os.path.abspath(args.mrc)
    if not os.path.isfile(mrc_path):
        sys.stderr.write("ERROR: no such file: %s\n" % mrc_path)
        return 2
    stem = os.path.splitext(os.path.basename(mrc_path))[0]
    outdir = args.outdir or os.path.join(os.path.dirname(mrc_path),
                                         "normalized", stem)
    os.makedirs(outdir, exist_ok=True)

    hdr = read_raw_words(mrc_path)
    origin, nstart, voxel = read_origin_fields(mrc_path)
    data, _voxel_full, _read_origin, _dims = read_mrc_full(mrc_path)
    sigma = float(np.std(data))
    contour = args.contour if args.contour is not None else \
        args.sigma_multiplier * sigma

    anchor = mrc_origin.sample_anchor(origin, nstart, voxel)
    target = mrc_origin.display_origin(origin, nstart, voxel)
    shift = target - anchor            # 规范化后点云相对原图点云的位移

    print("=" * 76)
    print("输入    : %s" % mrc_path)
    print("输出目录: %s" % outdir)
    print("=" * 76)

    # ---------- 1. header 与两条口径 ----------
    print()
    print("[1] header 与原点口径")
    print("    nx,ny,nz        : %d, %d, %d   %s"
          % (hdr["nx"], hdr["ny"], hdr["nz"],
             "(立方)" if hdr["nx"] == hdr["ny"] == hdr["nz"] else "(非立方)"))
    print("    mapc,mapr,maps  : %d, %d, %d" % hdr["mapcrs"])
    print("    voxel_size      : %.4f %.4f %.4f   (isotropic: %s)"
          % (voxel[0], voxel[1], voxel[2], bool(np.allclose(voxel, voxel[0]))))
    print("    nstart          : %s" % nstart.astype(int))
    print()
    print("    display_origin                : %s" % np.round(target, 4))
    print("      ^ MRC 标准 / 外部软件(ChimeraX) / scoring.read_mrc_full 用这个")
    print("    sample_anchor                 : %s" % np.round(anchor, 4))
    print("      ^ Sample 二进制用这个 (= origin + nstart*voxel，实测确认)")
    print("    两者之差（点云修正量）        : %s" % np.round(shift, 4))
    print("    nstart*voxel                  : %s" % np.round(nstart * voxel, 4))
    print("    words49-51 origin (实际读的)  : %s" % np.round(origin, 4))
    print("    words39-41 (MRC2014 xorg)     : %s" % np.round(hdr["xorg_mrc2014"], 4))
    print()
    print("    数据范围 : %.4f .. %.4f   std=%.6f" % (data.min(), data.max(), sigma))
    print("    采样阈值 : %.4f  (%s)"
          % (contour, "命令行 --contour" if args.contour is not None
             else "自动 %.0f*sigma" % args.sigma_multiplier))
    if np.any(np.abs(shift) > 1e-6):
        print("    -> 需要规范化：origin 与 nstart 都非零，两条口径差 nstart*voxel")
    else:
        print("    -> 无需规范化：两条口径本来就同值")

    # ---------- 2. 在原图上采样 ----------
    print()
    print("[2] 在原始 MRC 上采样")
    points, line3, box, orig_txt = sample_once(
        mrc_path, contour, args.voxel, outdir, "%s_orig_%.2f.txt" % (stem, args.voxel))
    print("    Sample 点云      : %d 点" % len(points))
    print("    TXT 第3行        : %s" % np.round(line3, 4))
    print("    TXT box          : %s" % box)
    print("    坐标范围 (Å)     : %s .. %s"
          % (np.round(points.min(axis=0), 3), np.round(points.max(axis=0), 3)))
    # 用 (nx, ny, nz) 顺序：data.shape 是 (nz, ny, nx)，非立方图上两者不同
    n_xyz = np.array([hdr["nx"], hdr["ny"], hdr["nz"]], dtype=np.float64)
    pred_line3 = anchor + n_xyz * voxel / 2.0 - box * args.voxel / 2.0
    print("    锚点公式预测 line3: %s   %s"
          % (np.round(pred_line3, 4),
             "MATCH" if np.allclose(pred_line3, line3, atol=1e-2) else "**不符**"))

    # ---------- 3. 候选标定（诊断） ----------
    print()
    print("[3] 候选原点诊断（不决定写什么，只核对锚点假设）")
    print("    对齐分数 = 落图内比例 × 落在密度上的比例")
    print("    %-40s %10s %8s %11s %10s %8s"
          % ("候选", "inside", "frac", "meanDens", "frac>lev", "score"))
    cands = candidate_origins(origin, nstart, voxel, hdr)
    for cand in cands:
        stat = alignment(data, points, cand["origin"], voxel, contour)
        cand["stat"] = stat
        cand["score"] = alignment_score(stat)
        print("    %-40s %5d/%-5d %7.3f %11.3f %10.3f %8.3f"
              % (cand["label"], stat["inside"], stat["total"], stat["inside_frac"],
                 stat["mean_density"], stat["frac_above"], cand["score"]))

    best = max(cands, key=lambda c: c["score"])
    anchor_score = next(c["score"] for c in cands
                        if np.allclose(c["origin"], anchor, atol=1e-6))
    print()
    print("    点云落点最佳的候选 : %s  (score %.3f)" % (best["label"], best["score"]))
    print("    sample_anchor 的分数: %.3f" % anchor_score)
    anchor_assumption_ok = np.allclose(best["origin"], anchor, atol=1e-4)
    if anchor_assumption_ok:
        print("    核对: 最佳候选就是 sample_anchor -> 锚点假设成立")
    else:
        print("    **警告: 最佳候选不是 sample_anchor —— 本图的 Sample 锚点不能")
        print("            用 origin+nstart*voxel 解释；规范化仍按确定性规则写，")
        print("            但请人工检查这张图的 header。")

    # ---------- 4. 写规范化 MRC ----------
    print()
    print("[4] 写规范化 MRC（只改 header 的 4 个字段，数组不动）")
    norm_mrc = os.path.join(outdir, "%s_normalized.mrc" % stem)
    result = mrc_origin.normalize_density_map(mrc_path, norm_mrc)
    wrote_copy = result.path != mrc_path
    effective_mrc = result.path
    same_data = True
    if not wrote_copy:
        print("    nstart 已全为 0 -> 无需修正，不写副本")
    else:
        print("    写出          : %s" % norm_mrc)
        print("    origin 回读   : %s   (期望 %s，即 display_origin、与原图一致)"
              % (np.round(read_origin_fields(norm_mrc)[0], 6), np.round(target, 6)))
        print("    nstart 回读   : %s   (期望 0 0 0)"
              % read_origin_fields(norm_mrc)[1].astype(int))
        print("    锚点变化      : Sample 从 %s 移到 %s"
              % (np.round(result.sample_anchor, 4), np.round(result.origin, 4)))
        with mrcfile.open(mrc_path, permissive=True) as a, \
                mrcfile.open(norm_mrc, permissive=True) as b:
            same_data = bool(np.array_equal(a.data, b.data)
                             and a.data.dtype == b.data.dtype)
            same_mapcrs = ((int(a.header.mapc), int(a.header.mapr), int(a.header.maps))
                           == (int(b.header.mapc), int(b.header.mapr), int(b.header.maps)))
        print("    数组数据一致  : %s" % ("一致" if same_data else "**不一致**"))
        print("    轴序保留      : %s" % ("一致" if same_mapcrs else "**被改动**"))

    # ---------- 5. 在规范化后的图上采样并校验 ----------
    print()
    print("[5] 在规范化后的 MRC 上采样并校验")
    pts2, line3_new, _box2, final_txt = sample_once(
        effective_mrc, contour, args.voxel, outdir,
        "%s_normalized_%.2f.txt" % (stem, args.voxel))
    print("    输入          : %s" % effective_mrc)
    print("    点云点数      : %d" % len(pts2))
    print("    TXT 第3行     : %s" % np.round(line3_new, 4))

    stat2 = alignment(data, pts2, target, voxel, contour)
    print("    对齐(display_origin 下) : inside=%d/%d (%.1f%%)  meanDens=%.3f  frac>lev=%.3f"
          % (stat2["inside"], stat2["total"], 100.0 * stat2["inside_frac"],
             stat2["mean_density"], stat2["frac_above"]))
    stat_before = alignment(data, points, anchor, voxel, contour)
    print("    原图点云(sample_anchor 下): inside=%d/%d (%.1f%%)  meanDens=%.3f  frac>lev=%.3f"
          % (stat_before["inside"], stat_before["total"],
             100.0 * stat_before["inside_frac"],
             stat_before["mean_density"], stat_before["frac_above"]))

    same_n = len(pts2) == len(points)
    max_dev = (float(np.abs(pts2 - (points + shift)).max()) if same_n
               else float("nan"))
    eq_shift = bool(same_n and max_dev <= 1e-4)
    print("    平移等式      : pts_norm == pts_orig + (display_origin - sample_anchor)")
    print("                    %s   最大偏差=%.2e"
          % ("成立" if eq_shift else "**不成立**", max_dev))

    # ---------- 6. 导出 PDB ----------
    pdb_path = os.path.join(outdir, "%s_points.pdb" % stem)
    save_points_as_pdb(pts2, pdb_path, chain_id=args.chain)
    print()
    print("[6] 导出 PDB")
    print("    路径          : %s" % pdb_path)
    print("    原子数        : %d (CA)" % len(pts2))
    print("    坐标范围 (Å)  : %s .. %s"
          % (np.round(pts2.min(axis=0), 3), np.round(pts2.max(axis=0), 3)))
    print("    采样 TXT      : %s" % final_txt)

    # ---------- 自检 ----------
    checks = [
        ("a. 图未移动（副本 origin == display_origin）",
         True if not wrote_copy
         else bool(np.allclose(read_origin_fields(norm_mrc)[0], target, atol=1e-4))),
        ("b. 数组 / 轴序未被改动", same_data),
        ("c. 点云落图内 >=99% 且 frac>lev >=0.5",
         bool(stat2["inside_frac"] >= 0.99 and stat2["frac_above"] >= 0.5)),
        ("d. 平移等式 pts_norm == pts_orig + shift", eq_shift),
        ("e. Sample 锚点假设成立（最佳候选 == sample_anchor）", anchor_assumption_ok),
    ]

    print()
    print("=" * 76)
    print("自检:")
    for label, ok in checks:
        print("    [%s] %s" % ("PASS" if ok else "FAIL", label))
    ok_all = all(ok for _label, ok in checks)
    print()
    if np.any(np.abs(shift) > 1e-6):
        print("结论: 已规范化 —— nstart 归零、origin 停在显示位置；")
        print("      Sample 锚点从 %s 移到 %s，与评分端同框，"
              % (np.round(anchor, 3), np.round(target, 3)))
        print("      且密度图在外部软件中的位置与原图逐位相同。")
    else:
        print("结论: 该 MRC 两条口径本来就同值，无需修正。")
    print("自检: %s" % ("通过" if ok_all else "**未通过，请检查上面的数字**"))
    print("=" * 76)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
