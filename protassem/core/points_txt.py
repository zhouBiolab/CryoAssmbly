"""点云 TXT 的统一读写（Sample 点云格式的单一实现）。

文件布局（仓库内 `Sample` 二进制实测，5 行头部 + 数据行）：

    line 0  sample        坐标缩放（采样体素边长，Å）
    line 1  盒子尺寸      实测恒为立方；不解析，原样保留
    line 2  未使用        不解析，原样保留
    line 3  origin x y z  盒子原点（Å）
    line 4  未使用        不解析，原样保留
    line 5+ 成对数据      (index x y z) 与 (vx vy vz density)

只有 line 0 与 line 3 被解释；line 1/2/4 逐字保留，便于"过滤后原样写回"。

数据行按**顺序两两成对**，不依赖绝对行号的奇偶：缺行或字段数不对会带行号报错，
不会像旧实现那样把点、法向量和密度静默错位。
"""

import collections

import numpy as np

HEADER_LINES = 5

POINT_DTYPE = [("index", np.int32),
               ("point", np.float32, (3,)),
               ("vector", np.float32, (3,)),
               ("density", np.float32)]

COORD_FIELDS = "index x y z"
VECTOR_FIELDS = "vx vy vz density"

PointCloud = collections.namedtuple(
    "PointCloud",
    "points vectors densities indices sample origin header_lines data_lines")


def read_point_cloud(path):
    """读取点云 TXT。

    Returns:
        PointCloud:
            points/vectors/densities: float64 数组，坐标已按
                ``voxel * sample + origin`` 换算为 Å；
            indices: int64 数组（体素索引）；
            sample/origin: 头部解析值；
            header_lines: 原始 5 行头部；
            data_lines: 原始数据行对 ``[(坐标行, 向量行), ...]``。

    Raises:
        ValueError: 头部不足 5 行、数据行不成对、字段数不是 4、数值无法解析；
            消息包含 ``文件:行号``。
    """
    with open(path, encoding="utf-8") as handle:
        lines = handle.read().splitlines()

    if len(lines) < HEADER_LINES:
        raise ValueError("%s: expected %d header lines, got %d"
                         % (path, HEADER_LINES, len(lines)))

    header_lines = lines[:HEADER_LINES]
    sample = _parse_float(path, 1, header_lines[0], "sample")
    origin = _parse_vector(path, 4, header_lines[3], "origin")

    data_lines = list(lines[HEADER_LINES:])
    while data_lines and not data_lines[-1].strip():
        data_lines.pop()
    for offset, line in enumerate(data_lines):
        if not line.strip():
            raise ValueError("%s:%d: blank line inside the data region"
                             % (path, HEADER_LINES + offset + 1))
    if len(data_lines) % 2:
        raise ValueError("%s:%d: data lines must come in pairs "
                         "(coordinate line + vector line), got %d line(s)"
                         % (path, HEADER_LINES + 1, len(data_lines)))

    points, vectors, densities, indices, pairs = [], [], [], [], []
    for offset in range(0, len(data_lines), 2):
        coord_lineno = HEADER_LINES + offset + 1
        vector_lineno = coord_lineno + 1
        coord_parts = _split(path, coord_lineno, data_lines[offset], COORD_FIELDS)
        vector_parts = _split(path, vector_lineno, data_lines[offset + 1], VECTOR_FIELDS)
        voxel = _to_float(path, coord_lineno, coord_parts[1:])
        indices.append(int(coord_parts[0]))
        points.append([voxel[i] * sample + origin[i] for i in range(3)])
        vectors.append(_to_float(path, vector_lineno, vector_parts[:3]))
        densities.append(_to_float(path, vector_lineno, vector_parts[3:])[0])
        pairs.append((data_lines[offset], data_lines[offset + 1]))

    return PointCloud(points=np.array(points, dtype=np.float64),
                      vectors=np.array(vectors, dtype=np.float64),
                      densities=np.array(densities, dtype=np.float64),
                      indices=np.array(indices, dtype=np.int64),
                      sample=sample,
                      origin=np.array(origin, dtype=np.float64),
                      header_lines=header_lines,
                      data_lines=pairs)


def line_mapping(cloud):
    """每个点对应的原始行下标 ``(坐标行, 向量行)``，用于按点过滤后原样写回。"""
    return [(HEADER_LINES + 2 * i, HEADER_LINES + 2 * i + 1)
            for i in range(len(cloud.data_lines))]


def read_point_cloud_file(path):
    """读取点云并返回 POINT_DTYPE 结构化数组（demo_mask/sw_mask 的既有格式）。"""
    cloud = read_point_cloud(path)
    data = np.zeros(len(cloud.points), dtype=POINT_DTYPE)
    data["index"] = cloud.indices
    data["point"] = cloud.points
    data["vector"] = cloud.vectors
    data["density"] = cloud.densities
    return data


def write_filtered(path, cloud, keep_indices):
    """按 keep_indices 写回原始数据行对，头部原样复制（保留文本精度）。

    Args:
        path: 输出路径
        cloud: read_point_cloud 的结果
        keep_indices: 要保留的点下标（顺序即写出顺序）
    """
    with open(path, "w", encoding="utf-8") as handle:
        for line in cloud.header_lines:
            handle.write(line + "\n")
        for index in keep_indices:
            coord_line, vector_line = cloud.data_lines[index]
            handle.write(coord_line + "\n")
            handle.write(vector_line + "\n")


def write_point_cloud(path, points, vectors, densities=None, indices=None,
                      sample=1.0, origin=(0.0, 0.0, 0.0)):
    """由数组生成点云 TXT（坐标按 ``(point - origin) / sample`` 写回体素坐标）。

    Args:
        path: 输出路径
        points: (N, 3) 绝对坐标（Å）
        vectors: (N, 3) 法向量
        densities: (N,) 密度；None 写 0
        indices: (N,) 体素索引；None 写 0..N-1
        sample: 头部坐标缩放；默认 1.0（坐标按 Å 原样写出）
        origin: 头部原点偏移
    """
    points = np.asarray(points, dtype=np.float64)
    vectors = np.asarray(vectors, dtype=np.float64)
    count = len(points)
    if densities is None:
        densities = np.zeros(count, dtype=np.float64)
    if indices is None:
        indices = np.arange(count, dtype=np.int64)
    origin = np.asarray(origin, dtype=np.float64)

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("%.6f\n" % sample)
        handle.write("0 0 0\n")
        handle.write("0.000000 0.000000 0.000000\n")
        handle.write("%.6f %.6f %.6f\n" % (origin[0], origin[1], origin[2]))
        handle.write("0.000000 0.000000 0.000000\n")
        for i in range(count):
            voxel = (points[i] - origin) / sample
            handle.write("%d %.6f %.6f %.6f\n"
                         % (int(indices[i]), voxel[0], voxel[1], voxel[2]))
            handle.write("%.6f %.6f %.6f %.6f\n"
                         % (vectors[i][0], vectors[i][1], vectors[i][2],
                            densities[i]))


def _split(path, lineno, line, expected):
    """按空白切分并校验字段数（严格 4 列，不做宽容解析）。"""
    parts = line.split()
    if len(parts) != 4:
        raise ValueError("%s:%d: expected 4 fields <%s>, got %d"
                         % (path, lineno, expected, len(parts)))
    return parts


def _to_float(path, lineno, parts):
    values = []
    for part in parts:
        try:
            values.append(float(part))
        except ValueError as exc:
            raise ValueError("%s:%d: %r is not a number" % (path, lineno, part)) from exc
    return values


def _parse_float(path, lineno, line, name):
    text = line.strip()
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError("%s:%d: %s must be a number, got %r"
                         % (path, lineno, name, text)) from exc


def _parse_vector(path, lineno, line, name):
    parts = line.split()
    if len(parts) != 3:
        raise ValueError("%s:%d: %s must have 3 numbers, got %d"
                         % (path, lineno, name, len(parts)))
    return _to_float(path, lineno, parts)
