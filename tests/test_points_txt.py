"""点云 TXT 统一读写的测试：数值、精度、错位反例与域 TXT 兼容接口。"""

import os
import tempfile
import unittest

import numpy as np

from protassem.core import points_txt
from tests import fixtures


THREE_PAIRS = [
    ("1 0 0 0", "1.000000 0.000000 0.000000 5.000000"),
    ("2 1 0 0", "0.000000 1.000000 0.000000 6.000000"),
    ("3 0 1 0", "0.000000 0.000000 1.000000 7.000000"),
]


def _legacy_parity_parse(text):
    """S3 之前 core/io.load_sample_points 的解析逻辑（仅用于回归取证）。"""
    points, normals = [], []
    lines = text.splitlines()
    for i in range(5, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        parts = line.split()
        if i % 2 == 1:
            if len(parts) >= 4:
                points.append([float(v) for v in parts[1:4]])
        else:
            if len(parts) >= 4:
                normals.append([float(v) for v in parts[:3]])
    return points, normals


class ReadPointCloudTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

    def path(self, name="cloud.txt"):
        return os.path.join(self.tmp, name)

    def write(self, text, name="cloud.txt"):
        path = self.path(name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return path

    def test_values_and_headers(self):
        path = self.write(fixtures.sample_txt_text(pairs=THREE_PAIRS))
        cloud = points_txt.read_point_cloud(path)
        self.assertEqual(cloud.sample, 2.0)
        np.testing.assert_allclose(cloud.origin, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(cloud.points[0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(cloud.points[1], [3.0, 2.0, 3.0])
        np.testing.assert_allclose(cloud.points[2], [1.0, 4.0, 3.0])
        np.testing.assert_allclose(cloud.vectors[1], [0.0, 1.0, 0.0])
        np.testing.assert_allclose(cloud.densities, [5.0, 6.0, 7.0])
        self.assertEqual(list(cloud.indices), [1, 2, 3])
        self.assertEqual(len(cloud.header_lines), points_txt.HEADER_LINES)
        self.assertEqual(len(cloud.data_lines), 3)

    def test_trailing_blank_line_is_allowed(self):
        path = self.write(fixtures.sample_txt_text(pairs=THREE_PAIRS) + "\n\n")
        self.assertEqual(len(points_txt.read_point_cloud(path).points), 3)

    def test_blank_line_in_the_middle_is_an_error(self):
        lines = fixtures.sample_txt_text(pairs=THREE_PAIRS).splitlines()
        lines.insert(7, "")
        with self.assertRaises(ValueError) as ctx:
            points_txt.read_point_cloud(self.write("\n".join(lines) + "\n"))
        self.assertIn("8:", str(ctx.exception))

    def test_short_point_line_is_rejected_where_legacy_silently_misaligned(self):
        text = fixtures.sample_txt_text(pairs=[THREE_PAIRS[0], ("2 1 0", THREE_PAIRS[1][1])])
        path = self.write(text)
        with self.assertRaises(ValueError) as ctx:
            points_txt.read_point_cloud(path)
        self.assertIn("expected 4 fields <index x y z>", str(ctx.exception))
        legacy_points, legacy_normals = _legacy_parity_parse(text)
        self.assertEqual(len(legacy_points), 1)
        self.assertEqual(len(legacy_normals), 2)   # 旧实现：点与法向量错位且不报错

    def test_odd_number_of_data_lines_is_rejected(self):
        lines = fixtures.sample_txt_text(pairs=THREE_PAIRS).splitlines()
        text = "\n".join(lines[:-1]) + "\n"
        with self.assertRaises(ValueError) as ctx:
            points_txt.read_point_cloud(self.write(text))
        self.assertIn("come in pairs", str(ctx.exception))

    def test_short_header_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            points_txt.read_point_cloud(self.write("2.0\n8 8 8\n"))
        self.assertIn("header lines", str(ctx.exception))

    def test_non_numeric_sample_is_rejected(self):
        lines = fixtures.sample_txt_text(pairs=THREE_PAIRS).splitlines()
        lines[0] = "two"
        with self.assertRaises(ValueError) as ctx:
            points_txt.read_point_cloud(self.write("\n".join(lines) + "\n"))
        self.assertIn("sample must be a number", str(ctx.exception))

    def test_non_numeric_coordinate_is_rejected(self):
        lines = fixtures.sample_txt_text(pairs=THREE_PAIRS).splitlines()
        lines[5] = "1 x 0 0"
        with self.assertRaises(ValueError) as ctx:
            points_txt.read_point_cloud(self.write("\n".join(lines) + "\n"))
        self.assertIn("is not a number", str(ctx.exception))

    def test_header_only_file_has_no_points(self):
        cloud = points_txt.read_point_cloud(self.write(fixtures.sample_txt_text(pairs=[])))
        self.assertEqual(len(cloud.points), 0)
        self.assertEqual(len(cloud.data_lines), 0)

    def test_structured_view_matches_arrays(self):
        path = self.write(fixtures.sample_txt_text(pairs=THREE_PAIRS))
        cloud = points_txt.read_point_cloud(path)
        data = points_txt.read_point_cloud_file(path)
        self.assertEqual(data.dtype.names, ("index", "point", "vector", "density"))
        np.testing.assert_allclose(data["point"], cloud.points)
        np.testing.assert_allclose(data["density"], cloud.densities)
        self.assertEqual(list(data["index"]), [1, 2, 3])


class WritePointCloudTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

    def test_generated_cloud_round_trips(self):
        points = [[1.0, 2.0, 3.0], [4.5, 5.5, 6.5]]
        vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        densities = [5.0, 6.0]
        path = os.path.join(self.tmp, "generated.txt")
        points_txt.write_point_cloud(path, points, vectors, densities=densities,
                                     indices=[7, 8])
        cloud = points_txt.read_point_cloud(path)
        np.testing.assert_allclose(cloud.points, points)
        np.testing.assert_allclose(cloud.vectors, vectors)
        np.testing.assert_allclose(cloud.densities, densities)
        self.assertEqual(list(cloud.indices), [7, 8])

    def test_filtered_write_keeps_original_bytes(self):
        path = os.path.join(self.tmp, "original.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(fixtures.sample_txt_text(pairs=THREE_PAIRS))
        with open(path, encoding="utf-8") as handle:
            original_lines = handle.readlines()
        cloud = points_txt.read_point_cloud(path)

        out = os.path.join(self.tmp, "kept.txt")
        points_txt.write_filtered(out, cloud, [0, 2])
        with open(out, encoding="utf-8") as handle:
            kept = handle.read()
        expected = "".join(original_lines[:points_txt.HEADER_LINES]
                           + [original_lines[5], original_lines[6],
                              original_lines[9], original_lines[10]])
        self.assertEqual(kept, expected)


class DomainTxtCompatibilityTest(unittest.TestCase):
    """domain_pdb_txt 的历史 6 元组接口保留，过滤写回逐字节保真。"""

    def test_info_reader_and_filtered_writer(self):
        from protassem.assembly.domain_parser import domain_pdb_txt
        with tempfile.TemporaryDirectory() as tmp:
            src = fixtures.make_sample_txt(os.path.join(tmp, "chain_A.txt"),
                                           pairs=THREE_PAIRS)
            with open(src, encoding="utf-8") as handle:
                original_lines = handle.readlines()
            result = domain_pdb_txt.load_sample_points_with_info(src)
            points, vectors, densities, lines, mapping, header = result
            self.assertEqual(len(points), 3)
            self.assertEqual(mapping, [(5, 6), (7, 8), (9, 10)])
            self.assertEqual(header["sample"], 2.0)
            self.assertEqual(len(lines), len(original_lines))

            out = os.path.join(tmp, "sub", "domain_d_1.txt")
            domain_pdb_txt.save_domain_txt(lines, [0, 2], mapping, header, out)
            with open(out, encoding="utf-8") as handle:
                kept = handle.read()
            expected = "".join(original_lines[:5]
                               + [original_lines[5], original_lines[6],
                                  original_lines[9], original_lines[10]])
            self.assertEqual(kept, expected)


if __name__ == "__main__":
    unittest.main()
