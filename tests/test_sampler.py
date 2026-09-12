"""Boundary tests for the external Sample binary call in sampling/sampler.py."""

import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from protassem.sampling import sampler
from tests import fixtures


class SampleDensityMapTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name
        self.mrc = fixtures.make_mrc(os.path.join(self.tmp, "map.mrc"))

    def _sample(self, binary_name="Sample", output_dir=None, **binary_kwargs):
        binary = fixtures.make_fake_sample_bin(
            os.path.join(self.tmp, binary_name), **binary_kwargs)
        if output_dir is None:
            output_dir = os.path.join(self.tmp, "out")
        with mock.patch.object(sampler, "SAMPLE_BINARY", binary):
            return sampler.sample_density_map(self.mrc, contour=1.0,
                                              voxel_size=2.0,
                                              output_dir=output_dir)

    def test_returns_points_and_normals(self):
        points, normals, txt_path = self._sample()
        self.assertEqual(os.path.basename(txt_path), "map_2.00.txt")
        np.testing.assert_allclose(points, [[1.0, 2.0, 3.0], [3.0, 2.0, 3.0]])
        np.testing.assert_allclose(normals, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def test_output_directory_with_spaces(self):
        spaced = os.path.join(self.tmp, "out dir with spaces")
        points, _, txt_path = self._sample(output_dir=spaced)
        self.assertTrue(os.path.isfile(txt_path))
        self.assertEqual(len(points), 2)

    def test_nonzero_returncode_raises_with_stderr(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._sample(exit_code=3, stderr_text="boom from sample")
        self.assertIn("boom from sample", str(ctx.exception))

    def test_empty_output_raises(self):
        with self.assertRaises(RuntimeError):
            self._sample(output="", exit_code=0)

    def test_missing_binary_raises(self):
        missing = os.path.join(self.tmp, "does-not-exist")
        with mock.patch.object(sampler, "SAMPLE_BINARY", missing):
            with self.assertRaises(FileNotFoundError):
                sampler.sample_density_map(self.mrc, contour=1.0,
                                           output_dir=self.tmp)


if __name__ == "__main__":
    unittest.main()
