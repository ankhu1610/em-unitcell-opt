"""Checks for the separate deterministic normal extension; NumPy only."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from invdes.stable_projector import (
    nearest_normal, periodic_nearest_normal, normal_projector_grid, _select,
)


class StableProjectorTests(unittest.TestCase):
    def test_known_boundary_normals(self):
        h, r = 1/(2*np.sqrt(3)), 2/(3*np.sqrt(3))
        points = np.array([[0., -h], [0., r]])
        d, n = nearest_normal(points, 1., tie_tolerance=64*np.finfo(float).eps)
        np.testing.assert_allclose(d, 0, atol=1e-28)
        np.testing.assert_allclose(n, [[0, -1], [0, 1]], atol=1e-14)
        dd, nn = nearest_normal(-points, 1., False, tie_tolerance=64*np.finfo(float).eps)
        np.testing.assert_allclose(dd, d, atol=1e-28)
        np.testing.assert_allclose(nn, -n, atol=1e-14)

    def test_projector_rank_trace_and_normals(self):
        points = np.random.default_rng(42).random((40, 2))*242.5
        _, n = periodic_nearest_normal(points, 242.5, [161., 112.])
        np.testing.assert_allclose(np.sum(n*n, axis=-1), 1., atol=1e-14)
        p = n[..., :, None]*n[..., None, :]
        np.testing.assert_allclose(np.trace(p, axis1=-2, axis2=-1), 1., atol=1e-14)
        np.testing.assert_allclose(p@p, p, atol=1e-14)
        xx, xy = normal_projector_grid(242.5, [161., 112.], 16)
        np.testing.assert_allclose(xx*(1-xx), xy*xy, atol=1e-14)

    def test_periodicity_and_scale_covariance(self):
        a = 242.5
        sides = np.array([161.0085563202569, 112.0059522227874])
        points = np.array([[121.25, 208.37044822696288], [23.1, 71.7], [184., 32.]])
        d, n = periodic_nearest_normal(points, a, sides)
        for shift in ([a, 0], [a/2, np.sqrt(3)*a/2], [-a, 0]):
            ds, ns = periodic_nearest_normal(points+shift, a, sides)
            np.testing.assert_allclose(ds, d, rtol=1e-12, atol=1e-10)
            np.testing.assert_allclose(ns, n, atol=1e-12)
        for scale in (1e-6, 1e3):
            ds, ns = periodic_nearest_normal(points*scale, a*scale, sides*scale)
            np.testing.assert_allclose(ds/scale**2, d, rtol=1e-12, atol=1e-10)
            np.testing.assert_allclose(ns, n, atol=1e-12)

    def test_prior_tie_point_stays_on_same_branch(self):
        point = np.array([[121.25, 208.37044822696288]])
        sides = np.array([161.0085563202569, 112.0059522227874])
        normals = []
        for step in (-1e-4, 0., 1e-4):
            shifted = sides+np.array([step, 0.])
            _, n = periodic_nearest_normal(point, 242.5, shifted)
            normals.append(n)
        np.testing.assert_allclose(normals[0], normals[1], atol=1e-14)
        np.testing.assert_allclose(normals[2], normals[1], atol=1e-14)

    def test_two_pass_avoids_tolerance_chains(self):
        def candidates():
            for distance, normal in ((2., [1., 0.]), (1.25, [0., 1.]), (.5, [-1., 0.])):
                yield np.array([distance]), np.array(normal)
        d, n = _select(candidates, (1,), 1.)
        np.testing.assert_array_equal(d, [.5])
        np.testing.assert_array_equal(n, [[0., 1.]])


if __name__ == "__main__":
    unittest.main()
