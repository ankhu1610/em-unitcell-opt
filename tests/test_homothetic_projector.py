"""Geometry-only checks for the fixed-domain homothetic normal extension."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from invdes.homothetic_projector import radial_normal, periodic_normal, normal_projector_grid


class HomotheticProjectorTests(unittest.TestCase):
    def test_known_normals_and_invariants(self):
        points = np.array([[0., -1.], [0., 1.], [1., 0.], [0., 0.]])
        n = radial_normal(points)
        np.testing.assert_allclose(n[:3], [[0., -1.], [0., 1.], [np.sqrt(3)/2, .5]], atol=1e-14)
        np.testing.assert_allclose(radial_normal(-points, False), -n, atol=1e-14)
        np.testing.assert_allclose(np.sum(n*n, axis=-1), 1., atol=1e-14)
        p = n[..., :, None]*n[..., None, :]
        np.testing.assert_allclose(np.trace(p, axis1=-2, axis2=-1), 1., atol=1e-14)
        np.testing.assert_allclose(p@p, p, atol=1e-14)

    def test_size_independence(self):
        a = 242.5
        b = a/np.sqrt(3)
        samples = [normal_projector_grid(a, np.array(s)*b, 32)
                   for s in ([1., .65], [1.15, .8], [1.35, 1.])]
        for sample in samples[1:]:
            for actual, expected in zip(sample, samples[0]):
                np.testing.assert_array_equal(actual, expected)
        xx, xy = samples[0]
        np.testing.assert_allclose(xx*(1-xx), xy*xy, atol=1e-14)

    def test_periodicity_and_units(self):
        a = 242.5
        sides = a/np.sqrt(3)*np.array([1.15, .8])
        points = np.array([[0., 0.], [23., 81.], [121.25, 208.37044822696288]])
        n = periodic_normal(points, a, sides)
        for shift in ([a, 0], [a/2, np.sqrt(3)*a/2], [-a, 0]):
            np.testing.assert_allclose(periodic_normal(points+shift, a, sides), n, atol=1e-12)
        np.testing.assert_allclose(periodic_normal(points*1e-6, a*1e-6, sides*1e-6), n, atol=1e-12)

    def test_maximum_boundary_ownership_and_normals(self):
        a = 242.5
        sides = a/np.sqrt(3)*np.array([1.35, 1.])
        centers = np.array([[a, a/np.sqrt(3)], [a/2, a/(2*np.sqrt(3))]])
        normals = np.array([[0., -1.], [np.sqrt(3)/2, .5], [-np.sqrt(3)/2, .5]])
        # Boundary points sampled by rays, then independently checked against
        # every center in a larger 7x7 periodic stencil.
        theta = np.arange(4096)*2*np.pi/4096 + .000123
        direction = np.column_stack((np.cos(theta), np.sin(theta)))
        maximum = np.max(direction@normals.T, axis=1)
        for hole, up in enumerate((False, True)):
            radius = np.minimum(2/(3*np.sqrt(3)), 1/(2*np.sqrt(3)*maximum))*sides[hole]
            sign = 1 if up else -1
            displacement = sign*radius[:, None]*direction
            points = centers[hole]+displacement
            own_distance = np.sum(displacement**2, axis=1)
            for other, center in enumerate(centers):
                for du in range(-3, 4):
                    for dv in range(-3, 4):
                        if other == hole and du == dv == 0:
                            continue
                        shifted = center+a*np.array([du+dv/2, np.sqrt(3)*dv/2])
                        self.assertTrue(np.all(np.sum((points-shifted)**2, axis=1) > own_distance))
            expected = sign*np.where((maximum > .75)[:, None],
                                     normals[np.argmax(direction@normals.T, axis=1)], direction)
            np.testing.assert_allclose(periodic_normal(points, a, sides), expected, atol=1e-12)

    def test_reject_unsupported_sizes(self):
        a = 242.5
        b = a/np.sqrt(3)
        for multipliers in ([.99, .8], [1.36, .8], [1.2, .64], [1.2, 1.01], [3., 3.]):
            with self.assertRaises(ValueError):
                normal_projector_grid(a, np.array(multipliers)*b, 8)


if __name__ == "__main__":
    unittest.main()
