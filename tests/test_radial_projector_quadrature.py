"""Small-basis direct-integration checks; no spectral reference data."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from invdes.radial_projector_quadrature import (
    radial_integral_factor, triangle_projector_integrals,
    projector_coefficients, projector_matrices,
)
from invdes.homothetic_projector import normal_projector_grid
from invdes.interface_pwe import projector_matrices as fft_projector_matrices


def indices(cutoff):
    return np.array([(i, j) for i in range(-cutoff, cutoff+1)
                     for j in range(-cutoff, cutoff+1)])


class RadialProjectorQuadratureTests(unittest.TestCase):
    def test_radial_integral_against_independent_quadrature(self):
        t = np.array([0., 1e-10, -.49, .5, 2., -17., 55.])
        nodes, weights = np.polynomial.legendre.leggauss(128)
        s = (nodes+1)/2
        expected = np.exp(-1j*t[:, None]*s)@(weights*s/2)
        np.testing.assert_allclose(radial_integral_factor(t), expected, atol=4e-15)

    def test_zero_area_and_raw_trace(self):
        a = 242.5
        integrals = triangle_projector_integrals(np.zeros((1, 2)), a, 64)[0]
        cell_area = np.sqrt(3)*a*a/4
        np.testing.assert_allclose(integrals, [cell_area/2, 0, cell_area/2], atol=2e-11)
        diff = indices(4)
        coefficients = projector_coefficients(diff, a, 64)
        expected = np.all(diff == 0, axis=1).astype(float)
        np.testing.assert_allclose(coefficients[:, 0]+coefficients[:, 2], expected, atol=3e-14)
        np.testing.assert_allclose(projector_coefficients([[0, 0]], a)[0],
                                   [.5, 0, .5], atol=1e-14)

    def test_conjugacy_hermiticity_scaling(self):
        diff = indices(3)
        c = projector_coefficients(diff, 242.5)
        np.testing.assert_allclose(projector_coefficients(-diff, 242.5), c.conj(), atol=2e-15)
        np.testing.assert_allclose(projector_coefficients(diff, 242.5e-6), c, atol=2e-14)
        p = projector_matrices(indices(2), 242.5)
        for matrix in p:
            np.testing.assert_allclose(matrix, matrix.conj().T, atol=2e-15)
        np.testing.assert_array_equal(p[0]+p[2], np.eye(len(p[0])))

    def test_angular_convergence(self):
        diff = indices(4)
        c32, c64, c128, c256 = [projector_coefficients(diff, 242.5, order)
                              for order in (32, 64, 128, 256)]
        np.testing.assert_allclose(c64, c128, atol=2e-13)
        np.testing.assert_allclose(c128, c256, atol=2e-13)
        self.assertLess(np.max(abs(c64-c256)), np.max(abs(c32-c256))+2e-14)

    def test_spatial_fft_approaches_direct_integrals(self):
        a = 242.5
        ids = indices(2)
        direct = projector_matrices(ids, a, 128)
        errors = []
        for grid in (128, 512):
            samples = normal_projector_grid(a, a/np.sqrt(3)*np.array([1.15, .8]), grid)
            fft = fft_projector_matrices(ids, samples)
            errors.append(max(np.max(abs(x-y)) for x, y in zip(fft, direct)))
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[1], .003)


if __name__ == "__main__":
    unittest.main()
