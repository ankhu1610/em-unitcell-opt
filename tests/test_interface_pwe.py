"""Standalone unittest checks: no torch, pytest, COMSOL, or learned data."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from invdes.interface_pwe import (
    nearest_normal, normal_projector_grid, projector_matrices,
    factorized_impermittivity, frequencies, hz_operator,
)


class InterfacePweTests(unittest.TestCase):
    def test_boundary_normals(self):
        side = 1.
        h = side/(2*np.sqrt(3))
        r = side/(1.5*np.sqrt(3))
        points = np.array([[0, -h], [r*np.cos(np.pi/6+np.pi/3),
                                     r*np.sin(np.pi/6+np.pi/3)]])
        d, n = nearest_normal(points, side)
        np.testing.assert_allclose(d, 0, atol=1e-28)
        np.testing.assert_allclose(n, [[0, -1], [0, 1]], atol=1e-14)
        _, down = nearest_normal(-points, side, False)
        np.testing.assert_allclose(down, -n, atol=1e-14)

    def test_projectors(self):
        ids = np.array([[0, 0], [1, 0], [0, 1], [-1, 0]])
        p = projector_matrices(ids, normal_projector_grid(1, [.55, .4], 64))
        np.testing.assert_allclose(p[0]+p[2], np.eye(4), atol=1e-14)
        for matrix in p:
            np.testing.assert_allclose(matrix, matrix.conj().T, atol=1e-14)

    def test_homogeneous(self):
        ids = np.array([[0, 0], [1, 0], [0, 1], [-1, 0]])
        p = projector_matrices(ids, normal_projector_grid(1, [.55, .4], 64))
        tensor = factorized_impermittivity(np.eye(4)/9, np.eye(4)*9, p)
        g = ids*2*np.pi
        k = np.array([.23, .41])
        f, checks = frequencies(g, [k], tensor, 4)
        expected = np.sort(299792458*np.linalg.norm(g+k, axis=1)/(6*np.pi*1e9))
        np.testing.assert_allclose(f[0], expected, rtol=1e-12)

    def test_flat_interface_inverse_rule(self):
        eta = np.array([[.6, .1], [.1, .6]])
        eps = np.array([[5., -.3], [-.3, 5.]])
        tensor = factorized_impermittivity(eta, eps,
                                           (np.eye(2), np.zeros((2, 2)), np.zeros((2, 2))))
        np.testing.assert_allclose(tensor[0], eta)
        np.testing.assert_allclose(tensor[3], np.linalg.inv(eps))
        np.testing.assert_allclose(tensor[1], 0)

    def test_complex_modes_rejected(self):
        with self.assertRaises(ValueError):
            frequencies(np.array([[1., 0.]]), [[0., 0.]],
                        (np.ones((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)),
                         np.array([[1+1j]])), 1)

    def test_complex_diagnostics_preserve_imaginary_part(self):
        f, checks = frequencies(
            np.array([[1., 0.]]), [[0., 0.]],
            (np.ones((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)),
             np.array([[1+1j]])), 1, diagnostic_complex=True)
        self.assertGreater(abs(f[0, 0].imag), 0)
        self.assertGreater(checks[0]["low_eigenvalue_relative_imaginary"], 0)

    def test_normal_grid_undersampling_rejected(self):
        with self.assertRaises(ValueError):
            projector_matrices(np.array([[-3, 0], [3, 0]]),
                               (np.ones((8, 8)), np.zeros((8, 8))))

    def test_layered_convergence_against_analytic_dispersion(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
        from audit_interface_pwe import layered_check
        runs = layered_check()["runs"]
        for old, new in zip(runs[:-1], runs[1:]):
            self.assertLess(new["factorized_max_relative_error_pct"],
                            old["factorized_max_relative_error_pct"])
        self.assertLess(runs[-1]["factorized_max_relative_error_pct"],
                        runs[-1]["direct_max_relative_error_pct"])


if __name__ == "__main__":
    unittest.main()
