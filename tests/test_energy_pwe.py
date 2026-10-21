import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
from pathlib import Path
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"scripts"))
from invdes.interface_pwe import (
    energy_factorized_impermittivity, factorized_impermittivity,
    normal_projector_grid, projector_matrices, real_frequencies, hz_operator,
)
from audit_pure_pwe import reciprocal_indices, reciprocal_basis, eta_matrix


class EnergyTests(unittest.TestCase):
    def setUp(self):
        self.ids = reciprocal_indices(2)
        self.g = self.ids@reciprocal_basis(1.)
        self.p = projector_matrices(self.ids, normal_projector_grid(1., [.57, .39], 64))
        self.eta = eta_matrix(self.ids, 1., [.57, .39])
        self.eps = 10*np.eye(len(self.ids))-9*self.eta

    def test_hermiticity_positive_and_general_eigenvalues(self):
        t = energy_factorized_impermittivity(self.eta, self.eps, self.p)
        k = np.array([.37, .23])
        matrix = hz_operator(self.g, k, t)
        np.testing.assert_allclose(matrix, matrix.conj().T, atol=1e-12)
        eigenvalues = np.linalg.eigvals(matrix)
        self.assertLess(np.max(abs(eigenvalues.imag)), 1e-10)
        self.assertGreater(eigenvalues.real.min(), 0)
        f, checks = real_frequencies(self.g, [k], t)
        np.testing.assert_allclose(f[0], np.sqrt(np.sort(eigenvalues.real)[:6])*299792458/(2*np.pi*1e9), rtol=1e-10)
        self.assertLess(checks[0]["max_eigenpair_residual_over_operator_scale"], 1e-12)

    def test_homogeneous(self):
        n = len(self.ids)
        t = energy_factorized_impermittivity(np.eye(n)/9, np.eye(n)*9, self.p)
        k = np.array([.3, .4])
        f, _ = real_frequencies(self.g, [k], t)
        expected = np.sort(np.linalg.norm(self.g+k, axis=1)/3)[:6]*299792458/(2*np.pi*1e9)
        np.testing.assert_allclose(f[0], expected, rtol=1e-12)

    def test_flat_interface_reduction(self):
        n = len(self.ids)
        p = (np.eye(n), np.zeros((n, n)), np.zeros((n, n)))
        old = factorized_impermittivity(self.eta, self.eps, p)
        new = energy_factorized_impermittivity(self.eta, self.eps, p)
        for x, y in zip(old, new):
            np.testing.assert_allclose(x, y, atol=1e-13)

    def test_gamma_and_time_reversal(self):
        t = energy_factorized_impermittivity(self.eta, self.eps, self.p)
        f, _ = real_frequencies(self.g, [[0, 0], [.3, .4], [-.3, -.4]], t)
        self.assertLess(f[0, 0], 1e-8)
        np.testing.assert_allclose(f[1], f[2], atol=1e-12)

    def test_rejects_nonhermitian_operator(self):
        n = len(self.ids)
        t = (np.eye(n), np.zeros((n, n)), np.zeros((n, n)), np.eye(n)*(1+1j))
        with self.assertRaises(ValueError):
            real_frequencies(self.g, [[.3, .4]], t)

    def test_rejects_original_ordered_product(self):
        original = factorized_impermittivity(self.eta, self.eps, self.p)
        with self.assertRaises(ValueError):
            real_frequencies(self.g, [[.3, .4]], original)

    def test_quadratic_energy_assembly(self):
        tensor = energy_factorized_impermittivity(self.eta, self.eps, self.p)
        n = len(self.ids)
        p, q, r = self.p
        projector = np.block([[p, q], [q, r]])
        b = np.linalg.solve(self.eps, np.eye(n))
        delta = self.eta-b
        zero = np.zeros_like(b)
        expected = np.block([[b, zero], [zero, b]]) + projector.conj().T @ np.block(
            [[delta, zero], [zero, delta]]) @ projector
        actual = np.block([[tensor[0], tensor[1]], [tensor[2], tensor[3]]])
        np.testing.assert_allclose(actual, expected, atol=1e-13)
        self.assertGreater(np.linalg.eigvalsh(actual)[0], 0)

    def test_rejects_negative_operator(self):
        n = len(self.ids)
        t = (-np.eye(n), np.zeros((n, n)), np.zeros((n, n)), -np.eye(n))
        with self.assertRaises(ValueError):
            real_frequencies(self.g, [[.3, .4]], t)

    def test_rejects_inconsistent_material(self):
        with self.assertRaises(ValueError):
            energy_factorized_impermittivity(-self.eta, self.eps, self.p)

    def test_analytic_layered_convergence(self):
        from audit_interface_pwe import layered_check
        runs = layered_check(energy_factorized_impermittivity, real_frequencies)["runs"]
        for old, new in zip(runs[:-1], runs[1:]):
            self.assertLess(new["factorized_max_relative_error_pct"],
                            old["factorized_max_relative_error_pct"])


if __name__ == "__main__":
    unittest.main()
