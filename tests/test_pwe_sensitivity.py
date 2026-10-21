import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.linalg import eigh
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from audit_pure_pwe import hole_ft, eta_matrix, reciprocal_indices, reciprocal_basis
from validate_energy_pwe_full import path_coordinates
from invdes.interface_pwe import (
    normal_projector_grid, projector_matrices, energy_factorized_impermittivity, hz_operator)
from invdes.pwe_sensitivity import (
    hole_ft_derivative, eta_derivative, energy_eigenvalue_derivative)


class SensitivityTests(unittest.TestCase):
    def test_area_derivative_and_conjugacy(self):
        side = 150.
        q = np.array([[0., 0.], [.1, .2], [-.3, .1]])
        for up in [True, False]:
            derivative = hole_ft_derivative(q, side, up)
            self.assertAlmostEqual(derivative[0].real, 2*hole_ft(q[:1], side, up)[0].real/side, places=10)
            np.testing.assert_allclose(hole_ft_derivative(-q, side, up), derivative.conj(), atol=1e-12)

    def test_shape_derivative_vs_independent_boundary_formula(self):
        q = np.array([[0., 0.], [.1, .2], [-.3, .1]])
        h = 1e-4
        for up in [True, False]:
            fd = (hole_ft(q, 150+h, up)-hole_ft(q, 150-h, up))/(2*h)
            np.testing.assert_allclose(hole_ft_derivative(q, 150., up), fd, rtol=1e-7, atol=1e-7)

    def test_material_both_lengths(self):
        ids, sides, h = reciprocal_indices(2), np.array([161., 112.]), 1e-4
        for axis in (0, 1):
            shift = np.eye(2)[axis]*h
            fd = (eta_matrix(ids, 242.5, sides+shift)-eta_matrix(ids, 242.5, sides-shift))/(2*h)
            analytic = eta_derivative(ids, 242.5, sides, axis)
            np.testing.assert_allclose(analytic, fd, rtol=2e-6, atol=1e-10)
            np.testing.assert_allclose(analytic, analytic.conj().T, atol=1e-12)

    def test_energy_chain_including_projector_derivative(self):
        ids, sides, a = reciprocal_indices(2), np.array([161., 112.]), 242.5
        eta = eta_matrix(ids, a, sides)
        identity = np.eye(len(ids))
        eps = 10*identity-9*eta
        b = np.linalg.inv(eps)
        projectors = projector_matrices(ids, normal_projector_grid(a, sides, 64))
        tensor = energy_factorized_impermittivity(eta, eps, projectors)
        g, k = ids@reciprocal_basis(a*1e-6), path_coordinates([.37], a*1e-6)[0]
        values, vectors = eigh(hz_operator(g, k, tensor), subset_by_index=[0, 1])
        deta = eta_derivative(ids, a, sides, 0)
        # A controlled smooth projector perturbation validates all product terms.
        dp = (identity*.001, identity*.002, -identity*.001)
        for dproj in [None, dp]:
            h = 1e-4
            matrices = []
            for sign in [-1, 1]:
                p = projectors if dproj is None else tuple(x+sign*h*y for x, y in zip(projectors, dproj))
                e = eta+sign*h*deta
                matrices.append(hz_operator(g, k, energy_factorized_impermittivity(e, 10*identity-9*e, p)))
            operator_fd = (matrices[1]-matrices[0])/(2*h)
            expected = np.sum(vectors.conj()*(operator_fd@vectors), axis=0).real
            got = energy_eigenvalue_derivative(g, k, vectors, b, eta-b, projectors, deta, dproj)
            np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
