"""Analytic energy-PWE sensitivities versus full-rebuild finite differences.

This validates the supported fixed radial-projector model, not autograd through
the historical raster mask. No saved physical results or tolerances are changed.
"""
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_pure_pwe import C, eta_matrix, reciprocal_indices, reciprocal_basis, path_coordinates
from invdes.interface_pwe import energy_factorized_impermittivity, hz_operator
from invdes.radial_projector_quadrature import projector_matrices
from invdes.pwe_sensitivity import eta_derivative, energy_eigenvalue_derivative, frequency_derivative


@pytest.mark.parametrize("multipliers", [[1.05, .70], [1.15, .82], [1.30, .95]])
@pytest.mark.parametrize("path_parameter", [.37, 1.43, 2.61])
def test_analytic_band_gradients_vs_full_rebuild(multipliers, path_parameter):
    spec = yaml.safe_load((ROOT / "configs/geometry_spec.yaml").read_text())
    assert spec["structures"]["para"]["background"]["eps_r"] == 9
    assert all(v["eps_r"] == 1 for v in spec["structures"]["para"]["inclusions"])
    a = spec["a_um"]
    sides = a / np.sqrt(3) * np.array(multipliers)
    ids = reciprocal_indices(3)  # Small CPU basis; not a cutoff-convergence claim.
    g = ids @ reciprocal_basis(a * 1e-6)
    k = path_coordinates([path_parameter], a * 1e-6)[0]
    projectors = projector_matrices(ids, a, 64)
    identity = np.eye(len(ids))

    def rebuild(lengths):
        eta = eta_matrix(ids, a, lengths, order=64)
        eps = 10 * identity - 9 * eta
        tensor = energy_factorized_impermittivity(eta, eps, projectors)
        operator = hz_operator(g, k, tensor)
        np.testing.assert_allclose(operator, operator.conj().T, rtol=1e-12, atol=1e-5)
        values, vectors = eigh(operator, subset_by_index=[0, 2])
        assert np.all(values > 0)
        assert np.all(np.diff(values) / values[1:] > 1e-6), "Use simple eigenmodes for derivative comparisons"
        return values, vectors, eta, np.linalg.inv(eps)

    values, vectors, eta, b = rebuild(sides)
    for axis in (0, 1):
        deta = eta_derivative(ids, a, sides, axis, order=64)
        dvalue = energy_eigenvalue_derivative(
            g, k, vectors[:, :2], b, eta - b, projectors, deta)
        analytic = frequency_derivative(values[:2], dvalue)
        assert np.all(np.isfinite(analytic))
        for h in (1e-3, 3e-4):  # um; independent perturbations of each length.
            step = np.eye(2)[axis] * h
            plus = rebuild(sides + step)[0][:2]
            minus = rebuild(sides - step)[0][:2]
            fd = C / (2 * np.pi * 1e9) * (np.sqrt(plus) - np.sqrt(minus)) / (2 * h)
            # Stricter than the old 1e-3 test; matches the nominal 1e-4
            # gradient criterion away from degeneracies. Small absolute floor
            # handles vanishing derivatives without silently setting errors to zero.
            np.testing.assert_allclose(analytic, fd, rtol=1e-4, atol=1e-7)


def test_zero_frequency_derivative_is_rejected_not_silently_zeroed():
    # Gamma zero modes are outside the square-root derivative's domain.
    with pytest.raises(ValueError, match="Positive eigenvalues"):
        frequency_derivative(np.array([0., 1.]), np.ones(2))
