"""Independent area-integral tests for the supported clipped-triangle geometry.

Replaces tests of the removed rectangular/single-triangle API. The original
test is preserved in docs/legacy_tests/, not silently declared passing.
"""
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_pure_pwe import hole_ft, eta_matrix, reciprocal_indices, reciprocal_basis


def area_integral(q, side, up):
    """Independent polar area quadrature, not the solver's boundary integral.

    Triangle half-planes limit a ray to h/max(n.dot(ray)); the disk limits
    it to R. Integrate exp(-i q.r)*r dr dtheta over that intersection.
    Work in side-normalized coordinates to keep integration scales moderate.
    """
    normals_at = np.array([-np.pi / 2, np.pi / 6, 5 * np.pi / 6])
    normals = np.column_stack((np.cos(normals_at), np.sin(normals_at)))
    radius, apothem = 2 / (3 * np.sqrt(3)), 1 / (2 * np.sqrt(3))
    switch = np.arccos(apothem / radius)
    breaks = np.unique(np.r_[0., 2 * np.pi,
                            np.mod(normals_at - switch, 2 * np.pi),
                            np.mod(normals_at + switch, 2 * np.pi)])
    angular_nodes, angular_weights = np.polynomial.legendre.leggauss(48)
    radial_nodes, radial_weights = np.polynomial.legendre.leggauss(64)
    q = np.atleast_2d(q) * side * (1 if up else -1)
    result = np.zeros(len(q), dtype=np.complex128)
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        theta = (hi + lo) / 2 + (hi - lo) / 2 * angular_nodes
        directions = np.column_stack((np.cos(theta), np.sin(theta)))
        bounds = np.minimum(radius, apothem / np.max(directions @ normals.T, axis=1))
        radii = bounds[:, None] * (radial_nodes + 1) / 2
        weights = ((hi - lo) / 2 * angular_weights[:, None]
                   * bounds[:, None] / 2 * radial_weights * radii)
        phase = (q @ directions.T)[:, :, None] * radii[None, :, :]
        result += np.sum(np.exp(-1j * phase) * weights, axis=(1, 2))
    return side**2 * result


@pytest.mark.parametrize("side", [145., 160., 185.])
@pytest.mark.parametrize("up", [False, True])
def test_boundary_fourier_vs_independent_area_quadrature(side, up):
    q = np.array([[0., 0.], [.001, 0.], [0., .002], [.015, .03],
                  [-.025, .015], [.05, -.04], [.12, .08]])
    expected = area_integral(q, side, up)
    actual = hole_ft(q, side, up, order=64)
    # Retains the former 1e-6 relative requirement; no contour/COMSOL fitting.
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(hole_ft(-q, side, up), actual.conj(), atol=1e-9)


def test_material_matrix_phase_normalization_and_hexagonal_lattice():
    spec = yaml.safe_load((ROOT / "configs/geometry_spec.yaml").read_text())
    a = spec["a_um"]
    geometry = spec["structures"]["para"]
    assert geometry["background"]["eps_r"] == 9
    assert [v["eps_r"] for v in geometry["inclusions"]] == [1, 1]
    ids = reciprocal_indices(1)
    # Independent reciprocal construction from real-space primitive vectors.
    direct = np.array([[a, 0.], [a / 2, np.sqrt(3) * a / 2]])
    reciprocal = 2 * np.pi * np.linalg.inv(direct).T
    np.testing.assert_allclose(reciprocal_basis(a), reciprocal, atol=1e-15)
    differences = ids[:, None, :] - ids[None, :, :]
    unique, inverse = np.unique(differences.reshape(-1, 2), axis=0, return_inverse=True)
    q = unique @ reciprocal
    sides = [160., 115.]
    # Ideal primitive-cell centres used by this evaluator; rounded YAML centres
    # are provenance labels, not a higher-precision alternative geometry.
    centers = np.array([[a, a / np.sqrt(3)], [a / 2, a / (2 * np.sqrt(3))]])
    coefficients = np.zeros(len(q), dtype=np.complex128)
    for side, up, center in zip(sides, [False, True], centers):
        coefficients += area_integral(q, side, up) * np.exp(-1j * (q @ center))
    coefficients *= (1 - 1 / 9) / abs(np.linalg.det(direct))
    coefficients[np.all(unique == 0, axis=1)] += 1 / 9
    expected = coefficients[inverse].reshape(len(ids), len(ids))
    actual = eta_matrix(ids, a, sides)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(actual, actual.conj().T, atol=1e-13)
