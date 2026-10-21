"""Separate conforming scalar Hz PWE and analytic material sensitivities.

A_GG' = eta(G-G') (k+G).(k+G'). No interface extension, fitted correction,
or material smoothing is introduced. Exact matrix assembly on nested Fourier
spaces has the Rayleigh--Ritz eigenvalue ordering; quadrature and numerical
eigensolves require independent checks. Derivative convergence does not follow
from a finite-basis Hellmann--Feynman check.
"""
import numpy as np
from scipy.linalg import eigh
from invdes.interface_pwe import C, _require_hermitian
from invdes.pwe_sensitivity import frequency_derivative


def operator(g, k, eta):
    """Assemble the SI Hz operator (or its derivative when eta is eta')."""
    g, k = np.asarray(g, dtype=np.float64), np.asarray(k, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.complex128)
    if g.ndim != 2 or g.shape[1] != 2 or k.shape != (2,):
        raise ValueError("Expected g=(N,2), k=(2,)")
    if eta.shape != (len(g), len(g)) or not np.all(np.isfinite(g+k)):
        raise ValueError("Invalid geometry/material matrix dimensions or values")
    _require_hermitian(eta, "variational eta")
    kg = g+k
    return eta*(kg@kg.T)


def checked_modes(matrix, n_bands=2):
    """Lowest positive modes plus one neighboring eigenvalue for separation."""
    matrix = np.asarray(matrix, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Operator must be square")
    if not isinstance(n_bands, int) or not 1 <= n_bands < len(matrix):
        raise ValueError("Require at least one additional mode for separation")
    defect = _require_hermitian(matrix, "variational Hz operator")
    values, vectors = eigh(matrix, subset_by_index=[0, n_bands],
                           driver="evr", check_finite=False)
    if values[0] <= 0:
        raise ValueError("Positive modes away from Gamma required")
    residual = np.linalg.norm(matrix@vectors-vectors*values, axis=0)/np.abs(values)
    if np.max(residual) > 1e-7:
        raise ValueError(f"Eigenpair residual failed: {np.max(residual)}")
    differences = np.diff(values)
    separation = np.array([
        min(differences[max(0, i-1):i+1])/values[i] for i in range(n_bands)])
    low = values[:n_bands]
    return (C*np.sqrt(low)/(2*np.pi*1e9), low, vectors[:, :n_bands],
            {"hermitian_defect": defect, "max_relative_eigenpair_residual": float(np.max(residual)),
             "relative_eigenvalue_separations": separation.tolist()})


def mode_derivative(g, k, values, vectors, deta):
    """GHz/um when deta is the material-matrix derivative per micrometre."""
    da = operator(g, k, deta)
    dlambda = np.sum(vectors.conj()*(da@vectors), axis=0).real
    return frequency_derivative(values, dlambda)


def spectrum(g, kpoints, eta, derivatives=()):
    """Checked first-two-band spectrum and optional analytic sensitivities."""
    frequencies, sensitivities, diagnostics = [], [], []
    for k in kpoints:
        f, values, vectors, diagnostic = checked_modes(operator(g, k, eta))
        frequencies.append(f)
        sensitivities.append([mode_derivative(g, k, values, vectors, de) for de in derivatives])
        diagnostics.append(diagnostic)
    gradients = np.asarray(sensitivities).transpose(1, 0, 2) if len(derivatives) else np.empty((0, len(kpoints), 2))
    return np.array(frequencies), gradients, diagnostics
