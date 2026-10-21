"""Normal-vector Fourier factorization for lossless 2D Hz PWE.

Implements Model B, Eq. (33), Antos & Veis, Optics Express 18 (2010),
https://arxiv.org/abs/1005.4219 . No learned quantities or reference spectra.
Material convolution matrices are supplied by the caller. Geometry-only
normal projectors are sampled separately from the material coefficients.

The original factorized_impermittivity retains the published non-Hermitian
finite product for comparison. energy_factorized_impermittivity implements
an explicitly different, Hermitian energy discretization. Neither supplies
a continuum error bound.
"""
import numpy as np

C = 299792458.0


def nearest_normal(points, side, up=True):
    """Closest boundary normal of a centred disk/triangle intersection.

    Exact closest points on the three finite chords and three circular arcs.
    Ties at corners/medial axes select the first primitive deterministically.
    """
    sign = 1 if up else -1
    p = np.asarray(points, dtype=float) * sign
    r = side / (1.5*np.sqrt(3))
    h = side / (2*np.sqrt(3))
    s = np.sqrt(r*r-h*h)
    gamma = np.arccos(h/r)
    best = np.full(p.shape[:-1], np.inf)
    normal = np.zeros_like(p)
    for angle in [-np.pi/2, np.pi/6, 5*np.pi/6]:
        n = np.array([np.cos(angle), np.sin(angle)])
        t = np.array([-n[1], n[0]])
        closest = h*n + np.clip(p @ t, -s, s)[..., None]*t
        d = np.sum((p-closest)**2, axis=-1)
        take = d < best
        normal[take] = n
        best = np.minimum(best, d)
        lo, hi = angle+gamma, angle+2*np.pi/3-gamma
        mid = (lo+hi)/2
        theta = mid + (np.arctan2(p[..., 1], p[..., 0])-mid+np.pi) % (2*np.pi)-np.pi
        theta = np.clip(theta, lo, hi)
        n_arc = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
        d = np.sum((p-r*n_arc)**2, axis=-1)
        take = d < best
        normal[take] = n_arc[take]
        best = np.minimum(best, d)
    return best, sign*normal


def normal_projector_grid(a, sides, grid):
    """Periodic nearest-interface extension; input lengths in common units.

    Normals are correct on each smooth boundary piece. The extension has
    medial-axis discontinuities; grid and basis refinement are both required.
    """
    u, v = np.meshgrid(np.arange(grid)/grid, np.arange(grid)/grid, indexing="ij")
    points = np.stack([a*(u+v/2), a*np.sqrt(3)*v/2], axis=-1)
    best = np.full((grid, grid), np.inf)
    normal = np.zeros_like(points)
    for side, center, up in zip(
            sides, [(a, a/np.sqrt(3)), (a/2, a/(2*np.sqrt(3)))], [False, True]):
        for du in [-1, 0, 1]:
            for dv in [-1, 0, 1]:
                shift = np.array(center) + a*np.array([du+dv/2, np.sqrt(3)*dv/2])
                distance, candidate = nearest_normal(points-shift, side, up)
                take = distance < best
                normal[take] = candidate[take]
                best = np.minimum(best, distance)
    return normal[..., 0]**2, normal[..., 0]*normal[..., 1]


def projector_matrices(indices, samples):
    """Toeplitz matrices Pxx, Pxy, Pyy; Pxx + Pyy = I exactly."""
    grid = samples[0].shape[0]
    diff = indices[:, None, :] - indices[None, :, :]
    if grid <= 2*np.max(np.abs(diff)):
        raise ValueError("Normal-field grid too small for requested Fourier differences")
    matrices = []
    for values in samples:
        coeff = np.fft.fft2(values)/grid**2
        matrices.append(coeff[diff[..., 0] % grid, diff[..., 1] % grid])
    return matrices[0], matrices[1], np.eye(len(indices))-matrices[0]


def factorized_impermittivity(eta, epsilon, projectors):
    """Eq. (33), preserving the published matrix multiplication order."""
    identity = np.eye(len(eta))
    inv_epsilon = np.linalg.solve(epsilon, identity)
    pxx, pxy, pyy = projectors
    delta = eta-inv_epsilon
    xx = inv_epsilon + delta @ pxx
    xy = delta @ pxy
    yy = inv_epsilon + delta @ pyy
    return xx, xy, xy, yy


def hz_operator(g, k, tensor):
    xx, xy, yx, yy = tensor
    x, y = (g+np.asarray(k)).T
    return (y[:, None]*xx*y[None, :] - y[:, None]*xy*x[None, :]
            - x[:, None]*yx*y[None, :] + x[:, None]*yy*x[None, :])


def _require_hermitian(matrix, name):
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains nonfinite values")
    relative = np.linalg.norm(matrix-matrix.conj().T)/max(np.linalg.norm(matrix), 1e-300)
    if relative > 1e-12:
        raise ValueError(f"{name} is not Hermitian: relative defect={relative}")
    return float(relative)


def energy_factorized_impermittivity(eta, epsilon, projectors):
    """Positive-energy normal-projector variant, NOT Eq. (33) itself.

    B = inverse(T(epsilon)), Delta = T(1/epsilon)-B, P = T(n n^T).
    Define W(D) = D* diag(B,B) D + (P D)* diag(Delta,Delta) (P D).
    Its matrix is B_block + P* Delta_block P. For positive scalar material,
    Delta >= 0 by the compression/inverse inequality, hence W is positive.
    This guarantees a Hermitian positive operator in exact arithmetic.

    For a constant normal this reduces to the normal/tangential inverse rule.
    For homogeneous material Delta=0, so truncated projectors cannot distort
    the analytic spectrum. For variable normals this is a distinct finite
    discretization: P^2 != P. Accuracy MUST be assessed by refinement.
    No eigenvalue clipping, spectral fitting, or post-hoc operator averaging.
    """
    for name, matrix in zip(("eta", "epsilon", "Pxx", "Pxy", "Pyy"),
                            (eta, epsilon, *projectors)):
        _require_hermitian(matrix, name)
    np.linalg.cholesky(epsilon)
    b = np.linalg.solve(epsilon, np.eye(len(epsilon)))
    delta = eta-b
    # Detect incompatible supplied material matrices; tiny negative roundoff
    # is permitted, never projected out of Delta.
    eigenvalues = np.linalg.eigvalsh(delta)
    if eigenvalues[0] < -1e-11*max(np.linalg.norm(eta), 1.):
        raise ValueError("Material matrices violate T(1/epsilon) >= inverse(T(epsilon))")
    p, q, r = projectors
    dp, dq, dr = delta@p, delta@q, delta@r
    xx = b + p@dp + q@dq
    xy = p@dq + q@dr
    yx = q@dp + r@dq
    yy = b + q@dq + r@dr
    return xx, xy, yx, yy


def real_frequencies(g, kpoints, tensor, n_bands=6):
    """Checked Hermitian eigensolve with residuals; invalid matrices raise.

    Only negative eigenvalues within roundoff relative to the full operator
    spectrum may be rounded to zero (the physical Gamma zero mode).
    """
    results, diagnostics = [], []
    for k in kpoints:
        operator = hz_operator(g, k, tensor)
        defect = _require_hermitian(operator, "Hz operator")
        values, vectors = np.linalg.eigh(operator)
        scale = max(float(np.max(abs(values))), 1.)
        if values[0] < -1e-12*scale:
            raise ValueError(f"Negative squared frequency: {values[0]}")
        low, v = values[:n_bands], vectors[:, :n_bands]
        residual = np.linalg.norm(operator@v-v*low, axis=0)/scale
        results.append(C*np.sqrt(np.maximum(low, 0))/(2*np.pi*1e9))
        diagnostics.append({
            "relative_nonhermiticity": defect,
            "max_eigenpair_residual_over_operator_scale": float(np.max(residual)),
            "minimum_eigenvalue": float(values[0]),
            "roundoff_negative_modes": int(np.sum(low < 0)),
        })
    return np.array(results), diagnostics


def frequencies(g, kpoints, tensor, n_bands=6, diagnostic_complex=False):
    """Return frequencies plus diagnostics, rejecting nonphysical low modes.

    diagnostic_complex=True is for refinement experiments ONLY: it returns
    complex frequencies unchanged, not validated physical predictions.
    """
    result, checks = [], []
    for k in kpoints:
        operator = hz_operator(g, k, tensor)
        vals = np.linalg.eigvals(operator)
        vals = vals[np.argsort(vals.real)][:n_bands]
        scale = max(float(np.max(np.abs(vals))), 1.)
        imag = float(np.max(np.abs(vals.imag))/scale)
        if not diagnostic_complex and (imag > 1e-8 or np.min(vals.real) < -1e-10*scale):
            raise ValueError(f"Nonphysical low eigenvalues: {vals}")
        result.append(C*np.sqrt(vals if diagnostic_complex else np.maximum(vals.real, 0))/(2*np.pi*1e9))
        checks.append({
            "low_eigenvalue_relative_imaginary": imag,
            "operator_relative_nonhermiticity":
                float(np.linalg.norm(operator-operator.conj().T)/np.linalg.norm(operator)),
        })
    return np.array(result), checks
