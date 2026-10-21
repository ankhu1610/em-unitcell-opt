"""Independent shape and energy sensitivities; validated solver is untouched.

Geometry units: micrometres. For the homothetic disk/triangle intersection,
Reynolds transport gives dF/dL = integral_boundary exp(-iq.r) (r.n/L) ds.
No finite differences enter hole_ft_derivative.
"""
import numpy as np
from invdes.interface_pwe import C


def hole_ft_derivative(q, side, up=True, order=64):
    q = np.asarray(q, dtype=float) * (1 if up else -1)
    r = side/(1.5*np.sqrt(3))
    h = side/(2*np.sqrt(3))
    s = np.sqrt(r*r-h*h)
    gamma = np.arccos(h/r)
    result = np.zeros(len(q), dtype=complex)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    for angle in [-np.pi/2, np.pi/6, 5*np.pi/6]:
        n = np.array([np.cos(angle), np.sin(angle)])
        t = np.array([-n[1], n[0]])
        qn, qt = q@n, q@t
        result += (h/side)*2*s*np.sinc(qt*s/np.pi)*np.exp(-1j*qn*h)
        lo, hi = angle+gamma, angle+2*np.pi/3-gamma
        theta = (hi+lo)/2+nodes*(hi-lo)/2
        normals = np.column_stack((np.cos(theta), np.sin(theta)))
        result += (r*r/side)*(hi-lo)/2*np.sum(
            np.exp(-1j*r*(q@normals.T))*weights, axis=1)
    return result


def eta_derivative(indices, a, sides, axis, order=64):
    if axis not in (0, 1):
        raise ValueError("axis must be L1=0 or L2=1")
    differences = indices[:, None, :]-indices[None, :, :]
    unique, inverse = np.unique(differences.reshape(-1, 2), axis=0, return_inverse=True)
    reciprocal = 2*np.pi/a*np.array([[1., -1/np.sqrt(3)], [0., 2/np.sqrt(3)]])
    q = unique@reciprocal
    center = [(a, a/np.sqrt(3)), (a/2, a/(2*np.sqrt(3)))][axis]
    coefficient = ((1-1/9)*hole_ft_derivative(q, sides[axis], axis == 1, order)
                   * np.exp(-1j*(q@center))/(np.sqrt(3)/2*a*a))
    return coefficient[inverse].reshape(len(indices), len(indices))


def energy_eigenvalue_derivative(g, k, vectors, b, delta, projectors,
                                deta, dprojectors=None):
    """Hellmann-Feynman energy derivative, including optional projector term.

    E=T(eps)=10I-9eta, B=E^-1, B'=9 B eta' B, Delta'=eta'-B'.
    W=diag(B,B)+P*diag(Delta,Delta)P. Avoid constructing W' explicitly.
    dprojectors=None gives the fixed-assignment/branch derivative. Supplying
    central projector secants is a diagnostic, not an analytic shape derivative.
    """
    x, y = (g+np.asarray(k)).T
    dx, dy = y[:, None]*vectors, -x[:, None]*vectors
    p, q, r = projectors
    ux, uy = p@dx+q@dy, q@dx+r@dy

    def quadratic(v, matrix):
        return np.sum(v.conj()*(matrix@v), axis=0).real

    value = quadratic(ux, deta)+quadratic(uy, deta)
    for v, sign in [(dx, 1), (dy, 1), (ux, -1), (uy, -1)]:
        value += sign*9*quadratic(b@v, deta)
    if dprojectors is not None:
        dp, dq, dr = dprojectors
        dux, duy = dp@dx+dq@dy, dq@dx+dr@dy
        value += 2*np.real(np.sum(dux.conj()*(delta@ux)+duy.conj()*(delta@uy), axis=0))
    return value


def frequency_derivative(values, eigenvalue_derivative):
    """Return GHz/um for eigenvalues in inverse metres squared."""
    if np.any(np.asarray(values) <= 0):
        raise ValueError("Positive eigenvalues away from Gamma required")
    return C/(4*np.pi*1e9*np.sqrt(values))*eigenvalue_derivative
