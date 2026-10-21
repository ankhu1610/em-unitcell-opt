"""Direct Fourier quadrature of the fixed homothetic normal extension.

The two nearest-center Voronoi cells are equilateral triangles of area
sqrt(3)*a**2/4 and inradius a/(2*sqrt(3)). Their three edges bisect the nearest
opposite-center pairs. Same-sublattice bisectors touch only vertices.
They tile the primitive cell periodically.

Within each triangle, integrate along rays analytically and integrate angle
with Gauss-Legendre quadrature. Angular intervals split at both nearest-normal
sector boundaries and circle/chord transitions, so no discontinuity lies
inside an integration interval. No spatial raster or fitted constants.

This is the extension in homothetic_projector for the documented fixed-center,
fixed-clipping-ratio family, L1/b in [1,1.35], L2/b in [.65,1], b=a/sqrt(3).
Its Fourier coefficients do not depend on side lengths; derivatives with
respect to those lengths are zero. This does not assert finite-basis accuracy.
"""
import numpy as np


def radial_integral_factor(t):
    """Integral from 0 to 1 of s*exp(-i*t*s) ds, stable also near zero."""
    t = np.asarray(t, dtype=float)
    result = np.empty(t.shape, dtype=complex)
    small = np.abs(t) < .5
    z = -1j*t[small]
    term = np.ones_like(z, dtype=complex)
    value = term/2
    for power in range(1, 19):
        term *= z/power
        value += term/(power+2)
    result[small] = value
    ts = t[~small]
    result[~small] = (np.exp(-1j*ts)*(1+1j*ts)-1)/(ts*ts)
    return result


def _angular_rule(order):
    if not isinstance(order, (int, np.integer)) or order < 2:
        raise ValueError("order must be an integer at least 2")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    gamma = np.arccos(.75)
    directions, radii, weighted_dyads = [], [], []
    for angle in (-np.pi/2, np.pi/6, 5*np.pi/6):
        chord = np.array((np.cos(angle), np.sin(angle)))
        for lo, hi, is_chord in ((-np.pi/3, -gamma, False),
                                (-gamma, gamma, True),
                                (gamma, np.pi/3, False)):
            delta = (lo+hi)/2+(hi-lo)/2*nodes
            theta = angle+delta
            e = np.column_stack((np.cos(theta), np.sin(theta)))
            radius = 1/(2*np.sqrt(3)*np.cos(delta))  # a=1
            normal = np.broadcast_to(chord, e.shape) if is_chord else e
            dyads = np.column_stack((normal[:, 0]**2,
                                    normal[:, 0]*normal[:, 1], normal[:, 1]**2))
            directions.append(e)
            radii.append(radius)
            weighted_dyads.append(dyads*(weights*(hi-lo)/2*radius**2)[:, None])
    return np.concatenate(directions), np.concatenate(radii), np.concatenate(weighted_dyads)


def triangle_projector_integrals(q, a, order=64, *, chunk_size=256):
    """Unnormalized up-cell Fourier integrals [Pxx,Pxy,Pyy], units area.

    q has shape (N,2), in inverse units of a. The down-cell integrals for real
    q are the complex conjugates, by central inversion.
    """
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or q.shape[1] != 2 or not np.all(np.isfinite(q)):
        raise ValueError("q must be a finite (N,2) array")
    if not np.isfinite(a) or a <= 0:
        raise ValueError("a must be positive and finite")
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    directions, radii, weighted_dyads = _angular_rule(order)
    result = np.empty((len(q), 3), dtype=complex)
    for start in range(0, len(q), chunk_size):
        stop = min(start+chunk_size, len(q))
        t = ((q[start:stop]*a)@directions.T)*radii
        result[start:stop] = radial_integral_factor(t)@weighted_dyads*a*a
    return result


def projector_coefficients(differences, a, order=64, *, chunk_size=256):
    """Normalized periodic Fourier coefficients [Pxx,Pxy,Pyy]."""
    differences = np.asarray(differences)
    if (differences.ndim != 2 or differences.shape[1] != 2
            or not np.all(np.isfinite(differences))
            or not np.all(differences == np.rint(differences))):
        raise ValueError("differences must be integer (N,2) indices")
    if not np.isfinite(a) or a <= 0:
        raise ValueError("a must be positive and finite")
    basis = 2*np.pi/a*np.array(((1., -1/np.sqrt(3)), (0., 2/np.sqrt(3))))
    q = differences@basis
    up = triangle_projector_integrals(q, a, order, chunk_size=chunk_size)
    down_center = np.array((a, a/np.sqrt(3)))
    up_center = np.array((a/2, a/(2*np.sqrt(3))))
    return (up.conj()*np.exp(-1j*(q@down_center))[:, None]
            + up*np.exp(-1j*(q@up_center))[:, None])/(np.sqrt(3)/2*a*a)


def projector_matrices(indices, a, order=64):
    """Return Hermitian Toeplitz Pxx,Pxy,I-Pxx without a spatial FFT grid.

    Trace is imposed algebraically; raw Pyy coefficients remain accessible
    through projector_coefficients for independent quadrature diagnostics.
    """
    indices = np.asarray(indices)
    if (indices.ndim != 2 or indices.shape[1] != 2 or len(indices) == 0
            or not np.all(np.isfinite(indices))
            or not np.all(indices == np.rint(indices))):
        raise ValueError("indices must be a nonempty integer (N,2) array")
    diff = indices[:, None, :]-indices[None, :, :]
    unique, inverse = np.unique(diff.reshape(-1, 2), axis=0, return_inverse=True)
    coefficients = projector_coefficients(unique, a, order)
    xx = coefficients[inverse, 0].reshape(len(indices), len(indices))
    xy = coefficients[inverse, 1].reshape(len(indices), len(indices))
    return xx, xy, np.eye(len(indices))-xx
