"""Separate, tolerance-stable nearest-interface normal extension.

No material coefficients or eigensolver are changed. Ties select the first
candidate within a fixed squared-distance tolerance of the true global minimum,
at both primitive and periodic-image levels. The default tolerance is
64 * float64 epsilon * a**2, independent of perturbation steps or spectra.
This removes roundoff tie flips, not genuine moving nearest-interface switches.
Corner normals are nonunique; candidate ordering defines an extension convention.
"""
import numpy as np

DEFAULT_TIE_ULPS = 64
_SQRT3 = np.sqrt(3.)
_RADIUS = 2 / (3 * _SQRT3)
_HEIGHT = 1 / (2 * _SQRT3)
_HALF_CHORD = np.sqrt(7.) / (6 * _SQRT3)
_GAMMA = np.arccos(.75)
_ANGLES = (-np.pi/2, np.pi/6, 5*np.pi/6)
_NORMALS = ((0., -1.), (_SQRT3/2, .5), (-_SQRT3/2, .5))


def _select(candidates, shape, tolerance):
    """Two passes prevent accumulated tolerance-chain selection errors."""
    best = np.full(shape, np.inf)
    for distance, _ in candidates():
        best = np.minimum(best, distance)
    selected = np.zeros(shape, dtype=bool)
    normal = np.zeros(shape + (2,))
    for distance, candidate in candidates():
        take = (~selected) & (distance <= best + tolerance)
        normal[take] = np.broadcast_to(candidate, normal.shape)[take]
        selected |= take
    if not np.all(selected):
        raise ValueError("No finite nearest-interface candidate")
    return best, normal


def nearest_normal(points, side, up=True, *, tie_tolerance):
    """Return true minimum squared distance and deterministically chosen normal.

    ``tie_tolerance`` has units of length squared. The selected normal may
    correspond to a candidate slightly farther than the returned true minimum.
    With fixed centers, each normal branch is independent of side length;
    branch boundaries still move, so this is not a globally smooth map.
    """
    points = np.asarray(points, dtype=float)
    if points.shape[-1:] != (2,) or not np.all(np.isfinite(points)):
        raise ValueError("points must be finite with final dimension 2")
    if not np.isfinite(side) or side <= 0:
        raise ValueError("side must be positive and finite")
    if not np.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError("tie_tolerance must be finite and nonnegative")
    sign = 1 if up else -1
    p = points * sign
    r, h, s = side*_RADIUS, side*_HEIGHT, side*_HALF_CHORD

    def candidates():
        for angle, components in zip(_ANGLES, _NORMALS):
            n = np.array(components)
            t = np.array([-n[1], n[0]])
            closest = h*n + np.clip(p@t, -s, s)[..., None]*t
            yield np.sum((p-closest)**2, axis=-1), n
            lo, hi = angle+_GAMMA, angle+2*np.pi/3-_GAMMA
            mid = (lo+hi)/2
            theta = mid + (np.arctan2(p[..., 1], p[..., 0])-mid+np.pi) % (2*np.pi)-np.pi
            theta = np.clip(theta, lo, hi)
            arc = np.stack((np.cos(theta), np.sin(theta)), axis=-1)
            yield np.sum((p-r*arc)**2, axis=-1), arc

    distance, normal = _select(candidates, p.shape[:-1], tie_tolerance)
    return distance, sign*normal


def periodic_nearest_normal(points, a, sides, *, tie_ulps=DEFAULT_TIE_ULPS):
    """Pointwise periodic extension for the documented two-hole geometry.

    Points are folded into the primitive cell before image enumeration, making
    tie ordering periodic. The 3x3 image stencil covers the documented size
    range; this helper does not claim validity for arbitrary large inclusions.
    """
    points = np.asarray(points, dtype=float)
    sides = np.asarray(sides, dtype=float)
    if not np.isfinite(a) or a <= 0:
        raise ValueError("a must be positive and finite")
    if sides.shape != (2,) or np.any(~np.isfinite(sides)) or np.any(sides <= 0):
        raise ValueError("sides must contain two positive finite lengths")
    if not np.isfinite(tie_ulps) or tie_ulps < 0:
        raise ValueError("tie_ulps must be finite and nonnegative")
    if points.shape[-1:] != (2,) or not np.all(np.isfinite(points)):
        raise ValueError("points must be finite with final dimension 2")
    v = 2*points[..., 1]/(_SQRT3*a)
    u = points[..., 0]/a-v/2
    u, v = u % 1, v % 1
    folded = np.stack((a*(u+v/2), a*_SQRT3*v/2), axis=-1)
    tolerance = tie_ulps*np.finfo(float).eps*a*a

    def candidates():
        for side, center, up in zip(
                sides, ((a, a/_SQRT3), (a/2, a/(2*_SQRT3))), (False, True)):
            for du in (-1, 0, 1):
                for dv in (-1, 0, 1):
                    shift = np.array(center)+a*np.array((du+dv/2, _SQRT3*dv/2))
                    yield nearest_normal(folded-shift, side, up, tie_tolerance=tolerance)

    return _select(candidates, folded.shape[:-1], tolerance)


def normal_projector_grid(a, sides, grid, *, tie_ulps=DEFAULT_TIE_ULPS):
    """Return Pxx and Pxy samples, compatible with existing projector assembly."""
    if not isinstance(grid, (int, np.integer)) or grid <= 0:
        raise ValueError("grid must be a positive integer")
    u, v = np.meshgrid(np.arange(grid)/grid, np.arange(grid)/grid, indexing="ij")
    points = np.stack((a*(u+v/2), a*_SQRT3*v/2), axis=-1)
    _, normal = periodic_nearest_normal(points, a, sides, tie_ulps=tie_ulps)
    return normal[..., 0]**2, normal[..., 0]*normal[..., 1]
