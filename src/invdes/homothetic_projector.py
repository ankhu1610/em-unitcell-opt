"""Experimental fixed-domain normal extension for the documented two holes.

Nearest-center Voronoi domains and radial chord/arc sectors depend only on
fixed centers and shape ratios, never on L1 or L2. On each smooth interface,
the radial construction gives the actual boundary normal for every supported
size. Domain and corner-sector discontinuities remain spatially fixed.

Containment argument: the three nearest opposite centers are distance
a/sqrt(3) away, aligned with the hole's outward chord normals. The corresponding
Voronoi half-plane bounds are a/(2sqrt(3)), whereas the largest triangle
support is L/(2sqrt(3)) < a/(2sqrt(3)). Other centers are at least a away;
the largest clipping radius is 0.3*a < a/2. Thus each supported hole lies
strictly within its own nearest-center domain. Sizes are checked against the
documented ranges; this is not a generic geometry implementation.

This supplies a geometry-only extension, not an accuracy guarantee for a
finite Fourier discretization. No reference spectra or fitted parameters.
"""
import numpy as np

DEFAULT_TIE_ULPS = 64
_SQRT3 = np.sqrt(3.)
_NORMALS = np.array(((0., -1.), (_SQRT3/2, .5), (-_SQRT3/2, .5)))


def _validate(a, sides, tie_ulps):
    sides = np.asarray(sides, dtype=float)
    if not np.isfinite(a) or a <= 0:
        raise ValueError("a must be positive and finite")
    if sides.shape != (2,) or np.any(~np.isfinite(sides)):
        raise ValueError("sides must contain two finite lengths")
    if not np.isfinite(tie_ulps) or tie_ulps < 0:
        raise ValueError("tie_ulps must be finite and nonnegative")
    # These are saved control ranges, not fitted limits. Permit roundoff only.
    lo = a/_SQRT3*np.array((1., .65))
    hi = a/_SQRT3*np.array((1.35, 1.))
    allowance = 64*np.finfo(float).eps*a
    if np.any(sides < lo-allowance) or np.any(sides > hi+allowance):
        raise ValueError("Sizes outside documented L1=[1,1.35]b, L2=[.65,1]b, b=a/sqrt(3)")
    return sides


def radial_normal(displacement, up=True):
    """Boundary normal at the ray/shape intersection, independent of side.

    Circle/chord transitions have nonunique normals; equality selects the
    radial normal. The center itself uses a fixed unit direction.
    """
    p = np.asarray(displacement, dtype=float)
    if p.shape[-1:] != (2,) or not np.all(np.isfinite(p)):
        raise ValueError("displacement must be finite with final dimension 2")
    sign = 1 if up else -1
    p = sign*p
    length = np.linalg.norm(p, axis=-1)
    direction = np.divide(p, length[..., None], out=np.zeros_like(p),
                          where=length[..., None] > 0)
    direction = np.where((length == 0)[..., None], np.array((0., 1.)), direction)
    dots = direction@_NORMALS.T
    which = np.argmax(dots, axis=-1)
    maximum = np.max(dots, axis=-1)
    normal = np.where((maximum > .75)[..., None], _NORMALS[which], direction)
    return sign*normal


def periodic_normal(points, a, sides, *, tie_ulps=DEFAULT_TIE_ULPS):
    """Return nearest-center radial normals with periodic first-candidate ties."""
    _validate(a, sides, tie_ulps)
    points = np.asarray(points, dtype=float)
    if points.shape[-1:] != (2,) or not np.all(np.isfinite(points)):
        raise ValueError("points must be finite with final dimension 2")
    v = 2*points[..., 1]/(_SQRT3*a)
    u = points[..., 0]/a-v/2
    u, v = u % 1, v % 1
    folded = np.stack((a*(u+v/2), a*_SQRT3*v/2), axis=-1)
    tolerance = tie_ulps*np.finfo(float).eps*a*a

    def candidates():
        for center, up in zip(((a, a/_SQRT3), (a/2, a/(2*_SQRT3))), (False, True)):
            for du in (-1, 0, 1):
                for dv in (-1, 0, 1):
                    shift = np.array(center)+a*np.array((du+dv/2, _SQRT3*dv/2))
                    displacement = folded-shift
                    yield np.sum(displacement**2, axis=-1), displacement, up

    best = np.full(points.shape[:-1], np.inf)
    for distance, _, _ in candidates():
        best = np.minimum(best, distance)
    selected = np.zeros(points.shape[:-1], dtype=bool)
    normal = np.zeros_like(points)
    for distance, displacement, up in candidates():
        take = (~selected) & (distance <= best+tolerance)
        if np.any(take):
            normal[take] = radial_normal(displacement[take], up)
        selected |= take
    if not np.all(selected):
        raise ValueError("No finite nearest-center candidate")
    return normal


def normal_projector_grid(a, sides, grid, *, tie_ulps=DEFAULT_TIE_ULPS):
    """Return Pxx and Pxy samples; supported side lengths do not affect them."""
    if not isinstance(grid, (int, np.integer)) or grid <= 0:
        raise ValueError("grid must be a positive integer")
    u, v = np.meshgrid(np.arange(grid)/grid, np.arange(grid)/grid, indexing="ij")
    points = np.stack((a*(u+v/2), a*_SQRT3*v/2), axis=-1)
    n = periodic_normal(points, a, sides, tie_ulps=tie_ulps)
    return n[..., 0]**2, n[..., 0]*n[..., 1]
