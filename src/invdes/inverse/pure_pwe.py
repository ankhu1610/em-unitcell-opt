"""Deterministic gap-only inverse search. No learned model or reference labels.

Search a line segment in (L1,L2) between two PWE-verified designs bracketing
the target. This finds one candidate, not all solutions or a global optimum.
"""
import numpy as np


def candidate(sides_um, frequencies, source):
    sides = np.asarray(sides_um, dtype=float)
    f = np.asarray(frequencies, dtype=float)
    if sides.shape != (2,) or not np.isfinite(sides).all() or np.any(sides <= 0):
        raise ValueError("Expected two positive finite side lengths")
    if f.ndim != 2 or f.shape[1] != 2 or not len(f) or not np.isfinite(f).all():
        raise ValueError("Expected finite (k, 2) frequencies")
    if np.any(f <= 0) or np.any(f[:, 1] < f[:, 0]):
        raise ValueError("Expected positive, sorted bands away from Gamma")
    low, high = float(f[:, 0].max()), float(f[:, 1].min())
    return {"L1_um": float(sides[0]), "L2_um": float(sides[1]),
            "gap_ghz": high-low, "lower_edge_ghz": low, "upper_edge_ghz": high,
            "source": source}


def design_gap(target, library, evaluate, tolerance=0.05, max_evaluations=8,
               fraction_limits=(0.1, 0.9)):
    """evaluate([L1,L2]) must return a full-path PWE candidate.

    Saved endpoints may be reused; every proposed interior geometry is actually
    evaluated. Secant interpolation chooses geometry only, never predicts a gap.
    Out-of-library-range targets are not declared physically unreachable.
    """
    if not np.isfinite(target) or target <= 0:
        raise ValueError("Band-gap width must be finite and strictly positive (GHz)")
    if not library or not np.isfinite(tolerance) or tolerance <= 0 or max_evaluations < 0:
        raise ValueError("Need a library, positive tolerance and nonnegative budget")
    floor, ceiling = fraction_limits
    if not 0 < floor < ceiling < 1:
        raise ValueError("Secant safeguard must lie strictly within (0,1)")
    best = dict(min(library, key=lambda c: abs(c["gap_ghz"]-target)))
    trace = []

    def result(status):
        return {"status": status, "target_gap_ghz": float(target),
                "tolerance_ghz": float(tolerance), "candidate": best,
                "absolute_error_ghz": abs(best["gap_ghz"]-target),
                "new_evaluations": len(trace), "trace": trace,
                "scope": "para; sampled 281-point path, not full Brillouin zone",
                "uniqueness": "One candidate only; geometry need not be unique"}

    if best["gap_ghz"] > 0 and abs(best["gap_ghz"]-target) <= tolerance:
        return result("matched_saved_pwe")
    lower = [c for c in library if c["gap_ghz"] < target]
    upper = [c for c in library if c["gap_ghz"] > target]
    if not lower or not upper:
        return result("no_saved_bracket_not_proven_unreachable")
    coordinates = np.array([[c["L1_um"], c["L2_um"]] for c in library])
    scale = np.maximum(np.ptp(coordinates, axis=0), 1e-12)

    def position(c):
        return np.array([c["L1_um"], c["L2_um"]])

    # Favor a short geometric segment, then endpoints near the target.
    lo, hi = min(((a, b) for a in lower for b in upper),
                 key=lambda pair: (round(float(np.linalg.norm(
                     (position(pair[0])-position(pair[1]))/scale)), 10),
                     pair[1]["gap_ghz"]-pair[0]["gap_ghz"]))
    bounds = [coordinates.min(axis=0), coordinates.max(axis=0)]
    for _ in range(max_evaluations):
        fraction = np.clip((target-lo["gap_ghz"])/(hi["gap_ghz"]-lo["gap_ghz"]),
                           floor, ceiling)
        sides = position(lo)+fraction*(position(hi)-position(lo))
        trial = evaluate(sides)
        if not np.allclose(position(trial), sides, rtol=0, atol=1e-10):
            raise ValueError("Evaluator returned a different geometry")
        if not np.isfinite(trial["gap_ghz"]):
            raise ValueError("Nonfinite PWE gap")
        if np.any(position(trial) < bounds[0]) or np.any(position(trial) > bounds[1]):
            raise ValueError("Candidate escaped library geometry bounds")
        trace.append(dict(trial))
        if abs(trial["gap_ghz"]-target) < abs(best["gap_ghz"]-target):
            best = dict(trial)
        if trial["gap_ghz"] > 0 and abs(trial["gap_ghz"]-target) <= tolerance:
            best = dict(trial)
            return result("matched_new_pwe")
        if trial["gap_ghz"] < target:
            lo = trial
        else:
            hi = trial
    return result("budget_exhausted")
