"""Math-only, anchor-free scalar inverse search in a two-dimensional box.

The evaluator supplies actual physics responses and optionally reliable
derivatives; this module contains no PWE data, interpolation, ML, or fitting.
One residual and two variables need not have a unique inverse. Minimum-norm
local steps do not imply a globally nearest design or global feasibility.
"""
from itertools import product
import numpy as np


def _point(x):
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (2,) or not np.all(np.isfinite(x)):
        raise ValueError("x must contain two finite normalized coordinates")
    if np.any(x < 0) or np.any(x > 1):
        raise ValueError("Normalized coordinates must lie in [0,1]^2")
    return x.copy()


def bounded_minimum_norm_step(residual, gradient, x, radius):
    """Minimize |residual + gradient @ step| subject to box/trust bounds.

    Among the minimizers, return the step with smallest Euclidean norm.
    Enumerating the nine free/lower/upper active patterns solves this rank-one
    problem without a singular Hessian inverse. The trust region is L-infinity.
    Gradient scaling avoids squaring very large or very small input gradients.
    """
    x = _point(x)
    gradient = np.asarray(gradient, dtype=np.float64)
    if gradient.shape != (2,) or not np.all(np.isfinite(gradient)):
        raise ValueError("gradient must contain two finite values")
    if not np.isfinite(residual):
        raise ValueError("residual must be finite")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive and finite")
    lower, upper = np.maximum(-x, -radius), np.minimum(1-x, radius)
    scale = float(np.max(np.abs(gradient)))
    if scale == 0 or residual == 0:
        return np.zeros(2)
    g = gradient/scale
    reachable_low = float(np.sum(np.where(g >= 0, g*lower, g*upper)))
    reachable_high = float(np.sum(np.where(g >= 0, g*upper, g*lower)))
    with np.errstate(over="ignore", under="ignore"):
        desired = np.divide(-np.float64(residual), np.float64(scale))
    target = float(np.clip(desired, reachable_low, reachable_high))
    slack = 128*np.finfo(float).eps*max(1., abs(target))
    candidates = []
    for pattern in product((-1, 0, 1), repeat=2):
        pattern = np.array(pattern)
        fixed, free = pattern != 0, pattern == 0
        step = np.where(pattern < 0, lower, np.where(pattern > 0, upper, 0.))
        remaining = target-float(g[fixed]@step[fixed])
        norm2 = float(g[free]@g[free])
        if norm2:
            step[free] = remaining*g[free]/norm2
        elif abs(remaining) > slack:
            continue
        if np.any(step < lower-slack) or np.any(step > upper+slack):
            continue
        step = np.clip(step, lower, upper)
        if abs(float(g@step)-target) <= slack:
            candidates.append(step)
    if not candidates:
        raise ArithmeticError("No feasible active-set solution within floating-point tolerance")
    return min(candidates, key=lambda s: float(s@s)).copy()


def solve_gap_target(evaluate, target_gap_ghz, initial_x=(.5, .5), *,
                     tolerance_ghz=.01, max_trial_evaluations=12,
                     initial_radius=.1, maximum_radius=.25, minimum_radius=1e-4,
                     acceptance_ratio=.1, shrink_ratio=.25, expand_ratio=.75,
                     gradient_norm_floor=1e-8, rejected_models_before_poll=2):
    """Run a bounded local search using caller-supplied physics evaluations.

    evaluate(x) returns a dict with finite ``gap_GHz``, optional finite
    ``gradient_normalized`` (two GHz/unit-coordinate derivatives), and
    ``gradient_reliable`` (default False). The caller must assess band/edge
    degeneracy and numerical derivative reliability before setting this flag.
    The evaluator is called at every proposed trial; no internal cache is used.
    Its owner may store full spectra independently of this JSON-ready trace.

    Unreliable/near-zero gradients, blocked models, or repeated model rejection
    trigger deterministic coordinate polling. Polls accept the first strict
    improvement; unsuccessful polls shrink the radius. All trial calls count
    toward the budget, including rejected trials. Initial evaluation is extra.
    A returned nonmatch means unresolved, never mathematically infeasible.
    """
    x = _point(initial_x)
    if not callable(evaluate) or not np.isfinite(target_gap_ghz):
        raise ValueError("Require an evaluator and finite target")
    if not np.isfinite(tolerance_ghz) or tolerance_ghz < 0:
        raise ValueError("tolerance_ghz must be finite and nonnegative")
    if not isinstance(max_trial_evaluations, int) or max_trial_evaluations < 0:
        raise ValueError("max_trial_evaluations must be a nonnegative integer")
    radii = np.array([minimum_radius, initial_radius, maximum_radius], dtype=float)
    if np.any(~np.isfinite(radii)) or not 0 < minimum_radius <= initial_radius <= maximum_radius:
        raise ValueError("Require 0 < minimum_radius <= initial_radius <= maximum_radius")
    if not 0 < acceptance_ratio <= shrink_ratio < expand_ratio < 1:
        raise ValueError("Invalid trust-region ratio thresholds")
    if not np.isfinite(gradient_norm_floor) or gradient_norm_floor < 0:
        raise ValueError("gradient_norm_floor must be finite and nonnegative")
    if not isinstance(rejected_models_before_poll, int) or rejected_models_before_poll < 1:
        raise ValueError("rejected_models_before_poll must be a positive integer")
    target = float(target_gap_ghz)
    trace, trials = [], 0
    radius, rejected_models = float(initial_radius), 0

    def call(point):
        value = evaluate(point.copy())
        if not isinstance(value, dict) or "gap_GHz" not in value or not np.isfinite(value["gap_GHz"]):
            raise ValueError("Evaluator must return a finite gap_GHz")
        derivative = value.get("gradient_normalized")
        if derivative is not None:
            derivative = np.asarray(derivative, dtype=float)
            if derivative.shape != (2,) or not np.all(np.isfinite(derivative)):
                raise ValueError("Supplied gradient_normalized must be finite with shape (2,)")
        reliable = value.get("gradient_reliable", False)
        if not isinstance(reliable, (bool, np.bool_)):
            raise ValueError("gradient_reliable must be a boolean")
        reliable = bool(reliable and derivative is not None)
        return {"gap": float(value["gap_GHz"]), "gradient": derivative, "reliable": reliable}

    def improved(new, old):
        # Reject changes at the scale of roundoff rather than accept cycling.
        slack = 8*np.finfo(float).eps*max(1., abs(old))
        return abs(new) < abs(old)-slack

    def append_trial(method, origin, point, value, accepted, residual_before,
                     predicted_residual=None, ratio=None, reason=None):
        trace.append({
            "event": "trial", "trial": trials, "method": method,
            "origin_x": origin.tolist(), "x": point.tolist(), "radius": radius,
            "gap_GHz": value["gap"], "residual_GHz": value["gap"]-target,
            "residual_before_GHz": float(residual_before),
            "gradient_reliable": value["reliable"],
            "origin_gradient_reliable": current["reliable"],
            "model_gradient_normalized": (current["gradient"].tolist()
                                           if method == "trust_region" else None),
            "accepted": bool(accepted),
            "predicted_residual_GHz": predicted_residual, "reduction_ratio": ratio,
            "reason": reason})

    current = call(x)
    trace.append({"event": "initial", "x": x.tolist(), "gap_GHz": current["gap"],
                  "residual_GHz": current["gap"]-target, "radius": radius,
                  "gradient_reliable": current["reliable"]})
    while True:
        residual = current["gap"]-target
        if abs(residual) <= tolerance_ghz:
            status = "matched"
            break
        if trials >= max_trial_evaluations:
            status = "budget_exhausted"
            break
        if radius < minimum_radius:
            status = "radius_exhausted"
            break
        gradient = current["gradient"]
        use_model = (current["reliable"] and np.linalg.norm(gradient) > gradient_norm_floor
                     and rejected_models < rejected_models_before_poll)
        step, predicted_residual, predicted_reduction = None, None, 0.
        if use_model:
            step = bounded_minimum_norm_step(residual, gradient, x, radius)
            predicted_residual = float(residual+gradient@step)
            predicted_reduction = 1-(predicted_residual/residual)**2
            use_model = bool(predicted_reduction > 128*np.finfo(float).eps
                             and np.any(step != 0))
        if use_model:
            candidate = np.clip(x+step, 0., 1.)
            value = call(candidate)
            trials += 1
            new_residual = value["gap"]-target
            with np.errstate(over="ignore"):
                ratio = float((1-np.square(new_residual/residual))/predicted_reduction)
            accepted = improved(new_residual, residual) and ratio >= acceptance_ratio
            append_trial("trust_region", x, candidate, value, accepted, residual,
                         predicted_residual, ratio if np.isfinite(ratio) else None,
                         "model_agreement" if accepted else "insufficient_actual_reduction")
            if accepted:
                x, current, rejected_models = candidate, value, 0
                if ratio < shrink_ratio:
                    radius *= .5
                elif ratio > expand_ratio and np.linalg.norm(step, ord=np.inf) >= .9*radius:
                    radius = min(maximum_radius, 2*radius)
            else:
                radius *= .5
                rejected_models += 1
            continue

        reason = ("rejected_linear_models" if rejected_models >= rejected_models_before_poll
                  else "unreliable_or_flat_gradient" if not current["reliable"]
                  or np.linalg.norm(gradient) <= gradient_norm_floor else "blocked_linear_model")
        accepted = False
        for axis, sign in product((0, 1), (1., -1.)):
            if trials >= max_trial_evaluations:
                break
            candidate = x.copy()
            candidate[axis] = np.clip(candidate[axis]+sign*radius, 0., 1.)
            if np.array_equal(candidate, x):
                continue
            value = call(candidate)
            trials += 1
            accepted = improved(value["gap"]-target, residual)
            append_trial("coordinate_poll", x, candidate, value, accepted, residual, reason=reason)
            if accepted:
                x, current, rejected_models = candidate, value, 0
                break
        if not accepted:
            radius *= .5
            rejected_models = 0

    return {
        "status": status, "matched": status == "matched", "target_gap_GHz": target,
        "x": x.tolist(), "gap_GHz": current["gap"],
        "absolute_error_GHz": abs(current["gap"]-target), "tolerance_GHz": float(tolerance_ghz),
        "trial_evaluations": trials, "total_evaluation_calls": trials+1,
        "final_radius": radius,
        "gradient_normalized": current["gradient"].tolist() if current["gradient"] is not None else None,
        "gradient_reliable": current["reliable"], "trace": trace,
        "unique_inverse_claimed": False, "global_feasibility_certified": False,
        "interpretation": ("Local target match; independent physics refinement is still required."
                           if status == "matched" else
                           "Target unresolved within this local search; not evidence of infeasibility.")}
