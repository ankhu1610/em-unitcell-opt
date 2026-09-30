"""
Continuous gradient and local refinement (T5.3).
Refines candidate (L1, L2) coordinates using L-BFGS-B, strictly box-bounded to
the physical sweep domain: L1 in [140, 189] um, L2 in [91, 140] um.
"""

from typing import Callable, Dict, Any, Tuple, Optional
import numpy as np
from scipy.optimize import minimize

from invdes.inverse.spec import SpecType

# Exact physical sweep boundaries
L1_BOUNDS = (140.0, 189.0)
L2_BOUNDS = (91.0, 140.0)


def refine_candidate(
    spec: SpecType,
    init_l1_um: float,
    init_l2_um: float,
    predict_fn: Callable[[float, float], Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
    grad_fn: Optional[Callable[[float, float], np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Refines candidate (L1, L2) using L-BFGS-B.
    predict_fn: (l1_um, l2_um) -> (f1_281, f2_281, std_281 or None)
    """
    # Evaluate initial candidate
    init_f1, init_f2, init_std = predict_fn(init_l1_um, init_l2_um)
    init_score = spec.evaluate_loss(init_f1, init_f2)

    def objective(x: np.ndarray) -> float:
        l1, l2 = float(x[0]), float(x[1])
        f1, f2, _ = predict_fn(l1, l2)
        return spec.evaluate_loss(f1, f2)

    x0 = np.array([init_l1_um, init_l2_um], dtype=np.float64)
    bounds = [L1_BOUNDS, L2_BOUNDS]

    # Run SciPy L-BFGS-B bounded refinement
    res = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 40, "ftol": 1e-6, "eps": 0.05}
    )

    opt_l1 = float(np.clip(res.x[0], L1_BOUNDS[0], L1_BOUNDS[1]))
    opt_l2 = float(np.clip(res.x[1], L2_BOUNDS[0], L2_BOUNDS[1]))
    opt_f1, opt_f2, opt_std = predict_fn(opt_l1, opt_l2)
    opt_score = spec.evaluate_loss(opt_f1, opt_f2)

    # Acceptance check: refinement must never increase the objective
    if opt_score > init_score:
        return {
            "l1_um": init_l1_um,
            "l2_um": init_l2_um,
            "score": init_score,
            "f1": init_f1,
            "f2": init_f2,
            "std": init_std,
            "improved": False
        }

    return {
        "l1_um": opt_l1,
        "l2_um": opt_l2,
        "score": opt_score,
        "f1": opt_f1,
        "f2": opt_f2,
        "std": opt_std,
        "improved": bool(opt_score < init_score)
    }
