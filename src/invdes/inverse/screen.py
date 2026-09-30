"""
Table screening module (T5.2).
Evaluates dense candidate grids against a specification and extracts distinct
local minima using Non-Maximum Suppression (NMS).
"""

import os
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from invdes.inverse.spec import SpecType, TargetCurve, GapSpec


def screen_candidates(
    spec: SpecType,
    f1_grid: np.ndarray,
    f2_grid: np.ndarray,
    l1_vals_um: np.ndarray,
    l2_vals_um: np.ndarray,
    top_m: int = 10,
    nms_radius_um: float = 3.5
) -> List[Dict[str, Any]]:
    """
    Screens precomputed (N_L1, N_L2, 281) dispersion curves against spec.
    f1_grid: (N_L1, N_L2, 281) in GHz
    f2_grid: (N_L1, N_L2, 281) in GHz
    l1_vals_um: (N_L1,)
    l2_vals_um: (N_L2,)
    """
    n1, n2, k_len = f1_grid.shape
    scores = np.zeros((n1, n2), dtype=np.float64)

    if isinstance(spec, TargetCurve):
        w = spec.weights
        w_sum = 0.0
        weighted_sq = np.zeros((n1, n2), dtype=np.float64)

        if spec.f1 is not None:
            diff1 = (f1_grid - spec.f1[None, None, :]) ** 2
            weighted_sq += np.sum(diff1 * w[None, None, :], axis=-1)
            w_sum += np.sum(w)

        if spec.f2 is not None:
            diff2 = (f2_grid - spec.f2[None, None, :]) ** 2
            weighted_sq += np.sum(diff2 * w[None, None, :], axis=-1)
            w_sum += np.sum(w)

        if w_sum > 0:
            scores = np.sqrt(weighted_sq / w_sum)

    elif isinstance(spec, GapSpec):
        f1_max = np.max(f1_grid, axis=-1)
        f2_min = np.min(f2_grid, axis=-1)
        path_gap = f2_min - f1_max
        mid = 0.5 * (f2_min + f1_max)

        center_err = np.abs(mid - spec.center_ghz)
        width_penalty = np.maximum(0.0, spec.min_width_ghz - path_gap)
        scores = spec.w_center * center_err + spec.w_width * width_penalty
    else:
        raise ValueError(f"Unknown spec type: {type(spec)}")

    # Flatten coordinates and scores
    flat_scores = scores.ravel()
    l1_mesh, l2_mesh = np.meshgrid(l1_vals_um, l2_vals_um, indexing="ij")
    flat_l1 = l1_mesh.ravel()
    flat_l2 = l2_mesh.ravel()

    # Sort indices ascending by score
    sorted_order = np.argsort(flat_scores)

    selected: List[Dict[str, Any]] = []
    selected_coords: List[Tuple[float, float]] = []

    for idx in sorted_order:
        cand_l1 = float(flat_l1[idx])
        cand_l2 = float(flat_l2[idx])
        score = float(flat_scores[idx])

        # NMS check: distance in (L1, L2) space
        too_close = False
        for s_l1, s_l2 in selected_coords:
            dist = np.hypot(cand_l1 - s_l1, cand_l2 - s_l2)
            if dist < nms_radius_um:
                too_close = True
                break

        if not too_close:
            selected_coords.append((cand_l1, cand_l2))
            # Extract 2D indices
            i1, i2 = np.unravel_index(idx, (n1, n2))
            selected.append({
                "l1_um": cand_l1,
                "l2_um": cand_l2,
                "score": score,
                "f1": f1_grid[i1, i2],
                "f2": f2_grid[i1, i2]
            })

            if len(selected) >= top_m:
                break

    return selected
