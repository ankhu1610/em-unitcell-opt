"""
Inverse design specifications (T5.1).
Defines target curves and band-gap requirements with weighted objectives.
"""

from dataclasses import dataclass, field
from typing import Optional, Union
import numpy as np

# Kink exclusion: ik in [88, 92] and [188, 192]
DEFAULT_KINK_NEIGHBORS = set(range(88, 93)).union(range(188, 193))


def get_default_weights(n_points: int = 281) -> np.ndarray:
    """Default weights with zero weight on the 2 samples either side of kinks (90, 190)."""
    w = np.ones(n_points, dtype=np.float64)
    for ik in DEFAULT_KINK_NEIGHBORS:
        if 0 <= ik < n_points:
            w[ik] = 0.0
    return w


@dataclass
class TargetCurve:
    """
    Target curve specification for bands 1 and 2.
    Objective is weighted RMSE across k-points and bands.
    """
    f1: Optional[np.ndarray] = None  # (281,) in GHz
    f2: Optional[np.ndarray] = None  # (281,) in GHz
    weights: np.ndarray = field(default_factory=get_default_weights)

    def __post_init__(self):
        if self.f1 is None and self.f2 is None:
            raise ValueError("At least one target band (f1 or f2) must be specified.")
        if self.weights is None:
            self.weights = get_default_weights(281)
        self.weights = np.asarray(self.weights, dtype=np.float64)

    def evaluate_loss(self, pred_f1: np.ndarray, pred_f2: np.ndarray) -> float:
        """
        Compute weighted RMSE between predicted and target dispersion curves.
        pred_f1, pred_f2: (281,) arrays in GHz.
        """
        w_sum = 0.0
        weighted_sq_err = 0.0

        if self.f1 is not None:
            diff1 = (pred_f1 - self.f1) ** 2
            weighted_sq_err += np.sum(diff1 * self.weights)
            w_sum += np.sum(self.weights)

        if self.f2 is not None:
            diff2 = (pred_f2 - self.f2) ** 2
            weighted_sq_err += np.sum(diff2 * self.weights)
            w_sum += np.sum(self.weights)

        if w_sum == 0:
            return 0.0
        return float(np.sqrt(weighted_sq_err / w_sum))


@dataclass
class GapSpec:
    """
    Target bandgap specification per AGENTS.md metrics:
      f1_max = max_k f1
      f2_min = min_k f2
      path_gap = f2_min - f1_max (signed)
      mid = (f2_min + f1_max) / 2
    """
    center_ghz: float
    min_width_ghz: float
    tolerance_ghz: float = 1.0
    w_center: float = 1.0
    w_width: float = 2.0

    def evaluate_loss(self, pred_f1: np.ndarray, pred_f2: np.ndarray) -> float:
        """
        Compute penalty objective:
        loss = w_center * |mid - center| + w_width * max(0, min_width - path_gap)
        """
        f1_max = np.max(pred_f1)
        f2_min = np.min(pred_f2)
        path_gap = f2_min - f1_max
        mid = 0.5 * (f2_min + f1_max)

        center_err = abs(mid - self.center_ghz)
        # Hinge loss on path gap width
        width_penalty = max(0.0, self.min_width_ghz - path_gap)

        loss = self.w_center * center_err + self.w_width * width_penalty
        return float(loss)


SpecType = Union[TargetCurve, GapSpec]
