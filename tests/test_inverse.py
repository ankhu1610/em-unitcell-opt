"""
Tests for inverse design specification, screening, refinement, and engine (T5.1-T5.4).
"""

import numpy as np
import pytest

from invdes.inverse.spec import TargetCurve, GapSpec, get_default_weights
from invdes.inverse.screen import screen_candidates
from invdes.inverse.refine import refine_candidate, L1_BOUNDS, L2_BOUNDS
from invdes.inverse.engine import InverseDesignEngine


def test_target_curve_spec():
    k_len = 281
    f1 = np.full(k_len, 100.0)
    f2 = np.full(k_len, 200.0)

    spec = TargetCurve(f1=f1, f2=f2)
    # Exact match should give 0.0 loss
    assert spec.evaluate_loss(f1, f2) == pytest.approx(0.0, abs=1e-7)

    # Offset by 2.0 GHz
    f1_shift = f1 + 2.0
    f2_shift = f2 + 2.0
    assert spec.evaluate_loss(f1_shift, f2_shift) == pytest.approx(2.0, abs=1e-5)


def test_gap_spec():
    k_len = 281
    # f1 in [80, 100], f2 in [120, 140]
    # f1_max = 100, f2_min = 120 -> path_gap = 20 GHz, mid = 110 GHz
    f1 = np.linspace(80.0, 100.0, k_len)
    f2 = np.linspace(120.0, 140.0, k_len)

    spec = GapSpec(center_ghz=110.0, min_width_ghz=15.0)
    # Exactly meets center and exceeds min_width -> loss should be 0.0
    assert spec.evaluate_loss(f1, f2) == pytest.approx(0.0, abs=1e-7)

    # Center mismatch
    spec_offset = GapSpec(center_ghz=115.0, min_width_ghz=15.0)
    assert spec_offset.evaluate_loss(f1, f2) == pytest.approx(5.0, abs=1e-5)

    # Gap width violation (requires 25 GHz width, actual is 20 GHz)
    spec_wide = GapSpec(center_ghz=110.0, min_width_ghz=25.0, w_width=2.0)
    assert spec_wide.evaluate_loss(f1, f2) == pytest.approx(2.0 * (25.0 - 20.0), abs=1e-5)


def test_screen_candidates():
    n1, n2, k_len = 5, 5, 281
    l1_vals = np.linspace(140.0, 189.0, n1)
    l2_vals = np.linspace(91.0, 140.0, n2)

    f1_grid = np.zeros((n1, n2, k_len))
    f2_grid = np.zeros((n1, n2, k_len))

    for i in range(n1):
        for j in range(n2):
            f1_grid[i, j] = 100.0 + 10.0 * i + j
            f2_grid[i, j] = 200.0 + 10.0 * i + j

    # Target matching grid point i=2, j=2
    target_f1 = f1_grid[2, 2]
    target_f2 = f2_grid[2, 2]
    spec = TargetCurve(f1=target_f1, f2=target_f2)

    candidates = screen_candidates(
        spec=spec,
        f1_grid=f1_grid,
        f2_grid=f2_grid,
        l1_vals_um=l1_vals,
        l2_vals_um=l2_vals,
        top_m=3,
        nms_radius_um=2.0
    )

    assert len(candidates) > 0
    top = candidates[0]
    assert top["l1_um"] == pytest.approx(l1_vals[2], abs=1e-5)
    assert top["l2_um"] == pytest.approx(l2_vals[2], abs=1e-5)
    assert top["score"] == pytest.approx(0.0, abs=1e-6)


def test_refine_candidate_bounds():
    def mock_predict(l1, l2):
        # Parabolic minimum at (160.0, 115.0)
        dist = np.hypot(l1 - 160.0, l2 - 115.0)
        f1 = np.full(281, 100.0 + dist)
        f2 = np.full(281, 200.0 + dist)
        return f1, f2, np.zeros(281)

    target_f1 = np.full(281, 100.0)
    target_f2 = np.full(281, 200.0)
    spec = TargetCurve(f1=target_f1, f2=target_f2)

    # Start off-center
    refined = refine_candidate(spec, 155.0, 120.0, mock_predict)
    assert refined["improved"] is True
    assert refined["l1_um"] == pytest.approx(160.0, abs=0.5)
    assert refined["l2_um"] == pytest.approx(115.0, abs=0.5)
    assert L1_BOUNDS[0] <= refined["l1_um"] <= L1_BOUNDS[1]
    assert L2_BOUNDS[0] <= refined["l2_um"] <= L2_BOUNDS[1]
