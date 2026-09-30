import numpy as np
import pytest
from invdes.splits import get_lodo_splits, get_checkerboard_split, get_edge_out_splits
from invdes.metrics import compute_metrics, compute_group_velocity, VALID_VG_INDICES

def test_lodo_splits():
    """Verify LODO covers all 64 designs without overlap."""
    folds = list(get_lodo_splits(n_designs=64))
    assert len(folds) == 64
    all_tested = []
    for train_idx, test_idx in folds:
        assert len(test_idx) == 1
        assert len(train_idx) == 63
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        all_tested.append(test_idx[0])
    assert sorted(all_tested) == list(range(64))

def test_checkerboard_split():
    """Verify checkerboard splits 32/32 with no overlap."""
    train_idx, test_idx = get_checkerboard_split((8, 8))
    assert len(train_idx) == 32
    assert len(test_idx) == 32
    assert len(np.intersect1d(train_idx, test_idx)) == 0
    assert len(np.union1d(train_idx, test_idx)) == 64

def test_edge_out_splits():
    """Verify all 4 edge-out splits leave out 8 designs with no overlap."""
    edge_splits = get_edge_out_splits((8, 8))
    assert len(edge_splits) == 4
    for edge_name, (train_idx, test_idx) in edge_splits.items():
        assert len(test_idx) == 8
        assert len(train_idx) == 56
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        assert len(np.union1d(train_idx, test_idx)) == 64

def test_metrics_hand_computable():
    """Verify metrics with a simple hand-computable synthetic case."""
    # 2 designs, 281 k-points, 2 bands
    y_true = np.zeros((2, 281, 2), dtype=np.float64)
    # Band 1: 100 to 200 linear
    y_true[:, :, 0] = np.linspace(100.0, 200.0, 281)[None, :]
    # Band 2: 300 to 400 linear
    y_true[:, :, 1] = np.linspace(300.0, 400.0, 281)[None, :]

    # Prediction with constant +1.0 GHz offset everywhere
    y_pred = y_true + 1.0

    metrics = compute_metrics(y_true, y_pred, dk=0.01)

    # MAE should be exactly 1.0
    assert np.isclose(metrics["pointwise"]["mae_ghz"], 1.0)
    assert np.isclose(metrics["pointwise"]["band1"]["mae_ghz"], 1.0)
    assert np.isclose(metrics["pointwise"]["band2"]["mae_ghz"], 1.0)

    # Path gap:
    # true: f2_min (300) - f1_max (200) = 100
    # pred: f2_min (301) - f1_max (201) = 100
    # error on path gap should be 0.0
    assert np.isclose(metrics["bandgap_features"]["path_gap"]["mae_ghz"], 0.0)

    # Mid frequency:
    # true: (300 + 200)/2 = 250
    # pred: (301 + 201)/2 = 251
    # error on mid should be 1.0
    assert np.isclose(metrics["bandgap_features"]["mid_frequency"]["mae_ghz"], 1.0)

    # Group velocity:
    # Since both are linear with identical slope, derivative diff is 0.0
    assert np.isclose(metrics["group_velocity"]["rmse_ghz_per_k"], 0.0)
