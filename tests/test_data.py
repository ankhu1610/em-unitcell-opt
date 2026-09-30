import numpy as np
import pytest
from invdes.data import load_bands, get_structure_data

def test_load_bands_contract():
    """Verify data/processed/bands.npz matches AGENTS.md contract."""
    bands = load_bands()
    F = bands["F"]
    L1 = bands["L1_um"]
    L2 = bands["L2_um"]
    k = bands["k"]

    assert F.shape == (2, 8, 8, 281, 2)
    assert F.dtype == np.float64
    assert not np.isnan(F).any()
    assert (F[..., 1] >= F[..., 0]).all()

    assert len(L1) == 8
    assert L1[0] == 140.0
    assert L1[-1] == 189.0

    assert len(L2) == 8
    assert L2[0] == 91.0
    assert L2[-1] == 140.0

    assert len(k) == 281
    assert np.isclose(k[0], 0.10)
    assert np.isclose(k[-1], 2.90)

def test_get_structure_data():
    """Verify structure data slicing and normalization."""
    for s_id in [0, 1]:
        X_norm, Y, raw = get_structure_data(s_id, normalize_inputs=True)
        assert X_norm.shape == (64, 2)
        assert Y.shape == (64, 281, 2)
        assert np.min(X_norm) >= 0.0
        assert np.max(X_norm) <= 1.0

        X_phys, _, _ = get_structure_data(s_id, normalize_inputs=False)
        assert np.min(X_phys[:, 0]) == 140.0
        assert np.max(X_phys[:, 0]) == 189.0
