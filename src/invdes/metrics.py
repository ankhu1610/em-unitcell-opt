import numpy as np
from typing import Dict, List, Tuple, Union

# Exclude 2 samples either side of kink indices (ik = 90 and ik = 190)
# Valid central difference indices run from 1 to 279.
KINK_INDICES = [90, 190]
EXCLUDED_KINK_SAMPLES = set()
for kink in KINK_INDICES:
    for offset in range(-2, 3): # -2, -1, 0, 1, 2
        EXCLUDED_KINK_SAMPLES.add(kink + offset)

VALID_VG_INDICES = np.array([
    i for i in range(1, 280) if i not in EXCLUDED_KINK_SAMPLES
], dtype=np.int64)

def compute_group_velocity(f: np.ndarray, dk: float = 0.01) -> np.ndarray:
    """
    Central difference dω/dk along k-axis (axis -1).
    f: shape (..., 281)
    Returns: vg of shape (..., 279) corresponding to indices 1..279
    """
    return (f[..., 2:] - f[..., :-2]) / (2.0 * dk)

def compute_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    dk: float = 0.01
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Compute comprehensive metrics contract per AGENTS.md:
    y_true, y_pred: shape (N, 281, 2) in GHz
    where band 0 is f1, band 1 is f2.
    """
    assert y_true.shape == y_pred.shape, f"Shape mismatch: {y_true.shape} vs {y_pred.shape}"
    assert y_true.shape[-2:] == (281, 2), f"Expected last dims (281, 2), got {y_true.shape[-2:]}"

    f1_true = y_true[..., 0] # (N, 281)
    f2_true = y_true[..., 1]
    f1_pred = y_pred[..., 0]
    f2_pred = y_pred[..., 1]

    # --- 1. Pointwise metrics per band and overall ---
    err1 = np.abs(f1_pred - f1_true)
    err2 = np.abs(f2_pred - f2_true)
    err_all = np.abs(y_pred - y_true)

    mae_b1 = float(np.mean(err1))
    mae_b2 = float(np.mean(err2))
    mae_overall = float(np.mean(err_all))

    rmse_b1 = float(np.sqrt(np.mean((f1_pred - f1_true) ** 2)))
    rmse_b2 = float(np.sqrt(np.mean((f2_pred - f2_true) ** 2)))
    rmse_overall = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    max_err_b1 = float(np.max(err1))
    max_err_b2 = float(np.max(err2))
    max_err_overall = float(np.max(err_all))

    rel_err_b1 = float(np.mean(err1 / np.maximum(np.abs(f1_true), 1e-6)) * 100.0)
    rel_err_b2 = float(np.mean(err2 / np.maximum(np.abs(f2_true), 1e-6)) * 100.0)
    rel_err_overall = float(np.mean(err_all / np.maximum(np.abs(y_true), 1e-6)) * 100.0)

    # --- 2. Band-gap features per design ---
    # f1_max = max_k f1, f2_min = min_k f2
    # path_gap = f2_min - f1_max (signed, not clipped)
    # mid = (f2_min + f1_max) / 2
    f1_max_true = np.max(f1_true, axis=-1)
    f2_min_true = np.min(f2_true, axis=-1)
    path_gap_true = f2_min_true - f1_max_true
    mid_true = (f2_min_true + f1_max_true) / 2.0

    f1_max_pred = np.max(f1_pred, axis=-1)
    f2_min_pred = np.min(f2_pred, axis=-1)
    path_gap_pred = f2_min_pred - f1_max_pred
    mid_pred = (f2_min_pred + f1_max_pred) / 2.0

    err_path_gap = np.abs(path_gap_pred - path_gap_true)
    err_mid = np.abs(mid_pred - mid_true)

    mae_path_gap = float(np.mean(err_path_gap))
    rmse_path_gap = float(np.sqrt(np.mean(err_path_gap ** 2)))
    max_path_gap_err = float(np.max(err_path_gap))

    mae_mid = float(np.mean(err_mid))
    rmse_mid = float(np.sqrt(np.mean(err_mid ** 2)))
    max_mid_err = float(np.max(err_mid))

    # --- 3. Group velocity RMS error excluding kinks ---
    # Central difference in k (indices 1..279)
    vg1_true = compute_group_velocity(f1_true, dk=dk) # (N, 279)
    vg2_true = compute_group_velocity(f2_true, dk=dk)
    vg1_pred = compute_group_velocity(f1_pred, dk=dk)
    vg2_pred = compute_group_velocity(f2_pred, dk=dk)

    # Filter to valid indices (offset by -1 since vg starts at ik=1)
    valid_slice = VALID_VG_INDICES - 1
    vg1_true_valid = vg1_true[..., valid_slice]
    vg2_true_valid = vg2_true[..., valid_slice]
    vg1_pred_valid = vg1_pred[..., valid_slice]
    vg2_pred_valid = vg2_pred[..., valid_slice]

    vg_rmse_b1 = float(np.sqrt(np.mean((vg1_pred_valid - vg1_true_valid) ** 2)))
    vg_rmse_b2 = float(np.sqrt(np.mean((vg2_pred_valid - vg2_true_valid) ** 2)))
    vg_rmse_overall = float(np.sqrt(np.mean(
        np.concatenate([
            (vg1_pred_valid - vg1_true_valid).ravel(),
            (vg2_pred_valid - vg2_true_valid).ravel()
        ]) ** 2
    )))

    return {
        "pointwise": {
            "mae_ghz": mae_overall,
            "rmse_ghz": rmse_overall,
            "max_error_ghz": max_err_overall,
            "rel_error_pct": rel_err_overall,
            "band1": {
                "mae_ghz": mae_b1,
                "rmse_ghz": rmse_b1,
                "max_error_ghz": max_err_b1,
                "rel_error_pct": rel_err_b1
            },
            "band2": {
                "mae_ghz": mae_b2,
                "rmse_ghz": rmse_b2,
                "max_error_ghz": max_err_b2,
                "rel_error_pct": rel_err_b2
            }
        },
        "bandgap_features": {
            "path_gap": {
                "mae_ghz": mae_path_gap,
                "rmse_ghz": rmse_path_gap,
                "max_error_ghz": max_path_gap_err
            },
            "mid_frequency": {
                "mae_ghz": mae_mid,
                "rmse_ghz": rmse_mid,
                "max_error_ghz": max_mid_err
            }
        },
        "group_velocity": {
            "rmse_ghz_per_k": vg_rmse_overall,
            "band1_rmse": vg_rmse_b1,
            "band2_rmse": vg_rmse_b2,
            "kink_indices_excluded": KINK_INDICES,
            "n_valid_points": int(len(VALID_VG_INDICES))
        }
    }
