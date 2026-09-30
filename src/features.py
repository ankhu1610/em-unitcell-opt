import numpy as np
from typing import Dict, Any, Union

def extract_design_features(f1: np.ndarray, f2: np.ndarray, dk: float = 0.01) -> Dict[str, float]:
    """
    Extract physical EM band metrics from 1D dispersion curves f1(k) and f2(k).
    
    Parameters:
        f1: 1D array of shape (281,) representing fundamental band frequencies in GHz.
        f2: 1D array of shape (281,) representing second band frequencies in GHz.
        dk: Step size along k-path (default 0.01).
        
    Returns:
        dict containing physical scalar features.
    """
    f1_min = float(np.nanmin(f1))
    f1_max = float(np.nanmax(f1))
    f2_min = float(np.nanmin(f2))
    f2_max = float(np.nanmax(f2))
    
    # Path gap: band 1 max to band 2 min along the sampled path
    raw_gap = f2_min - f1_max
    path_gap = float(max(raw_gap, 0.0))
    mid_gap = float(0.5 * (f1_max + f2_min))
    gap_ratio = float(path_gap / mid_gap if mid_gap > 0 else 0.0)
    has_gap = bool(raw_gap > 0.0)
    
    # Group velocities: v_g ~ |df/dk|
    grad1 = np.abs(np.gradient(f1, dk))
    grad2 = np.abs(np.gradient(f2, dk))
    
    vg1_max = float(np.nanmax(grad1))
    vg2_max = float(np.nanmax(grad2))
    
    # Flatness fraction: where |df/dk| < 5% of max gradient
    flat1 = float(np.mean(grad1 < (0.05 * vg1_max))) if vg1_max > 0 else 0.0
    flat2 = float(np.mean(grad2 < (0.05 * vg2_max))) if vg2_max > 0 else 0.0
    
    # Bandwidths
    bw1 = float(f1_max - f1_min)
    bw2 = float(f2_max - f2_min)
    
    return {
        "f1_min": f1_min,
        "f1_max": f1_max,
        "f2_min": f2_min,
        "f2_max": f2_max,
        "path_gap_ghz": path_gap,
        "mid_gap_ghz": mid_gap,
        "gap_ratio": gap_ratio,
        "has_gap": has_gap,
        "vg1_max": vg1_max,
        "vg2_max": vg2_max,
        "flat1_frac": flat1,
        "flat2_frac": flat2,
        "bw1_ghz": bw1,
        "bw2_ghz": bw2
    }

def batch_extract_features(F_tensor: np.ndarray, dk: float = 0.01) -> Dict[str, np.ndarray]:
    """
    Extract features across all designs in tensor F.
    F shape: (n_structures, 8, 8, 281, n_bands)
    """
    n_struct, n_l1, n_l2, n_k, n_bands = F_tensor.shape
    features_list = []
    
    for s in range(n_struct):
        for i1 in range(n_l1):
            for i2 in range(n_l2):
                f1 = F_tensor[s, i1, i2, :, 0]
                f2 = F_tensor[s, i1, i2, :, 1]
                feats = extract_design_features(f1, f2, dk=dk)
                feats.update({
                    "structure": s,
                    "i1": i1,
                    "i2": i2,
                    "L1_um": 140.0 + i1 * 7.0,
                    "L2_um": 91.0 + i2 * 7.0
                })
                features_list.append(feats)
                
    # Convert list of dicts to dict of arrays
    keys = features_list[0].keys()
    return {k: np.array([item[k] for item in features_list]) for k in keys}
