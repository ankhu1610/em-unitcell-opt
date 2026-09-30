import os
from typing import Dict, Tuple
import numpy as np

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "processed", "bands.npz"
)

def load_bands(path: str = DEFAULT_DATA_PATH) -> Dict[str, np.ndarray]:
    """
    Load data/processed/bands.npz per the AGENTS.md data contract:
    - F: float64 (2, 8, 8, 281, 2) [structure, i1, i2, ik, band] in GHz
    - L1_um: (8,) = 140...189 step 7
    - L2_um: (8,) = 91...140 step 7
    - k: (281,) = 0.10...2.90 step 0.01
    """
    data = np.load(path)
    F = data["F"].astype(np.float64)
    L1_um = data["L1_um"].astype(np.float64)
    L2_um = data["L2_um"].astype(np.float64)
    k = data["k"].astype(np.float64)

    assert F.shape == (2, 8, 8, 281, 2), f"Expected F shape (2, 8, 8, 281, 2), got {F.shape}"
    assert F.dtype == np.float64, f"Expected float64, got {F.dtype}"
    assert len(L1_um) == 8, f"Expected 8 L1 values, got {len(L1_um)}"
    assert len(L2_um) == 8, f"Expected 8 L2 values, got {len(L2_um)}"
    assert len(k) == 281, f"Expected 281 k points, got {len(k)}"

    return {
        "F": F,
        "L1_um": L1_um,
        "L2_um": L2_um,
        "k": k
    }

def get_structure_data(
    structure_id: int, 
    path: str = DEFAULT_DATA_PATH,
    normalize_inputs: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Returns (X, Y, meta) for a given structure (0: para, 1: r1).
    X: shape (64, 2) in [0, 1] if normalize_inputs else physical um.
    Y: shape (64, 281, 2) in GHz.
    """
    raw = load_bands(path)
    F_struct = raw["F"][structure_id] # (8, 8, 281, 2)
    L1_vals = raw["L1_um"]
    L2_vals = raw["L2_um"]

    # Meshgrid indexing (i1, i2)
    X_list = []
    Y_list = []
    for i1 in range(8):
        for i2 in range(8):
            l1 = L1_vals[i1]
            l2 = L2_vals[i2]
            if normalize_inputs:
                l1_norm = (l1 - 140.0) / 49.0
                l2_norm = (l2 - 91.0) / 49.0
                X_list.append([l1_norm, l2_norm])
            else:
                X_list.append([l1, l2])
            Y_list.append(F_struct[i1, i2])

    X = np.array(X_list, dtype=np.float64)
    Y = np.array(Y_list, dtype=np.float64)
    return X, Y, raw
