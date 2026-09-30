import os
import numpy as np
from typing import Tuple, List, Generator

def load_band_tensor(data_dir: str = None) -> np.ndarray:
    """
    Load F tensor: shape (2, 8, 8, 281, 4)
    Structure 0: para, Structure 1: r1
    """
    if data_dir is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(root, "data", "processed")
    tensor_path = os.path.join(data_dir, "F.npy")
    return np.load(tensor_path)

def get_design_grid() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        L1_grid: shape (8,) in um [140 to 189 um]
        L2_grid: shape (8,) in um [91 to 140 um]
        k_grid:  shape (281,) [0.10 to 2.90]
    """
    L1 = np.linspace(140.0, 189.0, 8)
    L2 = np.linspace(91.0, 140.0, 8)
    k = np.linspace(0.1, 2.9, 281)
    return L1, L2, k

def prepare_curve_dataset(structure_id: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepares dataset for curve-wise models (e.g., GP on PCA).
    
    Inputs:
        structure_id: 0 for para, 1 for r1.
        
    Returns:
        X: shape (64, 2), normalized (L1n, L2n) in [0, 1]
        Y: shape (64, 562), stacked [f1(k), f2(k)] in GHz
    """
    F = load_band_tensor()
    # Extract structure (8, 8, 281, 2)
    sub = F[structure_id, :, :, :, :2]
    
    X_list = []
    Y_list = []
    for i1 in range(8):
        for i2 in range(8):
            l1_norm = i1 / 7.0  # normalized [0, 1]
            l2_norm = i2 / 7.0  # normalized [0, 1]
            f1 = sub[i1, i2, :, 0]
            f2 = sub[i1, i2, :, 1]
            stacked_curve = np.concatenate([f1, f2])  # 562 dim
            X_list.append([l1_norm, l2_norm])
            Y_list.append(stacked_curve)
            
    return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.float32)

def leave_one_design_out_folds(n_designs: int = 64) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generator yielding (train_idx, test_idx) for Leave-One-Design-Out (LODO).
    Total 64 folds.
    """
    indices = np.arange(n_designs)
    for i in range(n_designs):
        test_idx = np.array([i])
        train_idx = np.delete(indices, i)
        yield train_idx, test_idx

def checkerboard_split(n1: int = 8, n2: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Checkerboard hold-out: drop every other grid cell (i1 + i2 is odd).
    """
    train_idx = []
    test_idx = []
    for idx in range(n1 * n2):
        i1 = idx // n2
        i2 = idx % n2
        if (i1 + i2) % 2 == 0:
            train_idx.append(idx)
        else:
            test_idx.append(idx)
    return np.array(train_idx), np.array(test_idx)
