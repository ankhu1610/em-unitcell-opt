import numpy as np
from typing import Dict, List, Tuple, Generator

def get_lodo_splits(n_designs: int = 64) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Leave-One-Design-Out (LODO) cross-validation splits.
    Yields (train_indices, test_indices) for each design.
    """
    all_indices = np.arange(n_designs, dtype=np.int64)
    for i in range(n_designs):
        test_idx = np.array([i], dtype=np.int64)
        train_idx = np.delete(all_indices, i)
        yield train_idx, test_idx

def get_checkerboard_split(grid_shape: Tuple[int, int] = (8, 8)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Checkerboard hold-out split based on parity of (i1 + i2).
    train: (i1 + i2) % 2 == 0 (32 designs)
    test:  (i1 + i2) % 2 == 1 (32 designs)
    """
    n1, n2 = grid_shape
    train_list = []
    test_list = []
    for i1 in range(n1):
        for i2 in range(n2):
            idx = i1 * n2 + i2
            if (i1 + i2) % 2 == 0:
                train_list.append(idx)
            else:
                test_list.append(idx)
    return np.array(train_list, dtype=np.int64), np.array(test_list, dtype=np.int64)

def get_edge_out_splits(grid_shape: Tuple[int, int] = (8, 8)) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Leave-One-Edge-Out splits.
    Leaves out each of the 4 outer perimeter edges:
    - edge_bottom: i1 = 0
    - edge_top:    i1 = n1 - 1
    - edge_left:   i2 = 0
    - edge_right:  i2 = n2 - 1
    Returns dict mapping edge name to (train_indices, test_indices).
    """
    n1, n2 = grid_shape
    n_total = n1 * n2
    all_indices = np.arange(n_total, dtype=np.int64)

    edges = {
        "edge_bottom": [0 * n2 + i2 for i2 in range(n2)],
        "edge_top": [(n1 - 1) * n2 + i2 for i2 in range(n2)],
        "edge_left": [i1 * n2 + 0 for i1 in range(n1)],
        "edge_right": [i1 * n2 + (n2 - 1) for i1 in range(n1)],
    }

    edge_splits = {}
    for edge_name, test_indices_list in edges.items():
        test_indices = np.array(test_indices_list, dtype=np.int64)
        train_indices = np.setdiff1d(all_indices, test_indices)
        edge_splits[edge_name] = (train_indices, test_indices)

    return edge_splits

def get_blind_split(n_train: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hook for blind validation set (H3/T6.1).
    All sweep designs are train; blind designs are evaluated out-of-sample.
    """
    train_indices = np.arange(n_train, dtype=np.int64)
    test_indices = np.array([], dtype=np.int64)
    return train_indices, test_indices
