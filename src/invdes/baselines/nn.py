import numpy as np

class NearestNeighborBaseline:
    """
    Nearest-Neighbor baseline in normalized (L1, L2) space.
    Interface: fit(X, Y), predict(X) -> (N, 281, 2) GHz
    """
    def __init__(self):
        self.X_train = None
        self.Y_train = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        X: (N, 2) in [0, 1]
        Y: (N, 281, 2) in GHz
        """
        assert Y.ndim == 3 and Y.shape[1:] == (281, 2), f"Expected Y shape (N, 281, 2), got {Y.shape}"
        self.X_train = X.copy()
        self.Y_train = Y.copy()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        X: (M, 2) in [0, 1]
        Returns: (M, 281, 2)
        """
        # Pairwise Euclidean distances
        dists = np.linalg.norm(X[:, None, :] - self.X_train[None, :, :], axis=2) # (M, N)
        nearest_idx = np.argmin(dists, axis=1)
        return self.Y_train[nearest_idx]
