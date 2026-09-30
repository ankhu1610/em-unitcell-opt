import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from scipy.interpolate import RegularGridInterpolator
from typing import Tuple

class PolynomialResponseSurface:
    """
    Polynomial Response Surface regression (degree 2 or 3) per frequency point.
    Maps (L1n, L2n) -> f(k).
    """
    def __init__(self, degree: int = 2, alpha: float = 1e-4):
        self.degree = degree
        self.alpha = alpha
        self.poly = PolynomialFeatures(degree=degree, include_bias=True)
        self.model = Ridge(alpha=alpha)
        
    def fit(self, X: np.ndarray, Y: np.ndarray):
        X_poly = self.poly.fit_transform(X)
        self.model.fit(X_poly, Y)
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_poly = self.poly.transform(X)
        return self.model.predict(X_poly)

class GridInterpolatorBaseline:
    """
    Bilinear or cubic grid interpolation on the 8x8 design grid.
    Only applicable when training data covers a regular grid.
    """
    def __init__(self, method: str = "linear"):
        self.method = method
        self.interpolators = []
        
    def fit(self, F_grid: np.ndarray):
        """
        F_grid: shape (8, 8, 562)
        """
        l1_vals = np.linspace(0, 1, 8)
        l2_vals = np.linspace(0, 1, 8)
        self.interpolator = RegularGridInterpolator((l1_vals, l2_vals), F_grid, method=self.method, bounds_error=False, fill_value=None)
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.interpolator(X)

class NearestNeighborBaseline:
    def __init__(self):
        self.X_train = None
        self.Y_train = None
        
    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.X_train = X
        self.Y_train = Y
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        # For each point in X, find nearest point in X_train
        dists = np.linalg.norm(X[:, None, :] - self.X_train[None, :, :], axis=2)
        nearest_idx = np.argmin(dists, axis=1)
        return self.Y_train[nearest_idx]
