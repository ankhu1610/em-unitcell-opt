import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

class PolynomialBaseline:
    """
    Polynomial Response Surface regression (degree 2 or 3) per (k, band).
    Interface: fit(X, Y), predict(X) -> (N, 281, 2) GHz
    """
    def __init__(self, degree: int = 2, alpha: float = 1e-4):
        self.degree = degree
        self.alpha = alpha
        self.poly = PolynomialFeatures(degree=degree, include_bias=True)
        self.model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        X: (N, 2)
        Y: (N, 281, 2)
        """
        assert Y.ndim == 3 and Y.shape[1:] == (281, 2), f"Expected Y shape (N, 281, 2), got {Y.shape}"
        N = X.shape[0]
        # Flatten Y to (N, 562)
        Y_flat = Y.reshape(N, -1)
        X_poly = self.poly.fit_transform(X)
        self.model.fit(X_poly, Y_flat)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        X: (M, 2)
        Returns: (M, 281, 2)
        """
        M = X.shape[0]
        X_poly = self.poly.transform(X)
        pred_flat = self.model.predict(X_poly)
        return pred_flat.reshape(M, 281, 2)
