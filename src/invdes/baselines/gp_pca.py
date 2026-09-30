import numpy as np
from typing import Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class GPPCABaseline:
    """
    Curve-wise GP-on-PCA surrogate per PLAN.md T1.2.
    1. Compresses 562-dim EM dispersion curves via PCA/POD.
    2. Models each component with an ARD Matérn-5/2 Gaussian Process.
    3. Reconstructs curves and enforces physical non-crossing: f2 >= f1.
    Interface: fit(X, Y), predict(X) -> (N, 281, 2) GHz
    """
    def __init__(self, n_components: int = 12, nu: float = 2.5):
        self.n_components = n_components
        self.nu = nu
        self.pca = None
        self.gps = []

    def fit(self, X: np.ndarray, Y: np.ndarray, n_restarts: int = 0):
        """
        X: (N, 2) normalized in [0, 1]
        Y: (N, 281, 2) in GHz
        """
        assert Y.ndim == 3 and Y.shape[1:] == (281, 2), f"Expected Y shape (N, 281, 2), got {Y.shape}"
        N = X.shape[0]
        # Flatten to (N, 562) with [f1, f2] concatenated along k
        Y_flat = np.concatenate([Y[:, :, 0], Y[:, :, 1]], axis=1)

        max_comp = min(N - 1, self.n_components)
        self.pca = PCA(n_components=max_comp)
        Z = self.pca.fit_transform(Y_flat)

        self.gps = []
        for j in range(max_comp):
            kernel = (
                ConstantKernel(1.0, (1e-3, 1e3)) * 
                Matern(length_scale=[0.5, 0.5], length_scale_bounds=(1e-2, 10.0), nu=self.nu) + 
                WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 1e-1))
            )
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts, random_state=42 + j)
            gp.fit(X, Z[:, j])
            self.gps.append(gp)

        return self

    def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        X: (M, 2)
        Returns:
            Y_pred: (M, 281, 2)
            Y_std: (M, 281, 2) if return_std else None
        """
        M = X.shape[0]
        n_comp = len(self.gps)

        Z_pred = np.zeros((M, n_comp), dtype=np.float64)
        Z_var = np.zeros((M, n_comp), dtype=np.float64)

        for j, gp in enumerate(self.gps):
            mu, sigma = gp.predict(X, return_std=True)
            Z_pred[:, j] = mu
            Z_var[:, j] = sigma ** 2

        Y_flat_pred = self.pca.inverse_transform(Z_pred) # (M, 562)

        f1_pred = Y_flat_pred[:, :281]
        f2_pred = Y_flat_pred[:, 281:]
        # Physical non-crossing constraint
        f2_pred = np.maximum(f2_pred, f1_pred)

        Y_pred = np.stack([f1_pred, f2_pred], axis=-1) # (M, 281, 2)

        if not return_std:
            return Y_pred

        # Analytical variance propagation: Var(y) = sum_j (phi_j^2 * Var(z_j))
        components = self.pca.components_ # (n_comp, 562)
        comp_sq = components ** 2
        Y_var = np.dot(Z_var, comp_sq) # (M, 562)
        Y_std_flat = np.sqrt(np.maximum(Y_var, 1e-8))

        std1 = Y_std_flat[:, :281]
        std2 = Y_std_flat[:, 281:]
        Y_std = np.stack([std1, std2], axis=-1)

        return Y_pred, Y_std

GPonPCASurrogate = GPPCABaseline
