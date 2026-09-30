import numpy as np
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from typing import Tuple, List, Optional

class GPonPCASurrogate:
    """
    Curve-wise surrogate:
    1. Compresses 562-dim EM dispersion curves via PCA/POD.
    2. Models each principal component with an ARD Matérn-5/2 Gaussian Process.
    3. Reconstructs full curves and analytically propagates uncertainty.
    """
    def __init__(self, n_components: int = 12, nu: float = 2.5):
        self.n_components = n_components
        self.nu = nu
        self.pca = None
        self.gps: List[GaussianProcessRegressor] = []
        self.explained_variance_ratio_ = None
        
    def fit(self, X: np.ndarray, Y: np.ndarray, n_restarts: int = 0):
        """
        X: shape (n_samples, 2), normalized parameters (L1n, L2n)
        Y: shape (n_samples, 562), stacked [f1, f2] in GHz
        """
        # Determine optimal number of PCA components (up to min(n_samples - 1, self.n_components))
        max_comp = min(X.shape[0] - 1, self.n_components)
        self.pca = PCA(n_components=max_comp)
        Z = self.pca.fit_transform(Y)
        self.explained_variance_ratio_ = float(np.sum(self.pca.explained_variance_ratio_))
        
        # Fit an ARD GP for each PCA component
        self.gps = []
        for j in range(max_comp):
            # ARD kernel with separate length scale for L1 and L2
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.5, 0.5], length_scale_bounds=(1e-2, 10.0), nu=self.nu) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 1e-1))
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts, random_state=42 + j)
            gp.fit(X, Z[:, j])
            self.gps.append(gp)
            
        return self
        
    def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict full dispersion curves f(k).
        
        Returns:
            Y_pred: shape (n_samples, 562)
            Y_std: shape (n_samples, 562) if return_std=True, else None
        """
        n_samples = X.shape[0]
        n_comp = len(self.gps)
        
        Z_pred = np.zeros((n_samples, n_comp), dtype=np.float32)
        Z_var = np.zeros((n_samples, n_comp), dtype=np.float32)
        
        for j, gp in enumerate(self.gps):
            mu, sigma = gp.predict(X, return_std=True)
            Z_pred[:, j] = mu
            Z_var[:, j] = sigma ** 2
            
        Y_pred = self.pca.inverse_transform(Z_pred)
        
        # Enforce physical non-crossing constraint: f2 >= f1
        f1_pred = Y_pred[:, :281]
        f2_pred = Y_pred[:, 281:]
        f2_pred = np.maximum(f2_pred, f1_pred)
        Y_pred = np.concatenate([f1_pred, f2_pred], axis=1)
        
        if not return_std:
            return Y_pred, None
            
        # Analytical variance propagation through PCA projection:
        # y = sum_j (z_j * phi_j) + mean  =>  Var(y) = sum_j (phi_j^2 * Var(z_j))
        components = self.pca.components_  # shape: (n_comp, 562)
        comp_sq = components ** 2           # shape: (n_comp, 562)
        
        # Z_var shape: (n_samples, n_comp)
        # Y_var shape: (n_samples, 562)
        Y_var = np.dot(Z_var, comp_sq)
        Y_std = np.sqrt(np.maximum(Y_var, 1e-8))
        
        return Y_pred, Y_std
