import os
import sys
import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple, Union

from dataset import prepare_curve_dataset
from models.gp_pca import GPonPCASurrogate
from features import extract_design_features

class InverseDesignEngine:
    """
    EM Unit Cell Inverse Design Engine.
    Given a desired electromagnetic response spec, searches the continuous (L1, L2)
    parameter space to identify optimal physical geometries with calibrated uncertainty.
    """
    def __init__(self, structure_id: int = 0, n_components: int = 12):
        self.structure_id = structure_id
        self.structure_name = "para" if structure_id == 0 else "r1"
        self.X_train, self.Y_train = prepare_curve_dataset(structure_id=structure_id)
        
        # Train GP-on-PCA surrogate on all available sweep designs
        self.surrogate = GPonPCASurrogate(n_components=n_components)
        self.surrogate.fit(self.X_train, self.Y_train)
        self.k_grid = np.linspace(0.1, 2.9, 281)
        
    def _norm_to_um(self, l1_norm: float, l2_norm: float) -> Tuple[float, float]:
        """Convert [0, 1] normalized values to physical micrometers."""
        l1_um = 140.0 + l1_norm * 49.0
        l2_um = 91.0 + l2_norm * 49.0
        return float(l1_um), float(l2_um)
        
    def _um_to_norm(self, l1_um: float, l2_um: float) -> Tuple[float, float]:
        """Convert physical micrometers to [0, 1] normalized values."""
        l1_norm = (l1_um - 140.0) / 49.0
        l2_norm = (l2_um - 91.0) / 49.0
        return float(l1_norm), float(l2_norm)

    def predict_design(self, l1_um: float, l2_um: float, return_std: bool = True) -> Dict:
        """Evaluate forward surrogate at arbitrary (L1, L2)."""
        l1_norm, l2_norm = self._um_to_norm(l1_um, l2_um)
        X_query = np.array([[l1_norm, l2_norm]], dtype=np.float32)
        Y_pred, Y_std = self.surrogate.predict(X_query, return_std=return_std)
        
        f1 = Y_pred[0, :281]
        f2 = Y_pred[0, 281:]
        std1 = Y_std[0, :281] if Y_std is not None else None
        std2 = Y_std[0, 281:] if Y_std is not None else None
        
        feats = extract_design_features(f1, f2)
        return {
            "L1_um": l1_um,
            "L2_um": l2_um,
            "f1": f1,
            "f2": f2,
            "f1_std": std1,
            "f2_std": std2,
            "mean_uncertainty_ghz": float(np.mean(Y_std[0])) if Y_std is not None else 0.0,
            "features": feats
        }

    def inverse_target_curve(self, target_f1: Optional[np.ndarray] = None, 
                             target_f2: Optional[np.ndarray] = None,
                             grid_resolution: int = 200,
                             top_k: int = 5) -> List[Dict]:
        """
        Solves inverse problem: find (L1, L2) whose band structure matches target curve(s).
        
        Two-stage algorithm:
        1. Global grid screening: Evaluates surrogate across 200x200 = 40,000 designs.
        2. Continuous L-BFGS-B refinement from best candidates.
        """
        # 1. Create dense candidate grid
        l1_vals = np.linspace(0.0, 1.0, grid_resolution)
        l2_vals = np.linspace(0.0, 1.0, grid_resolution)
        L1_grid, L2_grid = np.meshgrid(l1_vals, l2_vals)
        X_grid = np.stack([L1_grid.ravel(), L2_grid.ravel()], axis=1)  # (N_grid, 2)
        
        # Batch predict with surrogate
        Y_preds, Y_stds = self.surrogate.predict(X_grid, return_std=True)
        F1_preds = Y_preds[:, :281]
        F2_preds = Y_preds[:, 281:]
        
        # Compute loss on grid
        loss = np.zeros(len(X_grid), dtype=np.float32)
        if target_f1 is not None:
            diff1 = F1_preds - target_f1[None, :]
            loss += np.mean(diff1 ** 2, axis=1)
        if target_f2 is not None:
            diff2 = F2_preds - target_f2[None, :]
            loss += np.mean(diff2 ** 2, axis=1)
            
        rmse_grid = np.sqrt(loss)
        
        # Pick top unique candidate seeds
        best_indices = np.argsort(rmse_grid)[:top_k * 4]
        
        # 2. Local continuous refinement with L-BFGS-B
        def objective(x):
            X_q = np.array([x], dtype=np.float32)
            Y_q, _ = self.surrogate.predict(X_q, return_std=False)
            val = 0.0
            if target_f1 is not None:
                val += np.mean((Y_q[0, :281] - target_f1) ** 2)
            if target_f2 is not None:
                val += np.mean((Y_q[0, 281:] - target_f2) ** 2)
            return val
            
        candidates = []
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        
        seen_points = []
        for idx in best_indices:
            seed = X_grid[idx]
            # Avoid duplicate seeds
            if any(np.linalg.norm(seed - prev) < 0.03 for prev in seen_points):
                continue
            seen_points.append(seed)
            
            res = minimize(objective, seed, method="L-BFGS-B", bounds=bounds)
            opt_x = res.x
            opt_loss = np.sqrt(res.fun)
            
            opt_l1_um, opt_l2_um = self._norm_to_um(opt_x[0], opt_x[1])
            pred_info = self.predict_design(opt_l1_um, opt_l2_um, return_std=True)
            
            candidates.append({
                "rank": len(candidates) + 1,
                "L1_um": opt_l1_um,
                "L2_um": opt_l2_um,
                "rmse_ghz": float(opt_loss),
                "uncertainty_ghz": pred_info["mean_uncertainty_ghz"],
                "f1_pred": pred_info["f1"],
                "f2_pred": pred_info["f2"],
                "f1_std": pred_info["f1_std"],
                "f2_std": pred_info["f2_std"],
                "features": pred_info["features"]
            })
            
            if len(candidates) >= top_k:
                break
                
        # Sort candidates by RMSE
        candidates.sort(key=lambda c: c["rmse_ghz"])
        for i, c in enumerate(candidates):
            c["rank"] = i + 1
            
        return candidates

    def run_roundtrip_test(self, held_out_index: int = 15) -> Dict:
        """
        Experiment E6: Verify that target response from a known design recovers
        the exact physical parameters (L1, L2).
        """
        true_x = self.X_train[held_out_index]
        true_y = self.Y_train[held_out_index]
        
        true_l1, true_l2 = self._norm_to_um(true_x[0], true_x[1])
        target_f1 = true_y[:281]
        target_f2 = true_y[281:]
        
        results = self.inverse_target_curve(target_f1=target_f1, target_f2=target_f2, top_k=3)
        top1 = results[0]
        
        l1_err = abs(top1["L1_um"] - true_l1)
        l2_err = abs(top1["L2_um"] - true_l2)
        
        return {
            "held_out_index": held_out_index,
            "true_L1_um": true_l1,
            "true_L2_um": true_l2,
            "recovered_L1_um": top1["L1_um"],
            "recovered_L2_um": top1["L2_um"],
            "L1_error_um": l1_err,
            "L2_error_um": l2_err,
            "reconstruction_rmse_ghz": top1["rmse_ghz"],
            "uncertainty_ghz": top1["uncertainty_ghz"]
        }
