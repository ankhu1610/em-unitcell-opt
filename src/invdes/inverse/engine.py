"""
Inverse Design Engine (T5.4).
Coordinates multi-structure screening, continuous refinement, uncertainty estimation,
and reachability classification.
"""

import os
from typing import List, Dict, Any, Optional, Union
import numpy as np

from invdes.inverse.spec import SpecType, TargetCurve, GapSpec
from invdes.inverse.screen import screen_candidates
from invdes.inverse.refine import refine_candidate, L1_BOUNDS, L2_BOUNDS
from invdes.baselines.gp_pca import GPonPCASurrogate
from invdes.data import get_structure_data


class InverseDesignEngine:
    """
    Unified Inverse Design Engine.
    Searches 2D inclusion parameters (L1, L2) in [140, 189] x [91, 140] um
    across structures 'para' (0) and 'r1' (1) to meet target band specifications.
    """
    def __init__(
        self,
        surrogates: Optional[Dict[str, Any]] = None,
        grid_density: int = 50,
        reachability_factor: float = 3.0,
        lodo_rmse: Optional[Dict[str, float]] = None
    ):
        self.reachability_factor = reachability_factor
        self.structures = ["para", "r1"]

        # Default LODO RMSE baselines (from criteria and T1.2)
        self.lodo_rmse = lodo_rmse or {
            "para": 1.15,  # GHz
            "r1": 15.65    # GHz
        }

        # Setup surrogate forward models for both structures if not provided
        self.surrogates = surrogates or {}
        for s_idx, s_name in enumerate(self.structures):
            if s_name not in self.surrogates:
                gp = GPonPCASurrogate(n_components=12)
                X, Y, _ = get_structure_data(s_idx, normalize_inputs=True)
                gp.fit(X, Y)
                self.surrogates[s_name] = gp

        # Precompute candidate evaluation grid for rapid screening
        self.grid_density = grid_density
        self.l1_grid = np.linspace(L1_BOUNDS[0], L1_BOUNDS[1], grid_density)
        self.l2_grid = np.linspace(L2_BOUNDS[0], L2_BOUNDS[1], grid_density)

        self.table_f1: Dict[str, np.ndarray] = {}
        self.table_f2: Dict[str, np.ndarray] = {}
        self._build_tables()

    def _build_tables(self):
        """Precomputes forward predictions across grid for fast screening."""
        l1_mesh, l2_mesh = np.meshgrid(self.l1_grid, self.l2_grid, indexing="ij")
        n1, n2 = l1_mesh.shape

        # Normalize to [0, 1]
        l1_norm = (l1_mesh.ravel() - L1_BOUNDS[0]) / (L1_BOUNDS[1] - L1_BOUNDS[0])
        l2_norm = (l2_mesh.ravel() - L2_BOUNDS[0]) / (L2_BOUNDS[1] - L2_BOUNDS[0])
        X_query = np.stack([l1_norm, l2_norm], axis=1)

        for s_name in self.structures:
            surr = self.surrogates[s_name]
            out = surr.predict(X_query, return_std=False)
            Y_pred = out[0] if isinstance(out, tuple) else out
            # Y_pred is (M, 281, 2)
            f1 = Y_pred[:, :, 0].reshape(n1, n2, 281)
            f2 = Y_pred[:, :, 1].reshape(n1, n2, 281)
            self.table_f1[s_name] = f1
            self.table_f2[s_name] = f2

    def _get_predict_fn(self, s_name: str):
        surr = self.surrogates[s_name]

        def predict_fn(l1_um: float, l2_um: float):
            l1_n = (l1_um - L1_BOUNDS[0]) / (L1_BOUNDS[1] - L1_BOUNDS[0])
            l2_n = (l2_um - L2_BOUNDS[0]) / (L2_BOUNDS[1] - L2_BOUNDS[0])
            X_q = np.array([[l1_n, l2_n]], dtype=np.float64)
            out = surr.predict(X_q, return_std=True)
            if isinstance(out, tuple):
                Y_pred, Y_std = out
                std = Y_std[0] if Y_std is not None else None
            else:
                Y_pred = out
                std = None
            f1 = Y_pred[0, :, 0]
            f2 = Y_pred[0, :, 1]
            return f1, f2, std

        return predict_fn

    def design(
        self,
        spec: SpecType,
        structure: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Executes inverse design for a target specification.
        Returns top_k candidates with predicted response, uncertainty, and reachability.
        """
        structs_to_search = [structure] if structure in self.structures else self.structures
        all_candidates = []

        for s_name in structs_to_search:
            predict_fn = self._get_predict_fn(s_name)

            # Step 1: Screen precomputed table
            screened = screen_candidates(
                spec=spec,
                f1_grid=self.table_f1[s_name],
                f2_grid=self.table_f2[s_name],
                l1_vals_um=self.l1_grid,
                l2_vals_um=self.l2_grid,
                top_m=top_k * 2
            )

            # Step 2: Continuous refinement for each screened seed
            ref_rmse = self.lodo_rmse[s_name]
            threshold = self.reachability_factor * ref_rmse

            for item in screened:
                refined = refine_candidate(
                    spec=spec,
                    init_l1_um=item["l1_um"],
                    init_l2_um=item["l2_um"],
                    predict_fn=predict_fn
                )

                score = refined["score"]
                is_reachable = bool(score <= threshold)
                sigma_val = float(np.mean(refined["std"])) if refined["std"] is not None else 0.0

                all_candidates.append({
                    "structure": s_name,
                    "L1_um": refined["l1_um"],
                    "L2_um": refined["l2_um"],
                    "predicted_f1": refined["f1"],
                    "predicted_f2": refined["f2"],
                    "objective": score,
                    "sigma": sigma_val,
                    "reachable": is_reachable,
                    "status": "Achievable" if is_reachable else "not achievable in this design family; closest found"
                })

        # Rank all candidates across structures by objective ascending
        all_candidates.sort(key=lambda c: c["objective"])
        return all_candidates[:top_k]
