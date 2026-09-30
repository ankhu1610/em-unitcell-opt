import torch
import torch.nn as nn
from typing import Optional

class PhysicsRegularizedLoss(nn.Module):
    """
    Compound physics loss per PLAN.md T2.2:
    L = Huber(f_pred, f_true) + lambda_deriv * MSE(df_pred/dk, df_true/dk) [excluding kinks]
        + lambda_mono * ReLU(-df_pred/dL)^2
    """
    def __init__(self, delta: float = 1.0, lambda_priors: float = 0.0):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta)
        self.lambda_priors = lambda_priors
        self.relu = nn.ReLU()

    def forward(
        self,
        f1_pred: torch.Tensor,
        f2_pred: torch.Tensor,
        f1_true: torch.Tensor,
        f2_true: torch.Tensor,
        k_deriv_pred: Optional[torch.Tensor] = None,
        k_deriv_true: Optional[torch.Tensor] = None,
        valid_k_mask: Optional[torch.Tensor] = None,
        L_deriv_pred: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Data loss
        loss_data = self.huber(f1_pred, f1_true) + self.huber(f2_pred, f2_true)
        total_loss = loss_data

        if self.lambda_priors > 0.0:
            # 1. Derivative matching loss on k (excluding kinks)
            if k_deriv_pred is not None and k_deriv_true is not None and valid_k_mask is not None:
                diff_deriv = (k_deriv_pred - k_deriv_true) ** 2
                loss_deriv = torch.mean(diff_deriv[valid_k_mask])
                total_loss = total_loss + self.lambda_priors * loss_deriv

            # 2. Monotonicity prior: penalize negative dF/dL (since P3 verified positive slope)
            if L_deriv_pred is not None:
                # L_deriv_pred: (..., 2) for L1, L2
                # Penalize negative values: ReLU(-dL)^2
                mono_penalty = torch.mean(self.relu(-L_deriv_pred) ** 2)
                total_loss = total_loss + self.lambda_priors * mono_penalty

        return total_loss
