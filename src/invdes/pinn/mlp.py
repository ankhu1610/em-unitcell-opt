import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

class FourierFeatureEncoder(nn.Module):
    """
    Encodes wavevector k in [0.1, 2.9] into:
    - raw k
    - segment index (floor(k))
    - intra-segment offset (k - floor(k))
    - harmonic sin/cos Fourier features
    """
    def __init__(self, n_freqs: int = 6):
        super().__init__()
        self.n_freqs = n_freqs

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        # k: shape (..., 1)
        seg = torch.clamp(torch.floor(k), 0.0, 2.0)
        t = k - seg

        harmonics = []
        for m in range(1, self.n_freqs + 1):
            harmonics.append(torch.sin(2.0 * np.pi * m * (k / 3.0)))
            harmonics.append(torch.cos(2.0 * np.pi * m * (k / 3.0)))

        feats = [k, seg, t] + harmonics
        return torch.cat(feats, dim=-1) # 1 + 1 + 1 + 2 * 6 = 15 dims

class ResNetBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.act(self.fc1(x))
        out = self.fc2(out)
        return self.norm(out + res)

class PhysicsMLP(nn.Module):
    """
    Physics-regularized MLP backbone.
    Input: (structure_onehot (2), L1_n (1), L2_n (1), k_features (15)) = 19 dims.
    Output: f1, and f2 = f1 + softplus(raw_gap) guaranteeing f2 >= f1.
    """
    def __init__(self, in_dim: int = 19, hidden_dim: int = 128, n_blocks: int = 3):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU()
        )
        self.blocks = nn.ModuleList([ResNetBlock(hidden_dim) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden_dim, 2)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
        out = self.head(h)
        f1 = out[..., 0:1]
        gap = self.softplus(out[..., 1:2])
        f2 = f1 + gap
        return f1, f2
