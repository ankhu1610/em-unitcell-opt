import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional

class FourierFeatureEncoder(nn.Module):
    def __init__(self, n_freqs: int = 6):
        super().__init__()
        self.n_freqs = n_freqs
        
    def forward(self, k: torch.Tensor) -> torch.Tensor:
        # k shape: (B, 1), scaled 0.1 .. 2.9
        # Segment features: which segment (0, 1, or 2) and intra-segment position
        seg = torch.clamp(torch.floor(k), 0.0, 2.0)
        t = k - seg
        
        # Fourier harmonics
        harmonics = []
        for m in range(1, self.n_freqs + 1):
            harmonics.append(torch.sin(2.0 * np.pi * m * (k / 3.0)))
            harmonics.append(torch.cos(2.0 * np.pi * m * (k / 3.0)))
            
        feats = [k, seg, t] + harmonics
        return torch.cat(feats, dim=-1)

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

class PointwiseBandNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, n_blocks: int = 3):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU()
        )
        self.blocks = nn.ModuleList([ResNetBlock(hidden_dim) for _ in range(n_blocks)])
        # Output: f1, and raw_gap where gap = softplus(raw_gap)
        self.head = nn.Linear(hidden_dim, 2)
        self.softplus = nn.Softplus()
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
        out = self.head(h)
        f1 = out[:, 0:1]
        gap = self.softplus(out[:, 1:2])
        f2 = f1 + gap
        return f1, f2

class MLPSurrogate:
    def __init__(self, hidden_dim: int = 128, epochs: int = 250, lr: float = 1e-3, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.fourier = FourierFeatureEncoder(n_freqs=6)
        # in_dim: 2 (L1, L2) + Fourier dims (1 + 1 + 1 + 2*6 = 15) = 17
        self.model = PointwiseBandNet(in_dim=17, hidden_dim=hidden_dim).to(self.device)
        self.k_grid = np.linspace(0.1, 2.9, 281, dtype=np.float32)
        
    def _prepare_inputs(self, X: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        # X: (N, 2), k_grid: (281,)
        # Expand all (N * 281) points
        N = X.shape[0]
        K = len(self.k_grid)
        
        X_rep = np.repeat(X, K, axis=0)  # (N*K, 2)
        k_rep = np.tile(self.k_grid, N)[:, None]  # (N*K, 1)
        
        X_t = torch.tensor(X_rep, dtype=torch.float32)
        k_t = torch.tensor(k_rep, dtype=torch.float32)
        
        k_feats = self.fourier(k_t)
        in_t = torch.cat([X_t, k_feats], dim=-1).to(self.device)
        return in_t, k_t.to(self.device)
        
    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        X: (N, 2) in [0, 1]
        Y: (N, 562) in GHz
        """
        N = X.shape[0]
        in_t, _ = self._prepare_inputs(X)
        
        # Flatten Y
        f1_true = Y[:, :281].reshape(-1, 1)
        f2_true = Y[:, 281:].reshape(-1, 1)
        
        target_f1 = torch.tensor(f1_true, dtype=torch.float32).to(self.device)
        target_f2 = torch.tensor(f2_true, dtype=torch.float32).to(self.device)
        
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.HuberLoss(delta=1.0)
        
        self.model.train()
        batch_size = 4096
        n_samples = in_t.shape[0]
        
        for epoch in range(self.epochs):
            perm = torch.randperm(n_samples)
            for i in range(0, n_samples, batch_size):
                idx = perm[i:i+batch_size]
                batch_in = in_t[idx]
                b_f1 = target_f1[idx]
                b_f2 = target_f2[idx]
                
                optimizer.zero_grad()
                pred_f1, pred_f2 = self.model(batch_in)
                loss = criterion(pred_f1, b_f1) + criterion(pred_f2, b_f2)
                loss.backward()
                optimizer.step()
                
            scheduler.step()
            
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            in_t, _ = self._prepare_inputs(X)
            pred_f1, pred_f2 = self.model(in_t)
            
            N = X.shape[0]
            f1_arr = pred_f1.cpu().numpy().reshape(N, 281)
            f2_arr = pred_f2.cpu().numpy().reshape(N, 281)
            
            return np.concatenate([f1_arr, f2_arr], axis=1)
