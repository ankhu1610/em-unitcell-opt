import os
import random
from typing import List, Optional
import numpy as np
import torch

DEFAULT_SEEDS: List[int] = [0, 1, 2]

def set_seed(seed: int = 0) -> None:
    """
    Set seed across Python, NumPy, and PyTorch for strict reproducibility.
    Per AGENTS.md, float64 is the standard for physics computations.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device(device: Optional[str] = None) -> torch.device:
    """
    Return torch.device. Default is CPU-first per AGENTS.md.
    GPU is selectable via device='cuda' or INVDES_DEVICE=cuda.
    """
    target = device or os.environ.get("INVDES_DEVICE", "cpu")
    if target == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if target == "cpu":
        return torch.device("cpu")
    return torch.device(target)
