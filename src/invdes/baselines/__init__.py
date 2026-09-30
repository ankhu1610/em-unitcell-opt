"""
invdes.baselines: Standard benchmark baselines
"""

from .nn import NearestNeighborBaseline
from .poly import PolynomialBaseline
from .gp_pca import GPPCABaseline

__all__ = [
    "NearestNeighborBaseline",
    "PolynomialBaseline",
    "GPPCABaseline"
]
