"""
invdes.pinn: Physics-Informed Neural Network (Method 2)
"""

from .mlp import PhysicsMLP, FourierFeatureEncoder
from .losses import PhysicsRegularizedLoss

__all__ = [
    "PhysicsMLP",
    "FourierFeatureEncoder",
    "PhysicsRegularizedLoss"
]
