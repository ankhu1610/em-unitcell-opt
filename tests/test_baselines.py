import numpy as np
import pytest
from invdes.baselines import NearestNeighborBaseline, PolynomialBaseline, GPPCABaseline
from invdes.data import get_structure_data

def test_baselines_fit_predict():
    """Verify all baselines fit and predict with identical output shapes."""
    X, Y, _ = get_structure_data(0, normalize_inputs=True) # (64, 2), (64, 281, 2)
    X_train, Y_train = X[:10], Y[:10]
    X_test = X[10:12]

    models = [
        ("NN", NearestNeighborBaseline()),
        ("Poly2", PolynomialBaseline(degree=2)),
        ("Poly3", PolynomialBaseline(degree=3)),
        ("GP_PCA", GPPCABaseline(n_components=5))
    ]

    for name, model in models:
        model.fit(X_train, Y_train)
        pred = model.predict(X_test)
        assert pred.shape == (2, 281, 2), f"{name} output shape mismatch: {pred.shape}"
        assert not np.isnan(pred).any(), f"{name} produced NaNs"

def test_gp_pca_return_std():
    """Verify GP-on-PCA uncertainty shape."""
    X, Y, _ = get_structure_data(0, normalize_inputs=True)
    model = GPPCABaseline(n_components=5)
    model.fit(X[:10], Y[:10])
    pred, std = model.predict(X[10:12], return_std=True)
    assert pred.shape == (2, 281, 2)
    assert std.shape == (2, 281, 2)
    assert (std >= 0.0).all()
