import os
import sys
import random
import numpy as np
import torch
import pytest

import invdes
from invdes.utils import set_seed, get_device, DEFAULT_SEEDS

def test_package_import():
    """Verify invdes imports and version is defined."""
    assert hasattr(invdes, "__version__")
    assert invdes.__version__ == "0.1.0"

def test_set_seed_reproducibility():
    """Verify set_seed makes python, numpy, and torch deterministic."""
    set_seed(42)
    py_val1 = random.random()
    np_val1 = np.random.rand(5)
    th_val1 = torch.rand(5)

    set_seed(42)
    py_val2 = random.random()
    np_val2 = np.random.rand(5)
    th_val2 = torch.rand(5)

    assert py_val1 == py_val2
    np.testing.assert_allclose(np_val1, np_val2)
    assert torch.equal(th_val1, th_val2)

def test_default_seeds():
    """Verify default seeds are [0, 1, 2] per AGENTS.md."""
    assert DEFAULT_SEEDS == [0, 1, 2]

def test_cpu_first_device():
    """Verify default device is CPU per AGENTS.md."""
    dev = get_device()
    assert dev.type == "cpu"
    dev_explicit = get_device("cpu")
    assert dev_explicit.type == "cpu"

def test_existing_codebase_presence():
    """Verify existing modules are present on disk for wrapping."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    legacy_dir = os.path.join(root_dir, "inverse-design-em", "src")
    assert os.path.exists(os.path.join(legacy_dir, "models", "gp_pca.py")), "gp_pca.py must exist"
    assert os.path.exists(os.path.join(legacy_dir, "models", "mlp.py")), "mlp.py must exist"
    assert os.path.exists(os.path.join(legacy_dir, "models", "baseline.py")), "baseline.py must exist"
    assert os.path.exists(os.path.join(legacy_dir, "inverse.py")), "inverse.py must exist"
    assert os.path.exists(os.path.join(legacy_dir, "aggregate.py")), "aggregate.py must exist"
