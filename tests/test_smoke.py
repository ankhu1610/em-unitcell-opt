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

def test_physics_codebase_presence():
    """The maintained package contains physics, not the retired ML interfaces."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    physics_dir = os.path.join(root_dir, "src", "invdes")
    for relative in ("interface_pwe.py", "radial_projector_quadrature.py",
                     "inverse/pure_pwe.py", "inverse/physics_trust_region.py",
                     "inverse/quadrature_forward.py"):
        assert os.path.isfile(os.path.join(physics_dir, relative)), relative
    assert not os.path.exists(os.path.join(physics_dir, "baselines"))
    assert not os.path.exists(os.path.join(physics_dir, "multifidelity"))
