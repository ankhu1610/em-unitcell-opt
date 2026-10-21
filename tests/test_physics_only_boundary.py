"""Enforce the user-requested physics-only package boundary."""
import ast
import importlib.metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {"sklearn", "pandas", "polars", "streamlit", "tensorflow", "keras"}


def test_maintained_source_has_no_ml_stack_imports():
    for folder in ("src", "scripts"):
        for path in (ROOT / folder).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                assert not (FORBIDDEN & {name.split('.')[0] for name in names}), path
                assert not any(name.startswith(("invdes.baselines", "invdes.pinn",
                                                "invdes.multifidelity", "src.models"))
                               for name in names), path


def test_installed_project_declares_only_physics_dependencies():
    requires = importlib.metadata.requires("invdes") or []
    lowered = "\n".join(requires).lower()
    for dependency in ("scikit-learn", "pandas", "polars", "matplotlib", "streamlit"):
        assert dependency not in lowered
    for dependency in ("numpy", "scipy", "torch", "pyyaml"):
        assert dependency in lowered


def test_retired_entry_points_are_absent():
    for relative in ("src/invdes/baselines", "src/invdes/pinn", "src/invdes/multifidelity",
                     "src/invdes/inverse/engine.py", "src/models", "src/pipeline",
                     "scripts/train_pinn.py", "scripts/train_pinn_pwe.py", "inverse-design-em"):
        assert not (ROOT / relative).exists(), relative
