"""Small end-to-end checks of the separate diagnostic harness."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from invdes.projector_experiment import Experiment, DeadlineReached, summarize_pairs

ROOT = Path(__file__).resolve().parents[1]


def experiment(tmp_path):
    config = json.loads((ROOT/"configs/projector_stabilization_v1.json").read_text())
    config["steps_um"] = [0.001]
    config["path_parameters"] = [.37, 1.53]
    config["deadline_utc"] = "2099-01-01T00:00:00+00:00"
    return Experiment(tmp_path, config, "test-fingerprint", {})


def test_radial_full_derivative_matches_rebuilt_pwe(tmp_path):
    audit = experiment(tmp_path)
    audit.current = "tiny"
    result = audit.sensitivity("homothetic_radial", 1, 2, 32, 1e-4)
    for parameter in result["parameters"]:
        row = parameter["steps"][0]
        assert row["projector_changed_grid_points"] == 0
        assert row["passes_simple_sampled_modes"]
        assert row["analytic_vs_full_fd"]["max_relative"] < 1e-4
        assert row["material_relative_error"] < 1e-6
    assert result["gap_scope"] == "five-point proxy"
    assert result["continuum_gradient_certified"] is False


def test_checkpoint_does_not_recompute_or_accept_changed_fingerprint(tmp_path):
    audit = experiment(tmp_path)
    assert audit.get_or_compute("test", lambda: {"value": 123})["value"] == 123
    assert audit.get_or_compute("test", lambda: pytest.fail("recomputed"))["value"] == 123
    audit.run_id = "different"
    with pytest.raises(ValueError, match="fingerprint"):
        audit.get_or_compute("test", lambda: {})


def test_deadline_and_refinement_reporting(tmp_path):
    audit = experiment(tmp_path)
    audit.deadline = time.time()-1
    with pytest.raises(DeadlineReached):
        audit.get_or_compute("unstarted", lambda: {})
    rows = {
        5: {"frequencies_GHz": [[10, 20]], "gap_GHz": 10},
        9: {"frequencies_GHz": [[11, 20]], "gap_GHz": 9},
        15: None,
    }
    comparisons = summarize_pairs(rows, "cutoff", [5, 9, 15], 1e-30)
    assert len(comparisons) == 1
    assert comparisons[0]["gap_difference_GHz"] == -1
    np.testing.assert_allclose(comparisons[0]["frequency"]["max_relative"], 1/11)


def test_declared_job_names_unique_and_complete(tmp_path):
    audit = experiment(tmp_path)
    jobs = audit.jobs(1e-4, "all")
    names = [name for name, _ in jobs]
    assert len(names) == len(set(names)) == 120
    assert len(audit.jobs(1e-4, "screen")) == 12
