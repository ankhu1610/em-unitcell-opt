"""Supplement harness tests use synthetic spectra, never the running audit."""
import copy
import json
from pathlib import Path
import sys
import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from audit_quadrature_full_order import (
    compare_results, content_hash, exclusive_lock, parse_deadline,
    validate_baseline, verify_hashes, write_report,
)
from invdes.projector_experiment import digest


def configuration():
    return json.loads((ROOT/"configs/quadrature_full_order_v1.json").read_text())


def synthetic(config):
    forward = json.loads((ROOT/config["forward_config_source"]).read_text())
    source = json.loads((ROOT/"configs/radial_quadrature_audit_v1.json").read_text())
    parameters = (forward["full_path_start"]+forward["full_path_step"]
                  * np.arange(forward["full_path_count"]))
    f = np.column_stack([100+np.arange(len(parameters))*.01,
                         120+np.arange(len(parameters))*.02])
    edges = [len(parameters)-1, 0]
    gap = float(f[0, 1]-f[-1, 0])
    baseline = {
        "job": {"name": "full_d1_M20", "kind": "spectrum", "index": 1, "cutoff": 20,
                "order": 128, "full": True, "gradients": True},
        "sides_um": (np.array(config["side_multipliers"])*forward["a_um"]/np.sqrt(3)).tolist(),
        "path_parameters": parameters.tolist(), "frequencies_GHz": f.tolist(),
        "gap_GHz": gap, "edge_indices": edges,
        "analytic_gap_gradient_GHz_per_um": [.25, -.15],
        "edge_uniqueness_margins_GHz": [.01, .02],
        "mode_relative_separations": [[.1, .1], [.1, .1]],
        "max_eigenpair_residual": 1e-10, "cpu_device_max_absolute_GHz": 1e-9,
    }
    comparison = {
        "cutoff": 20, "sides_um": baseline["sides_um"],
        "path_parameters": parameters.tolist(), "frequencies_GHz": f.tolist(),
        "gap_GHz": gap, "gradient_GHz_per_um": [.25, -.15],
        "gradient_reliable": True, "evaluation_key": "test", "evaluation_path": "test.json",
        "checks": {
            "edge_indices": edges, "edge_uniqueness_margins_GHz": [.01, .02],
            "active_mode_relative_separations": [.1, .1],
            "gradient_reliability_reasons": [],
            "cpu_device_max_absolute_GHz": 1e-9, "edge_three_mode_max_absolute_GHz": 1e-9,
            "max_relative_eigenpair_residual": 1e-10,
        },
    }
    upstream = {"config": source}
    saved = {"run_id": content_hash(upstream), "result": baseline}
    return forward, upstream, saved, comparison


def test_preregistered_scope_and_guards():
    config = configuration()
    assert (config["baseline_order"], config["comparison_order"], config["cutoff"]) == (128, 256, 20)
    assert config["side_multipliers"] == [1.15, .80]
    guards = config["exploratory_readiness_guards"]
    assert guards["frequency_max_absolute_GHz"] == guards["gap_max_absolute_GHz"] == 1e-5
    assert guards["gap_gradient_max_relative"] == 1e-4
    assert not any(config[name] for name in
                   ("ml_training", "comsol_spectra_used", "comsol_fitting",
                    "production_solver_replacement"))


def test_baseline_manifest_and_geometry_identity():
    config = configuration()
    forward, upstream, saved, _ = synthetic(config)
    validate_baseline(saved, upstream, config, forward)
    changed = copy.deepcopy(saved)
    changed["run_id"] = "wrong"
    with pytest.raises(ValueError, match="manifest"):
        validate_baseline(changed, upstream, config, forward)
    changed = copy.deepcopy(saved)
    changed["result"]["sides_um"][0] += 1e-8
    with pytest.raises(ValueError, match="side"):
        validate_baseline(changed, upstream, config, forward)
    changed = copy.deepcopy(saved)
    changed["result"]["job"]["order"] = 64
    with pytest.raises(ValueError, match="job"):
        validate_baseline(changed, upstream, config, forward)


def test_comparison_pass_and_failure_are_preserved():
    config = configuration()
    _, _, saved, comparison = synthetic(config)
    result = compare_results(saved["result"], comparison, config)
    assert result["exploratory_readiness_passed"]
    comparison["gradient_GHz_per_um"][0] += .01
    result = compare_results(saved["result"], comparison, config)
    assert not result["exploratory_readiness_passed"]
    assert not result["exploratory_readiness_guard_results"]["gap_gradient_max_relative"]
    assert result["analytic_gap_gradients_GHz_per_um"][1][0] == .26


def test_comparison_rejects_mismatched_spectrum_and_nonfinite_gradient():
    config = configuration()
    _, _, saved, comparison = synthetic(config)
    broken = copy.deepcopy(comparison)
    broken["checks"]["edge_indices"] = [0, 0]
    with pytest.raises(ValueError, match="edge"):
        compare_results(saved["result"], broken, config)
    broken = copy.deepcopy(comparison)
    broken["gap_GHz"] += .01
    with pytest.raises(ValueError, match="gap"):
        compare_results(saved["result"], broken, config)
    broken = copy.deepcopy(comparison)
    broken["gradient_GHz_per_um"][0] = float("nan")
    with pytest.raises(ValueError, match="gradients"):
        compare_results(saved["result"], broken, config)


def test_deadline_must_be_aware_and_within_authorization():
    limit = configuration()["deadline_utc"]
    assert parse_deadline(limit, limit) > 0
    with pytest.raises(ValueError, match="timezone"):
        parse_deadline("2026-10-20T17:20:00", limit)
    with pytest.raises(ValueError, match="exceeds"):
        parse_deadline("2026-10-20T17:20:01Z", limit)


def test_lock_refuses_overlap_and_preserves_other_lock(tmp_path):
    lock = tmp_path/"run.lock"
    with exclusive_lock(lock):
        with pytest.raises(FileExistsError):
            with exclusive_lock(lock):
                raise AssertionError("must not enter")
        assert lock.exists()
    assert not lock.exists()


def test_protected_hashes_detect_mutation(tmp_path):
    path = tmp_path/"protected.txt"
    path.write_text("original")
    hashes = {str(path): digest(path)}
    verify_hashes(hashes)
    path.write_text("different")
    with pytest.raises(ValueError, match="Protected"):
        verify_hashes(hashes)


def test_report_is_generated_from_json(tmp_path):
    config = configuration()
    _, _, saved, comparison = synthetic(config)
    result = {"run_id": "test", "config": config, "verified_file_count": 3,
              "comparison": compare_results(saved["result"], comparison, config)}
    write_report(tmp_path, result)
    assert json.loads((tmp_path/"results.json").read_text()) == result
    report = (tmp_path/"REPORT.md").read_text()
    assert "Maximum relative gap-gradient difference" in report
    assert "guards passed: True" in report
