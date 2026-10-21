"""Synthetic immutable angular evidence; never execute a forward PWE solver."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"scripts"))
from audit_quadrature_full_order import compare_results, content_hash
from invdes.inverse.angular_result_guard import validate_angular_evidence
from invdes.inverse.quadrature_forward import QuadratureForward
import invdes.inverse.quadrature_forward as forward_module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def no_physics(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("The evidence guard must not run PWE or write a checkpoint")
    for name in ("evaluate", "__call__", "_backend", "_save_checkpoint"):
        monkeypatch.setattr(QuadratureForward, name, forbidden)
    for name in ("modes", "lowest_two", "eta_matrix", "projector_matrices"):
        monkeypatch.setattr(forward_module, name, forbidden)


@pytest.fixture
def evidence(tmp_path):
    root = tmp_path.resolve()
    config = json.loads((ROOT/"configs/quadrature_full_order_v1.json").read_text())
    forward = json.loads((ROOT/config["forward_config_source"]).read_text())
    radial = json.loads((ROOT/"configs/radial_quadrature_audit_v1.json").read_text())
    config_path = root/"configs/quadrature_full_order_v1.json"
    forward_path = root/config["forward_config_source"]
    radial_path = root/"configs/radial_quadrature_audit_v1.json"
    write(config_path, config)
    write(forward_path, forward)
    write(radial_path, radial)
    protected_source = root/"frozen_physics.txt"
    protected_source.write_text("unchanged historical physics")
    source_paths = [root/"scripts/audit_quadrature_full_order.py",
                    root/"src/invdes/inverse/quadrature_forward.py"]
    for path in source_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic immutable source; real validation helpers are imported\n")
    upstream = {"config": radial, "protected_hashes": {str(protected_source): digest(protected_source)},
                "code_hashes": {str(radial_path): digest(radial_path)}}
    upstream_path, baseline_path = root/config["baseline_manifest"], root/config["baseline_case"]
    write(upstream_path, upstream)
    parameters = forward["full_path_start"]+forward["full_path_step"]*np.arange(281)
    frequencies = np.column_stack((100+np.arange(281)*.01, 120+np.arange(281)*.02))
    edges = [280, 0]
    margins = [float(np.sort(frequencies[:, 0])[-1]-np.sort(frequencies[:, 0])[-2]),
               float(np.sort(frequencies[:, 1])[1]-np.sort(frequencies[:, 1])[0])]
    gap = float(frequencies[0, 1]-frequencies[-1, 0])
    gradient = np.array([.25, -.15])
    baseline = {
        "job": {"name": config["baseline_job"], "kind": "spectrum", "index": 1,
                "cutoff": 20, "order": 128, "full": True, "gradients": True},
        "sides_um": (np.array(config["side_multipliers"])*forward["a_um"]/np.sqrt(3)).tolist(),
        "path_parameters": parameters.tolist(), "frequencies_GHz": frequencies.tolist(),
        "gap_GHz": gap, "edge_indices": edges, "edge_uniqueness_margins_GHz": margins,
        "analytic_gap_gradient_GHz_per_um": gradient.tolist(),
        "mode_relative_separations": [[.1, .1], [.1, .1]],
        "max_eigenpair_residual": 1e-10, "cpu_device_max_absolute_GHz": 1e-9,
    }
    write(baseline_path, {"run_id": content_hash(upstream), "result": baseline})
    forward["projector_quadrature_order"] = 256
    manifest = {
        "config": config, "effective_forward_config": forward,
        "baseline_run_id": content_hash(upstream),
        "protected_hashes": {**upstream["protected_hashes"], **upstream["code_hashes"],
                             str(upstream_path): digest(upstream_path),
                             str(baseline_path): digest(baseline_path)},
        "code_hashes": {str(path): digest(path) for path in (*source_paths, config_path, forward_path)},
        "runtime": {"dtype": "float64/complex128"},
    }
    run_id = content_hash(manifest)
    out = root/config["output_directory"]
    write(out/"manifest.json", manifest)
    write(out/"status.json", {"status": "complete", "run_id": run_id})
    validator = QuadratureForward(forward, out, run_id, device="cpu")
    bounds = np.asarray(forward["side_multiplier_bounds"])
    x = (np.asarray(config["side_multipliers"])-bounds[:, 0])/np.diff(bounds, axis=1).ravel()
    identity = validator._identity(x, 20)
    key = validator._identity_hash(identity)
    path = out/"evaluations"/f"{key}.json"
    diagnostics = [{"relative_eigenpair_residual": 1e-10, "hermitian_defect": 0.,
                    "backward_residual": 1e-10, "k_index": index,
                    "backend": "cpu", "requested_backend": "cpu"} for index in range(281)]
    comparison_result = {
        "identity_hash": key, "scientific_fingerprint": run_id, "evaluation_key": key,
        "evaluation_path": str(path.resolve()), "cutoff": 20, "normalized_x": x.tolist(),
        "sides_um": validator.sides_from_x(x).tolist(), "path_parameters": parameters.tolist(),
        "frequencies_GHz": frequencies.tolist(), "gap_GHz": gap,
        "gradient_GHz_per_um": gradient.tolist(),
        "gradient_normalized": (gradient*validator.width).tolist(),
        "gradient_normalized_x": (gradient*validator.width).tolist(),
        "gradient_reliable": True, "cached": False, "elapsed_seconds": 1.,
        "resumed_point_count": 0, "continuum_accuracy_certified": False,
        "gap_scope": "sampled path, not full Brillouin zone",
        "checks": {
            "edge_indices": edges, "edge_path_parameters": parameters[edges].tolist(),
            "edge_uniqueness_margins_GHz": margins, "active_mode_relative_separations": [.1, .1],
            "gradient_reliability_reasons": [], "point_backends": ["cpu"]*281,
            "cpu_device_max_absolute_GHz": 1e-9, "edge_three_mode_max_absolute_GHz": 1e-9,
            "edge_three_mode_max_relative_eigenpair_residual": 1e-10,
            "max_relative_eigenpair_residual": 1e-10,
        },
    }
    record = {"identity": identity, "identity_hash": key, "frequencies_GHz": frequencies.tolist(),
              "point_diagnostics": diagnostics, "elapsed_seconds": 1., "result": comparison_result}
    record["payload_sha256"] = QuadratureForward._payload_hash(record)
    write(path, record)
    comparison = compare_results(baseline, comparison_result, config)
    write(out/"results.json", {
        "status": "complete", "run_id": run_id, "config": config,
        "verified_file_count": len({**manifest["protected_hashes"], **manifest["code_hashes"]}),
        "passes_exploratory_guard": True, "comparison": comparison,
    })
    return {"root": root, "out": out, "evaluation": path, "protected": protected_source}


def edit(path, change):
    value = json.loads(path.read_text())
    change(value)
    write(path, value)


def test_valid_evidence_is_deterministic_read_only_and_not_launch_authorization(evidence):
    root = evidence["root"]
    before = {str(path): digest(path) for path in root.rglob("*") if path.is_file()}
    first, second = validate_angular_evidence(root), validate_angular_evidence(root)
    assert first["passed"], first["reasons"]
    assert first == second
    assert first["state"] == "passed"
    assert first["launch_authorized"] is False
    assert str(evidence["evaluation"]) in first["evidence_hashes"]
    for name in ("manifest.json", "status.json", "results.json"):
        assert str(evidence["out"]/name) in first["evidence_hashes"]
    assert before == {str(path): digest(path) for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("which", ["report_run_id", "status_run_id", "report_config",
                                  "status", "reported_metric", "pass_flag", "nested_pass_flag",
                                  "evaluation_key", "evaluation_path", "verified_count"])
def test_stale_or_inconsistent_report_and_status_fail_closed(evidence, which):
    out = evidence["out"]
    if which == "status":
        edit(out/"status.json", lambda r: r.__setitem__("status", "running"))
    elif which == "status_run_id":
        edit(out/"status.json", lambda r: r.__setitem__("run_id", "wrong"))
    else:
        def change(report):
            if which == "report_run_id":
                report["run_id"] = "wrong"
            elif which == "report_config":
                report["config"]["comparison_order"] = 64
            elif which == "reported_metric":
                report["comparison"]["metrics"]["frequency_max_absolute_GHz"] = 1e-10
            elif which == "pass_flag":
                report["passes_exploratory_guard"] = False
            elif which == "nested_pass_flag":
                report["comparison"]["exploratory_readiness_passed"] = False
            elif which == "evaluation_key":
                report["comparison"]["comparison_evaluation_key"] = "wrong"
            elif which == "evaluation_path":
                report["comparison"]["comparison_evaluation_path"] = str(out/"other.json")
            else:
                report["verified_file_count"] += 1
        edit(out/"results.json", change)
    result = validate_angular_evidence(evidence["root"])
    assert not result["passed"]
    assert result["reasons"]


@pytest.mark.parametrize("which", ["config", "forward_config", "protected", "manifest",
                                  "upstream", "baseline"])
def test_changed_historical_inputs_fail(evidence, which):
    root = evidence["root"]
    paths = {
        "config": root/"configs/quadrature_full_order_v1.json",
        "forward_config": root/"configs/pure_pwe_math_inverse_v1.json",
        "protected": evidence["protected"], "manifest": evidence["out"]/"manifest.json",
        "upstream": root/"results/radial_quadrature_audit_v1/manifest.json",
        "baseline": root/"results/radial_quadrature_audit_v1/cases/full_d1_M20.json",
    }
    path = paths[which]
    if which == "protected":
        path.write_text("changed")
    else:
        edit(path, lambda r: r.__setitem__("unexpected_mutation", True))
    assert not validate_angular_evidence(root)["passed"]


@pytest.mark.parametrize("which", ["checksum", "derived_gap", "normalized_gradient",
                                  "nonfinite", "point_index", "gradient_reliability"])
def test_checkpoint_payload_and_derived_fields_are_validated(evidence, which):
    def change(record):
        if which == "checksum":
            record["frequencies_GHz"][0][0] += .001
            return
        if which == "derived_gap":
            record["result"]["gap_GHz"] += .001
        elif which == "normalized_gradient":
            record["result"]["gradient_normalized"][0] += .001
        elif which == "nonfinite":
            record["result"]["gap_GHz"] = float("nan")
            return
        elif which == "point_index":
            record["point_diagnostics"][0]["k_index"] = 1
        else:
            record["result"]["gradient_reliable"] = False
        record["payload_sha256"] = QuadratureForward._payload_hash(record)
    edit(evidence["evaluation"], change)
    result = validate_angular_evidence(evidence["root"])
    assert not result["passed"]
    assert result["state"] == "invalid"


@pytest.mark.parametrize("which", ["manifest", "report", "status", "evaluation", "partial"])
def test_missing_or_incomplete_evidence_fails_closed(evidence, which):
    if which == "partial":
        def change(record):
            record.pop("result")
            record["payload_sha256"] = QuadratureForward._payload_hash(record)
        edit(evidence["evaluation"], change)
    else:
        names = {"manifest": "manifest.json", "report": "results.json", "status": "status.json"}
        path = evidence["evaluation"] if which == "evaluation" else evidence["out"]/names[which]
        path.unlink()
    result = validate_angular_evidence(evidence["root"])
    assert not result["passed"]
    assert result["state"] == "incomplete"


def test_forged_pass_for_a_recomputed_failed_comparison_is_rejected(evidence):
    def change(record):
        result = record["result"]
        result["gradient_GHz_per_um"][0] += .01
        width = (np.diff(np.array(record["identity"]["config"]["side_multiplier_bounds"]), axis=1).ravel()
                 *record["identity"]["config"]["a_um"]/np.sqrt(3))
        normalized = (np.array(result["gradient_GHz_per_um"])*width).tolist()
        result["gradient_normalized"] = normalized
        result["gradient_normalized_x"] = normalized
        record["payload_sha256"] = QuadratureForward._payload_hash(record)
    edit(evidence["evaluation"], change)
    root, out = evidence["root"], evidence["out"]
    baseline = json.loads((root/"results/radial_quadrature_audit_v1/cases/full_d1_M20.json").read_text())["result"]
    record = json.loads(evidence["evaluation"].read_text())
    def update(report):
        report["comparison"] = compare_results(baseline, record["result"], report["config"])
        assert report["comparison"]["exploratory_readiness_passed"] is False
        report["passes_exploratory_guard"] = True
    edit(out/"results.json", update)
    assert not validate_angular_evidence(root)["passed"]
    edit(out/"results.json", lambda r: r.__setitem__("passes_exploratory_guard", False))
    honest_failure = validate_angular_evidence(root)
    assert not honest_failure["passed"]
    assert honest_failure["state"] == "failed"


def test_empty_root_is_incomplete(tmp_path):
    result = validate_angular_evidence(tmp_path)
    assert not result["passed"]
    assert result["state"] == "incomplete"
