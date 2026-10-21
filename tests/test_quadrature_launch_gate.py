"""Small synthetic audit fixtures; these tests never solve a PWE problem."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"src"))
sys.path.insert(0, str(REPO/"scripts"))
from audit_radial_quadrature import planned_jobs
from invdes.inverse.quadrature_launch_gate import AuditProvenanceError, check_quadrature_launch_gate


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def audit(tmp_path):
    root = tmp_path
    configs = root/"configs"
    configs.mkdir()
    config = json.loads((REPO/"configs/radial_quadrature_audit_v1.json").read_text())
    inverse = json.loads((REPO/"configs/pure_pwe_math_inverse_v1.json").read_text())
    write(configs/"radial_quadrature_audit_v1.json", config)
    write(configs/"pure_pwe_math_inverse_v1.json", inverse)
    write(configs/"pure_pwe_inverse.json", {"gap_tolerance_ghz": .05})
    criteria = configs/"criteria.yaml"
    criteria.write_text("pwe_gradcheck: {max_rel_err: 1e-4}\n"
                        "pwe_convergence: {max_rel_diff_at_chosen_M: .005}\n")
    frozen = root/"frozen_source.py"
    frozen.write_text("# synthetic protected source\n")
    manifest = {
        "config": config, "gradient_tolerance": 1e-4,
        "protected_hashes": {str(p): sha(p) for p in (criteria, configs/"pure_pwe_inverse.json")},
        "code_hashes": {str(p): sha(p) for p in (frozen, configs/"radial_quadrature_audit_v1.json")},
        "runtime": {"dtype": "float64/complex128"},
    }
    run_id = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    out = root/"results/radial_quadrature_audit_v1"
    write(out/"manifest.json", manifest)
    write(out/"status.json", {"status": "complete", "stage": "all", "completed_jobs": 61, "total_jobs": 61})
    write(out/"results.json", {"run_id": run_id, "config": config,
                               "completed_jobs": 61, "existing_gradient_tolerance": 1e-4})
    for job in planned_jobs(config, "all"):
        if job["kind"] == "coefficients":
            result = {"kind": "coefficients", "basis_cutoff": 25, "difference_count": 10201,
                      "rows": [{"order": order, "raw_trace_max_absolute": 1e-15,
                                "conjugacy_max_absolute": 0.,
                                "previous_order_max_absolute_difference": 1e-15}
                               for order in config["coefficient_orders"]]}
        elif job["kind"] == "material":
            result = {"kind": "material", "cutoff": 25, "orders": [64, 128],
                      "sides_um": (np.array(config["material_check_side_multipliers"])*242.5/np.sqrt(3)).tolist(),
                      "max_absolute_eta_difference": 1e-16, "frobenius_eta_difference": 1e-14,
                      "relative_frobenius_eta_difference": 1e-15}
        else:
            sides = np.array(config["side_multipliers"][job["index"]])*242.5/np.sqrt(3)
            if "axis" in job:
                sides[job["axis"]] += job["sign"]*job["step"]
            parameters = (.1+.01*np.arange(281) if job["full"]
                          else np.array(config["path_parameters"]))
            n = len(parameters)
            index = np.arange(n)
            # Unique lower/upper edges separated from neighbors by .002 GHz.
            low, high = (190, 90) if job["full"] else (2, 1)
            f = np.stack((100-.002*(index-low)**2, 140+.002*(index-high)**2), axis=1)
            derivative = np.array([[.1, .4], [.2, .1]])
            f += (sides-np.array([160., 112.]))@derivative + .001/job["cutoff"]
            # Keep the long-path lower band strictly positive.
            gradients = (np.repeat(derivative[:, None, :], 2 if job["full"] else n, axis=1)
                         if job["gradients"] else np.array([]))
            edges = [int(np.argmax(f[:, 0])), int(np.argmin(f[:, 1]))]
            result = {
                "kind": "spectrum", "job": job, "sides_um": sides.tolist(),
                "path_parameters": parameters.tolist(), "frequencies_GHz": f.tolist(),
                "analytic_gradients_GHz_per_um": gradients.tolist(),
                "analytic_gap_gradient_GHz_per_um": (derivative[:, 1]-derivative[:, 0]).tolist()
                if job["gradients"] else [],
                "mode_relative_separations": np.full((2 if job["full"] else n, 2), .1).tolist(),
                "edge_indices": edges, "gap_GHz": float(f[edges[1], 1]-f[edges[0], 0]),
                "edge_uniqueness_margins_GHz": [
                    float(np.sort(f[:, 0])[-1]-np.sort(f[:, 0])[-2]),
                    float(np.sort(f[:, 1])[1]-np.sort(f[:, 1])[0])],
                "max_eigenpair_residual": 1e-10, "cpu_device_max_absolute_GHz": 1e-9,
            }
        write(out/"cases"/(job["name"]+".json"), {"run_id": run_id, "result": result})
    return root, out


def edit_case(out, name, change):
    path = out/"cases"/(name+".json")
    saved = json.loads(path.read_text())
    change(saved["result"])
    write(path, saved)


def recompute_edges(result):
    f = np.array(result["frequencies_GHz"])
    result["edge_indices"] = [int(np.argmax(f[:, 0])), int(np.argmin(f[:, 1]))]
    result["gap_GHz"] = float(f[:, 1].min()-f[:, 0].max())
    result["edge_uniqueness_margins_GHz"] = [
        float(np.sort(f[:, 0])[-1]-np.sort(f[:, 0])[-2]),
        float(np.sort(f[:, 1])[1]-np.sort(f[:, 1])[0])]


def test_complete_numeric_pass_is_not_launch_authorization(audit):
    root, _ = audit
    result = check_quadrature_launch_gate(root)
    assert result["passed"], result["reasons"]
    assert result["launch_authorized"] is False
    assert result["requires_manual_review"]
    assert result["requires_independent_full_path_order256_check"]
    assert result["measurements"]["point_fd_count"] == 12
    assert result["measurements"]["full_gap_fd_count"] == 4
    assert len(result["measurements"]["full_path_cutoff_refinements"]) == 3


def test_missing_audit_is_incomplete_not_exception(tmp_path):
    result = check_quadrature_launch_gate(tmp_path)
    assert not result["passed"]
    assert result["measurements"]["missing_files"]


@pytest.mark.parametrize("mode", ["running", "missing_case", "unexpected_case", "screen"])
def test_incomplete_or_wrong_stage_fails(audit, mode):
    root, out = audit
    if mode == "missing_case":
        (out/"cases/full_d0_M25.json").unlink()
    elif mode == "unexpected_case":
        write(out/"cases/unplanned.json", {})
    else:
        write(out/"status.json", {"status": "running" if mode == "running" else "complete",
                                  "stage": "screen" if mode == "screen" else "all",
                                  "completed_jobs": 61, "total_jobs": 61})
    assert not check_quadrature_launch_gate(root)["passed"]


def test_nonfinite_result_fails_without_provenance_exception(audit):
    root, out = audit
    edit_case(out, "base_d1_M20",
              lambda r: r["analytic_gradients_GHz_per_um"][0][0].__setitem__(0, float("nan")))
    result = check_quadrature_launch_gate(root)
    assert not result["passed"]
    assert any("Nonfinite" in reason for reason in result["reasons"])


def test_failed_point_fd_is_recomputed(audit):
    root, out = audit
    def change(result):
        result["frequencies_GHz"][0][0] += .005
        recompute_edges(result)
    edit_case(out, "fd_d1_M20_L1_h0.01_s1", change)
    result = check_quadrature_launch_gate(root)
    assert not result["passed"]
    assert any("Finite-difference" in reason for reason in result["reasons"])


def test_failed_full_gap_fd_is_recomputed(audit):
    root, out = audit
    def change(result):
        result["frequencies_GHz"] = (np.array(result["frequencies_GHz"])+[0., .001]).tolist()
        recompute_edges(result)
    edit_case(out, "full_fd_d1_M20_L1_h0.01_s1", change)
    result = check_quadrature_launch_gate(root)
    assert not result["passed"]
    assert any("full_fd" in reason for reason in result["reasons"])


@pytest.mark.parametrize("failure", ["source", "fingerprint", "job", "malformed"])
def test_changed_or_malformed_provenance_raises(audit, failure):
    root, out = audit
    if failure == "source":
        (root/"frozen_source.py").write_text("# changed\n")
    elif failure == "malformed":
        (out/"cases/base_d0_M5.json").write_text("{not json")
    elif failure == "job":
        edit_case(out, "base_d0_M5", lambda r: r["job"].__setitem__("cutoff", 6))
    else:
        path = out/"cases/base_d0_M5.json"
        saved = json.loads(path.read_text())
        saved["run_id"] = "wrong"
        write(path, saved)
    with pytest.raises(AuditProvenanceError):
        check_quadrature_launch_gate(root)


@pytest.mark.parametrize("failure", ["mask", "residual", "cpu", "coefficients"])
def test_numerical_safety_guards(audit, failure):
    root, out = audit
    if failure == "mask":
        edit_case(out, "base_d1_M20", lambda r: r["mode_relative_separations"][0].__setitem__(0, 1e-8))
    elif failure == "residual":
        edit_case(out, "full_d0_M25", lambda r: r.__setitem__("max_eigenpair_residual", 1e-6))
    elif failure == "cpu":
        edit_case(out, "full_d0_M25", lambda r: r.__setitem__("cpu_device_max_absolute_GHz", 1e-4))
    else:
        edit_case(out, "coefficient_refinement",
                  lambda r: r["rows"][-1].__setitem__("previous_order_max_absolute_difference", 1e-9))
    assert not check_quadrature_launch_gate(root)["passed"]


def test_full_fd_changed_edge_and_near_tie_are_not_certified(audit):
    root, out = audit
    def change(result):
        # Make the upper edge switch from 90 to 89 and leave a near tie.
        values = result["frequencies_GHz"]
        values[89][1] = values[90][1]-1e-5
        recompute_edges(result)
    edit_case(out, "full_fd_d1_M20_L1_h0.01_s1", change)
    result = check_quadrature_launch_gate(root)
    assert not result["passed"]
    assert any("not unique" in reason for reason in result["reasons"])
    assert any("edges changed" in reason for reason in result["reasons"])


def test_refinement_gap_and_frequency_guards_are_independent(audit):
    root, out = audit
    def change_gap(result):
        result["frequencies_GHz"] = (np.array(result["frequencies_GHz"])+[0., .1]).tolist()
        recompute_edges(result)
    def change_frequency(result):
        result["frequencies_GHz"] = (np.array(result["frequencies_GHz"])+10.).tolist()
        recompute_edges(result)
    edit_case(out, "full_d0_M25", change_gap)
    edit_case(out, "full_d1_M25", change_frequency)
    result = check_quadrature_launch_gate(root)
    assert not result["passed"]
    rows = result["measurements"]["full_path_cutoff_refinements"]
    assert abs(rows[0]["gap_change_GHz"]) > .05
    assert rows[0]["frequency"]["max_relative"] < .005
    assert abs(rows[1]["gap_change_GHz"]) < .05
    assert rows[1]["frequency"]["max_relative"] > .005
