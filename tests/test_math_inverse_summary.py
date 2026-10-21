"""Read-only saved-result verification tests; no physics or live-run writes."""
import copy
import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from summarize_math_inverse_v2 import (
    assessment, build_summary, file_hash, json_hash, parse_preflight_xml,
    render_report, save_new_summary, target_hash, timing_statistics,
)
from invdes.inverse.quadrature_forward import QuadratureForward
import invdes.inverse.quadrature_forward as forward_module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture(autouse=True)
def forbid_physics(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Summary must never compute or save physics")
    for name in ("evaluate", "__call__", "_backend", "_save_checkpoint"):
        monkeypatch.setattr(QuadratureForward, name, forbidden)
    for name in ("modes", "lowest_two", "eta_matrix", "projector_matrices"):
        monkeypatch.setattr(forward_module, name, forbidden)


def make_evaluation(validator, x, cutoff, gap):
    identity = validator._identity(x, cutoff)
    key = validator._identity_hash(identity)
    path = validator.output/f"{key}.json"
    f = np.column_stack([100+np.arange(281)*.01,
                         102.8+gap+np.arange(281)*.02])
    edges = [280, 0]
    margins = [float(np.sort(f[:, 0])[-1]-np.sort(f[:, 0])[-2]),
               float(np.sort(f[:, 1])[1]-np.sort(f[:, 1])[0])]
    gradient = np.array([.25, -.15])
    value = {
        "identity_hash": key, "scientific_fingerprint": validator.fingerprint,
        "evaluation_key": key, "evaluation_path": str(path.resolve()), "cutoff": cutoff,
        "normalized_x": list(x), "sides_um": validator.sides_from_x(x).tolist(),
        "path_parameters": validator.parameters.tolist(), "frequencies_GHz": f.tolist(),
        "gap_GHz": float(f[0, 1]-f[-1, 0]), "gradient_GHz_per_um": gradient.tolist(),
        "gradient_normalized": (gradient*validator.width).tolist(),
        "gradient_normalized_x": (gradient*validator.width).tolist(),
        "gradient_reliable": True, "cached": False, "elapsed_seconds": 1.,
        "resumed_point_count": 0, "continuum_accuracy_certified": False,
        "gap_scope": "sampled path, not full Brillouin zone",
        "checks": {
            "edge_indices": edges, "edge_path_parameters": validator.parameters[edges].tolist(),
            "edge_uniqueness_margins_GHz": margins, "active_mode_relative_separations": [.1, .1],
            "gradient_reliability_reasons": [], "point_backends": ["cpu"]*281,
            "cpu_device_max_absolute_GHz": 1e-9, "edge_three_mode_max_absolute_GHz": 1e-9,
            "edge_three_mode_max_relative_eigenpair_residual": 1e-10,
            "max_relative_eigenpair_residual": 1e-10,
        },
    }
    diagnostics = [{"relative_eigenpair_residual": 1e-10, "hermitian_defect": 0.,
                    "backward_residual": 1e-10, "k_index": i,
                    "backend": "cpu", "requested_backend": "cpu"} for i in range(281)]
    record = {"identity": identity, "identity_hash": key, "frequencies_GHz": f.tolist(),
              "point_diagnostics": diagnostics, "elapsed_seconds": 1., "result": value}
    record["payload_sha256"] = QuadratureForward._payload_hash(record)
    write(path, record)
    return value


@pytest.fixture
def completed(tmp_path):
    root = tmp_path.resolve()
    out = root/"results/pure_pwe_math_inverse_v2"
    config = json.loads((ROOT/"configs/pure_pwe_math_inverse_v1.json").read_text())
    config_path = root/"configs/pure_pwe_math_inverse_v1.json"
    final_path = root/config["final_tolerance_source"]
    write(config_path, config)
    write(final_path, {config["final_tolerance_key"]: .05})
    for relative in ("scripts/summarize_math_inverse_v2.py", "tests/test_math_inverse_summary.py"):
        path = root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic verification implementation")
    protected = root/"protected.txt"
    protected.write_text("immutable")
    manifest = {
        "experiment_version": "pure_pwe_math_inverse_v2", "config": config,
        "protected_hashes": {str(protected): file_hash(protected)},
        "code_hashes": {str(config_path): file_hash(config_path), str(final_path): file_hash(final_path)},
        "gate_review": {"passed": True, "root_review_acknowledged": True},
        "final_tolerance_GHz": .05,
    }
    fingerprint = json_hash(manifest)
    write(out/"manifest.json", manifest)
    validator = QuadratureForward(config, out, fingerprint)
    initial = make_evaluation(validator, [.5, .5], 20, 15.)
    evaluations = [initial]
    rows = []
    for index, target in enumerate(config["targets_GHz"]):
        x = [.2+.2*index, .7]
        search = make_evaluation(validator, x, 20, target)
        refined = make_evaluation(validator, x, 25, target+.002)
        evaluations += [search, refined]
        trace = [
            {"event": "initial", "x": [.5, .5], "gap_GHz": initial["gap_GHz"],
             "residual_GHz": initial["gap_GHz"]-target, "gradient_reliable": True},
            {"event": "trial", "trial": 1, "method": "trust_region", "origin_x": [.5, .5],
             "x": x, "gap_GHz": search["gap_GHz"], "residual_GHz": search["gap_GHz"]-target,
             "residual_before_GHz": initial["gap_GHz"]-target,
             "model_gradient_normalized": initial["gradient_normalized"],
             "gradient_reliable": True, "accepted": True},
        ]
        calls = [{**{field: value[field] for field in
                     ("normalized_x", "sides_um", "gap_GHz", "evaluation_key", "evaluation_path")},
                  "cached": value is initial and index > 0, "call_elapsed_seconds": 1.}
                 for value in (initial, search)]
        row = {
            "scientific_fingerprint": fingerprint, "target_gap_GHz": target,
            "search": {"trace": trace, "total_evaluation_calls": 2, "trial_evaluations": 1,
                       "x": x, "gap_GHz": search["gap_GHz"], "target_gap_GHz": target,
                       "absolute_error_GHz": abs(search["gap_GHz"]-target),
                       "tolerance_GHz": .01, "matched": True, "status": "matched",
                       "unique_inverse_claimed": False, "global_feasibility_certified": False},
            "assessment": assessment(search, refined, target, .05), "evaluation_calls": calls,
            "candidate_search_evaluation": search["evaluation_key"],
            "independent_refinement_evaluation": refined["evaluation_key"],
        }
        row["payload_sha256"] = target_hash(row)
        rows.append(row)
        write(out/f"target_{index}.json", row)
        write(out/f"target_{index}_calls.json",
              {"scientific_fingerprint": fingerprint, "target_gap_GHz": target, "calls": calls})
    result = {
        "status": "complete", "scientific_fingerprint": fingerprint, "targets": rows,
        "saved_anchors_used": False, "ml_training": False, "comsol_spectra_used": False,
        "production_replaced": False, "continuum_accuracy_certified": False,
        "elapsed_seconds_this_invocation": 8.,
        "completed_distinct_physics_evaluations": [{field: value[field] for field in
            ("evaluation_key", "cutoff", "normalized_x", "elapsed_seconds", "gap_GHz")}
            for value in evaluations],
    }
    write(out/"results.json", result)
    write(out/"status.json", {"status": "complete", "completed_targets": 3})
    (out/"preflight_tests.xml").write_text(
        '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0" time="1">'
        '<testcase name="synthetic"/></testsuite></testsuites>')
    return {"root": root, "out": out, "protected": protected, "evaluation": validator.output/f"{initial['evaluation_key']}.json"}


def edit(path, change):
    value = json.loads(path.read_text())
    change(value)
    write(path, value)


def test_completed_verification_is_deterministic_and_read_only(completed):
    root = completed["root"]
    before = {str(path): file_hash(path) for path in root.rglob("*") if path.is_file()}
    first = build_summary(root)
    assert first == build_summary(root)
    assert first["status"] == "verified_complete"
    assert first["target_count"] == first["matches_at_M25"] == 3
    assert first["fresh_distinct_full_path_evaluations"] == 7
    assert first["distinct_evaluations_by_cutoff"] == {"20": 4, "25": 3}
    assert first["cached_search_calls_recorded"] == 2
    assert first["preflight_tests"]["passed"] == 1
    assert not first["verification_ran_physics"]
    assert before == {str(path): file_hash(path) for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("failure", ["lock", "running", "protected", "checksum", "partial",
                                    "inventory", "target_hash", "flags", "preflight"])
def test_incomplete_or_changed_evidence_fails(completed, failure):
    out = completed["out"]
    if failure == "lock":
        (out/"run.lock").write_text("123")
    elif failure == "running":
        edit(out/"status.json", lambda v: v.update(status="running"))
    elif failure == "protected":
        completed["protected"].write_text("changed")
    elif failure == "checksum":
        edit(completed["evaluation"], lambda v: v.update(payload_sha256="wrong"))
    elif failure == "partial":
        def make_partial(value):
            value.pop("result")
            value["payload_sha256"] = QuadratureForward._payload_hash(value)
        edit(completed["evaluation"], make_partial)
    elif failure == "inventory":
        edit(out/"results.json", lambda v: v["completed_distinct_physics_evaluations"].pop())
    elif failure == "target_hash":
        edit(out/"target_0.json", lambda v: v.update(payload_sha256="wrong"))
    elif failure == "flags":
        edit(out/"results.json", lambda v: v.update(saved_anchors_used=True))
    else:
        (out/"preflight_tests.xml").write_text(
            '<testsuite tests="1" failures="1" errors="0"><testcase/></testsuite>')
    with pytest.raises(ValueError):
        build_summary(completed["root"])


@pytest.mark.parametrize("failure", ["assessment", "initial_point", "trace_gap", "call_gap"])
def test_checksum_alone_does_not_validate_claims(completed, failure):
    out = completed["out"]
    row = json.loads((out/"target_0.json").read_text())
    if failure == "assessment":
        row["assessment"]["error_M25_GHz"] = 0.
    elif failure == "initial_point":
        row["search"]["trace"][0]["x"] = [.1, .1]
    elif failure == "trace_gap":
        row["search"]["trace"][1]["gap_GHz"] += 1.
    else:
        row["evaluation_calls"][1]["gap_GHz"] += 1.
        edit(out/"target_0_calls.json", lambda v: v.update(calls=row["evaluation_calls"]))
    row["payload_sha256"] = target_hash(row)
    write(out/"target_0.json", row)
    edit(out/"results.json", lambda v: v["targets"].__setitem__(0, row))
    with pytest.raises(ValueError):
        build_summary(completed["root"])


def test_assessment_preserves_cancellation_failure():
    search = {"cutoff": 20, "normalized_x": [.3, .4], "sides_um": [160., 110.], "gap_GHz": 10.04}
    refined = {**search, "cutoff": 25, "gap_GHz": 10.}
    result = assessment(search, refined, 10., .05)
    assert result["matches_at_M25"]
    assert not result["passes_empirical_refinement_guard"]
    refined["sides_um"] = [160.01, 110.]
    with pytest.raises(ValueError, match="unchanged"):
        assessment(search, refined, 10., .05)


def test_save_only_new_artifacts_from_verified_json(completed):
    summary = build_summary(completed["root"])
    out = completed["out"]
    before = {str(path): file_hash(path) for path in out.rglob("*") if path.is_file()}
    save_new_summary(out, summary)
    assert json.loads((out/"verification_summary.json").read_text()) == summary
    assert (out/"VERIFICATION.md").read_text() == render_report(summary)
    for name, checksum in before.items():
        assert file_hash(name) == checksum
    with pytest.raises(ValueError, match="overwrite"):
        save_new_summary(out, summary)


def test_xml_counts_and_timing_errors():
    with pytest.raises(ValueError, match="count"):
        parse_preflight_xml('<testsuite tests="2" failures="0" errors="0"><testcase/></testsuite>')
    with pytest.raises(ValueError):
        timing_statistics([float("nan")])
    with pytest.raises(ValueError):
        timing_statistics([])
