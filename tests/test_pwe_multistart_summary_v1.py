"""Synthetic saved-evidence tests; all PWE computation is forbidden."""
import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from summarize_pwe_multistart_v1 import (
    OWN_SOURCES, assessment, build_summary, file_hash, json_hash,
    render_report, save_new_summary, summarize_target, target_hash, write_summary,
)
from invdes.inverse.quadrature_forward import QuadratureForward
import invdes.inverse.quadrature_forward as forward_module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def edit(path, change):
    value = json.loads(path.read_text())
    change(value)
    write(path, value)


@pytest.fixture(autouse=True)
def forbid_physics(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Verification must not execute or save PWE")
    for name in ("evaluate", "__call__", "_backend", "_save_checkpoint"):
        monkeypatch.setattr(QuadratureForward, name, forbidden)
    for name in ("modes", "lowest_two", "eta_matrix", "projector_matrices"):
        monkeypatch.setattr(forward_module, name, forbidden)


def make_evaluation(validator, x, cutoff, gap):
    identity = validator._identity(x, cutoff)
    key = validator._identity_hash(identity)
    path = validator.output/f"{key}.json"
    frequencies = np.column_stack([100+np.arange(281)*.01,
                                   102.8+gap+np.arange(281)*.02])
    edges = [280, 0]
    margins = [float(np.sort(frequencies[:, 0])[-1]-np.sort(frequencies[:, 0])[-2]),
               float(np.sort(frequencies[:, 1])[1]-np.sort(frequencies[:, 1])[0])]
    gradient = np.array([.25, -.15])
    value = {
        "identity_hash": key, "scientific_fingerprint": validator.fingerprint,
        "evaluation_key": key, "evaluation_path": str(path.resolve()), "cutoff": cutoff,
        "normalized_x": list(x), "sides_um": validator.sides_from_x(x).tolist(),
        "path_parameters": validator.parameters.tolist(), "frequencies_GHz": frequencies.tolist(),
        "gap_GHz": float(frequencies[0, 1]-frequencies[-1, 0]),
        "gradient_GHz_per_um": gradient.tolist(),
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
    diagnostics = [
        {"relative_eigenpair_residual": 1e-10, "hermitian_defect": 0.,
         "backward_residual": 1e-10, "k_index": i,
         "backend": "cpu", "requested_backend": "cpu"}
        for i in range(281)]
    record = {"identity": identity, "identity_hash": key,
              "frequencies_GHz": frequencies.tolist(), "point_diagnostics": diagnostics,
              "elapsed_seconds": 1., "result": value}
    record["payload_sha256"] = QuadratureForward._payload_hash(record)
    write(path, record)
    return value


def aggregate(rows):
    result = []
    for target in (7.5, 15., 22.5, 27.5):
        selected = [row for row in rows if row["target_gap_GHz"] == target]
        passed = sum(row["success"] for row in selected)
        result.append({
            "target_gap_GHz": target, "planned_starts": 3,
            "completed_starts": 3, "not_yet_completed": 0,
            "successful_starts": passed, "unsuccessful_completed_starts": 3-passed,
            "success_fraction_of_planned": passed/3, "success_fraction_of_completed": passed/3,
            "all_planned_starts_completed": True,
        })
    return result


@pytest.fixture
def completed(tmp_path):
    root = tmp_path.resolve()
    out = root/"results/pure_pwe_multistart_v1"
    config = json.loads((ROOT/"configs/pure_pwe_multistart_v1.json").read_text())
    forward = json.loads((ROOT/config["base_config"]).read_text())
    config_path = root/"configs/pure_pwe_multistart_v1.json"
    base_path = root/config["base_config"]
    final_path = root/forward["final_tolerance_source"]
    write(config_path, config)
    write(base_path, forward)
    write(final_path, {forward["final_tolerance_key"]: .05})
    protected = root/"protected.txt"
    protected.write_text("immutable")
    paths = [config_path, base_path, final_path]
    for relative in OWN_SOURCES:
        path = root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen synthetic verification source")
        paths.append(path)
    plan = [
        {"case_id": f"target_{ti:02d}__{start['id']}", "target_index": ti,
         "target_gap_GHz": target, "start_id": start["id"],
         "initial_normalized_x": start["normalized_x"]}
        for ti, target in enumerate(config["targets_GHz"]) for start in config["starts"]]
    manifest = {
        "experiment_version": "pure_pwe_multistart_v1", "config": config,
        "effective_forward_config": forward, "case_plan": plan,
        "protected_hashes": {str(protected): file_hash(protected)},
        "code_hashes": {str(p): file_hash(p) for p in paths},
        "gate_review": {"passed": True, "root_review_acknowledged": True},
        "final_tolerance_GHz": .05,
    }
    fingerprint = json_hash(manifest)
    write(out/"manifest.json", manifest)
    write(out/"gate_review.json", manifest["gate_review"])
    validator = QuadratureForward(forward, out, fingerprint)
    initial = [make_evaluation(validator, start["normalized_x"], 20, 16.+si)
               for si, start in enumerate(config["starts"])]
    values, rows = initial.copy(), []
    for index, case in enumerate(plan):
        si, ti = index % 3, index // 3
        target, start = case["target_gap_GHz"], initial[si]
        x = [.2+.15*ti+.01*si, .2+.02*si]
        search = make_evaluation(validator, x, 20, target)
        refined = make_evaluation(validator, x, 25, target+.002)
        values += [search, refined]
        trace = [
            {"event": "initial", "x": case["initial_normalized_x"],
             "gap_GHz": start["gap_GHz"], "residual_GHz": start["gap_GHz"]-target,
             "gradient_reliable": True},
            {"event": "trial", "trial": 1, "method": "trust_region",
             "origin_x": case["initial_normalized_x"], "x": x,
             "gap_GHz": search["gap_GHz"], "residual_GHz": search["gap_GHz"]-target,
             "residual_before_GHz": start["gap_GHz"]-target,
             "model_gradient_normalized": start["gradient_normalized"],
             "gradient_reliable": True, "accepted": True},
        ]
        calls = [
            {**{field: value[field] for field in
                ("normalized_x", "sides_um", "gap_GHz", "evaluation_key", "evaluation_path")},
             "cached": value is start and ti > 0, "call_elapsed_seconds": 1.}
            for value in (start, search)]
        row = {
            **case, "scientific_fingerprint": fingerprint,
            "initial_sides_um": validator.sides_from_x(case["initial_normalized_x"]).tolist(),
            "search": {"trace": trace, "total_evaluation_calls": 2, "trial_evaluations": 1,
                       "x": x, "gap_GHz": search["gap_GHz"], "target_gap_GHz": target,
                       "absolute_error_GHz": abs(search["gap_GHz"]-target),
                       "tolerance_GHz": .01, "matched": True, "status": "matched",
                       "unique_inverse_claimed": False, "global_feasibility_certified": False},
            "assessment": assessment(search, refined, target, .05), "evaluation_calls": calls,
            "candidate_search_evaluation": search["evaluation_key"],
            "independent_refinement_evaluation": refined["evaluation_key"],
            "success": True, "outcome": "success",
        }
        row["payload_sha256"] = target_hash(row)
        rows.append(row)
        write(out/f"{case['case_id']}.json", row)
        write(out/f"{case['case_id']}_calls.json",
              {"scientific_fingerprint": fingerprint, **case, "calls": calls})
    result = {
        "status": "complete", "scientific_fingerprint": fingerprint, "cases": rows,
        "saved_anchors_used": False, "ml_training": False, "comsol_spectra_used": False,
        "comsol_fitting": False, "production_replaced": False,
        "continuum_accuracy_certified": False, "unique_inverse_claimed": False,
        "global_feasibility_certified": False,
        "planned_case_count": 12, "completed_cases": 12, "successful_cases": 12,
        "per_target": aggregate(rows), "elapsed_seconds_this_invocation": 30.,
        "completed_distinct_physics_evaluations": [
            {field: value[field] for field in
             ("evaluation_key", "cutoff", "normalized_x", "elapsed_seconds", "gap_GHz")}
            for value in values],
    }
    write(out/"results.json", result)
    write(out/"status.json", {
        "status": "complete", "scientific_fingerprint": fingerprint,
        "completed_cases": 12, "planned_cases": 12, "successful_cases": 12})
    (out/"preflight_tests.xml").write_text(
        '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0" time="1">'
        '<testcase name="synthetic"/></testsuite></testsuites>')
    return {
        "root": root, "out": out, "protected": protected,
        "evaluation": validator.output/f"{initial[0]['evaluation_key']}.json",
        "validator": validator,
    }


def rewrite_case(out, index, change):
    result = json.loads((out/"results.json").read_text())
    row = result["cases"][index]
    change(row)
    row["payload_sha256"] = target_hash(row)
    write(out/f"{row['case_id']}.json", row)
    write(out/"results.json", result)
    return row


def test_readonly_complete_matrix_is_deterministic(completed):
    root = completed["root"]
    before = {str(p): file_hash(p) for p in root.rglob("*") if p.is_file()}
    summary = build_summary(root)
    assert summary == build_summary(root)
    assert summary["case_count"] == summary["expected_case_count"] == 12
    assert summary["target_count"] == 4 and summary["starts_per_target"] == 3
    assert summary["successful_cases"] == summary["matches_at_M25"] == 12
    assert summary["fresh_distinct_full_path_evaluations"] == 27
    assert summary["distinct_evaluations_by_cutoff"] == {"20": 15, "25": 12}
    assert summary["cached_search_calls_recorded"] == 9
    assert all(row["case_count"] == 3 for row in summary["per_target"])
    assert all(all(span > 0 for span in row["returned_side_spans_um"])
               for row in summary["per_target"])
    assert summary["verification_ran_physics"] is False
    assert before == {str(p): file_hash(p) for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("failure", [
    "lock", "running", "protected", "checksum", "partial", "inventory",
    "flags", "preflight", "case_count", "case_plan", "status_fingerprint",
    "aggregate", "success_total", "gate_review", "base_config", "source_freeze",
])
def test_malformed_or_incomplete_evidence_rejected(completed, failure):
    out = completed["out"]
    if failure == "lock":
        (out/"run.lock").write_text("123")
    elif failure == "running":
        edit(out/"status.json", lambda value: value.update(status="running"))
    elif failure == "protected":
        completed["protected"].write_text("changed")
    elif failure == "checksum":
        edit(completed["evaluation"], lambda value: value.update(payload_sha256="wrong"))
    elif failure == "partial":
        def remove_result(value):
            value.pop("result")
            value["payload_sha256"] = QuadratureForward._payload_hash(value)
        edit(completed["evaluation"], remove_result)
    elif failure == "inventory":
        edit(out/"results.json", lambda value: value["completed_distinct_physics_evaluations"].pop())
    elif failure == "flags":
        edit(out/"results.json", lambda value: value.update(comsol_fitting=True))
    elif failure == "preflight":
        (out/"preflight_tests.xml").write_text(
            '<testsuite tests="1" failures="1" errors="0"><testcase/></testsuite>')
    elif failure == "case_count":
        edit(out/"results.json", lambda value: value["cases"].pop())
    elif failure == "case_plan":
        edit(out/"manifest.json", lambda value: value["case_plan"].pop())
    elif failure == "status_fingerprint":
        edit(out/"status.json", lambda value: value.update(scientific_fingerprint="prior run"))
    elif failure == "aggregate":
        edit(out/"results.json", lambda value: value["per_target"][0].update(planned_starts=2))
    elif failure == "success_total":
        edit(out/"results.json", lambda value: value.update(successful_cases=11))
    elif failure == "gate_review":
        edit(out/"gate_review.json", lambda value: value.update(passed=False))
    elif failure == "base_config":
        edit(completed["root"]/"configs/pure_pwe_math_inverse_v1.json",
             lambda value: value.update(search_cutoff=19))
    else:
        (completed["root"]/OWN_SOURCES[0]).write_text("# changed")
    with pytest.raises(ValueError):
        build_summary(completed["root"])


@pytest.mark.parametrize("failure", [
    "checksum", "start", "trace_start", "initial_sides", "assessment", "success",
    "outcome", "trace_gap", "call_gap", "external_checkpoint",
])
def test_rehashed_case_cannot_forge_claims(completed, failure):
    out = completed["out"]

    def change(row):
        if failure == "start":
            row["initial_normalized_x"] = [.4, .4]
        elif failure == "trace_start":
            row["search"]["trace"][0]["x"] = [.1, .9]
        elif failure == "initial_sides":
            row["initial_sides_um"][0] += 1.
        elif failure == "assessment":
            row["assessment"]["error_M25_GHz"] = 0.
        elif failure == "success":
            row["success"] = False
        elif failure == "outcome":
            row["outcome"] = "infeasible"
        elif failure == "trace_gap":
            row["search"]["trace"][1]["gap_GHz"] += 1.
        elif failure == "call_gap":
            row["evaluation_calls"][1]["gap_GHz"] += 1.
        elif failure == "external_checkpoint":
            row["evaluation_calls"][0]["evaluation_path"] = str(
                completed["root"]/"results/pure_pwe_math_inverse_v2/evaluations/old.json")
    row = rewrite_case(out, 0, change)
    if failure == "checksum":
        edit(out/f"{row['case_id']}.json", lambda value: value.update(payload_sha256="wrong"))
    elif failure in ("call_gap", "external_checkpoint"):
        edit(out/f"{row['case_id']}_calls.json",
             lambda value: value.update(calls=row["evaluation_calls"]))
    with pytest.raises(ValueError):
        build_summary(completed["root"])


def test_new_unreferenced_evaluation_is_rejected(completed):
    value = make_evaluation(completed["validator"], [.99, .99], 20, 19.)
    edit(completed["out"]/"results.json", lambda result:
         result["completed_distinct_physics_evaluations"].append({
             key: value[key] for key in
             ("evaluation_key", "cutoff", "normalized_x", "elapsed_seconds", "gap_GHz")}))
    with pytest.raises(ValueError, match="Unreferenced"):
        build_summary(completed["root"])


@pytest.mark.parametrize("kind", ["local_search", "refinement"])
def test_unresolved_attempts_remain_failures_in_full_matrix(completed, kind):
    out, validator = completed["out"], completed["validator"]
    result = json.loads((out/"results.json").read_text())
    row = result["cases"][0]
    target, x = row["target_gap_GHz"], row["search"]["x"]
    if kind == "local_search":
        search = make_evaluation(validator, x, 20, target+.03)
        refined = make_evaluation(validator, x, 25, target+.03)
        row["search"].update(
            gap_GHz=search["gap_GHz"], absolute_error_GHz=abs(search["gap_GHz"]-target),
            matched=False, status="radius_exhausted")
        row["search"]["trace"][-1].update(
            gap_GHz=search["gap_GHz"], residual_GHz=search["gap_GHz"]-target)
        row["evaluation_calls"][-1]["gap_GHz"] = search["gap_GHz"]
        row["outcome"] = "unresolved_local_search"
    else:
        search = json.loads((validator.output/f"{row['candidate_search_evaluation']}.json")
                            .read_text())["result"]
        refined = make_evaluation(validator, x, 25, target+.1)
        row["outcome"] = "independent_refinement_failed"
    row["assessment"] = assessment(search, refined, target, .05)
    row["success"] = False
    row["payload_sha256"] = target_hash(row)
    write(out/f"{row['case_id']}.json", row)
    edit(out/f"{row['case_id']}_calls.json",
         lambda value: value.update(calls=row["evaluation_calls"]))
    replacements = {value["evaluation_key"]: value for value in (search, refined)}
    for item in result["completed_distinct_physics_evaluations"]:
        if item["evaluation_key"] in replacements:
            item["gap_GHz"] = replacements[item["evaluation_key"]]["gap_GHz"]
    result["per_target"] = aggregate(result["cases"])
    result["successful_cases"] = 11
    write(out/"results.json", result)
    edit(out/"status.json", lambda value: value.update(successful_cases=11))
    summary = build_summary(completed["root"])
    assert summary["successful_cases"] == 11 and summary["case_count"] == 12
    assert summary["unresolved_case_ids"] == [row["case_id"]]
    assert summary["per_target"][0]["successful_starts"] == 2
    assert summary["per_target"][0]["case_count"] == 3
    assert not summary["per_target"][0]["all_starts_successful"]
    if kind == "local_search":
        assert summary["cases"][0]["passes_empirical_refinement_guard"]
        assert summary["passes_empirical_refinement_guard"] == 12
    assert not summary["global_feasibility_certified"]


def test_summary_saves_only_new_final_artifacts(completed):
    out = completed["out"]
    before = {str(p): file_hash(p) for p in out.rglob("*") if p.is_file()}
    summary = write_summary(completed["root"])
    assert json.loads((out/"verification_summary.json").read_text()) == summary
    assert (out/"VERIFICATION.md").read_text() == render_report(summary)
    assert "not as proof of infeasibility" in (out/"VERIFICATION.md").read_text()
    for name, digest in before.items():
        assert file_hash(name) == digest
    with pytest.raises(ValueError, match="overwrite"):
        save_new_summary(out, summary)


def test_save_refuses_active_lock(completed):
    summary = build_summary(completed["root"])
    (completed["out"]/"run.lock").write_text("123")
    with pytest.raises(ValueError, match="worker"):
        save_new_summary(completed["out"], summary)
    assert not (completed["out"]/"verification_summary.json").exists()


def test_failed_guard_retained_in_denominator():
    rows = [
        {"target_gap_GHz": 7.5, "case_id": str(i), "search_status": "matched",
         "sides_um": [150.+i, 110.-i], "matches_at_M25": True,
         "passes_empirical_refinement_guard": i != 1, "error_M25_GHz": .002,
         "success": i != 1}
        for i in range(3)]
    summary = summarize_target(rows)
    assert summary["case_count"] == 3 and summary["matches_at_M25"] == 3
    assert summary["passes_empirical_refinement_guard"] == summary["successful_starts"] == 2
    assert not summary["all_starts_pass_empirical_refinement_guard"]
    assert summary["returned_side_spans_um"] == [2., 2.]
    assert summary["maximum_pairwise_length_distance_um"] == pytest.approx(np.sqrt(8))
    assert summary["global_feasibility_certified"] is False


def test_empty_target_aggregation_rejected():
    with pytest.raises(ValueError):
        summarize_target([])
