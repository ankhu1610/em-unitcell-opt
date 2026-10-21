"""Verify a completed inverse run from saved data only; never execute PWE."""
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from invdes.inverse.quadrature_forward import QuadratureForward


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_hashes(hashes):
    for name, expected in hashes.items():
        path = Path(name)
        if not path.is_absolute() or file_hash(path) != expected:
            raise ValueError(f"Protected/source hash mismatch: {name}")


def target_hash(row):
    return json_hash({key: value for key, value in row.items() if key != "payload_sha256"})


def assessment(search, refined, target, tolerance):
    """Recompute independently, not by trusting the experiment's assessment."""
    if search["cutoff"] != 20 or refined["cutoff"] != 25:
        raise ValueError("Expected M20 and M25")
    if (search["normalized_x"] != refined["normalized_x"]
            or search["sides_um"] != refined["sides_um"]):
        raise ValueError("M25 must use the unchanged candidate geometry")
    g20, g25 = float(search["gap_GHz"]), float(refined["gap_GHz"])
    if not np.all(np.isfinite([g20, g25, target, tolerance])) or tolerance <= 0:
        raise ValueError("Assessment requires finite values and positive tolerance")
    e20, e25, delta = abs(g20-target), abs(g25-target), abs(g25-g20)
    return {
        "target_gap_GHz": target, "sides_um": search["sides_um"],
        "normalized_x": search["normalized_x"], "gap_M20_GHz": g20, "gap_M25_GHz": g25,
        "error_M20_GHz": e20, "error_M25_GHz": e25, "cutoff_gap_change_GHz": delta,
        "error_plus_refinement_GHz": e20+delta, "existing_tolerance_GHz": tolerance,
        "passes_empirical_refinement_guard": e20+delta <= tolerance,
        "matches_at_M25": e25 <= tolerance, "continuum_accuracy_certified": False,
        "unchanged_candidate_independently_checked": True,
    }


def parse_preflight_xml(text):
    root = ET.fromstring(text)
    suites = [root] if root.tag == "testsuite" else list(root.findall(".//testsuite"))
    if not suites:
        raise ValueError("Preflight XML has no test suites")
    # Count leaf suites only, avoiding double-counting nested aggregate suites.
    suites = [suite for suite in suites if not suite.findall("testsuite")]
    totals = {key: sum(int(suite.get(key, "0")) for suite in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    totals["seconds"] = sum(float(suite.get("time", "0")) for suite in suites)
    if (totals["tests"] <= 0 or totals["failures"] or totals["errors"]
            or any(value < 0 or not np.isfinite(value) for value in totals.values())):
        raise ValueError("Preflight tests are missing or failed")
    actual_cases = sum(len(suite.findall("testcase")) for suite in suites)
    if actual_cases != totals["tests"]:
        raise ValueError("Preflight XML test count disagrees with test cases")
    totals["passed"] = totals["tests"]-totals["skipped"]
    return totals


def validate_trace(row, evaluations, config):
    """Check every recorded call/trace against its independently validated data."""
    search, calls = row["search"], row["evaluation_calls"]
    trace = search["trace"]
    target = row["target_gap_GHz"]
    if not trace or trace[0].get("event") != "initial":
        raise ValueError("Missing initial trace")
    if trace[0]["x"] != config["initial_normalized_x"]:
        raise ValueError("A target did not start from the fixed nominal point")
    if (len(trace) != len(calls) or search["total_evaluation_calls"] != len(calls)
            or search["trial_evaluations"] != len(trace)-1):
        raise ValueError("Trace/call counts disagree")
    if len(trace)-1 > config["trust_region"]["max_trial_evaluations"]:
        raise ValueError("Recorded search exceeds its preregistered trial budget")
    current_x, current_value = None, None
    methods = {}
    for index, (event, call) in enumerate(zip(trace, calls)):
        key = call["evaluation_key"]
        value = evaluations.get(key)
        if value is None or value["cutoff"] != config["search_cutoff"]:
            raise ValueError("Trace lacks a complete M20 evaluation")
        for field in ("normalized_x", "sides_um", "gap_GHz", "evaluation_path"):
            if call.get(field) != value[field]:
                raise ValueError(f"Call differs from checkpoint: {field}")
        if (not isinstance(call.get("cached"), bool)
                or not isinstance(call.get("call_elapsed_seconds"), (float, int))
                or not np.isfinite(call["call_elapsed_seconds"]) or call["call_elapsed_seconds"] < 0):
            raise ValueError("Malformed call timing/cache diagnostics")
        if (event.get("x") != value["normalized_x"] or event.get("gap_GHz") != value["gap_GHz"]
                or event.get("residual_GHz") != value["gap_GHz"]-target
                or event.get("gradient_reliable") is not value["gradient_reliable"]):
            raise ValueError("Trace differs from checked physics")
        if index == 0:
            current_x, current_value = event["x"], value
            continue
        if (event.get("event") != "trial" or event.get("trial") != index
                or event.get("origin_x") != current_x
                or event.get("residual_before_GHz") != current_value["gap_GHz"]-target
                or not isinstance(event.get("accepted"), bool)):
            raise ValueError("Trial chronology is inconsistent")
        method = event.get("method")
        if method not in ("trust_region", "coordinate_poll"):
            raise ValueError("Unexpected trial method")
        if (method == "trust_region"
                and event.get("model_gradient_normalized") != current_value["gradient_normalized"]):
            raise ValueError("Recorded model gradient differs from its origin evaluation")
        methods[method] = methods.get(method, 0)+1
        if event["accepted"]:
            if abs(value["gap_GHz"]-target) >= abs(current_value["gap_GHz"]-target):
                raise ValueError("Accepted trial did not reduce the target residual")
            current_x, current_value = event["x"], value
    residual = abs(current_value["gap_GHz"]-target)
    matched = residual <= config["internal_tolerance_GHz"]
    if (search["x"] != current_x or search["gap_GHz"] != current_value["gap_GHz"]
            or search["target_gap_GHz"] != target
            or search["absolute_error_GHz"] != residual
            or search["tolerance_GHz"] != config["internal_tolerance_GHz"]
            or search["matched"] is not matched
            or (search["status"] == "matched") != matched
            or search["status"] not in ("matched", "radius_exhausted", "budget_exhausted")
            or search.get("unique_inverse_claimed") is not False
            or search.get("global_feasibility_certified") is not False):
        raise ValueError("Search summary differs from its accepted trace")
    return {"initial_normalized_x": trace[0]["x"], "search_status": search["status"],
            "trial_evaluations": len(trace)-1, "total_evaluation_calls": len(calls),
            "cached_calls": sum(call["cached"] for call in calls), "trial_methods": methods,
            "accepted_trials": sum(event.get("accepted", False) for event in trace[1:]),
            "rejected_trials": sum(not event["accepted"] for event in trace[1:])}


def timing_statistics(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Missing or invalid evaluation timings")
    return {"count": len(values), "total_seconds": float(values.sum()),
            "minimum_seconds": float(values.min()), "median_seconds": float(np.median(values)),
            "mean_seconds": float(values.mean()), "maximum_seconds": float(values.max())}


def build_summary(root):
    """Read-only verification. Hashes are read as bytes, never as spectral labels."""
    root = Path(root).resolve()
    out = root/"results/pure_pwe_math_inverse_v2"
    if (out/"run.lock").exists():
        raise ValueError("Worker lock exists; do not summarize or edit a running experiment")
    read_hashes = {}

    def read_json(path):
        path = Path(path)
        raw = path.read_bytes()
        read_hashes[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    manifest = read_json(out/"manifest.json")
    result = read_json(out/"results.json")
    status = read_json(out/"status.json")
    if (manifest.get("experiment_version") != "pure_pwe_math_inverse_v2"
            or result.get("status") != "complete" or status.get("status") != "complete"
            or status.get("completed_targets") != 3):
        raise ValueError("Require the completed three-target v2 experiment")
    hashes = {**manifest["protected_hashes"], **manifest["code_hashes"]}
    for key in set(manifest["protected_hashes"]) & set(manifest["code_hashes"]):
        if manifest["protected_hashes"][key] != manifest["code_hashes"][key]:
            raise ValueError("Conflicting immutable source hashes")
    verify_hashes(hashes)
    fingerprint = json_hash(manifest)
    config = manifest["config"]
    if (fingerprint != result["scientific_fingerprint"]
            or config != read_json(root/"configs/pure_pwe_math_inverse_v1.json")
            or config["targets_GHz"] != [10., 20., 30.]
            or config["search_cutoff"] != 20 or config["independent_check_cutoff"] != 25
            or config["full_path_count"] != 281
            or config["initial_normalized_x"] != [.5, .5]):
        raise ValueError("Scientific configuration/fingerprint differs from this verification scope")
    false_config = ("saved_anchors_used", "ml_training", "comsol_spectra_used",
                    "comsol_fitting", "production_solver_replacement")
    false_result = ("saved_anchors_used", "ml_training", "comsol_spectra_used",
                    "production_replaced", "continuum_accuracy_certified")
    if any(config.get(k) is not False for k in false_config) or any(
            result.get(k) is not False for k in false_result):
        raise ValueError("No-anchor/no-ML/no-COMSOL scope flags changed")
    if (manifest["gate_review"].get("root_review_acknowledged") is not True
            or manifest["gate_review"].get("passed") is not True):
        raise ValueError("Missing reviewed launch evidence")
    tolerance = read_json(root/config["final_tolerance_source"])[config["final_tolerance_key"]]
    if tolerance != .05 or manifest["final_tolerance_GHz"] != tolerance:
        raise ValueError("The existing final tolerance has changed")
    evaluator = QuadratureForward(config, out, fingerprint, device="cpu")
    # Construction maps coordinates only. Never call evaluate, __call__, modes,
    # eigensolvers, projector assembly, or the experiment's cached_at helper.
    evaluations = {}
    paths = sorted((out/"evaluations").glob("*.json"))
    for path in paths:
        record = read_json(path)
        identity = record.get("identity", {})
        cutoff = identity.get("cutoff")
        if cutoff not in (20, 25):
            raise ValueError("Unexpected evaluation cutoff")
        x = np.asarray([float.fromhex(v) for v in identity["normalized_x_hex"]])
        sides = evaluator.sides_from_x(x)
        expected_identity = evaluator._identity(x, cutoff)
        key = evaluator._identity_hash(expected_identity)
        if path.stem != key:
            raise ValueError("Evaluation filename/identity mismatch")
        evaluator._validate_checkpoint(record, expected_identity, key, x, sides, cutoff, path)
        if "result" not in record:
            raise ValueError("Completed experiment contains a partial evaluation")
        if key in evaluations:
            raise ValueError("Duplicate physical evaluation identity")
        evaluations[key] = record["result"]
    if not evaluations:
        raise ValueError("No complete physical evaluations")
    completed = result.get("completed_distinct_physics_evaluations")
    if not isinstance(completed, list) or len(completed) != len(evaluations):
        raise ValueError("Completed evaluation inventory disagrees")
    completed_keys = []
    for item in completed:
        key = item.get("evaluation_key")
        if key not in evaluations or item != {
                field: evaluations[key][field] for field in
                ("evaluation_key", "cutoff", "normalized_x", "elapsed_seconds", "gap_GHz")}:
            raise ValueError("Completed evaluation inventory fields disagree")
        completed_keys.append(key)
    if len(set(completed_keys)) != len(completed_keys):
        raise ValueError("Duplicate inventory entries")
    if len(result.get("targets", [])) != 3:
        raise ValueError("Missing target results")
    rows = []
    for index, target in enumerate(config["targets_GHz"]):
        row = read_json(out/f"target_{index}.json")
        if (row != result["targets"][index] or row.get("target_gap_GHz") != target
                or row.get("scientific_fingerprint") != fingerprint
                or row.get("payload_sha256") != target_hash(row)):
            raise ValueError("Target identity/checksum/result mismatch")
        calls = read_json(out/f"target_{index}_calls.json")
        if (calls.get("scientific_fingerprint") != fingerprint
                or calls.get("target_gap_GHz") != target or calls.get("calls") != row["evaluation_calls"]):
            raise ValueError("Target call checkpoint mismatch")
        trace = validate_trace(row, evaluations, config)
        search = evaluations.get(row["candidate_search_evaluation"])
        refined = evaluations.get(row["independent_refinement_evaluation"])
        if search is None or refined is None or search["normalized_x"] != row["search"]["x"]:
            raise ValueError("Missing candidate or independent refinement")
        recomputed = assessment(search, refined, target, tolerance)
        if recomputed != row["assessment"]:
            raise ValueError("Stored assessment differs from independent recomputation")
        rows.append({**recomputed, **trace,
                     "candidate_search_evaluation": search["evaluation_key"],
                     "independent_refinement_evaluation": refined["evaluation_key"]})
    xml_path = out/"preflight_tests.xml"
    xml = xml_path.read_bytes()
    read_hashes[str(xml_path.resolve())] = hashlib.sha256(xml).hexdigest()
    preflight = parse_preflight_xml(xml)
    values = list(evaluations.values())
    checks = [value["checks"] for value in values]
    cutoff_counts = {str(cutoff): sum(v["cutoff"] == cutoff for v in values) for cutoff in (20, 25)}
    by_cutoff = {str(cutoff): timing_statistics([
        v["elapsed_seconds"] for v in values if v["cutoff"] == cutoff]) for cutoff in (20, 25)}
    backends = {}
    for check in checks:
        for backend in check["point_backends"]:
            backends[backend] = backends.get(backend, 0)+1
    own_paths = [root/"scripts/summarize_math_inverse_v2.py",
                 root/"tests/test_math_inverse_summary.py"]
    summary = {
        "status": "verified_complete", "scientific_fingerprint": fingerprint,
        "targets": rows, "target_count": len(rows),
        "matches_at_M25": sum(r["matches_at_M25"] for r in rows),
        "passes_empirical_refinement_guard": sum(r["passes_empirical_refinement_guard"] for r in rows),
        "existing_tolerance_GHz": tolerance,
        "protected_file_count": len(manifest["protected_hashes"]),
        "code_file_count": len(manifest["code_hashes"]), "unique_verified_input_count": len(hashes),
        "source_snapshot_hashes": read_hashes,
        "verification_implementation_hashes": {str(p): file_hash(p) for p in own_paths},
        "fresh_distinct_full_path_evaluations": len(values),
        "distinct_evaluations_by_cutoff": cutoff_counts,
        "distinct_normalized_geometries": len({tuple(v["normalized_x"]) for v in values}),
        "evaluation_timing_seconds": timing_statistics([v["elapsed_seconds"] for v in values]),
        "evaluation_timing_by_cutoff": by_cutoff,
        "time_scope": "Saved per-evaluation elapsed time includes assembly, sampled path, checks and derivatives; summed work time is not necessarily experiment wall time.",
        "elapsed_seconds_last_invocation": result["elapsed_seconds_this_invocation"],
        "search_calls_recorded": sum(r["total_evaluation_calls"] for r in rows),
        "cached_search_calls_recorded": sum(r["cached_calls"] for r in rows),
        "sampled_k_count_per_evaluation": config["full_path_count"],
        "point_backend_counts": backends,
        "worst_cpu_device_difference_GHz": max(c["cpu_device_max_absolute_GHz"] for c in checks),
        "worst_edge_three_mode_difference_GHz": max(c["edge_three_mode_max_absolute_GHz"] for c in checks),
        "worst_relative_eigenpair_residual": max(c["max_relative_eigenpair_residual"] for c in checks),
        "all_targets_started_at_fixed_x": True,
        "fixed_initial_normalized_x": config["initial_normalized_x"],
        "saved_anchors_used": False, "ml_training": False, "comsol_spectra_used": False,
        "comsol_fitting": False, "continuum_accuracy_certified": False,
        "unique_inverse_claimed": False, "global_feasibility_certified": False,
        "preflight_tests": preflight, "verification_ran_physics": False,
        "caching_note": "Repeated identical points within this experiment share checked PWE checkpoints; these are not historical saved-design anchors.",
    }
    if (out/"run.lock").exists() or sorted((out/"evaluations").glob("*.json")) != paths:
        raise ValueError("Experiment changed during verification")
    verify_hashes({**hashes, **read_hashes})
    return summary


def render_report(summary):
    lines = [
        "# Independent saved-result verification: anchor-free PWE inverse", "",
        f"Status: {summary['status']}. Targets verified: {summary['target_count']}.",
        f"Existing tolerance: {summary['existing_tolerance_GHz']:.6g} GHz. "
        f"M25 matches: {summary['matches_at_M25']}; empirical refinement guards passed: "
        f"{summary['passes_empirical_refinement_guard']}.", "",
        "|Target GHz|L1 um|L2 um|M20 gap GHz|M25 gap GHz|M20 error GHz|M25 error GHz|Cutoff change GHz|Error + change GHz|Guard|",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in summary["targets"]:
        lines.append(
            f"|{row['target_gap_GHz']:.6g}|{row['sides_um'][0]:.10g}|{row['sides_um'][1]:.10g}|"
            f"{row['gap_M20_GHz']:.10g}|{row['gap_M25_GHz']:.10g}|{row['error_M20_GHz']:.6g}|"
            f"{row['error_M25_GHz']:.6g}|{row['cutoff_gap_change_GHz']:.6g}|"
            f"{row['error_plus_refinement_GHz']:.6g}|{row['passes_empirical_refinement_guard']}|")
    lines += [
        "", f"Distinct full-path physics evaluations: {summary['fresh_distinct_full_path_evaluations']}; "
        f"by cutoff: {summary['distinct_evaluations_by_cutoff']}.",
        f"Fixed initial normalized coordinates for every request: {summary['fixed_initial_normalized_x']}.",
        f"Search calls: {summary['search_calls_recorded']}; cached calls: {summary['cached_search_calls_recorded']}.",
        f"Saved evaluation work time: {summary['evaluation_timing_seconds']['total_seconds']:.6g} seconds; "
        f"mean: {summary['evaluation_timing_seconds']['mean_seconds']:.6g} seconds.",
        summary["time_scope"], "",
        f"Worst CPU/device frequency difference: {summary['worst_cpu_device_difference_GHz']:.6g} GHz.",
        f"Worst relative eigenpair residual: {summary['worst_relative_eigenpair_residual']:.6g}.",
        f"Recorded path-point backends: {summary['point_backend_counts']}.",
        f"Immutable input/source files verified: {summary['unique_verified_input_count']}.",
        f"Preflight tests: {summary['preflight_tests']['passed']} passed; "
        f"{summary['preflight_tests']['failures']} failures; {summary['preflight_tests']['errors']} errors.", "",
        "Verification recomputed assessments from saved frequencies and checked checkpoint identities, "
        "checksums, full-path diagnostics, accepted traces, unchanged refinement geometries and protected hashes.",
        "It did not execute PWE, ML, COMSOL, fitting or inverse optimization.",
        summary["caching_note"], "",
        "These are sampled-path gaps and empirical two-cutoff checks, not certified continuum errors or "
        "complete Brillouin-zone gaps. A scalar gap need not identify a unique pair of side lengths; "
        "unresolved local searches do not prove a target infeasible.",
    ]
    return "\n".join(lines)+"\n"


def save_new_summary(output, summary):
    output = Path(output)
    paths = [output/"verification_summary.json", output/"VERIFICATION.md"]
    if (output/"run.lock").exists() or any(path.exists() for path in paths):
        raise ValueError("Do not overwrite existing verification files or write while the worker is active")
    encoded = json.dumps(summary, indent=2, allow_nan=False)+"\n"
    # Validate rendering before any output is written.
    render_report(json.loads(encoded))
    with paths[0].open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    saved = json.loads(paths[0].read_text(encoding="utf-8"))
    with paths[1].open("x", encoding="utf-8") as stream:
        stream.write(render_report(saved))


def main():
    summary = build_summary(ROOT)
    output = ROOT/"results/pure_pwe_math_inverse_v2"
    save_new_summary(output, summary)
    print(json.dumps({"status": summary["status"], "verified_targets": summary["target_count"],
                      "output": str(output/"verification_summary.json")}))


if __name__ == "__main__":
    main()
