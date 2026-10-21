"""Independently verify a completed multistart run without executing PWE."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from summarize_math_inverse_v2 import (
    assessment, file_hash, json_hash, parse_preflight_xml, target_hash,
    timing_statistics, validate_trace, verify_hashes,
)
from invdes.inverse.quadrature_forward import QuadratureForward

VERSION = "pure_pwe_multistart_v1"
CONFIG_PATH = "configs/pure_pwe_multistart_v1.json"
OWN_SOURCES = ("scripts/summarize_pwe_multistart_v1.py",
               "tests/test_pwe_multistart_summary_v1.py",
               "scripts/summarize_math_inverse_v2.py")


def summarize_target(rows):
    """Describe finite-sample starting-point sensitivity, not uniqueness."""
    if not rows or len({row["target_gap_GHz"] for row in rows}) != 1:
        raise ValueError("One target and at least one case required")
    sides = np.asarray([row["sides_um"] for row in rows], dtype=float)
    if sides.shape != (len(rows), 2) or not np.all(np.isfinite(sides)):
        raise ValueError("Malformed returned lengths")
    passed = [row for row in rows if row["passes_empirical_refinement_guard"]]
    distances = [float(np.linalg.norm(a-b)) for a in sides for b in sides]
    return {
        "target_gap_GHz": rows[0]["target_gap_GHz"], "case_count": len(rows),
        "case_ids": [row["case_id"] for row in rows],
        "search_matches": sum(row["search_status"] == "matched" for row in rows),
        "matches_at_M25": sum(row["matches_at_M25"] for row in rows),
        "passes_empirical_refinement_guard": len(passed),
        "successful_starts": sum(row["success"] for row in rows),
        "all_starts_successful": all(row["success"] for row in rows),
        "all_starts_match_at_M25": all(row["matches_at_M25"] for row in rows),
        "all_starts_pass_empirical_refinement_guard": len(passed) == len(rows),
        "M25_error_range_GHz": [
            min(row["error_M25_GHz"] for row in rows),
            max(row["error_M25_GHz"] for row in rows)],
        "returned_L1_range_um": [float(sides[:, 0].min()), float(sides[:, 0].max())],
        "returned_L2_range_um": [float(sides[:, 1].min()), float(sides[:, 1].max())],
        "returned_side_spans_um": np.ptp(sides, axis=0).tolist(),
        "maximum_pairwise_length_distance_um": max(distances),
        "length_spread_includes_unresolved_candidates": True,
        "global_feasibility_certified": False, "unique_inverse_claimed": False,
    }


def _merged_hashes(manifest):
    protected, code = manifest["protected_hashes"], manifest["code_hashes"]
    for name in set(protected) & set(code):
        if protected[name] != code[name]:
            raise ValueError("Conflicting immutable source hashes")
    return {**protected, **code}


def _inventory(evaluations, recorded):
    if not isinstance(recorded, list) or len(recorded) != len(evaluations):
        raise ValueError("Completed evaluation inventory disagrees")
    keys = []
    for item in recorded:
        key = item.get("evaluation_key")
        if key not in evaluations or item != {
                field: evaluations[key][field] for field in
                ("evaluation_key", "cutoff", "normalized_x", "elapsed_seconds", "gap_GHz")}:
            raise ValueError("Completed evaluation inventory fields disagree")
        keys.append(key)
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate inventory entries")


def build_summary(root):
    """Read and validate final evidence only. Never call an evaluator."""
    root = Path(root).resolve()
    out = root/"results"/VERSION
    if (out/"run.lock").exists():
        raise ValueError("Worker lock exists; wait for the experiment to finish")
    read_hashes = {}

    def read_json(path):
        path = Path(path)
        raw = path.read_bytes()
        read_hashes[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    manifest = read_json(out/"manifest.json")
    result = read_json(out/"results.json")
    status = read_json(out/"status.json")
    if (manifest.get("experiment_version") != VERSION
            or result.get("status") != "complete" or status.get("status") != "complete"):
        raise ValueError("Require the completed multistart experiment")
    hashes = _merged_hashes(manifest)
    verify_hashes(hashes)
    config = manifest["config"]
    fingerprint = json_hash(manifest)
    if (fingerprint != result["scientific_fingerprint"]
            or config != read_json(root/CONFIG_PATH)):
        raise ValueError("Scientific configuration or fingerprint changed")
    if (manifest["gate_review"].get("root_review_acknowledged") is not True
            or manifest["gate_review"].get("passed") is not True):
        raise ValueError("Missing reviewed launch evidence")
    for relative in OWN_SOURCES:
        path = str((root/relative).resolve())
        if path not in hashes:
            raise ValueError(f"Verification implementation was not frozen before launch: {path}")
    forward_config = manifest["effective_forward_config"]
    base_path = (root/config["base_config"]).resolve()
    if not base_path.is_relative_to(root) or forward_config != read_json(base_path):
        raise ValueError("The effective solver configuration differs from its unchanged base")
    if (forward_config["search_cutoff"] != 20
            or forward_config["independent_check_cutoff"] != 25
            or forward_config["full_path_count"] != 281):
        raise ValueError("Unexpected physics evaluation scope")
    if config["targets_GHz"] != [7.5, 15., 22.5, 27.5]:
        raise ValueError("Unexpected preregistered new targets")
    starts = config["starts"]
    if ([start["normalized_x"] for start in starts]
            != [[.5, .5], [.1, .9], [.9, .1]]):
        raise ValueError("Unexpected preregistered starts")
    if len({start["id"] for start in starts}) != len(starts):
        raise ValueError("Duplicate start identity")
    false_flags = ("saved_anchors_used", "ml_training", "comsol_spectra_used",
                   "comsol_fitting", "production_solver_replacement")
    if any(config.get(key) is not False for key in false_flags):
        raise ValueError("No-anchor/no-ML/no-COMSOL scope flags changed")
    if any(result.get(key) is not False for key in (
            "saved_anchors_used", "ml_training", "comsol_spectra_used",
            "comsol_fitting", "production_replaced", "continuum_accuracy_certified",
            "unique_inverse_claimed", "global_feasibility_certified")):
        raise ValueError("Result scope flags changed")
    if read_json(out/"gate_review.json") != manifest["gate_review"]:
        raise ValueError("Saved gate review differs from its frozen evidence")
    tolerance = read_json(root/forward_config["final_tolerance_source"])[
        forward_config["final_tolerance_key"]]
    if tolerance != .05 or manifest["final_tolerance_GHz"] != tolerance:
        raise ValueError("Existing final tolerance changed")
    evaluator = QuadratureForward(forward_config, out, fingerprint, device="cpu")
    evaluations = {}
    paths = sorted((out/"evaluations").glob("*.json"))
    for path in paths:
        if path.resolve().parent != (out/"evaluations").resolve():
            raise ValueError("External evaluation references are not allowed")
        record = read_json(path)
        identity = record.get("identity", {})
        cutoff = identity.get("cutoff")
        if cutoff not in (20, 25):
            raise ValueError("Unexpected evaluation cutoff")
        x = np.asarray([float.fromhex(v) for v in identity["normalized_x_hex"]])
        sides = evaluator.sides_from_x(x)
        expected = evaluator._identity(x, cutoff)
        key = evaluator._identity_hash(expected)
        if path.stem != key or key in evaluations:
            raise ValueError("Evaluation filename or identity mismatch")
        evaluator._validate_checkpoint(record, expected, key, x, sides, cutoff, path)
        if "result" not in record:
            raise ValueError("Complete experiment contains a partial evaluation")
        evaluations[key] = record["result"]
    if not evaluations:
        raise ValueError("No complete physical evaluations")
    _inventory(evaluations, result.get("completed_distinct_physics_evaluations"))
    prescribed = [
        {"case_id": f"target_{ti:02d}__{start['id']}", "target_index": ti,
         "target_gap_GHz": target, "start_id": start["id"],
         "initial_normalized_x": start["normalized_x"]}
        for ti, target in enumerate(config["targets_GHz"])
        for start in starts]
    if (len(result.get("cases", [])) != len(prescribed)
            or status.get("completed_cases") != len(prescribed)
            or result.get("completed_cases") != len(prescribed)
            or result.get("planned_case_count") != len(prescribed)
            or status.get("planned_cases") != len(prescribed)
            or status.get("scientific_fingerprint") != fingerprint
            or manifest.get("case_plan") != prescribed):
        raise ValueError("All prescribed target/start cases must be complete")
    rows, referenced_keys = [], set()
    for index, expected in enumerate(prescribed):
        case_id = expected["case_id"]
        row = read_json(out/f"{case_id}.json")
        if (row != result["cases"][index]
                or row.get("scientific_fingerprint") != fingerprint
                or row.get("payload_sha256") != target_hash(row)
                or any(row.get(k) != v for k, v in expected.items())):
            raise ValueError("Case identity/checksum/result mismatch")
        calls = read_json(out/f"{case_id}_calls.json")
        if (calls.get("scientific_fingerprint") != fingerprint
                or any(calls.get(k) != v for k, v in expected.items())
                or calls.get("calls") != row["evaluation_calls"]):
            raise ValueError("Case call checkpoint mismatch")
        trace_config = copy.deepcopy(forward_config)
        trace_config["initial_normalized_x"] = expected["initial_normalized_x"]
        trace = validate_trace(row, evaluations, trace_config)
        search = evaluations.get(row["candidate_search_evaluation"])
        refined = evaluations.get(row["independent_refinement_evaluation"])
        if search is None or refined is None or search["normalized_x"] != row["search"]["x"]:
            raise ValueError("Missing candidate or independent refinement")
        recomputed = assessment(search, refined, expected["target_gap_GHz"], tolerance)
        if recomputed != row["assessment"]:
            raise ValueError("Stored assessment differs from independent recomputation")
        initial_sides = evaluator.sides_from_x(expected["initial_normalized_x"]).tolist()
        success = bool(row["search"]["matched"] and recomputed["matches_at_M25"]
                       and recomputed["passes_empirical_refinement_guard"])
        outcome = ("success" if success else "unresolved_local_search"
                   if not row["search"]["matched"] else "independent_refinement_failed")
        if (row.get("initial_sides_um") != initial_sides or row.get("success") is not success
                or row.get("outcome") != outcome):
            raise ValueError("Case initial lengths or success/outcome claim disagrees")
        referenced_keys.update(call["evaluation_key"] for call in row["evaluation_calls"])
        referenced_keys.update((search["evaluation_key"], refined["evaluation_key"]))
        rows.append({**expected, **recomputed, **trace,
                     "initial_sides_um": initial_sides, "success": success, "outcome": outcome,
                     "candidate_search_evaluation": search["evaluation_key"],
                     "independent_refinement_evaluation": refined["evaluation_key"]})
    if referenced_keys != set(evaluations):
        raise ValueError("Unreferenced evaluations outside the prescribed case traces")
    successes = sum(row["success"] for row in rows)
    expected_aggregates = []
    for target in config["targets_GHz"]:
        selected = [row for row in rows if row["target_gap_GHz"] == target]
        passed = sum(row["success"] for row in selected)
        count = len(selected)
        expected_aggregates.append({
            "target_gap_GHz": target, "planned_starts": count,
            "completed_starts": count, "not_yet_completed": 0,
            "successful_starts": passed, "unsuccessful_completed_starts": count-passed,
            "success_fraction_of_planned": passed/count,
            "success_fraction_of_completed": passed/count,
            "all_planned_starts_completed": True,
        })
    if (result.get("per_target") != expected_aggregates
            or result.get("successful_cases") != successes
            or status.get("successful_cases") != successes):
        raise ValueError("Stored reliability counts or denominators disagree")
    xml_path = out/"preflight_tests.xml"
    xml = xml_path.read_bytes()
    read_hashes[str(xml_path.resolve())] = hashlib.sha256(xml).hexdigest()
    preflight = parse_preflight_xml(xml)
    values = list(evaluations.values())
    checks = [v["checks"] for v in values]
    backends = {}
    for check in checks:
        for backend in check["point_backends"]:
            backends[backend] = backends.get(backend, 0)+1
    elapsed = result["elapsed_seconds_this_invocation"]
    if not isinstance(elapsed, (int, float)) or not np.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Invalid last-invocation elapsed time")
    summary = {
        "status": "verified_complete", "scientific_fingerprint": fingerprint,
        "cases": rows, "case_count": len(rows), "target_count": len(config["targets_GHz"]),
        "starts_per_target": len(starts), "expected_case_count": len(prescribed),
        "per_target": [summarize_target([row for row in rows if row["target_gap_GHz"] == target])
                       for target in config["targets_GHz"]],
        "search_matches": sum(row["search_status"] == "matched" for row in rows),
        "successful_cases": successes,
        "matches_at_M25": sum(row["matches_at_M25"] for row in rows),
        "passes_empirical_refinement_guard": sum(row["passes_empirical_refinement_guard"]
                                               for row in rows),
        "unresolved_case_ids": [row["case_id"] for row in rows if not row["success"]],
        "existing_tolerance_GHz": tolerance,
        "protected_file_count": len(manifest["protected_hashes"]),
        "code_file_count": len(manifest["code_hashes"]),
        "unique_verified_input_count": len(hashes),
        "source_snapshot_hashes": read_hashes,
        "verification_implementation_hashes": {
            str(root/p): file_hash(root/p) for p in OWN_SOURCES},
        "fresh_distinct_full_path_evaluations": len(values),
        "distinct_evaluations_by_cutoff": {
            str(c): sum(v["cutoff"] == c for v in values) for c in (20, 25)},
        "distinct_normalized_geometries": len({tuple(v["normalized_x"]) for v in values}),
        "evaluation_timing_seconds": timing_statistics([v["elapsed_seconds"] for v in values]),
        "evaluation_timing_by_cutoff": {
            str(c): timing_statistics([v["elapsed_seconds"] for v in values if v["cutoff"] == c])
            for c in (20, 25)},
        "elapsed_seconds_last_invocation": elapsed,
        "search_calls_recorded": sum(row["total_evaluation_calls"] for row in rows),
        "cached_search_calls_recorded": sum(row["cached_calls"] for row in rows),
        "sampled_k_count_per_evaluation": forward_config["full_path_count"],
        "point_backend_counts": backends,
        "worst_cpu_device_difference_GHz": max(c["cpu_device_max_absolute_GHz"] for c in checks),
        "worst_edge_three_mode_difference_GHz": max(
            c["edge_three_mode_max_absolute_GHz"] for c in checks),
        "worst_relative_eigenpair_residual": max(c["max_relative_eigenpair_residual"] for c in checks),
        "all_cases_started_at_prescribed_x": True,
        "saved_anchors_used": False, "ml_training": False, "comsol_spectra_used": False,
        "comsol_fitting": False, "continuum_accuracy_certified": False,
        "unique_inverse_claimed": False, "global_feasibility_certified": False,
        "preflight_tests": preflight, "verification_ran_physics": False,
        "caching_note": "Only exact-point checkpoints generated within this new scientific run are shared; historical anchor libraries are not used.",
        "time_scope": "Per-evaluation elapsed time includes assembly, the sampled path, checks and derivatives. Summed work time need not equal wall time; last invocation excludes earlier resumed invocations.",
        "reliability_scope": "Finite deterministic target/start matrix, not a statistical guarantee over all requested gaps or starting geometries.",
    }
    if (out/"run.lock").exists() or sorted((out/"evaluations").glob("*.json")) != paths:
        raise ValueError("Experiment changed during verification")
    verify_hashes({**hashes, **read_hashes})
    return summary


def render_report(summary):
    lines = [
        "# Independent verification: pure-PWE multistart inverse", "",
        f"Status: {summary['status']}. Verified cases: {summary['case_count']}/"
        f"{summary['expected_case_count']}; targets: {summary['target_count']}; "
        f"starts per target: {summary['starts_per_target']}.",
        f"Existing final tolerance: {summary['existing_tolerance_GHz']:.6g} GHz. "
        f"M25 matches: {summary['matches_at_M25']}/{summary['case_count']}; "
        f"empirical refinement guards passed: {summary['passes_empirical_refinement_guard']}/"
        f"{summary['case_count']}.", "",
        f"Combined success (internal search, M25 match and refinement guard): "
        f"{summary['successful_cases']}/{summary['case_count']}.", "",
        "|Target GHz|Start|Initial L1 um|Initial L2 um|Returned L1 um|Returned L2 um|M25 gap GHz|M25 error GHz|M20-to-M25 change GHz|Guard|Search status|",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in summary["cases"]:
        lines.append(
            f"|{row['target_gap_GHz']:.6g}|{row['start_id']}|"
            f"{row['initial_sides_um'][0]:.10g}|{row['initial_sides_um'][1]:.10g}|"
            f"{row['sides_um'][0]:.10g}|{row['sides_um'][1]:.10g}|"
            f"{row['gap_M25_GHz']:.10g}|{row['error_M25_GHz']:.6g}|"
            f"{row['cutoff_gap_change_GHz']:.6g}|{row['passes_empirical_refinement_guard']}|"
            f"{row['search_status']}|")
    lines += ["", "## Starting-point sensitivity", "",
              "|Target GHz|Combined success / starts|M25 matches / starts|Guards passed / starts|Returned L1 span um|Returned L2 span um|Maximum pairwise length distance um|",
              "|---|---|---|---|---|---|---|"]
    for row in summary["per_target"]:
        lines.append(
            f"|{row['target_gap_GHz']:.6g}|{row['successful_starts']}/{row['case_count']}|"
            f"{row['matches_at_M25']}/{row['case_count']}|"
            f"{row['passes_empirical_refinement_guard']}/{row['case_count']}|"
            f"{row['returned_side_spans_um'][0]:.9g}|{row['returned_side_spans_um'][1]:.9g}|"
            f"{row['maximum_pairwise_length_distance_um']:.9g}|")
    lines += [
        "", "Length spread includes all returned candidates, including unresolved searches. "
        "Different lengths can realize the same scalar gap; these spans are not parameter-recovery errors.",
        f"Unresolved cases under the combined success criterion: {summary['unresolved_case_ids']}.",
        "A local search or refinement-guard failure is recorded as unresolved, not as proof of infeasibility.",
        "", "## Numerical evidence", "",
        f"Fresh distinct full-path evaluations: {summary['fresh_distinct_full_path_evaluations']}; "
        f"cutoff counts: {summary['distinct_evaluations_by_cutoff']}.",
        f"Search calls: {summary['search_calls_recorded']}; reused exact-point calls: "
        f"{summary['cached_search_calls_recorded']}.",
        f"Summed saved evaluation time: {summary['evaluation_timing_seconds']['total_seconds']:.8g} seconds; "
        f"last-invocation elapsed time: {summary['elapsed_seconds_last_invocation']:.8g} seconds.",
        summary["time_scope"],
        f"Worst CPU/device frequency difference: {summary['worst_cpu_device_difference_GHz']:.8g} GHz.",
        f"Worst relative eigenpair residual: {summary['worst_relative_eigenpair_residual']:.8g}.",
        f"Verified immutable source/input files: {summary['unique_verified_input_count']}.",
        f"Preflight tests: {summary['preflight_tests']['passed']} passed; "
        f"{summary['preflight_tests']['failures']} failures; "
        f"{summary['preflight_tests']['errors']} errors.", "",
        "Verification checked saved checkpoint identities, payload hashes, all path diagnostics, every "
        "search trace and its prescribed starting geometry, and independently recomputed both cutoff errors "
        "and the unchanged-candidate refinement guard. Verification did not execute PWE.",
        summary["caching_note"], "",
        "No ML training, COMSOL execution, COMSOL spectra, fitting, historical anchor initialization, "
        "or changes to protected earlier solver/results are part of this experiment.",
        summary["reliability_scope"],
        "The gaps refer to the existing sampled path. Two-cutoff agreement is empirical, not a continuum "
        "error bound or a complete Brillouin-zone gap certificate. No uniqueness is asserted.",
    ]
    return "\n".join(lines)+"\n"


def save_new_summary(output, summary):
    output = Path(output)
    paths = [output/"verification_summary.json", output/"VERIFICATION.md"]
    if (output/"run.lock").exists() or any(path.exists() for path in paths):
        raise ValueError("Do not overwrite verification files or write while a worker is active")
    encoded = json.dumps(summary, indent=2, allow_nan=False)+"\n"
    rendered = render_report(json.loads(encoded))
    with paths[0].open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    with paths[1].open("x", encoding="utf-8") as stream:
        stream.write(rendered)


def write_summary(root):
    """Verify a completed unlocked run and create only new final artifacts."""
    summary = build_summary(root)
    save_new_summary(Path(root)/"results"/VERSION, summary)
    return summary


def main():
    summary = write_summary(ROOT)
    print(json.dumps({"status": summary["status"], "verified_cases": summary["case_count"],
                      "output": str(ROOT/"results"/VERSION/"verification_summary.json")}))


if __name__ == "__main__":
    main()
