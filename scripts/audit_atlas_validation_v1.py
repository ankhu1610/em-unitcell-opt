"""Read-only physics audit and reproducible diagnostics for the atlas experiment.

This script NEVER evaluates PWE or runs COMSOL. Only its own derived artifacts
are written; all source experiments remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import Delaunay

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from invdes.inverse.atlas_validation_v1 import select_atlas_starts
from invdes.inverse.physics_trust_region import solve_gap_target
from invdes.inverse.quadrature_forward import QuadratureForward
from invdes.inverse.response_atlas_v1 import (
    candidate_grid, choose_next_point, deduplicate_observations,
    fit_pwe_surface, load_comsol_reference,
)
from summarize_math_inverse_v2 import assessment, file_hash, json_hash, verify_hashes
from summarize_pwe_multistart_v1 import build_summary as verify_multistart

VERSION = "pure_pwe_atlas_validation_v1"
CONTROLS = ("max_trial_evaluations", "initial_radius", "maximum_radius",
            "minimum_radius", "acceptance_ratio", "shrink_ratio", "expand_ratio",
            "gradient_norm_floor", "rejected_models_before_poll")


class Evidence:
    def __init__(self):
        self.hashes = {}

    def read(self, path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if str(path) in self.hashes and self.hashes[str(path)] != digest:
            raise ValueError(f"Evidence changed during audit: {path}")
        self.hashes[str(path)] = digest
        return json.loads(raw)

    def recheck(self):
        verify_hashes(self.hashes)


def verified_evaluations(out, manifest, evidence):
    """Validate all stored path eigenvalues, diagnostics and checkpoint identities."""
    out = Path(out).resolve()
    if (out / "run.lock").exists():
        raise ValueError(f"Worker still owns {out.name}")
    evaluator = QuadratureForward(manifest["effective_forward_config"], out,
                                  json_hash(manifest), device="cpu")
    values = {}
    paths = sorted((out / "evaluations").glob("*.json"))
    for path in paths:
        record = evidence.read(path)
        identity = record["identity"]
        cutoff = identity["cutoff"]
        if cutoff not in (20, 25):
            raise ValueError("Unexpected cutoff")
        x = np.array([float.fromhex(v) for v in identity["normalized_x_hex"]])
        expected = evaluator._identity(x, cutoff)
        key = evaluator._identity_hash(expected)
        if path.stem != key or key in values:
            raise ValueError("Checkpoint filename/identity mismatch")
        evaluator._validate_checkpoint(record, expected, key, x,
                                       evaluator.sides_from_x(x), cutoff, path)
        if "result" not in record:
            raise ValueError("Incomplete physics checkpoint")
        values[key] = record["result"]
    if not values or (out / "run.lock").exists():
        raise ValueError("No complete stable physics evidence")
    if paths != sorted((out / "evaluations").glob("*.json")):
        raise ValueError("Checkpoint inventory changed during audit")
    return evaluator, values


def checked_call(call, values):
    value = values[call["evaluation_key"]]
    if value["cutoff"] != 20:
        raise ValueError("Search call is not M20")
    for field in ("normalized_x", "sides_um", "gap_GHz"):
        if call[field] != value[field]:
            raise ValueError(f"Search call differs from physics: {field}")
    return value


def replay_search(row, values, config):
    """Re-execute ONLY deterministic optimizer arithmetic against saved values."""
    calls = row["evaluation_calls"]
    cursor = 0

    def lookup(x):
        nonlocal cursor
        if cursor >= len(calls):
            raise ValueError("Missing search call during deterministic replay")
        value = checked_call(calls[cursor], values)
        elapsed = calls[cursor]["call_elapsed_seconds"]
        if not isinstance(elapsed, (int, float)) or not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid call duration")
        if not np.array_equal(np.asarray(x), value["normalized_x"]):
            raise ValueError("Search requested a different geometry during replay")
        cursor += 1
        return value

    replayed = solve_gap_target(lookup, row["target_gap_GHz"], row["atlas_start_normalized"],
                               tolerance_ghz=config["internal_tolerance_GHz"],
                               **{k: config["trust_region"][k] for k in CONTROLS})
    if cursor != len(calls) or replayed != row["search"]:
        raise ValueError("Stored search differs from deterministic replay")


def error_metrics(errors):
    values = np.asarray(errors, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Finite nonempty scalar-error vector required")
    return {"count": len(values), "bias_GHz": float(np.mean(values)),
            "MAE_GHz": float(np.mean(np.abs(values))),
            "RMSE_GHz": float(np.sqrt(np.mean(values**2))),
            "max_absolute_GHz": float(np.max(np.abs(values)))}


def coordinate_groups(points, tolerance=1e-8):
    """Connected groups of numerically equivalent geometries, for CV only."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("Finite coordinate pairs required")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Nonnegative finite coordinate tolerance required")
    remaining, groups = set(range(len(points))), []
    while remaining:
        group, frontier = [], [min(remaining)]
        while frontier:
            i = frontier.pop()
            if i not in remaining:
                continue
            remaining.remove(i)
            group.append(i)
            frontier.extend(j for j in sorted(remaining)
                            if np.linalg.norm(points[i]-points[j]) <= tolerance)
        groups.append(sorted(group))
    return groups


def source_audit(root, evidence):
    """Verify the 58-point atlas and replay both adaptive selection sequences."""
    root = Path(root)
    multi_summary = verify_multistart(root)
    evidence.hashes.update(multi_summary["source_snapshot_hashes"])
    multi_dir = root / "results/pure_pwe_multistart_v1"
    multi = evidence.read(multi_dir / "results.json")
    multi_manifest = evidence.read(multi_dir / "manifest.json")
    evidence.hashes.update(multi_manifest["protected_hashes"])
    evidence.hashes.update(multi_manifest["code_hashes"])
    forward = multi_manifest["effective_forward_config"]
    observations = deduplicate_observations([
        {"normalized_x": c["normalized_x"], "gap_GHz": c["gap_GHz"]}
        for row in multi["cases"] for c in row["evaluation_calls"] if not c["cached"]])
    batch_diagnostics = []
    for version in ("pure_pwe_response_atlas_v1", "pure_pwe_response_atlas_v2"):
        out = root / "results" / version
        manifest, result = (evidence.read(out / name) for name in ("manifest.json", "results.json"))
        if (result["status"] != "complete_for_requested_budget"
                or manifest["effective_forward_config"] != forward
                or result["scientific_fingerprint"] != json_hash(manifest)
                or result["protocol"] != manifest):
            raise ValueError("Atlas protocol/status mismatch")
        verify_hashes(manifest["seed_source_hashes"])
        config = manifest["atlas_config"]
        if config != evidence.read(root / "configs" / f"{version}.json"):
            raise ValueError("Atlas config changed")
        _, values = verified_evaluations(out, manifest, evidence)
        if result["seed_observations"] != len(observations):
            raise ValueError("Atlas seed count mismatch")
        seen, prospective = set(), []
        for index, row in enumerate(result["new_evaluations"]):
            expected, rationale = choose_next_point(observations, config, index)
            if expected != row["normalized_x"] or rationale != row["selection_rationale"]:
                raise ValueError("Adaptive point selection failed deterministic replay")
            value = checked_call(row, values)
            if row["elapsed_seconds"] != value["elapsed_seconds"]:
                raise ValueError("Atlas elapsed time differs from checkpoint")
            pred = float(fit_pwe_surface(observations, .02)([expected])[0])
            prospective.append({"normalized_x": expected, "prediction_GHz": pred,
                                "actual_M20_GHz": value["gap_GHz"],
                                "prediction_error_GHz": pred-value["gap_GHz"],
                                "selection_rationale": rationale})
            seen.add(row["evaluation_key"])
            observations = deduplicate_observations(observations + [row])
        if seen != set(values) or result["distinct_pwe_points"] != len(observations):
            raise ValueError("Atlas sample inventory mismatch")
        batch_diagnostics.append({"version": version, "prospective_samples": prospective,
                                  "prospective_error": error_metrics([
                                      r["prediction_error_GHz"] for r in prospective])})
    prior_x = [r["assessment"]["normalized_x"] for r in multi["cases"] if r["success"]]
    return observations, prior_x, multi_summary, batch_diagnostics, forward


def surface_diagnostics(root, observations, forward, evidence):
    points = np.array([r["normalized_x"] for r in observations])
    observed = np.array([r["gap_GHz"] for r in observations])
    model = fit_pwe_surface(observations, .02)
    grid = candidate_grid(81)
    predicted = np.asarray(model(grid))
    hull = Delaunay(points)
    supported = hull.find_simplex(grid) >= 0
    loo_errors, loo_rows = [], []
    for i, row in enumerate(observations):
        remaining = observations[:i]+observations[i+1:]
        pred = float(fit_pwe_surface(remaining, .02)([row["normalized_x"]])[0])
        loo_errors.append(pred-row["gap_GHz"])
        inside = bool(Delaunay(np.delete(points, i, axis=0)).find_simplex([row["normalized_x"]])[0] >= 0)
        loo_rows.append({"normalized_x": row["normalized_x"], "prediction_error_GHz": pred-row["gap_GHz"],
                         "inside_remaining_sample_hull": inside})
    groups, grouped_rows = coordinate_groups(points), []
    for indices in groups:
        keep = [i for i in range(len(points)) if i not in indices]
        remaining = [observations[i] for i in keep]
        query = points[indices[0]]
        pred = float(fit_pwe_surface(remaining,.02)([query])[0])
        inside = bool(Delaunay(points[keep]).find_simplex([query])[0] >= 0)
        grouped_rows.append({"normalized_x": query.tolist(), "held_out_record_count": len(indices),
                             "prediction_error_GHz": pred-observed[indices[0]],
                             "inside_remaining_sample_hull": inside})
    path = Path(root) / "data/processed/bands.npz"
    evidence.hashes[str(path.resolve())] = file_hash(path)
    reference = load_comsol_reference(path, 0)
    bounds = np.asarray(forward["side_multiplier_bounds"])*forward["a_um"]/np.sqrt(3.)
    lower, width = bounds[:, 0], np.diff(bounds, axis=1).ravel()
    physical = np.array([(a, b) for a in reference["L1_um"] for b in reference["L2_um"]])
    normalized = (physical-lower)/width
    mask = ((normalized >= 0).all(axis=1) & (normalized <= 1).all(axis=1)
            & (hull.find_simplex(normalized) >= 0))
    comsol_gaps = reference["signed_path_gap_GHz"].ravel()
    pwe_pred = model(normalized[mask])
    comparison = error_metrics(pwe_pred-comsol_gaps[mask]) if mask.any() else None
    return {
        "observed_M20_range_GHz": [float(observed.min()), float(observed.max())],
        "interpolated_grid_range_GHz": [float(predicted.min()), float(predicted.max())],
        "grid_size": 81, "grid_count": len(grid),
        "grid_fraction_inside_sample_convex_hull": float(supported.mean()),
        "grid_fraction_outside_observed_value_range": float(((predicted < observed.min()) | (predicted > observed.max())).mean()),
        "leave_one_design_out_error": error_metrics(loo_errors),
        "leave_one_design_out_rows": loo_rows,
        "leave_one_design_out_inside_hull_error": error_metrics([
            r["prediction_error_GHz"] for r in loo_rows if r["inside_remaining_sample_hull"]]),
        "coordinate_group_count": len(groups), "grouping_tolerance_normalized": 1e-8,
        "grouped_leave_one_geometry_out_rows": grouped_rows,
        "grouped_leave_one_geometry_out_error": error_metrics([r["prediction_error_GHz"] for r in grouped_rows]),
        "grouped_leave_one_geometry_out_inside_hull_error": error_metrics([
            r["prediction_error_GHz"] for r in grouped_rows if r["inside_remaining_sample_hull"]]),
        "loo_note": "Fixed RBF hyperparameters; spatially clustered samples. The legacy leave_one_design_out fields hold out one float-distinct record; grouped metrics remove all coordinates within 1e-8 normalized distance (connected groups) to avoid numerical near-duplicate leakage. Neither is a random/generalization guarantee or confidence interval.",
        "COMSOL_reference_range_GHz": [float(comsol_gaps.min()), float(comsol_gaps.max())],
        "common_supported_COMSOL_grid_count": int(mask.sum()),
        "PWE_interpolant_minus_COMSOL_gap": comparison,
        "comparison_note": "Scalar signed path-gap only, at COMSOL grid coordinates inside the PWE box and sample convex hull. PWE is interpolated: differences mix interpolation error and model/geometry/discretization differences. No pointwise-band, midband, or group-velocity validation is claimed.",
        "COMSOL_used_for_PWE_fitting": False,
        "convex_hull_note": "Inside the hull is geometric support, not a guarantee of interpolation accuracy.",
    }


def audit(root, *, sources_only=False):
    root = Path(root).resolve()
    evidence = Evidence()
    observations, prior_x, multi, batches, forward = source_audit(root, evidence)
    diagnostics = surface_diagnostics(root, observations, forward, evidence)
    out = root / "results" / VERSION
    manifest = evidence.read(out / "manifest.json")
    config = evidence.read(root / "configs" / f"{VERSION}.json")
    verify_hashes(manifest["source_hashes"])
    if manifest["config"] != config or manifest["effective_forward_config"] != forward:
        raise ValueError("Validation protocol config mismatch")
    starts = select_atlas_starts(observations, prior_x, config)
    if starts != manifest["atlas_starts_normalized"]:
        raise ValueError("Atlas proposal selection replay failed")
    summary = {
        "status": "sources_verified_validation_pending", "source_M20_count": len(observations),
        "prior_M25_validated_count": multi["successful_cases"],
        "prior_evaluation_timing_by_cutoff": multi["evaluation_timing_by_cutoff"],
        "source_batches": batches, "surface_diagnostics": diagnostics,
        "atlas_starts_normalized": starts, "source_observations": observations,
        "prior_validated_cases": multi["cases"], "cases": [],
        "scientific_fingerprint": json_hash(manifest),
        "verification_ran_physics": False,
        "audit_limitation": "The new driver manifest did not freeze implementation hashes or include its own reviewed gate before launch. This is a post-launch independent audit, not retrospective preregistration. The unchanged inherited physics files are checked against the earlier multistart manifest.",
    }
    if not sources_only:
        result = evidence.read(out / "results.json")
        if (result["status"] != "complete" or result["scientific_fingerprint"] != json_hash(manifest)
                or len(result["cases"]) != len(config["targets_GHz"])):
            raise ValueError("Validation is not complete")
        evaluator, values = verified_evaluations(out, manifest, evidence)
        tolerance = evidence.read(root / forward["final_tolerance_source"])[forward["final_tolerance_key"]]
        if tolerance != multi["existing_tolerance_GHz"]:
            raise ValueError("Final tolerance changed")
        used = set()
        for i, row in enumerate(result["cases"]):
            target = config["targets_GHz"][i]
            if (row["case_id"] != f"atlas_target_{i:02d}" or row["target_gap_GHz"] != target
                    or row["atlas_start_normalized"] != starts[i]
                    or row["atlas_start_sides_um"] != evaluator.sides_from_x(starts[i]).tolist()):
                raise ValueError("Case identity or prescribed start differs")
            replay_search(row, values, forward)
            m20, m25 = (values[row[k]] for k in ("candidate_search_evaluation", "independent_refinement_evaluation"))
            if m20["normalized_x"] != row["search"]["x"]:
                raise ValueError("Final candidate does not match search")
            recomputed = assessment(m20, m25, target, tolerance)
            success = bool(row["search"]["matched"] and recomputed["matches_at_M25"]
                           and recomputed["passes_empirical_refinement_guard"])
            outcome = "success" if success else ("unresolved_local_search" if not row["search"]["matched"] else "independent_refinement_failed")
            if recomputed != row["assessment"] or row["success"] is not success or row["outcome"] != outcome:
                raise ValueError("Assessment or outcome does not independently reproduce")
            initial = values[row["evaluation_calls"][0]["evaluation_key"]]
            pred = float(fit_pwe_surface(observations, .02)([starts[i]])[0])
            summary["cases"].append({**recomputed, "case_id": row["case_id"],
                "success": success, "outcome": outcome,
                "atlas_start_normalized": starts[i], "atlas_start_sides_um": row["atlas_start_sides_um"],
                "atlas_prediction_at_start_GHz": pred, "actual_M20_at_start_GHz": initial["gap_GHz"],
                "prediction_error_at_start_GHz": pred-initial["gap_GHz"],
                "search_calls": len(row["evaluation_calls"]),
                "M25_elapsed_seconds": m25["elapsed_seconds"],
                "search_status": row["search"]["status"]})
            used.update(c["evaluation_key"] for c in row["evaluation_calls"])
            used.update((m20["evaluation_key"], m25["evaluation_key"]))
        if used != set(values) or result["current_case"] is not None:
            raise ValueError("Unexpected evaluation inventory/current case")
        successes = sum(r["success"] for r in summary["cases"])
        if result["completed_cases"] != len(summary["cases"]) or result["successful_cases"] != successes:
            raise ValueError("Incorrect result counts")
        summary.update(status="verified_complete", successful_cases=successes,
                       distinct_evaluations=len(values),
                       evaluations_by_cutoff={str(c): sum(v["cutoff"] == c for v in values.values()) for c in (20,25)},
                       evaluation_seconds=sum(v["elapsed_seconds"] for v in values.values()),
                       sampled_k_count_per_evaluation=forward["full_path_count"],
                       point_backends=sorted({b for v in values.values() for b in v["checks"]["point_backends"]}),
                       worst_cpu_device_difference_GHz=max(v["checks"]["cpu_device_max_absolute_GHz"] for v in values.values()),
                       worst_relative_eigenpair_residual=max(v["checks"]["max_relative_eigenpair_residual"] for v in values.values()),
                       worst_edge_three_mode_difference_GHz=max(v["checks"]["edge_three_mode_max_absolute_GHz"] for v in values.values()),
                       prospective_start_error=error_metrics([r["prediction_error_at_start_GHz"] for r in summary["cases"]]))
    summary["source_snapshot_hashes"] = evidence.hashes
    summary["verification_implementation_hashes"] = {str(Path(__file__).resolve()): file_hash(__file__)}
    evidence.recheck()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = audit(ROOT, sources_only=args.sources_only)
    if args.output.exists():
        raise ValueError("Refusing to overwrite an existing audit artifact")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    d = summary["surface_diagnostics"]
    print(json.dumps({"status": summary["status"], "samples": summary["source_M20_count"],
                      "coordinate_groups":d["coordinate_group_count"],
                      "grouped_CV_inside_hull":d["grouped_leave_one_geometry_out_inside_hull_error"],
                      "PWE_interpolant_minus_COMSOL":d["PWE_interpolant_minus_COMSOL_gap"]}, indent=2))


if __name__ == "__main__":
    main()
