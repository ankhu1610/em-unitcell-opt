"""Validate deterministic inverse proposals selected from the v2 PWE response atlas."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from validate_energy_pwe_full import atomic_json
from invdes.inverse.math_inverse_experiment import independent_assessment
from invdes.inverse.physics_trust_region import solve_gap_target
from invdes.inverse.quadrature_forward import QuadratureForward
from invdes.inverse.response_atlas_v1 import candidate_grid, deduplicate_observations, fit_pwe_surface
from invdes.projector_experiment import DeadlineReached, digest


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def select_atlas_starts(observations, prior_validated_x, config):
    """Select one reproducible, atlas-derived and novel local start per target."""
    grid = candidate_grid(int(config["selection_grid_size"]))
    model = fit_pwe_surface(observations, smoothing=0.02)
    prediction = np.asarray(model(grid), dtype=float)
    prior = np.asarray(prior_validated_x, dtype=float)
    selected = []
    for target in config["targets_GHz"]:
        distances_prior = (np.min(np.linalg.norm(grid[:, None, :]-prior[None, :, :], axis=2), axis=1)
                           if len(prior) else np.full(len(grid), np.inf))
        distances_selected = (np.min(np.linalg.norm(grid[:, None, :]-np.asarray(selected)[None, :, :], axis=2), axis=1)
                              if selected else np.full(len(grid), np.inf))
        eligible = ((distances_prior >= float(config["minimum_distance_from_prior_M25"]))
                    & (distances_selected >= float(config["minimum_distance_between_selected_starts"])))
        if not np.any(eligible):
            raise ValueError("No atlas grid point is sufficiently separated from prior validation starts")
        residual = np.abs(prediction-float(target))
        best = np.min(residual[eligible])
        candidates = np.flatnonzero(eligible & np.isclose(residual, best, rtol=0., atol=1e-12))
        index = min(candidates, key=lambda i: (grid[i, 0], grid[i, 1]))
        selected.append(grid[index].tolist())
    return selected


def render_report(result):
    lines = [
        "# Atlas-derived pure-PWE candidate validation", "",
        f"Status: {result['status']}.",
        "Starts are deterministic proposals from the pure-PWE v2 atlas and deliberately separated",
        "from the earlier M25-validated multistart solutions. The atlas interpolation never accepts a result.", "",
        "|Target GHz|Atlas start L1 um|Atlas start L2 um|Final L1 um|Final L2 um|M20 gap GHz|M25 gap GHz|M25 error GHz|Guard|Outcome|",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in result.get("cases", []):
        start, assessment = row["atlas_start_sides_um"], row["assessment"]
        lines.append(
            f"|{row['target_gap_GHz']:.8g}|{start[0]:.9g}|{start[1]:.9g}|"
            f"{assessment['sides_um'][0]:.9g}|{assessment['sides_um'][1]:.9g}|"
            f"{assessment['gap_M20_GHz']:.9g}|{assessment['gap_M25_GHz']:.9g}|"
            f"{assessment['error_M25_GHz']:.6g}|{assessment['passes_empirical_refinement_guard']}|{row['outcome']}|")
    lines += ["", "Every M25 calculation evaluates the unchanged M20 candidate. A positive local outcome",
              "does not establish uniqueness, global feasibility, full-Brillouin-zone behavior, or continuum accuracy."]
    return "\n".join(lines)+"\n"


class AtlasValidationExperiment:
    def __init__(self, root, *, deadline, device="cpu"):
        self.root = Path(root).resolve()
        self.config_path = self.root/"configs/pure_pwe_atlas_validation_v1.json"
        self.config = json.loads(self.config_path.read_text())
        self.out = self.root/self.config["output_directory"]
        self.deadline = float(deadline)
        if self.deadline <= time.time():
            raise DeadlineReached("Validation deadline must be in the future")
        if self.deadline-time.time() > float(self.config["max_runtime_hours"])*3600:
            raise ValueError("Validation deadline exceeds its preregistered runtime")
        self.forward = json.loads((self.root/self.config["base_config"]).read_text())
        self.observations, self.prior_x, self.source_hashes = self._load_sources()
        self.starts = select_atlas_starts(self.observations, self.prior_x, self.config)
        self.protocol = {
            "experiment_version": "pure_pwe_atlas_validation_v1", "config": self.config,
            "effective_forward_config": self.forward, "source_hashes": self.source_hashes,
            "atlas_starts_normalized": self.starts,
            "runtime": {"dtype": "float64/complex128", "allowed_backends": ["cpu", "cuda"]},
        }
        self.fingerprint = json_hash(self.protocol)
        self.evaluator = QuadratureForward(self.forward, self.out, self.fingerprint, device,
                                            self.check_deadline)
        tolerance_path = self.root/self.forward["final_tolerance_source"]
        self.tolerance = float(json.loads(tolerance_path.read_text())[self.forward["final_tolerance_key"]])

    def check_deadline(self):
        if time.time() >= self.deadline:
            raise DeadlineReached("Atlas-validation deadline reached; checkpoints preserved")

    def _load_sources(self):
        atlas_dir, multi_dir = (self.root/self.config[key] for key in ("atlas_source", "multistart_source"))
        paths = [atlas_dir/"results.json", atlas_dir/"manifest.json", multi_dir/"results.json", multi_dir/"manifest.json"]
        if not all(path.is_file() for path in paths):
            raise ValueError("Completed atlas and multistart sources are required")
        atlas, atlas_manifest, multi = (json.loads(path.read_text()) for path in (paths[0], paths[1], paths[2]))
        if atlas.get("status") != "complete_for_requested_budget" or multi.get("status") != "complete":
            raise ValueError("Only completed source experiments may select validation starts")
        observations = []
        for case in multi["cases"]:
            for call in case["evaluation_calls"]:
                if not call["cached"]:
                    observations.append({"normalized_x": call["normalized_x"], "gap_GHz": call["gap_GHz"]})
        def append_atlas_rows(atlas_dir, result, manifest):
            for row in result["new_evaluations"]:
                observations.append({"normalized_x": row["normalized_x"], "gap_GHz": row["gap_GHz"]})
            parent_name = manifest.get("atlas_config", {}).get("seed_atlas")
            if parent_name:
                parent_dir = self.root/parent_name
                parent_paths = [parent_dir/"results.json", parent_dir/"manifest.json"]
                if not all(path.is_file() for path in parent_paths):
                    raise ValueError("Atlas validation source has an incomplete parent atlas")
                parent_result, parent_manifest = (json.loads(path.read_text()) for path in parent_paths)
                if parent_result.get("status") != "complete_for_requested_budget":
                    raise ValueError("Atlas validation source has an unfinished parent atlas")
                paths.extend(parent_paths)
                append_atlas_rows(parent_dir, parent_result, parent_manifest)
        append_atlas_rows(atlas_dir, atlas, atlas_manifest)
        prior = [case["assessment"]["normalized_x"] for case in multi["cases"] if case["success"]]
        return deduplicate_observations(observations), prior, {str(path.resolve()): digest(path) for path in paths}

    def _cached(self, x, cutoff):
        x = np.asarray(x, dtype=float)
        identity = self.evaluator._identity(x, cutoff)
        key = self.evaluator._identity_hash(identity)
        path = self.evaluator.output/f"{key}.json"
        record = json.loads(path.read_text())
        self.evaluator._validate_checkpoint(record, identity, key, x, self.evaluator.sides_from_x(x), cutoff, path)
        return record["result"]

    def _save(self, result):
        result["scientific_fingerprint"] = self.fingerprint
        result["completed_cases"] = len(result["cases"])
        result["successful_cases"] = sum(row["success"] for row in result["cases"])
        atomic_json(self.out/"results.json", result)
        (self.out/"REPORT.md").write_text(render_report(result), encoding="utf-8")

    def run(self):
        self.out.mkdir(parents=True, exist_ok=True)
        lock = self.out/"run.lock"
        with lock.open("x") as stream:
            stream.write(str(os.getpid()))
        try:
            manifest = self.out/"manifest.json"
            if manifest.exists() and json.loads(manifest.read_text()) != self.protocol:
                raise ValueError("Existing validation manifest differs from this protocol")
            atomic_json(manifest, self.protocol)
            result = {"status": "running", "cases": [], "current_case": None}
            self._save(result)
            controls = {key: self.forward["trust_region"][key] for key in (
                "max_trial_evaluations", "initial_radius", "maximum_radius", "minimum_radius",
                "acceptance_ratio", "shrink_ratio", "expand_ratio", "gradient_norm_floor",
                "rejected_models_before_poll")}
            for index, (target, start) in enumerate(zip(self.config["targets_GHz"], self.starts)):
                self.check_deadline()
                result["current_case"] = {"target_gap_GHz": target, "atlas_start_normalized": start}
                calls = []
                def evaluate(x):
                    value = self.evaluator(x, cutoff=self.forward["search_cutoff"])
                    calls.append({key: value[key] for key in ("normalized_x", "sides_um", "gap_GHz", "evaluation_key", "call_elapsed_seconds")})
                    self._save(result)
                    return value
                search = solve_gap_target(evaluate, target, start,
                                          tolerance_ghz=self.forward["internal_tolerance_GHz"], **controls)
                m20 = self._cached(search["x"], self.forward["search_cutoff"])
                m25 = self.evaluator(search["x"], cutoff=self.forward["independent_check_cutoff"])
                assessment = independent_assessment(m20, m25, target, self.tolerance)
                success = bool(search["matched"] and assessment["matches_at_M25"]
                               and assessment["passes_empirical_refinement_guard"])
                outcome = "success" if success else ("unresolved_local_search" if not search["matched"] else "independent_refinement_failed")
                result["cases"].append({
                    "case_id": f"atlas_target_{index:02d}", "target_gap_GHz": target,
                    "atlas_start_normalized": start, "atlas_start_sides_um": self.evaluator.sides_from_x(start).tolist(),
                    "search": search, "assessment": assessment, "success": success, "outcome": outcome,
                    "evaluation_calls": calls, "candidate_search_evaluation": m20["evaluation_key"],
                    "independent_refinement_evaluation": m25["evaluation_key"],
                })
                self._save(result)
            result["status"], result["current_case"] = "complete", None
            self._save(result)
            return result
        except DeadlineReached:
            result = locals().get("result", {"cases": [], "current_case": None})
            result["status"] = "deadline_checkpointed"
            self._save(result)
            return result
        finally:
            lock.unlink(missing_ok=True)
