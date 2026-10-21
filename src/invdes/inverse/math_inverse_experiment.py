"""Isolated, explicitly gated anchor-free PWE research orchestration.

All candidate responses are calculated by the pure-PWE evaluator. Existing
COMSOL data, saved inverse anchors and ML models are never loaded.
"""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import scipy

from validate_energy_pwe_full import atomic_json
from invdes.projector_experiment import DeadlineReached, digest
from invdes.inverse.physics_trust_region import solve_gap_target
from invdes.inverse.quadrature_forward import QuadratureForward


def independent_assessment(search, refined, target, tolerance):
    """Keep finite-cutoff errors and empirical refinement change separate."""
    g20, g25 = float(search["gap_GHz"]), float(refined["gap_GHz"])
    if not np.all(np.isfinite([g20, g25, target, tolerance])) or tolerance <= 0:
        raise ValueError("Assessment requires finite gaps and positive tolerance")
    if not np.array_equal(search["normalized_x"], refined["normalized_x"]):
        raise ValueError("Refinement must evaluate the unchanged candidate")
    if (search.get("cutoff") != 20 or refined.get("cutoff") != 25
            or not np.array_equal(search["sides_um"], refined["sides_um"])):
        raise ValueError("Require M20/M25 evaluations with identical physical side lengths")
    e20, e25, delta = abs(g20-target), abs(g25-target), abs(g25-g20)
    return {
        "target_gap_GHz": target, "sides_um": search["sides_um"],
        "normalized_x": search["normalized_x"], "gap_M20_GHz": g20,
        "gap_M25_GHz": g25, "error_M20_GHz": e20, "error_M25_GHz": e25,
        "cutoff_gap_change_GHz": delta, "error_plus_refinement_GHz": e20+delta,
        "existing_tolerance_GHz": tolerance,
        "passes_empirical_refinement_guard": e20+delta <= tolerance,
        "matches_at_M25": e25 <= tolerance,
        "continuum_accuracy_certified": False,
        "unchanged_candidate_independently_checked": True,
    }


def render_report(result):
    lines = [
        "# Anchor-free mathematical PWE inverse experiment", "",
        "Exploratory local search only. No ML, COMSOL spectra or saved design anchors.",
        "Band gaps are signed gaps on the existing sampled path, not full-Brillouin-zone certificates.",
        "The same nominal geometry starts every request. A scalar gap does not identify a unique pair of lengths.",
        "", f"Status: {result['status']}.",
        "", "|Target GHz|L1 um|L2 um|M20 gap GHz|M25 gap GHz|M25 error GHz|M change GHz|Error plus change GHz|Empirical guard|",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in result.get("targets", []):
        a = row["assessment"]
        lines.append(
            f"|{a['target_gap_GHz']:.6g}|{a['sides_um'][0]:.9g}|{a['sides_um'][1]:.9g}|"
            f"{a['gap_M20_GHz']:.9g}|{a['gap_M25_GHz']:.9g}|{a['error_M25_GHz']:.6g}|"
            f"{a['cutoff_gap_change_GHz']:.6g}|{a['error_plus_refinement_GHz']:.6g}|"
            f"{a['passes_empirical_refinement_guard']}|")
    lines += ["", "Two-cutoff agreement is an empirical check, not an error bound for the continuum.",
              "Unresolved targets are not proved infeasible. Trial traces retain rejected steps.",
              "Completed evaluation checkpoints can be reused only for an identical scientific fingerprint."]
    return "\n".join(lines)+"\n"


class MathInverseExperiment:
    """An explicit root-reviewed gate is required before constructing a run."""
    def __init__(self, root, gate_review, *, deadline, device):
        self.root = Path(root)
        self.out = self.root/"results/pure_pwe_math_inverse_v2"
        self.config = json.loads((self.root/"configs/pure_pwe_math_inverse_v1.json").read_text())
        if (not gate_review.get("passed") or not gate_review.get("root_review_acknowledged")
                or not gate_review.get("exploratory_only")):
            raise ValueError("Numerical checks and explicit exploratory root review are required")
        if not np.isfinite(deadline) or time.time() >= deadline:
            raise DeadlineReached("No new inverse experiment may start after the deadline")
        self.deadline = float(deadline)
        final_path = self.root/self.config["final_tolerance_source"]
        self.tolerance = float(json.loads(final_path.read_text())[self.config["final_tolerance_key"]])
        upstream = self.root/"results/radial_quadrature_audit_v1/manifest.json"
        audit_manifest = json.loads(upstream.read_text())
        self.protected = {**audit_manifest["protected_hashes"], **audit_manifest["code_hashes"],
                          str(upstream): digest(upstream)}
        # Preserve the completed research session as well as the older validated
        # solver/results. None of these folders are output locations for this run.
        for folder in ("projector_stabilization_v1", "radial_quadrature_audit_v1",
                       "variational_pwe_audit_v1", "quadrature_full_order_v1",
                       "pwe_projector_research_session"):
            for path in sorted((self.root/"results"/folder).rglob("*")):
                if path.is_file():
                    self.protected[str(path)] = digest(path)
        self.protected.update(gate_review.get("evidence_hashes", {}))
        paths = [
            Path(__file__), self.root/"scripts/run_math_pwe_inverse.py",
            self.root/"src/invdes/inverse/physics_trust_region.py",
            self.root/"src/invdes/inverse/quadrature_forward.py",
            self.root/"src/invdes/inverse/quadrature_launch_gate.py",
            self.root/"src/invdes/inverse/angular_result_guard.py",
            self.root/"tests/test_math_inverse_experiment.py",
            self.root/"tests/test_quadrature_forward.py",
            self.root/"tests/test_physics_trust_region.py",
            self.root/"tests/test_quadrature_launch_gate.py",
            self.root/"tests/test_angular_result_guard.py",
            self.root/"configs/pure_pwe_math_inverse_v1.json", final_path,
            self.root/"configs/criteria.yaml",
        ]
        self.protocol = {
            "experiment_version": "pure_pwe_math_inverse_v2",
            "config": self.config, "protected_hashes": self.protected,
            "code_hashes": {str(p): digest(p) for p in paths},
            "gate_review": gate_review, "final_tolerance_GHz": self.tolerance,
            "runtime": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                        "dtype": "float64/complex128", "allowed_backends": ["cpu", "cuda"]},
        }
        self.fingerprint = hashlib.sha256(json.dumps(self.protocol, sort_keys=True).encode()).hexdigest()
        self.verify()
        if (self.out/"run.lock").exists():
            raise ValueError("An inverse run lock already exists; inspect it before resuming")
        self.out.mkdir(parents=True, exist_ok=True)
        manifest = self.out/"manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != self.protocol:
                raise ValueError("Scientific protocol changed; preserve this output and use a new version")
        else:
            atomic_json(manifest, self.protocol)
        self.evaluator = QuadratureForward(self.config, self.out, self.fingerprint, device,
                                          self.check_deadline)

    def check_deadline(self):
        if time.time() >= self.deadline:
            raise DeadlineReached("Bounded research deadline; checkpoint preserved")

    def verify(self):
        hashes = {**self.protected, **getattr(self, "protocol", {}).get("code_hashes", {})}
        changed = [p for p, h in hashes.items() if digest(p) != h]
        if changed:
            raise ValueError(f"Protected or frozen implementation files changed: {changed}")

    def save(self, result):
        self.verify()
        result["protected_files_verified"] = len(self.protected)
        atomic_json(self.out/"results.json", result)
        (self.out/"REPORT.md").write_text(render_report(result), encoding="utf-8")

    @staticmethod
    def target_hash(row):
        return hashlib.sha256(json.dumps(
            {k: v for k, v in row.items() if k != "payload_sha256"},
            sort_keys=True, allow_nan=False).encode()).hexdigest()

    def validate_target(self, row, target):
        if (row.get("scientific_fingerprint") != self.fingerprint
                or row.get("target_gap_GHz") != target
                or row.get("payload_sha256") != self.target_hash(row)):
            raise ValueError("Target checkpoint identity or checksum mismatch")
        # A completed target must agree with its independently integrity-checked
        # physics evaluations, not merely contain a stored success flag.
        search = self.cached_at(row["search"]["x"], self.config["search_cutoff"])
        refined = self.cached_at(row["search"]["x"], self.config["independent_check_cutoff"])
        metadata = row["search"]
        residual = abs(search["gap_GHz"]-target)
        if (metadata.get("target_gap_GHz") != target or metadata.get("gap_GHz") != search["gap_GHz"]
                or metadata.get("absolute_error_GHz") != residual
                or metadata.get("tolerance_GHz") != self.config["internal_tolerance_GHz"]
                or metadata.get("matched") is not (residual <= self.config["internal_tolerance_GHz"])
                or (metadata.get("status") == "matched") != metadata.get("matched")):
            raise ValueError("Target search summary disagrees with checked physics results")
        if (row["assessment"] != independent_assessment(search, refined, target, self.tolerance)
                or row["candidate_search_evaluation"] != search["evaluation_key"]
                or row["independent_refinement_evaluation"] != refined["evaluation_key"]):
            raise ValueError("Target assessment disagrees with checked physics results")

    def cached_at(self, x, cutoff):
        identity = self.evaluator._identity(x, cutoff)
        key = self.evaluator._identity_hash(identity)
        path = self.evaluator.output/f"{key}.json"
        if not path.exists() or "result" not in json.loads(path.read_text()):
            raise ValueError("Completed target lacks a complete physics checkpoint")
        return self.evaluator(x, cutoff=cutoff)

    def run(self):
        result = {"status": "running", "scientific_fingerprint": self.fingerprint, "targets": [],
                  "ml_training": False, "comsol_spectra_used": False, "saved_anchors_used": False,
                  "production_replaced": False, "continuum_accuracy_certified": False,
                  "started_unix": time.time()}
        lock = self.out/"run.lock"
        with lock.open("x") as stream:
            stream.write(str(__import__("os").getpid()))
        try:
            self.save(result)
            for index, target in enumerate(self.config["targets_GHz"]):
                self.check_deadline()
                target_path = self.out/f"target_{index}.json"
                if target_path.exists():
                    saved = json.loads(target_path.read_text())
                    self.validate_target(saved, target)
                    result["targets"].append(saved)
                    self.save(result)
                    continue
                calls = []

                def evaluate(x):
                    self.check_deadline()
                    self.verify()
                    value = self.evaluator(x)
                    calls.append({k: value[k] for k in (
                        "normalized_x", "sides_um", "gap_GHz", "evaluation_key",
                        "evaluation_path", "cached", "call_elapsed_seconds")})
                    atomic_json(self.out/f"target_{index}_calls.json", {
                        "scientific_fingerprint": self.fingerprint, "target_gap_GHz": target,
                        "calls": calls, "kernel_replayed_from_fixed_start_after_resume": True})
                    atomic_json(self.out/"status.json", {
                        "status": "running", "target_gap_GHz": target, "target_index": index,
                        "evaluation_calls_current_replay": len(calls),
                        "last_gap_GHz": value["gap_GHz"], "updated_unix": time.time()})
                    print(f"target {target}: call {len(calls)}, gap {value['gap_GHz']:.9g}", flush=True)
                    return value

                controls = {k: self.config["trust_region"][k] for k in (
                    "max_trial_evaluations", "initial_radius", "maximum_radius", "minimum_radius",
                    "acceptance_ratio", "shrink_ratio", "expand_ratio", "gradient_norm_floor",
                    "rejected_models_before_poll")}
                search = solve_gap_target(
                    evaluate, target, self.config["initial_normalized_x"],
                    tolerance_ghz=self.config["internal_tolerance_GHz"], **controls)
                fine0 = self.evaluator(search["x"])
                fine1 = self.evaluator(search["x"], cutoff=self.config["independent_check_cutoff"])
                row = {
                    "scientific_fingerprint": self.fingerprint, "target_gap_GHz": target,
                    "search": search, "assessment": independent_assessment(fine0, fine1, target, self.tolerance),
                    "evaluation_calls": calls, "candidate_search_evaluation": fine0["evaluation_key"],
                    "independent_refinement_evaluation": fine1["evaluation_key"],
                }
                row["payload_sha256"] = self.target_hash(row)
                atomic_json(target_path, row)
                result["targets"].append(row)
                self.save(result)
            result["status"] = "complete"
        except DeadlineReached:
            result["status"] = "deadline_checkpointed"
        except BaseException as exc:
            result["status"] = "failed"
            result["error"] = repr(exc)
            raise
        finally:
            lock.unlink()
            result["elapsed_seconds_this_invocation"] = time.time()-result["started_unix"]
            completed_evaluations = []
            for path in self.evaluator.output.glob("*.json"):
                record = json.loads(path.read_text())
                if "result" in record:
                    value = record["result"]
                    value = self.cached_at(value["normalized_x"], value["cutoff"])
                    completed_evaluations.append({
                        key: value[key] for key in ("evaluation_key", "cutoff", "normalized_x",
                                                   "elapsed_seconds", "gap_GHz")})
            result["completed_distinct_physics_evaluations"] = completed_evaluations
            self.save(result)
            atomic_json(self.out/"status.json", {
                "status": result["status"], "completed_targets": len(result["targets"]),
                "updated_unix": time.time(), "deadline_unix": self.deadline})
        return result
