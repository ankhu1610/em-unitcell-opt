"""Isolated, preregistered multistart checks of the unchanged pure-PWE inverse.

Only orchestration changes: every start uses the same bounded local kernel and
full-path physics evaluator. No historical response is used as a search anchor.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import scipy

from validate_energy_pwe_full import atomic_json
from invdes.projector_experiment import DeadlineReached, digest
from invdes.inverse.math_inverse_experiment import independent_assessment
from invdes.inverse.physics_trust_region import solve_gap_target
from invdes.inverse.quadrature_forward import QuadratureForward


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def case_hash(row):
    return json_hash({key: value for key, value in row.items() if key != "payload_sha256"})


def planned_cases(config):
    targets, starts = config.get("targets_GHz"), config.get("starts")
    if (not isinstance(targets, list) or not targets
            or any(isinstance(t, bool) or not isinstance(t, (float, int))
                   or not np.isfinite(t) or t <= 0 for t in targets)
            or len(set(targets)) != len(targets)):
        raise ValueError("Targets must be distinct finite positive band gaps")
    if not isinstance(starts, list) or not starts:
        raise ValueError("Require preregistered starting geometries")
    ids, coordinates = set(), set()
    for start in starts:
        name, x = start.get("id"), np.asarray(start.get("normalized_x"), dtype=float)
        if (not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name)
                or name in ids or x.shape != (2,) or not np.all(np.isfinite(x))
                or np.any(x < 0) or np.any(x > 1) or tuple(x) in coordinates):
            raise ValueError("Starts require distinct safe IDs and finite coordinates in [0,1]^2")
        ids.add(name)
        coordinates.add(tuple(x))
    return [
        {"case_id": f"target_{i:02d}__{start['id']}", "target_index": i,
         "target_gap_GHz": float(target), "start_id": start["id"],
         "initial_normalized_x": list(start["normalized_x"])}
        for i, target in enumerate(targets) for start in starts
    ]


def aggregate_cases(plan, completed):
    """Retain the planned denominator even while some starts are unfinished."""
    completed_by_id = {row["case_id"]: row for row in completed}
    if len(completed_by_id) != len(completed):
        raise ValueError("Duplicate completed cases")
    plan_ids = {row["case_id"] for row in plan}
    if not set(completed_by_id).issubset(plan_ids):
        raise ValueError("Unplanned completed case")
    rows = []
    for target in dict.fromkeys(row["target_gap_GHz"] for row in plan):
        wanted = [row for row in plan if row["target_gap_GHz"] == target]
        done = [completed_by_id[row["case_id"]] for row in wanted
                if row["case_id"] in completed_by_id]
        success = sum(row["success"] for row in done)
        rows.append({
            "target_gap_GHz": target, "planned_starts": len(wanted),
            "completed_starts": len(done), "not_yet_completed": len(wanted)-len(done),
            "successful_starts": success, "unsuccessful_completed_starts": len(done)-success,
            "success_fraction_of_planned": success/len(wanted),
            "success_fraction_of_completed": success/len(done) if done else None,
            "all_planned_starts_completed": len(done) == len(wanted),
        })
    return rows


def render_report(result):
    lines = [
        "# Pure-PWE anchor-free multistart reliability experiment", "",
        "The existing full-path evaluator and inverse controls are unchanged.",
        "Each target/start pair is an independent local search; all attempts are retained.",
        "No ML, COMSOL fitting or historical response anchors are used.",
        "A signed sampled-path gap is not a full-Brillouin-zone or continuum certificate.",
        "", f"Status: {result['status']}.",
        f"Completed cases: {len(result['cases'])}/{result['planned_case_count']}.",
        "",
        "|Target GHz|Start|Initial L1 um|Initial L2 um|Final L1 um|Final L2 um|M20 gap GHz|M25 gap GHz|M25 error GHz|Error plus refinement GHz|Outcome|",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in result["cases"]:
        a, initial = row["assessment"], row["initial_sides_um"]
        lines.append(
            f"|{row['target_gap_GHz']:.8g}|{row['start_id']}|{initial[0]:.9g}|"
            f"{initial[1]:.9g}|{a['sides_um'][0]:.9g}|{a['sides_um'][1]:.9g}|"
            f"{a['gap_M20_GHz']:.9g}|{a['gap_M25_GHz']:.9g}|{a['error_M25_GHz']:.6g}|"
            f"{a['error_plus_refinement_GHz']:.6g}|{row['outcome']}|")
    lines += ["", "|Target GHz|Successful / planned|Completed|Unfinished|Success / completed|",
              "|---|---|---|---|---|"]
    for row in result["per_target"]:
        fraction = row["success_fraction_of_completed"]
        value = "not available" if fraction is None else f"{fraction:.6g}"
        lines.append(f"|{row['target_gap_GHz']:.8g}|{row['successful_starts']}/"
                     f"{row['planned_starts']}|{row['completed_starts']}|"
                     f"{row['not_yet_completed']}|{value}|")
    lines += [
        "",
        "Success requires the unchanged internal search tolerance, the unchanged final M25",
        "tolerance, and the empirical guard abs(G20-target)+abs(G25-G20).",
        "M25 checks the unchanged candidate, including locally unresolved searches; it never fits the geometry.",
        "Unresolved local searches do not prove infeasibility. Different successful geometries are",
        "expected because a scalar target generally does not uniquely determine two lengths.",
        "Only identical, integrity-checked evaluations inside this new run are reused.",
    ]
    if result.get("error"):
        lines += ["", f"Stopped failure: {result['error']}"]
    return "\n".join(lines)+"\n"


class MultistartExperiment:
    """An explicit reviewed gate is required; construction never executes PWE."""
    def __init__(self, root, gate_review, *, deadline, device):
        self.root = Path(root).resolve()
        self.out = self.root/"results/pure_pwe_multistart_v1"
        self._assert_unverified_output()
        if (not gate_review.get("passed") or not gate_review.get("root_review_acknowledged")
                or not gate_review.get("exploratory_only") or gate_review.get("launch_blockers")):
            raise ValueError("Passed numerical evidence and explicit exploratory root review required")
        config_path = self.root/"configs/pure_pwe_multistart_v1.json"
        if gate_review.get("multistart_config_sha256") != digest(config_path):
            raise ValueError("Reviewed multistart configuration hash does not match")
        self.config = json.loads(config_path.read_text())
        self.plan = planned_cases(self.config)
        maximum = self.config.get("max_runtime_hours")
        now = time.time()
        if (isinstance(maximum, bool) or not isinstance(maximum, (float, int))
                or not np.isfinite(maximum) or maximum <= 0):
            raise ValueError("Require a finite positive configured runtime bound")
        if not np.isfinite(deadline) or deadline <= now:
            raise DeadlineReached("No new experiment may start after its deadline")
        if deadline-now > maximum*3600:
            raise ValueError("Deadline exceeds the configured runtime bound")
        self.deadline = float(deadline)
        base_path = (self.root/self.config["base_config"]).resolve()
        if not base_path.is_relative_to(self.root):
            raise ValueError("Base config must remain inside the repository")
        self.forward_config = json.loads(base_path.read_text())
        prior = self.root/"results/pure_pwe_math_inverse_v2"
        if (prior/"run.lock").exists():
            raise ValueError("Prior inverse run is still locked")
        prior_manifest = json.loads((prior/"manifest.json").read_text())
        verification = json.loads((prior/"verification_summary.json").read_text())
        if (prior_manifest.get("experiment_version") != "pure_pwe_math_inverse_v2"
                or verification.get("status") != "verified_complete"
                or self.forward_config != prior_manifest["config"]
                or verification.get("scientific_fingerprint") != json_hash(prior_manifest)):
            raise ValueError("Require the unchanged, independently verified prior solver configuration")
        self.protected = {}
        # Check inherited hashes BEFORE adding current snapshots; never silently
        # re-baseline a changed historical source or result.
        for hashes in (prior_manifest["protected_hashes"], prior_manifest["code_hashes"],
                       verification["source_snapshot_hashes"],
                       verification["verification_implementation_hashes"],
                       gate_review.get("evidence_hashes", {})):
            self._merge_verified_hashes(hashes)
        for path in sorted(prior.rglob("*")):
            if path.is_file():
                self._merge_verified_hashes({str(path): digest(path)})
        tolerance_path = self.root/self.forward_config["final_tolerance_source"]
        self.tolerance = float(json.loads(tolerance_path.read_text())[
            self.forward_config["final_tolerance_key"]])
        if self.tolerance != prior_manifest["final_tolerance_GHz"]:
            raise ValueError("The existing final tolerance changed")
        paths = [Path(__file__), self.root/"tests/test_multistart_experiment_v1.py",
                 self.root/"scripts/run_pwe_multistart_v1.py",
                 self.root/"tests/test_pwe_multistart_cli_v1.py",
                 self.root/"scripts/summarize_pwe_multistart_v1.py",
                 self.root/"tests/test_pwe_multistart_summary_v1.py", config_path, base_path,
                 tolerance_path]
        self.protocol = {
            "experiment_version": "pure_pwe_multistart_v1",
            "config": self.config, "effective_forward_config": self.forward_config,
            "case_plan": self.plan, "protected_hashes": self.protected,
            "code_hashes": {str(p.resolve()): digest(p) for p in paths},
            "gate_review": copy.deepcopy(gate_review), "final_tolerance_GHz": self.tolerance,
            "runtime": {"python": sys.version, "numpy": np.__version__,
                        "scipy": scipy.__version__, "dtype": "float64/complex128",
                        "allowed_backends": ["cpu", "cuda"]},
        }
        self.fingerprint = json_hash(self.protocol)
        self.verify()
        if (self.out/"run.lock").exists():
            raise ValueError("Multistart run lock exists; inspect the worker before resuming")
        self.evaluator = QuadratureForward(
            self.forward_config, self.out, self.fingerprint, device, self.check_deadline)
        # The manifest is committed only after run() acquires its exclusive lock.
        manifest = self.out/"manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()) != self.protocol:
            raise ValueError("Scientific protocol changed; preserve output and use a new version")

    def _merge_verified_hashes(self, hashes):
        for name, expected in hashes.items():
            path = Path(name)
            if not path.is_absolute() or digest(path) != expected:
                raise ValueError(f"Protected/source hash mismatch: {name}")
            canonical = str(path.resolve())
            if canonical in self.protected and self.protected[canonical] != expected:
                raise ValueError(f"Conflicting protected hashes: {name}")
            self.protected[canonical] = expected

    def check_deadline(self):
        if time.time() >= self.deadline:
            raise DeadlineReached("Multistart deadline reached; checkpoints preserved")

    def _assert_unverified_output(self):
        if any((self.out/name).exists()
               for name in ("verification_summary.json", "VERIFICATION.md")):
            raise ValueError("Verified output is immutable; use read-only verification or a new output version")

    def verify(self):
        for hashes in (self.protected, getattr(self, "protocol", {}).get("code_hashes", {})):
            for path, expected in hashes.items():
                if digest(path) != expected:
                    raise ValueError(f"Protected or frozen implementation file changed: {path}")

    def _controls(self):
        return {key: self.forward_config["trust_region"][key] for key in (
            "max_trial_evaluations", "initial_radius", "maximum_radius", "minimum_radius",
            "acceptance_ratio", "shrink_ratio", "expand_ratio", "gradient_norm_floor",
            "rejected_models_before_poll")}

    def _solve(self, evaluate, case):
        return solve_gap_target(
            evaluate, case["target_gap_GHz"], case["initial_normalized_x"],
            tolerance_ghz=self.forward_config["internal_tolerance_GHz"], **self._controls())

    def cached_at(self, x, cutoff):
        """Read and fully validate a complete checkpoint without executing PWE."""
        x = np.asarray(x, dtype=float)
        sides = self.evaluator.sides_from_x(x)
        identity = self.evaluator._identity(x, cutoff)
        key = self.evaluator._identity_hash(identity)
        path = self.evaluator.output/f"{key}.json"
        if not path.exists():
            raise ValueError("Completed case lacks a physics checkpoint")
        record = json.loads(path.read_text())
        if "result" not in record:
            raise ValueError("Completed case lacks a complete physics checkpoint")
        self.evaluator._validate_checkpoint(record, identity, key, x, sides, cutoff, path)
        return copy.deepcopy(record["result"])

    @staticmethod
    def _outcome(search, assessment):
        success = bool(search["matched"] and assessment["matches_at_M25"]
                       and assessment["passes_empirical_refinement_guard"])
        outcome = ("success" if success else "unresolved_local_search"
                   if not search["matched"] else "independent_refinement_failed")
        return success, outcome

    def validate_case(self, row, case):
        if (row.get("scientific_fingerprint") != self.fingerprint
                or any(row.get(key) != value for key, value in case.items())
                or row.get("payload_sha256") != case_hash(row)):
            raise ValueError("Case checkpoint identity, start or checksum mismatch")
        expected_initial = self.evaluator.sides_from_x(case["initial_normalized_x"]).tolist()
        if row.get("initial_sides_um") != expected_initial:
            raise ValueError("Case initial lengths disagree with its fixed start")
        calls, cursor = row["evaluation_calls"], 0

        def replay(x):
            nonlocal cursor
            if cursor >= len(calls):
                raise ValueError("Missing recorded search call")
            value = self.cached_at(x, self.forward_config["search_cutoff"])
            recorded = calls[cursor]
            cursor += 1
            if any(recorded.get(key) != value[key] for key in (
                    "normalized_x", "sides_um", "gap_GHz", "evaluation_key", "evaluation_path")):
                raise ValueError("Recorded call disagrees with checked physics")
            elapsed = recorded.get("call_elapsed_seconds")
            if (not isinstance(recorded.get("cached"), bool)
                    or not isinstance(elapsed, (int, float)) or not np.isfinite(elapsed)
                    or elapsed < 0):
                raise ValueError("Invalid recorded call metadata")
            return value

        replayed = self._solve(replay, case)
        if replayed != row["search"] or cursor != len(calls):
            raise ValueError("Recorded search/trace differs from deterministic checked replay")
        search = self.cached_at(replayed["x"], self.forward_config["search_cutoff"])
        refined = self.cached_at(replayed["x"], self.forward_config["independent_check_cutoff"])
        assessment = independent_assessment(search, refined, case["target_gap_GHz"], self.tolerance)
        success, outcome = self._outcome(replayed, assessment)
        if (row.get("assessment") != assessment or row.get("success") is not success
                or row.get("outcome") != outcome
                or row.get("candidate_search_evaluation") != search["evaluation_key"]
                or row.get("independent_refinement_evaluation") != refined["evaluation_key"]):
            raise ValueError("Case assessment disagrees with checked physics")

    def save(self, result):
        self.verify()
        result["protected_files_verified"] = len(self.protected)
        result["per_target"] = aggregate_cases(self.plan, result["cases"])
        result["completed_cases"] = len(result["cases"])
        result["successful_cases"] = sum(row["success"] for row in result["cases"])
        atomic_json(self.out/"results.json", result)
        (self.out/"REPORT.md").write_text(render_report(result), encoding="utf-8")

    def _status(self, result, **extra):
        atomic_json(self.out/"status.json", {
            "status": result["status"], "scientific_fingerprint": self.fingerprint,
            "completed_cases": len(result["cases"]), "planned_cases": len(self.plan),
            "successful_cases": sum(row["success"] for row in result["cases"]),
            "current_case": result.get("current_case"), "updated_unix": time.time(),
            "deadline_unix": self.deadline, **extra})

    def run(self):
        self._assert_unverified_output()
        self.out.mkdir(parents=True, exist_ok=True)
        lock = self.out/"run.lock"
        # Exclusive creation occurs before any manifest/status/checkpoint write.
        with lock.open("x") as stream:
            stream.write(str(os.getpid()))
        result = {
            "status": "running", "scientific_fingerprint": self.fingerprint,
            "cases": [], "planned_case_count": len(self.plan),
            "ml_training": False, "comsol_spectra_used": False, "comsol_fitting": False,
            "saved_anchors_used": False, "production_replaced": False,
            "continuum_accuracy_certified": False, "unique_inverse_claimed": False,
            "global_feasibility_certified": False, "started_unix": time.time(),
            "current_case": None,
        }
        manifest_committed = False
        try:
            self._assert_unverified_output()
            self.verify()
            manifest = self.out/"manifest.json"
            if manifest.exists():
                if json.loads(manifest.read_text()) != self.protocol:
                    raise ValueError("Scientific protocol changed; use a new output version")
            else:
                atomic_json(manifest, self.protocol)
            manifest_committed = True
            gate_path = self.out/"gate_review.json"
            gate = self.protocol.get("gate_review", {})
            if gate_path.exists():
                if json.loads(gate_path.read_text()) != gate:
                    raise ValueError("Stored gate review differs from the frozen protocol")
            else:
                atomic_json(gate_path, gate)
            # Load every completed case before replacing aggregate progress, so
            # an operational failure or deadline cannot erase completed totals.
            completed_ids = set()
            for case in self.plan:
                path = self.out/f"{case['case_id']}.json"
                if path.exists():
                    row = json.loads(path.read_text())
                    self.validate_case(row, case)
                    result["cases"].append(row)
                    completed_ids.add(case["case_id"])
            if (self.out/"failure.json").exists():
                raise ValueError("Previous failure requires review before any automatic continuation")
            self.save(result)
            self._status(result)
            for case in self.plan:
                result["current_case"] = case
                path = self.out/f"{case['case_id']}.json"
                if case["case_id"] in completed_ids:
                    continue
                self.check_deadline()
                calls = []

                def evaluate(x):
                    self.check_deadline()
                    self.verify()
                    value = self.evaluator(x)
                    calls.append({key: value[key] for key in (
                        "normalized_x", "sides_um", "gap_GHz", "evaluation_key",
                        "evaluation_path", "cached", "call_elapsed_seconds")})
                    atomic_json(self.out/f"{case['case_id']}_calls.json", {
                        "scientific_fingerprint": self.fingerprint, **case, "calls": calls,
                        "kernel_replayed_from_preregistered_start_after_resume": True})
                    self._status(result, evaluation_calls_current_replay=len(calls),
                                 last_gap_GHz=value["gap_GHz"], phase="search")
                    print(f"{case['case_id']} target {case['target_gap_GHz']}: "
                          f"call {len(calls)}, gap {value['gap_GHz']:.9g}", flush=True)
                    return value

                self._status(result, phase="search", evaluation_calls_current_replay=0)
                search = self._solve(evaluate, case)
                fine0 = self.cached_at(search["x"], self.forward_config["search_cutoff"])
                self.check_deadline()
                self.verify()
                self._status(result, phase="independent_M25_check",
                             evaluation_calls_current_replay=len(calls))
                fine1 = self.evaluator(search["x"], cutoff=self.forward_config["independent_check_cutoff"])
                assessment = independent_assessment(fine0, fine1, case["target_gap_GHz"], self.tolerance)
                success, outcome = self._outcome(search, assessment)
                row = {
                    "scientific_fingerprint": self.fingerprint, **case,
                    "initial_sides_um": self.evaluator.sides_from_x(case["initial_normalized_x"]).tolist(),
                    "search": search, "assessment": assessment, "success": success, "outcome": outcome,
                    "evaluation_calls": calls, "candidate_search_evaluation": fine0["evaluation_key"],
                    "independent_refinement_evaluation": fine1["evaluation_key"],
                }
                row["payload_sha256"] = case_hash(row)
                self.validate_case(row, case)
                atomic_json(path, row)
                result["cases"].append(row)
                self.save(result)
                self._status(result, phase="case_complete")
            result["status"], result["current_case"] = "complete", None
        except DeadlineReached:
            result["status"] = "deadline_checkpointed"
        except BaseException as exc:
            result["status"], result["error"] = "failed", repr(exc)
            if manifest_committed and not (self.out/"failure.json").exists():
                atomic_json(self.out/"failure.json", {
                    "scientific_fingerprint": self.fingerprint, "current_case": result["current_case"],
                    "error": repr(exc), "failed_unix": time.time(),
                    "interpretation": "Numerical or operational failure; not evidence of infeasibility."})
            raise
        finally:
            # Lock remains held until every final status/result write finishes.
            try:
                if manifest_committed:
                    result["elapsed_seconds_this_invocation"] = time.time()-result["started_unix"]
                    completed = []
                    for path in sorted(self.evaluator.output.glob("*.json")):
                        record = json.loads(path.read_text())
                        if "result" in record:
                            value = record["result"]
                            value = self.cached_at(value["normalized_x"], value["cutoff"])
                            completed.append({key: value[key] for key in (
                                "evaluation_key", "cutoff", "normalized_x", "elapsed_seconds", "gap_GHz")})
                    result["completed_distinct_physics_evaluations"] = completed
                    self.save(result)
                    self._status(result)
            finally:
                lock.unlink()
        return result
