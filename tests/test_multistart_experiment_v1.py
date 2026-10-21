"""Toy-only orchestration checks; never launch a physical eigenproblem."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
sys.path.insert(0, str(ROOT/"src"))
from invdes.inverse import multistart_experiment_v1 as module


class ToyEvaluator:
    def __init__(self, output):
        self.output = output/"evaluations"
        self.values, self.fresh_calls = {}, []
        self.raise_after = None
        self.refinement_shift = .002

    def sides_from_x(self, x):
        return np.array([140., 91.])+49*np.asarray(x)

    def __call__(self, x, cutoff=None):
        cutoff = cutoff or 20
        key = module.json_hash([list(x), cutoff])
        cached = key in self.values
        if not cached and self.raise_after is not None and len(self.fresh_calls) >= self.raise_after:
            raise module.DeadlineReached("Toy deadline")
        if not cached:
            self.fresh_calls.append((list(x), cutoff))
        value = {
            "gap_GHz": 5.+30.*x[0]+(.002 if cutoff == 25 else 0.),
            "gradient_normalized": [30., 0.], "gradient_reliable": True,
            "normalized_x": list(x), "sides_um": self.sides_from_x(x).tolist(),
            "cutoff": cutoff, "evaluation_key": key, "evaluation_path": str(self.output/f"{key}.json"),
            "cached": cached, "call_elapsed_seconds": 0., "elapsed_seconds": .1,
        }
        if cutoff == 25:
            value["gap_GHz"] = 5.+30.*x[0]+self.refinement_shift
        self.values[key] = copy.deepcopy(value)
        return value

    def checked(self, x, cutoff):
        key = module.json_hash([list(x), cutoff])
        if key not in self.values:
            raise ValueError("Completed case lacks a physics checkpoint")
        return copy.deepcopy(self.values[key])


def toy_run(directory, targets=(7.5, 15., 22.5, 27.5)):
    run = object.__new__(module.MultistartExperiment)
    run.root, run.out = Path(directory), Path(directory)
    run.config = {"targets_GHz": list(targets), "starts": [
        {"id": "center", "normalized_x": [.5, .5]},
        {"id": "low_l1_high_l2", "normalized_x": [.1, .9]},
        {"id": "high_l1_low_l2", "normalized_x": [.9, .1]}]}
    run.forward_config = json.loads((ROOT/"configs/pure_pwe_math_inverse_v1.json").read_text())
    run.plan = module.planned_cases(run.config)
    run.fingerprint, run.tolerance = "toy", .05
    run.protected, run.protocol = {}, {"code_hashes": {}}
    run.deadline = time.time()+60
    run.evaluator = ToyEvaluator(run.out)
    run.cached_at = run.evaluator.checked
    return run


def constructor_fixture(directory):
    root = Path(directory)
    config = {"targets_GHz": [7.5], "starts": [
        {"id": "center", "normalized_x": [.5, .5]}],
        "base_config": "configs/pure_pwe_math_inverse_v1.json", "max_runtime_hours": 4}
    base = json.loads((ROOT/"configs/pure_pwe_math_inverse_v1.json").read_text())
    prior = root/"results/pure_pwe_math_inverse_v2"
    prior.mkdir(parents=True)
    for folder in ("configs", "scripts", "tests"):
        (root/folder).mkdir()
    protected = root/"old_source.txt"
    protected.write_text("unchanged")
    hashes = {str(protected): module.digest(protected)}
    manifest = {"experiment_version": "pure_pwe_math_inverse_v2", "config": base,
                "protected_hashes": hashes, "code_hashes": {}, "final_tolerance_GHz": .05}
    (prior/"manifest.json").write_text(json.dumps(manifest))
    verification = {"status": "verified_complete", "scientific_fingerprint": module.json_hash(manifest),
                    "source_snapshot_hashes": hashes, "verification_implementation_hashes": hashes}
    (prior/"verification_summary.json").write_text(json.dumps(verification))
    (root/"configs/pure_pwe_multistart_v1.json").write_text(json.dumps(config))
    (root/config["base_config"]).write_text(json.dumps(base))
    (root/base["final_tolerance_source"]).write_text(json.dumps({base["final_tolerance_key"]: .05}))
    for name in ("scripts/run_pwe_multistart_v1.py", "scripts/summarize_pwe_multistart_v1.py",
                 "tests/test_multistart_experiment_v1.py", "tests/test_pwe_multistart_cli_v1.py",
                 "tests/test_pwe_multistart_summary_v1.py"):
        (root/name).write_text("# fixture")
    gate = {"passed": True, "root_review_acknowledged": True, "exploratory_only": True,
            "multistart_config_sha256": module.digest(root/"configs/pure_pwe_multistart_v1.json")}
    return root, gate


class MultistartTests(unittest.TestCase):
    def test_cartesian_target_major_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            run = toy_run(directory)
            self.assertEqual(len(run.plan), 12)
            self.assertEqual([row["target_gap_GHz"] for row in run.plan[:4]], [7.5]*3+[15.])
            self.assertEqual(run.plan[2]["initial_normalized_x"], [.9, .1])

    def test_invalid_plans_fail_closed(self):
        base = {"targets_GHz": [7.5], "starts": [{"id": "safe", "normalized_x": [.5, .5]}]}
        invalid = [
            {**base, "targets_GHz": [7.5, 7.5]}, {**base, "targets_GHz": [float("nan")]},
            {**base, "targets_GHz": [True]}, {**base, "targets_GHz": []},
            {**base, "starts": [{"id": "../escape", "normalized_x": [.5, .5]}]},
            {**base, "starts": [{"id": "safe", "normalized_x": [1.1, .5]}]},
            {**base, "starts": base["starts"]*2},
        ]
        for config in invalid:
            with self.assertRaises(ValueError):
                module.planned_cases(config)

    def test_complete_all_twelve_cases_with_unchanged_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            run = toy_run(directory)
            result = run.run()
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(result["cases"]), 12)
            self.assertEqual(result["successful_cases"], 12)
            for expected, row in zip(run.plan, result["cases"]):
                self.assertEqual(row["search"]["trace"][0]["x"], expected["initial_normalized_x"])
                self.assertEqual(row["payload_sha256"], module.case_hash(row))
                self.assertEqual(row["assessment"]["existing_tolerance_GHz"], .05)
                self.assertFalse(row["search"]["unique_inverse_claimed"])
            for row in result["per_target"]:
                self.assertEqual(row["success_fraction_of_planned"], 1.)
                self.assertEqual(row["planned_starts"], 3)
            self.assertEqual(sum(cutoff == 25 for _, cutoff in run.evaluator.fresh_calls), 12)
            self.assertFalse((run.out/"run.lock").exists())

    def test_resume_completed_cases_never_calls_physics(self):
        with tempfile.TemporaryDirectory() as directory:
            run = toy_run(directory, targets=(15.,))
            original = run.run()
            with mock.patch.object(ToyEvaluator, "__call__", side_effect=AssertionError("No physics")):
                resumed = run.run()
            self.assertEqual(original["cases"], resumed["cases"])
            self.assertEqual(resumed["status"], "complete")

    def test_partial_denominators_do_not_hide_unfinished_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            run = toy_run(directory)
            row = {**run.plan[0], "success": True}
            summary = module.aggregate_cases(run.plan, [row])
            self.assertEqual(summary[0]["success_fraction_of_planned"], 1/3)
            self.assertEqual(summary[0]["success_fraction_of_completed"], 1.)
            self.assertEqual(summary[0]["not_yet_completed"], 2)
            self.assertIsNone(summary[1]["success_fraction_of_completed"])

    def test_local_nonmatches_still_receive_M25_check(self):
        with tempfile.TemporaryDirectory() as directory:
            run = toy_run(directory, targets=(100.,))
            run.forward_config["trust_region"]["max_trial_evaluations"] = 1
            result = run.run()
            self.assertEqual(len(result["cases"]), 3)
            self.assertEqual(result["successful_cases"], 0)
            for row in result["cases"]:
                self.assertEqual(row["outcome"], "unresolved_local_search")
                self.assertFalse(row["search"]["global_feasibility_certified"])
                self.assertAlmostEqual(
                    row["assessment"]["gap_M25_GHz"]-row["assessment"]["gap_M20_GHz"],
                    run.evaluator.refinement_shift)

    def test_independent_refinement_failure_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            run = toy_run(directory, targets=(15.,))
            run.evaluator.refinement_shift = .2
            result = run.run()
            self.assertEqual(result["successful_cases"], 0)
            self.assertTrue(all(row["search"]["matched"] for row in result["cases"]))
            self.assertTrue(all(row["outcome"] == "independent_refinement_failed"
                                for row in result["cases"]))

    def test_deadline_preserves_checkpoint_and_replays_fixed_start(self):
        with tempfile.TemporaryDirectory() as directory:
            run = toy_run(directory, targets=(15.,))
            run.evaluator.raise_after = 2
            result = run.run()
            self.assertEqual(result["status"], "deadline_checkpointed")
            self.assertEqual(result["completed_cases"], 0)
            self.assertFalse((run.out/"run.lock").exists())
            run.evaluator.raise_after = None
            result = run.run()
            self.assertEqual(result["status"], "complete")
            self.assertTrue(result["cases"][0]["evaluation_calls"][0]["cached"])
            self.assertEqual(result["cases"][0]["search"]["trace"][0]["x"], [.5, .5])

    def test_active_lock_refuses_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            run = toy_run(directory)
            lock = run.out/"run.lock"
            lock.write_text("another-worker")
            with self.assertRaises(FileExistsError):
                run.run()
            self.assertEqual(lock.read_text(), "another-worker")
            self.assertFalse((run.out/"manifest.json").exists())

    def test_case_payload_and_start_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run = toy_run(directory, targets=(15.,))
            row = run.run()["cases"][0]
            bad = copy.deepcopy(row)
            bad["assessment"]["error_M25_GHz"] = 0.
            with self.assertRaisesRegex(ValueError, "checksum"):
                run.validate_case(bad, run.plan[0])
            bad["payload_sha256"] = module.case_hash(bad)
            with self.assertRaisesRegex(ValueError, "assessment"):
                run.validate_case(bad, run.plan[0])
            bad = copy.deepcopy(row)
            bad["initial_normalized_x"] = [.1, .9]
            bad["payload_sha256"] = module.case_hash(bad)
            with self.assertRaisesRegex(ValueError, "start"):
                run.validate_case(bad, run.plan[0])

    def test_rehashed_trace_tampering_rejected_by_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            run = toy_run(directory, targets=(15.,))
            row = run.run()["cases"][0]
            row["search"]["trace"][1]["accepted"] = False
            row["payload_sha256"] = module.case_hash(row)
            with self.assertRaisesRegex(ValueError, "replay"):
                run.validate_case(row, run.plan[0])

    def test_missing_completed_physics_cannot_be_silently_recomputed(self):
        with tempfile.TemporaryDirectory() as directory:
            run = toy_run(directory, targets=(15.,))
            row = run.run()["cases"][0]
            del run.evaluator.values[row["independent_refinement_evaluation"]]
            with mock.patch.object(ToyEvaluator, "__call__", side_effect=AssertionError("No physics")):
                with self.assertRaisesRegex(ValueError, "lacks"):
                    run.validate_case(row, run.plan[0])

    def test_failure_record_blocks_automatic_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            run = toy_run(directory, targets=(15.,))
            with mock.patch.object(ToyEvaluator, "__call__", side_effect=ValueError("bad eigensolve")):
                with self.assertRaisesRegex(ValueError, "bad eigensolve"):
                    run.run()
            failure = json.loads((run.out/"failure.json").read_text())
            self.assertEqual(failure["current_case"]["start_id"], "center")
            self.assertFalse((run.out/"run.lock").exists())
            with self.assertRaisesRegex(ValueError, "Previous failure requires review"):
                run.run()

    def test_inherited_changed_hash_is_not_rebaselined(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"old.txt"
            path.write_text("protected")
            run = toy_run(directory)
            expected = module.digest(path)
            run._merge_verified_hashes({str(path): expected})
            path.write_text("changed")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                run._merge_verified_hashes({str(path): expected})
            self.assertEqual(run.protected[str(path.resolve())], expected)

    def test_gate_required_before_construction_or_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            for gate in (
                    {"passed": True},
                    {"passed": True, "root_review_acknowledged": True, "exploratory_only": True,
                     "launch_blockers": ["review not complete"]}):
                with self.assertRaisesRegex(ValueError, "review"):
                    module.MultistartExperiment(directory, gate, deadline=time.time()+60, device="cpu")
            self.assertFalse((Path(directory)/"results").exists())

    def test_cache_validation_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            run = toy_run(directory)
            evaluator = mock.Mock()
            evaluator.output = run.out/"evaluations"
            evaluator.output.mkdir()
            evaluator.sides_from_x.return_value = np.array([140., 91.])
            evaluator._identity.return_value = {"identity": "toy"}
            evaluator._identity_hash.return_value = "key"
            (evaluator.output/"key.json").write_text(json.dumps({"result": {"gap_GHz": 1.}}))
            run.evaluator = evaluator
            result = module.MultistartExperiment.cached_at(run, [.5, .5], 20)
            self.assertEqual(result, {"gap_GHz": 1.})
            evaluator._validate_checkpoint.assert_called_once()
            evaluator.assert_not_called()

    def test_constructor_copies_existing_forward_config_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root, gate = constructor_fixture(directory)
            with mock.patch.object(module, "QuadratureForward") as evaluator:
                run = module.MultistartExperiment(root, gate, deadline=time.time()+60, device="cpu")
            self.assertEqual(run.forward_config, json.loads(
                (root/"configs/pure_pwe_math_inverse_v1.json").read_text()))
            self.assertEqual(evaluator.call_args.args[0], run.forward_config)
            self.assertFalse(run.out.exists())
            self.assertTrue(any("verification_summary.json" in name for name in run.protected))

    def test_constructor_rejects_changed_historical_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root, gate = constructor_fixture(directory)
            (root/"old_source.txt").write_text("changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                module.MultistartExperiment(root, gate, deadline=time.time()+60, device="cpu")
            self.assertFalse((root/"results/pure_pwe_multistart_v1").exists())

    def test_constructor_rejects_forward_config_or_tolerance_changes(self):
        for name, key, value in (
                ("pure_pwe_math_inverse_v1.json", "search_cutoff", 19),
                ("pure_pwe_inverse.json", "gap_tolerance_ghz", .1)):
            with tempfile.TemporaryDirectory() as directory:
                root, gate = constructor_fixture(directory)
                path = root/"configs"/name
                config = json.loads(path.read_text())
                config[key] = value
                path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    module.MultistartExperiment(root, gate, deadline=time.time()+60, device="cpu")

    def test_constructor_rejects_unreviewed_plan_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root, gate = constructor_fixture(directory)
            gate["multistart_config_sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "configuration hash"):
                module.MultistartExperiment(root, gate, deadline=time.time()+60, device="cpu")

    def test_constructor_rejects_excess_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root, gate = constructor_fixture(directory)
            with self.assertRaisesRegex(ValueError, "runtime bound"):
                module.MultistartExperiment(root, gate, deadline=time.time()+5*3600, device="cpu")

    def test_verified_output_cannot_be_constructed_or_run_again(self):
        for marker in ("verification_summary.json", "VERIFICATION.md"):
            with tempfile.TemporaryDirectory() as directory:
                root, gate = constructor_fixture(directory)
                out = root/"results/pure_pwe_multistart_v1"
                out.mkdir()
                (out/marker).write_text("immutable verification")
                with self.assertRaisesRegex(ValueError, "Verified output is immutable"):
                    module.MultistartExperiment(root, gate, deadline=time.time()+60, device="cpu")
                run = toy_run(out, targets=(15.,))
                (out/"results.json").write_text("previous results")
                before = {path.name: path.read_bytes() for path in out.iterdir()}
                with self.assertRaisesRegex(ValueError, "Verified output is immutable"):
                    run.run()
                self.assertEqual(before, {path.name: path.read_bytes() for path in out.iterdir()})

    def test_verification_guard_is_rechecked_under_exclusive_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            run = toy_run(directory)
            with mock.patch.object(run, "_assert_unverified_output", side_effect=[
                    None, ValueError("Verified output is immutable")]) as guard:
                with self.assertRaisesRegex(ValueError, "Verified output is immutable"):
                    run.run()
            self.assertEqual(guard.call_count, 2)
            self.assertFalse((run.out/"run.lock").exists())
            self.assertFalse((run.out/"manifest.json").exists())
            self.assertFalse((run.out/"results.json").exists())


if __name__ == "__main__":
    unittest.main()
