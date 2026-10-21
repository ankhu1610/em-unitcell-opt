import copy
from pathlib import Path
import sys
import unittest
from unittest import mock
import tempfile
import time
import json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from invdes.inverse.math_inverse_experiment import independent_assessment, render_report, MathInverseExperiment


class MathInverseExperimentTests(unittest.TestCase):
    def test_complete_three_target_workflow_with_toy_physics(self):
        class ToyEvaluator:
            def __init__(self, output):
                self.output = output/"evaluations"
            def __call__(self, x, cutoff=None):
                cutoff = cutoff or 20
                return {"gap_GHz": 10.+20.*x[0]+(.002 if cutoff == 25 else 0.),
                        "gradient_normalized": [20., 0.], "gradient_reliable": True,
                        "normalized_x": list(x), "sides_um": [140.+49.*x[0], 91.+49.*x[1]],
                        "cutoff": cutoff, "evaluation_key": str((list(x), cutoff)),
                        "evaluation_path": "toy", "cached": False, "call_elapsed_seconds": 0.}
        with tempfile.TemporaryDirectory() as directory:
            run = object.__new__(MathInverseExperiment)
            run.out = Path(directory)
            root = Path(__file__).resolve().parents[1]
            run.config = json.loads((root/"configs/pure_pwe_math_inverse_v1.json").read_text())
            run.fingerprint, run.tolerance = "toy", .05
            run.protected = {}
            run.deadline = time.time()+30.
            run.evaluator = ToyEvaluator(run.out)
            with mock.patch.object(run, "verify"):
                result = run.run()
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(result["targets"]), 3)
            for target in result["targets"]:
                self.assertTrue(target["assessment"]["passes_empirical_refinement_guard"])
                self.assertEqual(target["search"]["trace"][0]["x"], [.5, .5])
                self.assertEqual(target["payload_sha256"], run.target_hash(target))
            self.assertFalse((run.out/"run.lock").exists())

    def test_launch_fails_closed_when_evidence_fails(self):
        import run_math_pwe_inverse as cli
        with mock.patch.object(sys, "argv", ["run_math_pwe_inverse.py", "--run",
                                             "--acknowledge-exploratory-review"]), \
                mock.patch.object(cli, "combined_evidence",
                                  return_value={"passed": False, "launch_blockers": ["stale evidence"]}), \
                mock.patch.object(cli, "MathInverseExperiment") as experiment:
            with self.assertRaisesRegex(ValueError, "readiness guard"):
                cli.main()
            experiment.assert_not_called()

    def test_diagnostic_does_not_launch(self):
        import run_math_pwe_inverse as cli
        with mock.patch.object(sys, "argv", ["run_math_pwe_inverse.py"]), \
                mock.patch.object(cli, "combined_evidence",
                                  return_value={"passed": True, "launch_blockers": []}), \
                mock.patch.object(cli, "MathInverseExperiment") as experiment, \
                mock.patch("builtins.print") as output:
            cli.main()
            experiment.assert_not_called()
            self.assertIn("launch_blockers", output.call_args.args[0])

    def test_same_geometry_refinement(self):
        base = {"gap_GHz": 10.009, "normalized_x": [.3, .7], "sides_um": [160., 110.], "cutoff": 20}
        fine = {**base, "gap_GHz": 10.039, "cutoff": 25}
        result = independent_assessment(base, fine, 10., .05)
        self.assertTrue(result["passes_empirical_refinement_guard"])
        self.assertAlmostEqual(result["error_plus_refinement_GHz"], .039)
        self.assertFalse(result["continuum_accuracy_certified"])
        fine["normalized_x"] = [.31, .7]
        with self.assertRaisesRegex(ValueError, "unchanged"):
            independent_assessment(base, fine, 10., .05)

    def test_refinement_cancellation_does_not_pass_conservative_guard(self):
        base = {"gap_GHz": 10.04, "normalized_x": [.3, .7], "sides_um": [160., 110.], "cutoff": 20}
        fine = {**base, "gap_GHz": 10.0, "cutoff": 25}
        result = independent_assessment(base, fine, 10., .05)
        self.assertTrue(result["matches_at_M25"])
        self.assertFalse(result["passes_empirical_refinement_guard"])
        report = render_report({"status": "complete", "targets": [{"assessment": result}]})
        self.assertIn("160", report)
        self.assertIn("not an error bound", report)
        for invalid in (float("nan"), float("inf")):
            altered = copy.deepcopy(fine)
            altered["gap_GHz"] = invalid
            with self.assertRaises(ValueError):
                independent_assessment(base, altered, 10., .05)

    def test_refinement_rejects_cutoff_or_physical_geometry_mismatch(self):
        base = {"gap_GHz": 10., "normalized_x": [.3, .7], "sides_um": [160., 110.], "cutoff": 20}
        for fine in ({**base}, {**base, "cutoff": 25, "sides_um": [161., 110.]}):
            with self.assertRaisesRegex(ValueError, "M20/M25"):
                independent_assessment(base, fine, 10., .05)

    def test_target_resume_checks_payload_and_recomputed_assessment(self):
        experiment = object.__new__(MathInverseExperiment)
        experiment.fingerprint = "test"
        experiment.config = {"search_cutoff": 20, "independent_check_cutoff": 25,
                             "internal_tolerance_GHz": .01}
        experiment.tolerance = .05
        base = {"gap_GHz": 10., "normalized_x": [.3, .7], "sides_um": [160., 110.],
                "cutoff": 20, "evaluation_key": "m20"}
        fine = {**base, "gap_GHz": 10.01, "cutoff": 25, "evaluation_key": "m25"}
        row = {"scientific_fingerprint": "test", "target_gap_GHz": 10.,
               "search": {"x": [.3, .7], "target_gap_GHz": 10., "gap_GHz": 10.,
                          "absolute_error_GHz": 0., "tolerance_GHz": .01, "matched": True,
                          "status": "matched"}, "assessment": independent_assessment(base, fine, 10., .05),
               "candidate_search_evaluation": "m20", "independent_refinement_evaluation": "m25"}
        row["payload_sha256"] = experiment.target_hash(row)
        with mock.patch.object(experiment, "cached_at", side_effect=[base, fine]):
            experiment.validate_target(row, 10.)
        row["assessment"]["error_M25_GHz"] = 0.
        with self.assertRaisesRegex(ValueError, "checksum"):
            experiment.validate_target(row, 10.)
        row["payload_sha256"] = experiment.target_hash(row)
        with mock.patch.object(experiment, "cached_at", side_effect=[base, fine]):
            with self.assertRaisesRegex(ValueError, "assessment disagrees"):
                experiment.validate_target(row, 10.)

if __name__ == "__main__":
    unittest.main()
