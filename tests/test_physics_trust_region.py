import json
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from invdes.inverse.physics_trust_region import bounded_minimum_norm_step, solve_gap_target


class BoundedStepTests(unittest.TestCase):
    def test_unbounded_minimum_norm_root(self):
        gradient = np.array([2., 3.])
        step = bounded_minimum_norm_step(.1, gradient, [.5, .5], .25)
        np.testing.assert_allclose(step, -.1*gradient/(gradient@gradient), atol=1e-15)

    def test_boundary_and_unused_coordinate(self):
        np.testing.assert_allclose(
            bounded_minimum_norm_step(.2, [1., 1.], [0., .5], .4), [0., -.2], atol=1e-14)
        np.testing.assert_allclose(
            bounded_minimum_norm_step(-.4, [0., 2.], [.3, .4], .25), [0., .2], atol=1e-14)
        np.testing.assert_allclose(
            bounded_minimum_norm_step(3., [1., 2.], [0., .5], .1), [0., -.1], atol=1e-14)

    def test_optimal_against_dense_grid(self):
        # Deterministic cases include infeasible roots, boundary starts and rank-one axes.
        cases = [
            (.7, [1., 2.], [.5, .5], .2),
            (-.6, [1., -2.], [.2, .8], .1),
            (.13, [-.2, .8], [0., 1.], .3),
            (-.9, [0., 3.], [1., .9], .25),
            (.04, [2., -1.], [.01, .7], .15)]
        for residual, gradient, x, radius in cases:
            g, x = np.array(gradient), np.array(x)
            step = bounded_minimum_norm_step(residual, g, x, radius)
            lower, upper = np.maximum(-x, -radius), np.minimum(1-x, radius)
            self.assertTrue(np.all(step >= lower-1e-14) and np.all(step <= upper+1e-14))
            p, q = np.meshgrid(np.linspace(lower[0], upper[0], 301),
                               np.linspace(lower[1], upper[1], 301), indexing="ij")
            dense = np.min(np.abs(residual+g[0]*p+g[1]*q))
            self.assertLessEqual(abs(residual+g@step), dense+1e-12)

    def test_gradient_rescaling_and_zero(self):
        expected = bounded_minimum_norm_step(.1, [2., -3.], [.5, .5], .2)
        for scale in [1e-200, 1e200]:
            np.testing.assert_allclose(
                bounded_minimum_norm_step(.1*scale, np.array([2., -3.])*scale, [.5, .5], .2),
                expected, rtol=1e-13, atol=1e-15)
        np.testing.assert_array_equal(bounded_minimum_norm_step(1., [0., 0.], [.5, .5], .1), [0., 0.])
        np.testing.assert_array_equal(bounded_minimum_norm_step(0., [1., 2.], [.5, .5], .1), [0., 0.])

    def test_invalid_inputs(self):
        for point in [[-.01, .5], [.5, 1.01], [np.nan, .5], [.5]]:
            with self.assertRaises(ValueError):
                bounded_minimum_norm_step(1., [1., 1.], point, .1)
        for gradient in [[1.], [np.nan, 2.]]:
            with self.assertRaises(ValueError):
                bounded_minimum_norm_step(1., gradient, [.5, .5], .1)
        with self.assertRaises(ValueError):
            bounded_minimum_norm_step(1., [1., 1.], [.5, .5], 0.)


class DriverTests(unittest.TestCase):
    def test_affine_gap_convergence_and_domain_bounds(self):
        calls = []
        def evaluate(x):
            calls.append(x)
            self.assertTrue(np.all(x >= 0) and np.all(x <= 1))
            return {"gap_GHz": 2*x[0]+3*x[1], "gradient_normalized": [2., 3.],
                    "gradient_reliable": True}
        result = solve_gap_target(evaluate, 4.5, tolerance_ghz=1e-9)
        self.assertEqual(result["status"], "matched")
        self.assertEqual(len(calls), result["total_evaluation_calls"])
        self.assertLessEqual(result["trial_evaluations"], 12)
        accepted = [r["residual_GHz"] for r in result["trace"] if r.get("accepted")]
        self.assertTrue(np.all(np.diff(np.abs(accepted)) < 0))
        self.assertFalse(result["unique_inverse_claimed"])
        json.dumps(result)

    def test_nonlinear_gap_convergence(self):
        def evaluate(x):
            return {"gap_GHz": x[0]**2+x[1], "gradient_normalized": [2*x[0], 1.],
                    "gradient_reliable": True}
        result = solve_gap_target(evaluate, 1.1, tolerance_ghz=1e-8)
        self.assertEqual(result["status"], "matched")
        self.assertLessEqual(result["absolute_error_GHz"], 1e-8)

    def test_missing_unreliable_and_zero_gradient_use_poll(self):
        for derivative, reliable in [(None, False), ([0., 0.], True), ([-1., 0.], False)]:
            def evaluate(x):
                return {"gap_GHz": x[0], "gradient_normalized": derivative,
                        "gradient_reliable": reliable}
            result = solve_gap_target(evaluate, .65, tolerance_ghz=1e-9)
            self.assertEqual(result["status"], "matched")
            self.assertTrue(all(r["method"] == "coordinate_poll"
                                for r in result["trace"] if r["event"] == "trial"))

    def test_reject_wrong_linear_model(self):
        def evaluate(x):
            return {"gap_GHz": x[0], "gradient_normalized": [-1., 0.],
                    "gradient_reliable": True}
        result = solve_gap_target(evaluate, .8, max_trial_evaluations=1)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertFalse(result["trace"][1]["accepted"])
        self.assertLess(result["trace"][1]["reduction_ratio"], 0)
        np.testing.assert_allclose(result["x"], [.5, .5])

    def test_repeated_wrong_models_trigger_poll(self):
        def evaluate(x):
            return {"gap_GHz": x[0], "gradient_normalized": [-1., 0.],
                    "gradient_reliable": True}
        result = solve_gap_target(evaluate, .8, max_trial_evaluations=3)
        self.assertEqual([r["method"] for r in result["trace"][1:]],
                         ["trust_region", "trust_region", "coordinate_poll"])
        self.assertTrue(result["trace"][-1]["accepted"])

    def test_initial_match_and_zero_budget(self):
        evaluator = lambda x: {"gap_GHz": 1.}
        matched = solve_gap_target(evaluator, 1., max_trial_evaluations=0)
        self.assertEqual(matched["status"], "matched")
        result = solve_gap_target(evaluator, 2., max_trial_evaluations=0)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["total_evaluation_calls"], 1)
        self.assertFalse(result["global_feasibility_certified"])
        self.assertIn("not evidence of infeasibility", result["interpretation"])

    def test_flat_response_stops_at_radius_not_infeasibility(self):
        result = solve_gap_target(lambda x: {"gap_GHz": 1.}, 2.,
                                  initial_radius=.1, minimum_radius=.06,
                                  max_trial_evaluations=12)
        self.assertEqual(result["status"], "radius_exhausted")
        self.assertEqual(result["trial_evaluations"], 4)

    def test_invalid_evaluator_and_parameters(self):
        for evaluate in [lambda x: {}, lambda x: {"gap_GHz": np.nan},
                         lambda x: {"gap_GHz": 1., "gradient_normalized": [1.]},
                         lambda x: {"gap_GHz": 1., "gradient_reliable": "yes"}]:
            with self.assertRaises(ValueError):
                solve_gap_target(evaluate, 2.)
        with self.assertRaises(ValueError):
            solve_gap_target(lambda x: {"gap_GHz": 1.}, 2., initial_x=[-1., 0.])
        with self.assertRaises(ValueError):
            solve_gap_target(lambda x: {"gap_GHz": 1.}, 2., max_trial_evaluations=-1)

    def test_preregistered_launch_gate_and_no_anchors(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root/"configs/pure_pwe_math_inverse_v1.json").read_text())
        self.assertEqual(config["targets_GHz"], [10., 20., 30.])
        self.assertEqual(config["initial_normalized_x"], [.5, .5])
        self.assertEqual(config["launch_gate"]["status"], "not_authorized_by_this_config")
        self.assertFalse(config["saved_anchors_used"])
        tolerance = json.loads((root/config["final_tolerance_source"]).read_text())[config["final_tolerance_key"]]
        self.assertGreaterEqual(tolerance, config["internal_tolerance_GHz"])


if __name__ == "__main__":
    unittest.main()
