import sys
from pathlib import Path
import unittest
from unittest.mock import Mock
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from invdes.inverse.pure_pwe import candidate, design_gap


def point(x, gap):
    return candidate([x, 2.], [[10., 20.+gap], [20., 21.+gap]], "synthetic")


class PureInverseTests(unittest.TestCase):
    def test_gap_uses_global_edges_not_direct_difference(self):
        self.assertEqual(point(1, 3)["gap_ghz"], 3)

    def test_exact_retrieval_needs_no_solver(self):
        evaluate = Mock(side_effect=AssertionError("Must reuse saved PWE"))
        r = design_gap(3, [point(1, 3), point(2, 5)], evaluate)
        self.assertEqual(r["status"], "matched_saved_pwe")
        evaluate.assert_not_called()

    def test_continuous_root_really_calls_evaluator(self):
        evaluate = Mock(side_effect=lambda sides: point(sides[0], sides[0]**2))
        r = design_gap(2., [point(1, 1), point(2, 4)], evaluate, tolerance=1e-4,
                       max_evaluations=20)
        self.assertEqual(r["status"], "matched_new_pwe")
        self.assertLessEqual(r["absolute_error_ghz"], 1e-4)
        self.assertGreater(evaluate.call_count, 0)
        self.assertTrue(1 < r["candidate"]["L1_um"] < 2)

    def test_negative_endpoint_can_bracket_positive_gap(self):
        r = design_gap(.5, [point(1, -1), point(3, 1)],
                       lambda sides: point(sides[0], sides[0]-2))
        self.assertEqual(r["status"], "matched_new_pwe")

    def test_invalid_targets(self):
        for target in [0, -1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                design_gap(target, [point(1, 1)], Mock())

    def test_no_bracket_does_not_claim_impossibility(self):
        r = design_gap(10, [point(1, 1), point(2, 4)], Mock())
        self.assertEqual(r["status"], "no_saved_bracket_not_proven_unreachable")

    def test_exhausted_budget_is_not_success(self):
        r = design_gap(2, [point(1, 1), point(2, 4)], Mock(), max_evaluations=0)
        self.assertEqual(r["status"], "budget_exhausted")

    def test_bad_spectrum_rejected(self):
        for spectrum in [[[2, 1]], [[1, np.nan]], [[0, 1]]]:
            with self.assertRaises(ValueError):
                candidate([1, 2], spectrum, "bad")


if __name__ == "__main__":
    unittest.main()
