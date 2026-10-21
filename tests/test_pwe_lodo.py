import sys
from pathlib import Path
from unittest.mock import Mock
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from benchmark_inverse_pwe_lodo import folds, run_fold, summarize


class LodoTests(unittest.TestCase):
    def setUp(self):
        self.library = [{"source": f"D{i}", "gap_ghz": gap, "L1_um": i+1., "L2_um": 2.}
                        for i, gap in enumerate([-1., 1., 2.])]
        self.settings = {"gap_tolerance_ghz": .05, "max_new_evaluations": 0,
                         "secant_fraction_limits": [.1, .9]}

    def test_exclusion_by_whole_design(self):
        cases = list(folds(self.library))
        self.assertEqual(len(cases), 2)
        for held, entries in cases:
            self.assertEqual(len(entries), 2)
            self.assertNotIn(held["source"], [c["source"] for c in entries])

    def test_leakage_rejected(self):
        with self.assertRaises(ValueError):
            run_fold(self.library[1], self.library, Mock(), self.settings)

    def test_unbracketed_is_counted_not_omitted(self):
        held, entries = list(folds(self.library))[-1]
        r = run_fold(held, entries, Mock(), self.settings)
        self.assertEqual(r["status"], "no_saved_bracket_not_proven_unreachable")
        summary = summarize([{"result": r}, {"execution_error": "test"}], 2)
        self.assertEqual(summary["nonmatching_cases"], 1)
        self.assertEqual(summary["execution_errors"], 1)
        self.assertEqual(summary["gap_mae_ghz"], 1.)
        self.assertEqual(summary["success_rate_completed_pct"], 0.)


if __name__ == "__main__":
    unittest.main()
