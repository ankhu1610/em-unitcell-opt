import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from validate_energy_pwe_saved import metrics, gap_comparison


class ValidationTests(unittest.TestCase):
    def test_metrics(self):
        m = metrics(np.array([[100., 200.]]), np.array([[101., 198.]]))
        self.assertEqual(m["mae_GHz"], 1.5)
        self.assertEqual(m["max_abs_relative_pct"], 1.)
        self.assertEqual(m["count"], 2)

    def test_gap_definitions(self):
        truth = np.array([[100., 150.], [120., 160.]])
        pred = np.array([[101., 151.], [122., 164.]])
        gaps = gap_comparison(truth, pred, [1., 2.])
        self.assertEqual(gaps["direct_K"]["COMSOL_GHz"], 40.)
        self.assertEqual(gaps["direct_K"]["PWE_GHz"], 42.)
        self.assertEqual(gaps["indirect_M_K"]["COMSOL_GHz"], 30.)
        self.assertEqual(gaps["indirect_M_K"]["PWE_GHz"], 29.)

    def test_shape_rejected(self):
        with self.assertRaises(ValueError):
            metrics(np.ones((2, 2)), np.ones((2, 1)))


if __name__ == "__main__":
    unittest.main()
