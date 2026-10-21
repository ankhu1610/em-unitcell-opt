import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from validate_energy_pwe_full import lowest_two
from invdes.interface_pwe import real_frequencies


class FullValidationTests(unittest.TestCase):
    def test_subset_matches_dense(self):
        g = np.array([[0., 0.], [1., 0.], [-1., 0.], [0., 1.]])*2*np.pi
        b = np.eye(4)/9
        z = np.zeros((4, 4))
        tensor = b, z, z, b
        k = [.3, .4]
        subset, check = lowest_two(g, k, tensor)
        dense, _ = real_frequencies(g, [k], tensor, 2)
        np.testing.assert_allclose(subset, dense[0], rtol=1e-12)
        self.assertLess(check["relative_eigenpair_residual"], 1e-12)

    def test_rejects_negative(self):
        b = -np.eye(4)
        z = np.zeros((4, 4))
        with self.assertRaises(ValueError):
            lowest_two(np.ones((4, 2)), [.3, .4], (b, z, z, b))


if __name__ == "__main__":
    unittest.main()
