import sys
from pathlib import Path
import unittest
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from audit_pure_pwe import eta_matrix, reciprocal_indices, reciprocal_basis, path_coordinates
from invdes.interface_pwe import C
from invdes.pwe_sensitivity import eta_derivative
from invdes.variational_pwe import operator, checked_modes, mode_derivative, spectrum


class VariationalTests(unittest.TestCase):
    def setUp(self):
        self.a = 242.5
        self.ids = reciprocal_indices(2)
        self.g = self.ids@reciprocal_basis(self.a*1e-6)
        self.k = path_coordinates([.37], self.a*1e-6)[0]
        self.sides = np.array([161., 112.])
        self.eta = eta_matrix(self.ids, self.a, self.sides)

    def test_homogeneous(self):
        f, _, _, diag = checked_modes(operator(self.g, self.k, np.eye(len(self.ids))/9))
        expected = np.sort(C*np.linalg.norm(self.g+self.k, axis=1)/(6*np.pi*1e9))[:2]
        np.testing.assert_allclose(f, expected, rtol=1e-12)
        self.assertLess(diag["max_relative_eigenpair_residual"], 1e-10)

    def test_positive_hermitian_and_reciprocity(self):
        matrix = operator(self.g, self.k, self.eta)
        np.testing.assert_allclose(matrix, matrix.conj().T, atol=1e-7)
        forward = checked_modes(matrix)
        reverse = checked_modes(operator(self.g, -self.k, self.eta))
        self.assertTrue(np.all(forward[1] > 0))
        np.testing.assert_allclose(forward[0], reverse[0], rtol=1e-12)

    def test_operator_derivative_both_lengths(self):
        step = 1e-3
        for axis in (0, 1):
            shift = np.eye(2)[axis]*step
            minus = operator(self.g, self.k, eta_matrix(self.ids, self.a, self.sides-shift))
            plus = operator(self.g, self.k, eta_matrix(self.ids, self.a, self.sides+shift))
            fd = (plus-minus)/(2*step)
            analytic = operator(self.g, self.k, eta_derivative(self.ids, self.a, self.sides, axis))
            self.assertLess(np.linalg.norm(fd-analytic)/np.linalg.norm(analytic), 1e-7)

    def test_spectrum_derivative_both_lengths(self):
        step = 1e-3
        _, values, vectors, _ = checked_modes(operator(self.g, self.k, self.eta))
        for axis in (0, 1):
            shift = np.eye(2)[axis]*step
            minus = checked_modes(operator(self.g, self.k, eta_matrix(self.ids, self.a, self.sides-shift)))[0]
            plus = checked_modes(operator(self.g, self.k, eta_matrix(self.ids, self.a, self.sides+shift)))[0]
            analytic = mode_derivative(self.g, self.k, values, vectors,
                                       eta_derivative(self.ids, self.a, self.sides, axis))
            np.testing.assert_allclose(analytic, (plus-minus)/(2*step), rtol=1e-6, atol=1e-7)

    def test_spectrum_layout(self):
        ks = path_coordinates([.37, 1.53], self.a*1e-6)
        derivatives = [eta_derivative(self.ids, self.a, self.sides, axis) for axis in (0, 1)]
        f, df, diagnostics = spectrum(self.g, ks, self.eta, derivatives)
        self.assertEqual(f.shape, (2, 2))
        self.assertEqual(df.shape, (2, 2, 2))
        self.assertEqual(len(diagnostics), 2)
        self.assertEqual(spectrum(self.g, ks, self.eta)[1].shape, (0, 2, 2))

    def test_nested_basis_eigenvalues_do_not_increase(self):
        ids = reciprocal_indices(3)
        large = checked_modes(operator(ids@reciprocal_basis(self.a*1e-6), self.k,
                                       eta_matrix(ids, self.a, self.sides)))[0]
        small = checked_modes(operator(self.g, self.k, self.eta))[0]
        self.assertTrue(np.all(large <= small+1e-9))

    def test_invalid_operator_rejected(self):
        with self.assertRaises(ValueError):
            checked_modes(-np.eye(4))
        broken = np.eye(4, dtype=complex)
        broken[0, 1] = 1j
        with self.assertRaises(ValueError):
            checked_modes(broken)
        with self.assertRaises(ValueError):
            operator(self.g, self.k, np.eye(2))

    def test_audit_job_names_and_scope(self):
        from audit_variational_pwe import jobs, difference
        config = json.loads((ROOT/"configs/variational_pwe_audit_v1.json").read_text())
        planned = jobs(config)
        self.assertEqual(len(planned), 34)
        self.assertEqual(len({job[0] for job in planned}), 34)
        self.assertEqual(sum(job[4] for job in planned), 18)
        for name, index, cutoff, sides, gradients in planned:
            if not gradients:
                self.assertEqual(index, config["finite_difference_design_index"])
                self.assertIn(cutoff, config["finite_difference_cutoffs"])
                self.assertEqual(sides.shape, (2,))
        self.assertEqual(difference([1., 2.], [1., 2.], 1e-8)["max_relative"], 0.)


if __name__ == "__main__":
    unittest.main()
