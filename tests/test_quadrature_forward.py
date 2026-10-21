"""Only small M2/few-point CPU checks; scratch output, no search launch."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
sys.path.insert(0, str(ROOT/"src"))
from invdes.inverse.quadrature_forward import QuadratureForward, assess_gradient_reliability


def small_config():
    c = json.loads((ROOT/"configs/pure_pwe_math_inverse_v1.json").read_text())
    c.update(search_cutoff=2, independent_check_cutoff=3, full_path_start=.37,
             full_path_step=.41, full_path_count=5, projector_quadrature_order=32)
    return c


class QuadratureForwardTests(unittest.TestCase):
    def test_bounds_and_cache_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluator = QuadratureForward(small_config(), directory, "test")
            b = small_config()["a_um"]/np.sqrt(3)
            np.testing.assert_allclose(evaluator.sides_from_x([0, 0]), [b, .65*b])
            np.testing.assert_allclose(evaluator.sides_from_x([1, 1]), [1.35*b, b])
            for x in ([-.01, .5], [.5, 1.01], [np.nan, .5], [.5]):
                with self.assertRaises(ValueError):
                    evaluator.sides_from_x(x)
            result = evaluator([.5, .5])
            cached = evaluator([.5, .5])
            self.assertFalse(result["cached"])
            self.assertTrue(cached["cached"])
            self.assertEqual(result["identity_hash"], cached["identity_hash"])
            self.assertEqual(len(evaluator._projectors), 1)
            other = QuadratureForward(small_config(), directory, "different")
            self.assertNotEqual(evaluator._identity_hash(evaluator._identity([.5, .5], 2)),
                                other._identity_hash(other._identity([.5, .5], 2)))
            changed = np.nextafter(.5, 1.)
            self.assertNotEqual(evaluator._identity_hash(evaluator._identity([.5, .5], 2)),
                                evaluator._identity_hash(evaluator._identity([changed, .5], 2)))
            path = Path(directory)/"evaluations"/f'{result["identity_hash"]}.json'
            record = json.loads(path.read_text())
            record["identity"]["cutoff"] = 999
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                evaluator([.5, .5])

    def test_normalized_gradient_full_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluator = QuadratureForward(small_config(), directory, "gradient")
            x = np.array([.47, .53])
            result = evaluator(x)
            gradient = np.asarray(result["gradient_normalized_x"])
            np.testing.assert_allclose(gradient,
                                       np.asarray(result["gradient_GHz_per_um"])*evaluator.width)
            for axis in (0, 1):
                dx = np.eye(2)[axis]*1e-5
                minus, plus = evaluator(x-dx), evaluator(x+dx)
                self.assertEqual(minus["checks"]["edge_indices"], plus["checks"]["edge_indices"])
                fd = (plus["gap_GHz"]-minus["gap_GHz"])/(2e-5)
                np.testing.assert_allclose(fd, gradient[axis], rtol=2e-5, atol=2e-6)

    def test_resume_after_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            def deadline():
                calls.append(1)
                if len(calls) == 4:
                    raise TimeoutError("test deadline")
            evaluator = QuadratureForward(small_config(), directory, "resume", deadline_check=deadline)
            with self.assertRaises(TimeoutError):
                evaluator([.5, .5])
            path = next((Path(directory)/"evaluations").glob("*.json"))
            record = json.loads(path.read_text())
            self.assertEqual(len(record["frequencies_GHz"]), 2)
            resumed = QuadratureForward(small_config(), directory, "resume")([.5, .5])
            self.assertEqual(resumed["resumed_point_count"], 2)
            self.assertEqual(resumed["frequencies_GHz"][:2], record["frequencies_GHz"])

    def test_reliability_flags(self):
        self.assertTrue(assess_gradient_reliability([.1, .2], [.01, .02], [1., 2.],
                                                   1e-6, 1e-4)["gradient_reliable"])
        for separation, margin, gradient in (([0., .2], [.01, .02], [1., 2.]),
                                             ([.1, .2], [0., .02], [1., 2.]),
                                             ([.1, .2], [.01, .02], [np.nan, 2.])):
            self.assertFalse(assess_gradient_reliability(separation, margin, gradient,
                                                        1e-6, 1e-4)["gradient_reliable"])

    def test_payload_and_derived_field_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluator = QuadratureForward(small_config(), directory, "integrity")
            result = evaluator([.5, .5])
            path = Path(result["evaluation_path"])
            original = json.loads(path.read_text())
            altered = json.loads(json.dumps(original))
            altered["result"]["gap_GHz"] += 1.
            path.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "checksum"):
                evaluator([.5, .5])
            # Even a freshly recomputed checksum must not bypass consistency checks.
            changes = [
                lambda r: r["result"].update(gap_GHz=r["result"]["gap_GHz"]+1),
                lambda r: r["result"].update(gradient_normalized=[0., 0.]),
                lambda r: r["result"].update(gradient_reliable=not r["result"]["gradient_reliable"]),
                lambda r: r["point_diagnostics"][0].update(relative_eigenpair_residual=1.),
                lambda r: r["point_diagnostics"][0].update(k_index=2),
                lambda r: r["point_diagnostics"][0].update(backend="unknown"),
                lambda r: r.update(elapsed_seconds=-1.),
                lambda r: r["result"]["checks"].update(edge_uniqueness_margins_GHz=[0., 0.]),
            ]
            for change in changes:
                altered = json.loads(json.dumps(original))
                change(altered)
                altered["payload_sha256"] = evaluator._payload_hash(altered)
                path.write_text(json.dumps(altered))
                with self.assertRaises(ValueError):
                    evaluator([.5, .5])
            altered = json.loads(json.dumps(original))
            altered["point_diagnostics"][0]["relative_eigenpair_residual"] = float("nan")
            path.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "finite JSON"):
                evaluator([.5, .5])

    def test_cleanup_runs_on_backend_constructor_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluator = QuadratureForward(small_config(), directory, "gpu-failure", device="cuda")
            cleared = []
            fake_torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: cleared.append(True)))
            with patch.object(evaluator, "_backend", side_effect=RuntimeError("constructor failed")):
                with patch.dict(sys.modules, {"torch": fake_torch}):
                    with self.assertRaisesRegex(RuntimeError, "constructor failed"):
                        evaluator([.5, .5])
            self.assertEqual(cleared, [True])
            record = json.loads(next((Path(directory)/"evaluations").glob("*.json")).read_text())
            self.assertEqual(record["frequencies_GHz"], [])
            self.assertEqual(record["payload_sha256"], evaluator._payload_hash(record))


if __name__ == "__main__":
    unittest.main()
