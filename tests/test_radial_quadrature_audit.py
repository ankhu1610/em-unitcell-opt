"""Light harness tests; no experiment directories or large eigensolves."""
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from audit_radial_quadrature import planned_jobs, RadialAudit


class RadialAuditTests(unittest.TestCase):
    def test_job_scopes_and_predeclared_order(self):
        config = json.loads((ROOT/"configs/radial_quadrature_audit_v1.json").read_text())
        screen, all_jobs = planned_jobs(config, "screen"), planned_jobs(config, "all")
        self.assertEqual(len(screen), 47)
        self.assertEqual(len(all_jobs), 61)
        self.assertEqual(len(set(j["name"] for j in all_jobs)), len(all_jobs))
        self.assertFalse(any(j.get("full") for j in screen))
        self.assertEqual(sum(j.get("full", False) for j in all_jobs), 14)
        self.assertTrue(all(j["order"] == 128 for j in all_jobs
                            if j["kind"] == "spectrum" and not j["name"].startswith("angular")))

    def test_small_spectrum_and_rebuilt_gradient(self):
        config = json.loads((ROOT/"configs/radial_quadrature_audit_v1.json").read_text())
        audit = RadialAudit(ROOT, config, "unit-test", {})
        audit.progress = lambda stage: None
        job = dict(name="unit", kind="spectrum", index=1, cutoff=2,
                   order=32, full=False, gradients=True)
        base = audit.spectrum(job)
        perturbed = [audit.spectrum(dict(job, gradients=False, axis=0, step=.001, sign=s))
                     for s in (-1, 1)]
        fd = (np.array(perturbed[1]["frequencies_GHz"])
              - np.array(perturbed[0]["frequencies_GHz"]))/.002
        analytic = np.array(base["analytic_gradients_GHz_per_um"])[0]
        np.testing.assert_allclose(fd, analytic, rtol=2e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
