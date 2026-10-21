"""CPU-only, stdlib-unittest audit regressions; never launch a physics solve."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import audit_atlas_validation_v1 as audit
import render_atlas_validation_report as rendering
import compare_atlas_interpolants as interpolants
from invdes.inverse.atlas_validation_v1 import select_atlas_starts
from invdes.inverse.physics_trust_region import solve_gap_target
from invdes.inverse.quadrature_forward import QuadratureForward


class AtlasAuditTests(unittest.TestCase):
    def setUp(self):
        self.forward = json.loads((ROOT / "configs/pure_pwe_math_inverse_v1.json").read_text())

    def fake_search(self):
        values, calls = {}, []
        def evaluate(x):
            key = str(len(calls))
            value = {"evaluation_key": key, "cutoff": 20, "normalized_x": list(x),
                     "sides_um": (140+49*np.array(x)).tolist(), "gap_GHz": 10*float(x[0]),
                     "gradient_normalized": [10., 0.], "gradient_reliable": True}
            values[key] = value
            calls.append({k: value[k] for k in ("evaluation_key", "normalized_x", "sides_um", "gap_GHz")})
            calls[-1]["call_elapsed_seconds"] = 0.
            return value
        search = solve_gap_target(evaluate, 7.5, [.5,.5],
                                  tolerance_ghz=self.forward["internal_tolerance_GHz"],
                                  **{k: self.forward["trust_region"][k] for k in audit.CONTROLS})
        return {"target_gap_GHz": 7.5, "atlas_start_normalized": [.5,.5],
                "search": search, "evaluation_calls": calls}, values

    def test_deterministic_replay(self):
        row, values = self.fake_search()
        audit.replay_search(row, values, self.forward)

    def test_replay_rejects_modified_trace(self):
        row, values = self.fake_search()
        row["search"]["trace"][0]["gap_GHz"] += 1
        with self.assertRaisesRegex(ValueError, "differs from deterministic replay"):
            audit.replay_search(row, values, self.forward)

    def test_replay_rejects_swapped_start(self):
        row, values = self.fake_search()
        row["atlas_start_normalized"] = [.3,.4]
        with self.assertRaisesRegex(ValueError, "different geometry"):
            audit.replay_search(row, values, self.forward)

    def test_replay_rejects_false_gap(self):
        row, values = self.fake_search()
        row["evaluation_calls"][0]["gap_GHz"] += .1
        with self.assertRaisesRegex(ValueError, "differs from physics"):
            audit.replay_search(row, values, self.forward)

    def test_replay_rejects_extra_call(self):
        row, values = self.fake_search()
        row["evaluation_calls"].append(copy.deepcopy(row["evaluation_calls"][-1]))
        with self.assertRaises(ValueError):
            audit.replay_search(row, values, self.forward)

    def test_replay_rejects_nan_timing(self):
        row, values = self.fake_search()
        row["evaluation_calls"][0]["call_elapsed_seconds"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Invalid call duration"):
            audit.replay_search(row, values, self.forward)

    def test_refinement_must_be_same_geometry(self):
        a = {"cutoff":20,"normalized_x":[.2,.3],"sides_um":[150.,110.],"gap_GHz":7.5}
        b = {**a,"cutoff":25,"normalized_x":[.2,.4]}
        with self.assertRaisesRegex(ValueError, "unchanged candidate"):
            audit.assessment(a,b,7.5,.05)

    def test_guard_is_stricter_than_m25_match(self):
        a = {"cutoff":20,"normalized_x":[.2,.3],"sides_um":[150.,110.],"gap_GHz":7.54}
        b = {**a,"cutoff":25,"gap_GHz":7.48}
        result = audit.assessment(a,b,7.5,.05)
        self.assertTrue(result["matches_at_M25"])
        self.assertFalse(result["passes_empirical_refinement_guard"])

    def test_selection_is_reproducible_and_separated(self):
        rows = [{"normalized_x":[x,y],"gap_GHz":10*x+5*y} for x in (0.,.5,1.) for y in (0.,.5,1.)]
        config = {"selection_grid_size":21,"targets_GHz":[3.,6.,9.,12.],
                  "minimum_distance_from_prior_M25":.12,"minimum_distance_between_selected_starts":.15}
        starts = select_atlas_starts(rows, [[.5,.5]], config)
        self.assertEqual(starts, select_atlas_starts(rows, [[.5,.5]], config))
        for i, x in enumerate(starts):
            self.assertGreaterEqual(np.linalg.norm(np.array(x)-[.5,.5]),.12)
            for y in starts[:i]:
                self.assertGreaterEqual(np.linalg.norm(np.array(x)-y),.15)

    def test_selection_rejects_empty_eligible_region(self):
        rows = [{"normalized_x":[x,y],"gap_GHz":10*x+5*y} for x,y in ((0.,0.),(0.,1.),(1.,0.),(1.,1.))]
        config = {"selection_grid_size":5,"targets_GHz":[3.],
                  "minimum_distance_from_prior_M25":2.,"minimum_distance_between_selected_starts":.1}
        with self.assertRaisesRegex(ValueError,"No atlas grid point"):
            select_atlas_starts(rows, [[.5,.5]], config)

    def test_error_metrics_reject_nonfinite(self):
        for errors in ([],[float("nan")],[float("inf")]):
            with self.subTest(errors=errors), self.assertRaises(ValueError):
                audit.error_metrics(errors)

    def test_error_metric_bias_sign(self):
        m = audit.error_metrics([1.,-3.])
        self.assertEqual(m["bias_GHz"],-1.)
        self.assertEqual(m["MAE_GHz"],2.)
        self.assertAlmostEqual(m["RMSE_GHz"],np.sqrt(5.))

    def test_near_duplicate_grouping(self):
        points = [[.4,.6],[.4,.6000000000000001],[0.,0.],[.4,.6001]]
        self.assertEqual(audit.coordinate_groups(points),[[0,1],[2],[3]])

    def test_grouping_rejects_invalid_coordinates(self):
        with self.assertRaises(ValueError):
            audit.coordinate_groups([[0.,float("nan")]])

    def test_contour_axis_orientation(self):
        segments = rendering.contour_segments([[0.,1.],[0.,1.]],.5)
        self.assertEqual(len(segments),2)
        self.assertTrue(all(abs(p[0]-.5)<1e-12 for s in segments for p in s))
        self.assertEqual(sorted({p[1] for s in segments for p in s}),[0.,.5,1.])

    def test_plot_physical_l2_increases_upwards(self):
        self.assertEqual(rendering.pixel((0.,0.),85),(85.,430.))
        self.assertEqual(rendering.pixel((1.,1.),85),(525.,100.))

    def test_contour_saddle_is_finite(self):
        segments = rendering.contour_segments([[-1.,1.],[1.,-1.]],0.)
        self.assertEqual(len(segments),2)
        self.assertTrue(np.all(np.isfinite(segments)))

    def test_constant_grid_has_no_false_contours(self):
        self.assertEqual(rendering.contour_segments(np.ones((4,4)),2.),[])

    def test_contours_reject_nan(self):
        with self.assertRaises(ValueError):
            rendering.contour_segments([[1.,2.],[3.,float("nan")]],2.)

    def test_alternative_models_reproduce_an_affine_plane(self):
        points=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.],[.5,.5]])
        values=2*points[:,0]+3*points[:,1]+4
        query=np.array([[.2,.3],[.6,.1]])
        for name in ("RBF_linear_polynomial","piecewise_linear"):
            with self.subTest(model=name):
                np.testing.assert_allclose(interpolants.fit(name,points,values)(query),
                                           2*query[:,0]+3*query[:,1]+4,rtol=0.,atol=1e-12)

    def test_piecewise_linear_refuses_extrapolation(self):
        model=interpolants.fit("piecewise_linear",np.array([[0.,0.],[0.,1.],[1.,0.]]),np.array([0.,1.,2.]))
        self.assertTrue(np.isnan(model([[1.,1.]])[0]))

    def test_comparison_rejects_unknown_model(self):
        with self.assertRaises(ValueError):
            interpolants.fit("unknown",np.zeros((4,2)),np.zeros(4))

    def test_complete_source_audit_never_calls_physics(self):
        with patch.object(QuadratureForward,"__call__",side_effect=AssertionError("Physics must not run")):
            summary = audit.audit(ROOT,sources_only=True)
        self.assertEqual(summary["source_M20_count"],58)
        self.assertEqual(summary["prior_M25_validated_count"],12)
        self.assertFalse(summary["verification_ran_physics"])

    def test_final_audit_refuses_incomplete_validation(self):
        original = audit.Evidence.read
        def unfinished(evidence,path):
            value = original(evidence,path)
            if Path(path).resolve() == (ROOT/"results/pure_pwe_atlas_validation_v1/results.json").resolve():
                value["status"] = "running"
            return value
        with patch.object(audit.Evidence,"read",unfinished), self.assertRaisesRegex(ValueError,"Validation is not complete"):
            audit.audit(ROOT)

    def test_checkpoint_payload_tampering_rejected(self):
        out = ROOT / "results/pure_pwe_response_atlas_v2"
        manifest = json.loads((out/"manifest.json").read_text())
        evidence = audit.Evidence()
        original = evidence.read
        def corrupted(path):
            record = original(path)
            record["frequencies_GHz"][0][0] += 1.
            return record
        with patch.object(evidence,"read",side_effect=corrupted), self.assertRaises(ValueError):
            audit.verified_evaluations(out,manifest,evidence)

    def test_forged_payload_hash_cannot_hide_false_gap(self):
        out = ROOT / "results/pure_pwe_response_atlas_v2"
        manifest = json.loads((out/"manifest.json").read_text())
        evidence = audit.Evidence()
        original = evidence.read
        def corrupted(path):
            record = original(path)
            record["result"]["gap_GHz"] += 1.
            record["payload_sha256"] = QuadratureForward._payload_hash(record)
            return record
        with patch.object(evidence,"read",side_effect=corrupted), self.assertRaises(ValueError):
            audit.verified_evaluations(out,manifest,evidence)


if __name__ == "__main__":
    unittest.main(verbosity=2)
