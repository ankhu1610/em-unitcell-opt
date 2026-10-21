"""Audit, test, and publish new derived artifacts once the physics worker finishes."""
from __future__ import annotations
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import time
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from audit_atlas_validation_v1 import audit, file_hash, verify_hashes
from invdes.inverse.atlas_validation_v1 import select_atlas_starts
from render_atlas_validation_report import report, render_html, load_comsol_reference
from compare_atlas_interpolants import compare, report as comparison_report


def main():
    out = ROOT/"results/pure_pwe_atlas_validation_v1/final"
    if out.exists():
        raise ValueError("Final output directory already exists; do not overwrite final evidence")
    summary = audit(ROOT)
    suite = unittest.defaultTestLoader.discover(str(ROOT/"tests"),pattern="test_atlas_validation_audit_unittest.py")
    capture = io.StringIO()
    started = time.perf_counter()
    result = unittest.TextTestRunner(stream=capture,verbosity=2).run(suite)
    test_result = {"tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
                   "skipped":len(result.skipped),"passed":result.wasSuccessful(),
                   "elapsed_seconds":time.perf_counter()-started,"transcript":capture.getvalue(),
                   "implementation_sha256":file_hash(ROOT/"tests/test_atlas_validation_audit_unittest.py")}
    print(capture.getvalue())
    if not result.wasSuccessful():
        raise ValueError("Focused tests failed; no final report published")
    summary["focused_tests"] = {k:v for k,v in test_result.items() if k != "transcript"}
    comparison = compare(summary)
    summary["interpolant_comparison"] = {"scope":comparison["scope"],"limitations":comparison["limitations"],
        "models":[{k:v for k,v in r.items() if k != "grouped_CV_rows"} for r in comparison["models"]]}
    config = json.loads((ROOT/"configs/pure_pwe_atlas_validation_v1.json").read_text())
    timings = []
    for _ in range(20):
        t0 = time.perf_counter()
        selected = select_atlas_starts(summary["source_observations"],
            [r["normalized_x"] for r in summary["prior_validated_cases"] if r["success"]],config)
        timings.append(time.perf_counter()-t0)
        if selected != summary["atlas_starts_normalized"]:
            raise ValueError("Proposal benchmark changed selection")
    summary["proposal_selection_benchmark"] = {"repeats":20,"median_seconds":float(np.median(timings)),
        "min_seconds":min(timings),"max_seconds":max(timings),
        "scope":"Fit 58-record frozen RBF and choose all four starts on the 81x81 grid; excludes physics and source-audit I/O. One host under current load, not a controlled hardware benchmark."}
    summary["generated_utc"] = datetime.now(timezone.utc).isoformat()
    summary["verification_implementation_hashes"].update({str(ROOT/p):file_hash(ROOT/p) for p in (
        "scripts/finalize_atlas_validation_v1.py","scripts/render_atlas_validation_report.py",
        "scripts/compare_atlas_interpolants.py",
        "tests/test_atlas_validation_audit_unittest.py","scripts/rasterize_atlas_svg.cjs",
        "src/invdes/inverse/atlas_validation_v1.py","src/invdes/inverse/response_atlas_v1.py")})
    forward = json.loads((ROOT/"configs/pure_pwe_math_inverse_v1.json").read_text())
    ref = load_comsol_reference(ROOT/"data/processed/bands.npz",0)
    artifacts = {"audit_summary.json":json.dumps(summary,indent=2,allow_nan=False)+"\n",
                 "focused_tests.json":json.dumps(test_result,indent=2,allow_nan=False)+"\n",
                 "interpolant_comparison.json":json.dumps(comparison,indent=2,allow_nan=False)+"\n",
                 "INTERPOLANT_COMPARISON.md":comparison_report(comparison),
                 "VALIDATION_REPORT.md":report(summary),
                 "validated_response_atlas.html":render_html(summary,forward,ref)}
    verify_hashes(summary["source_snapshot_hashes"])
    verify_hashes(summary["verification_implementation_hashes"])
    out.mkdir()
    for name,content in artifacts.items():
        with (out/name).open("x",encoding="utf-8") as stream:
            stream.write(content)
    receipt = {"status":"generated_pending_final_visual_review","generated_utc":summary["generated_utc"],
               "artifacts":{name:file_hash(out/name) for name in artifacts},
               "note":"Static figure must be rasterized and inspected; browser-layout QA is blocked for local file URLs."}
    with (out/"generation_receipt.json").open("x",encoding="utf-8") as stream:
        json.dump(receipt,stream,indent=2)
    print(json.dumps({"status":summary["status"],"successful_cases":summary["successful_cases"],
                      "output":str(out)},indent=2))


if __name__ == "__main__":
    main()
