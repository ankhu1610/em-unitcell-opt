"""Check evidence or run the separately reviewed, anchor-free pure-PWE prototype."""
import os
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "2"
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from validate_energy_pwe_full import atomic_json
from invdes.projector_experiment import digest
from invdes.inverse.quadrature_launch_gate import check_quadrature_launch_gate
from invdes.inverse.angular_result_guard import validate_angular_evidence
from invdes.inverse.math_inverse_experiment import MathInverseExperiment


def combined_evidence():
    numerical = check_quadrature_launch_gate(ROOT)
    angular = validate_angular_evidence(ROOT)
    angular_passed = angular["passed"]
    evidence_paths = [
        ROOT/"results/radial_quadrature_audit_v1/manifest.json",
        ROOT/"results/radial_quadrature_audit_v1/results.json",
        ROOT/"results/radial_quadrature_audit_v1/status.json",
    ]
    return {
        "passed": bool(numerical["passed"] and angular_passed),
        "numerical_audit": numerical, "full_path_angular_audit": angular,
        "full_path_angular_passed": angular_passed,
        "evidence_hashes": {
            **{str(p): digest(p) for p in evidence_paths if p.exists()},
            **angular["evidence_hashes"],
        },
        "launch_blockers": numerical["reasons"] + angular["reasons"],
        "exploratory_only": True, "root_review_acknowledged": False,
        "scope": "Readiness for three local inverse requests, not a production, continuum or COMSOL accuracy certificate.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--save-review", action="store_true",
                        help="Save diagnostic readiness evidence without launching physics")
    parser.add_argument("--acknowledge-exploratory-review", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--until-utc")
    args = parser.parse_args()
    evidence = combined_evidence()
    if not args.run:
        if args.save_review:
            path = ROOT/"results/pure_pwe_math_inverse_v2/prelaunch_review.json"
            atomic_json(path, evidence)
            print(json.dumps({"passed": evidence["passed"], "launch_authorized": False,
                              "review_path": str(path)}, indent=2))
        else:
            print(json.dumps(evidence, indent=2))
        return
    if not args.acknowledge_exploratory_review:
        parser.error("Launching needs explicit root review acknowledgement")
    if not evidence["passed"] or evidence["launch_blockers"]:
        raise ValueError("Evidence is incomplete or fails the exploratory readiness guard")
    if not args.until_utc:
        parser.error("A bounded --until-utc is required")
    deadline = datetime.fromisoformat(args.until_utc.replace("Z", "+00:00"))
    if deadline.tzinfo is None:
        parser.error("Runtime deadline must include a timezone")
    # New explicit user authorization supersedes the completed earlier session.
    # The runtime stop remains separate from all scientific acceptance criteria.
    if deadline.timestamp()-time.time() > 3*60*60:
        parser.error("Choose a checkpointing runtime window of at most three hours")
    if deadline.timestamp() <= time.time():
        parser.error("The authorized research deadline has passed")
    from invdes.utils import set_seed
    set_seed(0)
    evidence["root_review_acknowledged"] = True
    evidence["root_review"] = (
        "Analytic versus finite-difference agreement, simple unique active sampled edges, "
        "full-path cutoff and angular refinement, CPU/device checks and immutable files reviewed. "
        "Run the preregistered three requests from the fixed nominal point, with fresh pure PWE "
        "at each distinct trial and unchanged-candidate M25 verification. No promotion to production.")
    out = ROOT/"results/pure_pwe_math_inverse_v2"
    review_path = out/"gate_review.json"
    if review_path.exists():
        saved = json.loads(review_path.read_text())
        if saved != evidence:
            raise ValueError("Gate evidence changed after launch; preserve original experiment")
    else:
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(review_path, evidence)
    experiment = MathInverseExperiment(
        ROOT, evidence, deadline=deadline.timestamp(), device=args.device)
    result = experiment.run()
    print(json.dumps({"status": result["status"], "output_directory": str(out),
                      "completed_targets": len(result["targets"])}, indent=2))


if __name__ == "__main__":
    main()
