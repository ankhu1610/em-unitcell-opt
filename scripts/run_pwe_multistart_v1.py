"""Run the isolated multistart extension without modifying the validated PWE."""
import os
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "2"
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from run_math_pwe_inverse import combined_evidence
from invdes.inverse.multistart_experiment_v1 import MultistartExperiment
from invdes.projector_experiment import digest
from summarize_math_inverse_v2 import parse_preflight_xml


def reviewed_evidence():
    """Refresh old numerical gates; diagnostic readiness never launches work."""
    evidence = combined_evidence()
    evidence["scope"] = (
        "Twelve local searches: four new prescribed gaps by three fixed starts. "
        "Unchanged frozen solver/config/controls; full M20 paths and independent "
        "same-candidate M25 checks. No production or continuum certification.")
    evidence["multistart_config_sha256"] = digest(
        ROOT/"configs/pure_pwe_multistart_v1.json")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--acknowledge-exploratory-review", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--until-utc")
    args = parser.parse_args()
    evidence = reviewed_evidence()
    if not args.run:
        print(json.dumps(evidence, indent=2))
        return
    if not args.acknowledge_exploratory_review:
        parser.error("Launching requires explicit exploratory review acknowledgement")
    if not evidence["passed"] or evidence["launch_blockers"]:
        raise ValueError("Readiness guard failed; no multistart physics may launch")
    if not args.until_utc:
        parser.error("A bounded --until-utc is required")
    deadline = datetime.fromisoformat(args.until_utc.replace("Z", "+00:00"))
    if deadline.tzinfo is None:
        parser.error("Deadline must include its timezone")
    config = json.loads((ROOT/"configs/pure_pwe_multistart_v1.json").read_text())
    remaining = deadline.timestamp()-time.time()
    if not 0 < remaining <= config["max_runtime_hours"]*3600:
        parser.error("Deadline must be in the future within the configured runtime window")
    preflight_path = ROOT/"results/pure_pwe_multistart_v1/preflight_tests.xml"
    preflight_text = preflight_path.read_text()
    evidence["preflight_tests"] = parse_preflight_xml(preflight_text)
    tested_classes = {
        case.get("classname", "") for case in ET.fromstring(preflight_text).iter("testcase")}
    for module in ("tests.test_multistart_experiment_v1",
                   "tests.test_pwe_multistart_cli_v1",
                   "tests.test_pwe_multistart_summary_v1"):
        if not any(name == module or name.startswith(module+".") for name in tested_classes):
            raise ValueError(f"Required preflight module missing: {module}")
    evidence["evidence_hashes"][str(preflight_path)] = digest(preflight_path)
    evidence["root_review_acknowledged"] = True
    evidence["root_review"] = (
        "User-authorized multi-start extension reviewed against completed numerical "
        "and angular audits. Preserve the existing solver, thresholds, historical "
        "results and every new attempt. Do not warm-start across targets or starts. "
        "Unresolved local searches do not establish infeasibility.")
    from invdes.utils import set_seed
    set_seed(0)
    experiment = MultistartExperiment(
        ROOT, evidence, deadline=deadline.timestamp(), device=args.device)
    print(json.dumps({"status": "launching", "pid": os.getpid(),
                      "output_directory": str(experiment.out),
                      "deadline_utc": deadline.isoformat(),
                      "requested_device": args.device}), flush=True)
    result = experiment.run()
    if result["status"] == "complete":
        from summarize_pwe_multistart_v1 import write_summary
        write_summary(ROOT)
    print(json.dumps({"status": result["status"],
                      "output_directory": str(experiment.out)}, indent=2))


if __name__ == "__main__":
    main()
