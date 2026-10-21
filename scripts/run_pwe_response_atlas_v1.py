"""Build a bounded adaptive PWE response atlas and a separate COMSOL reference panel."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from invdes.inverse.response_atlas_v1 import ResponseAtlasExperiment
from invdes.utils import set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--acknowledge-exploratory-atlas", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--until-utc")
    parser.add_argument("--max-new-evaluations", type=int)
    parser.add_argument("--config", default="configs/pure_pwe_response_atlas_v1.json")
    args = parser.parse_args()
    config = json.loads((ROOT/args.config).read_text())
    if not args.run:
        print(json.dumps({
            "launch_authorized": False, "pilot_new_evaluations": config["pilot_new_evaluations"],
            "maximum_new_evaluations": config["maximum_new_evaluations"],
            "comsol_used_as_reference_only": config["comsol_used_as_reference_only"],
        }, indent=2))
        return
    if not args.acknowledge_exploratory_atlas:
        parser.error("Launching requires explicit acknowledgement of the exploratory atlas scope")
    if not args.until_utc:
        parser.error("A bounded --until-utc is required")
    deadline = datetime.fromisoformat(args.until_utc.replace("Z", "+00:00"))
    if deadline.tzinfo is None or deadline.timestamp() <= time.time():
        parser.error("Deadline must be a future timestamp with timezone")
    if deadline.timestamp()-time.time() > float(config["max_runtime_hours"])*3600:
        parser.error("Deadline exceeds the preregistered response-atlas runtime")
    budget = args.max_new_evaluations or int(config["pilot_new_evaluations"])
    set_seed(0)
    experiment = ResponseAtlasExperiment(ROOT, deadline=deadline.timestamp(), device=args.device,
                                         config_path=args.config)
    result = experiment.run(budget)
    print(json.dumps({"status": result["status"], "new_evaluations": len(result["new_evaluations"]),
                      "output_directory": str(experiment.out)}, indent=2))


if __name__ == "__main__":
    main()
