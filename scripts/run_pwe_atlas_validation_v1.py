"""Run map-derived M20/M25 PWE candidate validation."""
import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from invdes.inverse.atlas_validation_v1 import AtlasValidationExperiment
from invdes.utils import set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--acknowledge-exploratory-validation", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--until-utc")
    args = parser.parse_args()
    if not args.run:
        print("No physics launched. Use --run with explicit exploratory acknowledgement.")
        return
    if not args.acknowledge_exploratory_validation or not args.until_utc:
        parser.error("Launch requires acknowledgement and a bounded --until-utc")
    deadline = datetime.fromisoformat(args.until_utc.replace("Z", "+00:00"))
    if deadline.tzinfo is None or deadline.timestamp() <= time.time():
        parser.error("Deadline must be future and timezone-aware")
    set_seed(0)
    result = AtlasValidationExperiment(ROOT, deadline=deadline.timestamp(), device=args.device).run()
    print({"status": result["status"], "completed_cases": len(result["cases"])})


if __name__ == "__main__":
    main()
