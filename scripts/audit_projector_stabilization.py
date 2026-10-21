"""Run isolated projector experiments, with protected historical checksums."""
import os
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "2"
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from validate_energy_pwe_full import atomic_json
from invdes.projector_experiment import Experiment, DeadlineReached, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["screen", "convergence", "all"], default="all")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--until-utc", help="Explicit new runtime deadline for an authorized resume; scientific fingerprint unchanged")
    args = parser.parse_args()
    config_path = ROOT/"configs/projector_stabilization_v1.json"
    config = json.loads(config_path.read_text())
    criteria = (ROOT/"configs/criteria.yaml").read_text()
    gradient_tolerance = float(re.search(r"pwe_gradcheck:.*max_rel_err:\s*([0-9.eE+-]+)", criteria).group(1))
    convergence_tolerance = float(re.search(r"pwe_convergence:.*max_rel_diff_at_chosen_M:\s*([0-9.eE+-]+)", criteria).group(1))
    reference_m = int(re.search(r"pwe_convergence:.*ref_M:\s*(\d+)", criteria).group(1))
    frozen = json.loads((ROOT/"configs/energy_pwe_audit.json").read_text())
    if config["a_um"] != frozen["a_um"] or (frozen["eps_background"], frozen["eps_hole"]) != (9, 1):
        raise ValueError("Frozen physics mismatch")
    folders = ["energy_pwe_full_validation", "energy_pwe_audit", "pure_pwe_inverse",
               "pure_pwe_inverse_lodo", "pwe_sensitivity_audit"]
    files = [ROOT/f for f in [
        "src/invdes/interface_pwe.py", "src/invdes/pwe_sensitivity.py",
        "scripts/audit_pure_pwe.py", "scripts/validate_energy_pwe_full.py",
        "scripts/benchmark_pwe_cuda.py", "scripts/audit_pwe_sensitivity.py",
        "scripts/inverse_pwe.py", "src/invdes/inverse/pure_pwe.py",
        "configs/criteria.yaml", "configs/geometry_spec.yaml", "configs/energy_pwe_audit.json",
        "configs/pure_pwe_inverse.json", "configs/pwe_sensitivity_audit.json"]]
    for folder in folders:
        files.extend(p for p in (ROOT/"results"/folder).rglob("*") if p.is_file())
    protected = {str(p): digest(p) for p in files}
    code = [Path(__file__), config_path, ROOT/"src/invdes/stable_projector.py",
            ROOT/"src/invdes/homothetic_projector.py", ROOT/"src/invdes/projector_experiment.py"]
    protocol = {"config": config, "gradient_tolerance": gradient_tolerance,
                "convergence_tolerance": convergence_tolerance, "reference_m": reference_m,
                "code_hashes": {str(p): digest(p) for p in code}, "protected_hashes": protected}
    out = ROOT/"results/projector_stabilization_v1"
    out.mkdir(parents=True, exist_ok=True)
    manifest = out/"manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != protocol:
            raise ValueError("Experiment changed; retain results and use a separate version")
    else:
        atomic_json(manifest, protocol)
    run_id = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    experiment = Experiment(ROOT, config, run_id, protected, args.device)
    if args.until_utc:
        deadline = datetime.fromisoformat(args.until_utc.replace("Z", "+00:00"))
        if deadline.tzinfo is None:
            parser.error("--until-utc requires timezone information")
        experiment.deadline = deadline.timestamp()
    jobs = experiment.jobs(gradient_tolerance, args.stage)
    experiment.total = len(jobs)
    lock = out/"run.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    try:
        for name, compute in jobs:
            experiment.get_or_compute(name, compute)
            experiment.report(gradient_tolerance, convergence_tolerance, reference_m)
        state = "complete"
    except DeadlineReached as exc:
        state = "deadline_checkpointed"
        print(str(exc), flush=True)
    except BaseException as exc:
        atomic_json(out/"status.json", {"status": "failed", "error": repr(exc),
                    "case": experiment.current, "completed_jobs": experiment.completed,
                    "updated_unix": time.time()})
        raise
    finally:
        lock.unlink()
    experiment.report(gradient_tolerance, convergence_tolerance, reference_m)
    atomic_json(out/"status.json", {
        "status": state, "stage": args.stage, "completed_jobs": experiment.completed,
        "total_jobs": len(jobs), "elapsed_seconds": time.time()-experiment.started,
        "runtime_deadline_unix": experiment.deadline,
        "updated_unix": time.time(), "production_solver_replaced": False})


if __name__ == "__main__":
    main()
