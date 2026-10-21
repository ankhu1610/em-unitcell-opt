"""Fingerprinted no-projector PWE baseline; never writes historical outputs."""
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
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from audit_pure_pwe import eta_matrix, reciprocal_indices, reciprocal_basis, path_coordinates
from validate_energy_pwe_full import atomic_json
from invdes.pwe_sensitivity import eta_derivative
from invdes.variational_pwe import spectrum

OUT = ROOT/"results/variational_pwe_audit_v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(protected):
    changed = [p for p, expected in protected.items() if digest(p) != expected]
    if changed:
        raise ValueError(f"Protected files changed: {changed}")


def difference(a, b, floor):
    a, b = np.asarray(a), np.asarray(b)
    delta = np.abs(a-b)
    return {"max_absolute": float(delta.max()),
            "max_relative": float(np.max(delta/np.maximum(np.maximum(abs(a), abs(b)), floor)))}


def evaluate(config, index, cutoff, sides=None, gradients=True):
    a = config["a_um"]
    if sides is None:
        sides = np.array(config["side_multipliers"][index])*a/np.sqrt(3)
    ids = reciprocal_indices(cutoff)
    eta = eta_matrix(ids, a, sides, config["arc_order"])
    de = [eta_derivative(ids, a, sides, axis, config["arc_order"])
          for axis in (0, 1)] if gradients else []
    f, df, diagnostics = spectrum(ids@reciprocal_basis(a*1e-6),
                                  path_coordinates(config["path_parameters"], a*1e-6), eta, de)
    edges = [int(np.argmax(f[:, 0])), int(np.argmin(f[:, 1]))]
    return {"design_index": index, "sides_um": sides.tolist(), "cutoff": cutoff,
            "path_parameters": config["path_parameters"], "frequencies_GHz": f.tolist(),
            "analytic_gradients_GHz_per_um": df.tolist(), "diagnostics": diagnostics,
            "gap_proxy_GHz": float(f[edges[1], 1]-f[edges[0], 0]), "edge_indices": edges,
            "gap_scope": "five-point proxy",
            "analytic_selected_edge_gap_gradient_GHz_per_um": [
                float(x[edges[1], 1]-x[edges[0], 0]) for x in df]}


def jobs(config):
    result = []
    for i in range(len(config["side_multipliers"])):
        for cutoff in config["cutoffs"]:
            result.append((f"base_d{i}_M{cutoff}", i, cutoff, None, True))
    i = config["finite_difference_design_index"]
    for cutoff in config["finite_difference_cutoffs"]:
        for axis in (0, 1):
            for step in config["finite_difference_steps_um"]:
                for sign in (-1, 1):
                    sides = np.array(config["side_multipliers"][i])*config["a_um"]/np.sqrt(3)
                    sides[axis] += sign*step
                    result.append((f"fd_d{i}_M{cutoff}_L{axis+1}_h{step}_s{sign}",
                                   i, cutoff, sides, False))
    return result


def write_report(config, run_id, protected, tolerance):
    cases = {}
    for path in sorted((OUT/"cases").glob("*.json")):
        saved = json.loads(path.read_text())
        if saved["run_id"] != run_id:
            raise ValueError("Mismatched checkpoint in report")
        cases[path.stem] = saved["result"]
    refinements, gradchecks = [], []
    floor = config["derivative_absolute_floor_ghz_per_um"]
    for i in range(len(config["side_multipliers"])):
        for left, right in zip(config["cutoffs"][:-1], config["cutoffs"][1:]):
            a, b = cases.get(f"base_d{i}_M{left}"), cases.get(f"base_d{i}_M{right}")
            if a is None or b is None:
                continue
            fa, fb = np.array(a["frequencies_GHz"]), np.array(b["frequencies_GHz"])
            relative_increase = (fb-fa)/fa
            refinements.append({
                "design_index": i, "cutoff_from": left, "cutoff_to": right,
                "frequency_difference": difference(fa, fb, 1e-30),
                "gap_proxy_change_GHz": b["gap_proxy_GHz"]-a["gap_proxy_GHz"],
                "gradient_difference": difference(a["analytic_gradients_GHz_per_um"],
                                                  b["analytic_gradients_GHz_per_um"], floor),
                "selected_edge_gap_gradient_difference": difference(
                    a["analytic_selected_edge_gap_gradient_GHz_per_um"],
                    b["analytic_selected_edge_gap_gradient_GHz_per_um"], floor),
                "edge_indices_unchanged": a["edge_indices"] == b["edge_indices"],
                "max_signed_frequency_increase_GHz": float((fb-fa).max()),
                "monotonicity_violations_above_roundoff_slack": int(np.sum(
                    relative_increase > config["monotonicity_roundoff_relative_slack"]))})
    i = config["finite_difference_design_index"]
    for cutoff in config["finite_difference_cutoffs"]:
        base = cases.get(f"base_d{i}_M{cutoff}")
        if base is None:
            continue
        for axis in (0, 1):
            for step in config["finite_difference_steps_um"]:
                minus = cases.get(f"fd_d{i}_M{cutoff}_L{axis+1}_h{step}_s-1")
                plus = cases.get(f"fd_d{i}_M{cutoff}_L{axis+1}_h{step}_s1")
                if minus is None or plus is None:
                    continue
                fd = (np.array(plus["frequencies_GHz"])-np.array(minus["frequencies_GHz"]))/(2*step)
                analytic = np.array(base["analytic_gradients_GHz_per_um"])[axis]
                separation = np.array([[d["relative_eigenvalue_separations"] for d in row["diagnostics"]]
                                       for row in (base, minus, plus)])
                mask = np.all(separation > config["near_degenerate_relative_separation"], axis=0)
                err = difference(analytic[mask], fd[mask], floor) if mask.any() else None
                gap_fd = (plus["gap_proxy_GHz"]-minus["gap_proxy_GHz"])/(2*step)
                gradchecks.append({
                    "design_index": i, "cutoff": cutoff, "parameter": f"L{axis+1}", "step_um": step,
                    "simple_base_and_endpoint_mask": mask.tolist(),
                    "analytic_GHz_per_um": analytic.tolist(), "finite_difference_GHz_per_um": fd.tolist(),
                    "error": err, "passes_existing_gradient_tolerance": err["max_relative"] <= tolerance if err else None,
                    "selected_edge_analytic_gap_gradient_GHz_per_um":
                        base["analytic_selected_edge_gap_gradient_GHz_per_um"][axis],
                    "finite_difference_gap_proxy_gradient_GHz_per_um": gap_fd,
                    "edge_indices_unchanged": base["edge_indices"] == minus["edge_indices"] == plus["edge_indices"]})
    verify(protected)
    payload = {"run_id": run_id, "config": config, "completed_jobs": len(cases),
               "existing_gradient_tolerance": tolerance, "cutoff_refinements": refinements,
               "gradient_checks": gradchecks, "protected_hashes_verified": True,
               "protected_file_count": len(protected), "ml_training": False,
               "comsol_spectra_used": False, "production_solver_replaced": False,
               "continuum_accuracy_certified": False, "updated_unix": time.time()}
    atomic_json(OUT/"results.json", payload)
    lines = ["# Independent variational Hz PWE audit", "", config["scope"], "",
             f"Completed checkpointed jobs: {len(cases)}. Protected files unchanged: {len(protected)}.",
             f"Unchanged gradient tolerance: {tolerance:g}.",
             "No projectors, ML, COMSOL spectra or fitting. Production files/results are unchanged.",
             "Nested-basis eigenvalue monotonicity applies to exact Galerkin assembly; numerical violations are recorded.",
             "Derivative agreement at finite M is not a continuum accuracy certificate.",
             "Base/end-point simplicity checks do not exclude intermediate crossings.",
             "Gap derivatives refer to selected active edges; unchanged indices alone do not prove unique extrema.", "",
             "|Design|M change|Frequency max relative change|Gap proxy change GHz|Gradient max relative change|Monotonicity violations|",
             "|---|---|---|---|---|---|"]
    for r in refinements:
        lines.append(f"|{r['design_index']}|{r['cutoff_from']} to {r['cutoff_to']}|"
                     f"{r['frequency_difference']['max_relative']:.6g}|{r['gap_proxy_change_GHz']:.6g}|"
                     f"{r['gradient_difference']['max_relative']:.6g}|"
                     f"{r['monotonicity_violations_above_roundoff_slack']}|")
    lines += ["", "|M|Parameter|FD step um|Max relative derivative error|Pass existing tolerance|",
              "|---|---|---|---|---|"]
    for r in gradchecks:
        error = r["error"]
        lines.append(f"|{r['cutoff']}|{r['parameter']}|{r['step_um']}|"
                     f"{error['max_relative'] if error else 'excluded'}|{r['passes_existing_gradient_tolerance']}|")
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--until-utc", help="Timezone-aware runtime deadline; not part of scientific fingerprint")
    args = parser.parse_args()
    deadline = float("inf")
    if args.until_utc:
        value = datetime.fromisoformat(args.until_utc.replace("Z", "+00:00"))
        if value.tzinfo is None:
            parser.error("--until-utc requires a timezone")
        deadline = value.timestamp()
    config_path = ROOT/"configs/variational_pwe_audit_v1.json"
    config = json.loads(config_path.read_text())
    frozen = json.loads((ROOT/"configs/energy_pwe_audit.json").read_text())
    for key in ("a_um", "eps_background", "eps_hole"):
        if config[key] != frozen[key]:
            raise ValueError(f"Frozen physics mismatch: {key}")
    main_manifest = ROOT/"results/projector_stabilization_v1/manifest.json"
    upstream = json.loads(main_manifest.read_text())
    # Hash only frozen manifest inputs, NEVER the actively changing main outputs.
    protected = dict(upstream["protected_hashes"])
    protected.update(upstream["code_hashes"])
    protected[str(main_manifest)] = digest(main_manifest)
    verify(protected)
    tolerance = float(re.search(r"pwe_gradcheck:.*max_rel_err:\s*([0-9.eE+-]+)",
                                (ROOT/"configs/criteria.yaml").read_text()).group(1))
    code = [Path(__file__), config_path, ROOT/"src/invdes/variational_pwe.py",
            ROOT/"tests/test_variational_pwe.py"]
    protocol = {"config": config, "gradient_tolerance": tolerance,
                "protected_hashes": protected, "code_hashes": {str(p): digest(p) for p in code},
                "runtime": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                            "backend": "scipy.linalg.eigh/evr", "dtype": "float64/complex128"}}
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT/"manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != protocol:
            raise ValueError("Audit inputs changed; do not overwrite previous experiment")
    else:
        atomic_json(manifest, protocol)
    run_id = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    lock = OUT/"run.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    started, completed, state = time.time(), 0, "running"
    planned = jobs(config)
    try:
        for name, i, cutoff, sides, gradients in planned:
            path = OUT/"cases"/f"{name}.json"
            if path.exists():
                if json.loads(path.read_text())["run_id"] != run_id:
                    raise ValueError(f"Checkpoint fingerprint mismatch: {name}")
                completed += 1
                continue
            if time.time() >= deadline:
                state = "deadline_checkpointed"
                break
            atomic_json(OUT/"status.json", {"status": "running", "pid": os.getpid(), "case": name,
                        "completed_jobs": completed, "total_jobs": len(planned),
                        "started_unix": started, "updated_unix": time.time()})
            print(name, flush=True)
            start = time.perf_counter()
            result = evaluate(config, i, cutoff, sides, gradients)
            result["elapsed_seconds"] = time.perf_counter()-start
            verify(protected)
            atomic_json(path, {"run_id": run_id, "result": result})
            completed += 1
            write_report(config, run_id, protected, tolerance)
        else:
            state = "complete"
        write_report(config, run_id, protected, tolerance)
        atomic_json(OUT/"status.json", {"status": state, "completed_jobs": completed,
                    "total_jobs": len(planned), "elapsed_seconds": time.time()-started,
                    "runtime_deadline_unix": deadline if np.isfinite(deadline) else None,
                    "updated_unix": time.time(), "production_solver_replaced": False})
    except BaseException as exc:
        atomic_json(OUT/"status.json", {"status": "failed", "error": repr(exc),
                    "completed_jobs": completed, "updated_unix": time.time()})
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
