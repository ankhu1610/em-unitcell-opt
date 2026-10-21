"""Independent sensitivity audit. Never writes validated solver/results.

Analytic material derivative + exact energy chain on a fixed projector branch.
Full-rebuild central differences test whether this is useful beyond that branch.
Projector-secanted chain is diagnostic only, not independent analytic verification.
"""
import os
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "2"
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
from scipy.linalg import eigh
from validate_energy_pwe_full import ROOT, atomic_json
from audit_pure_pwe import reciprocal_indices, reciprocal_basis, path_coordinates, eta_matrix
from invdes.interface_pwe import (
    C, normal_projector_grid, projector_matrices, energy_factorized_impermittivity,
    hz_operator, _require_hermitian)
from invdes.pwe_sensitivity import eta_derivative, energy_eigenvalue_derivative, frequency_derivative

OUT = ROOT/"results/pwe_sensitivity_audit"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def modes(g, kpoints, tensor):
    frequencies, eigenpairs, residuals, separations = [], [], [], []
    for k in kpoints:
        operator = hz_operator(g, k, tensor)
        _require_hermitian(operator, "audit Hz")
        values, vectors = eigh(operator, subset_by_index=[0, 2], driver="evr", check_finite=False)
        if values[0] <= 0:
            raise ValueError("Nonpositive eigenvalue away from Gamma")
        residual = np.max(np.linalg.norm(operator@vectors-vectors*values, axis=0)/np.abs(values))
        if residual > 1e-7:
            raise ValueError(f"Eigenpair residual failed: {residual}")
        frequencies.append(C*np.sqrt(values[:2])/(2*np.pi*1e9))
        eigenpairs.append((values[:2], vectors[:, :2]))
        residuals.append(float(residual))
        differences = np.diff(values)
        separations.append([differences[0]/values[0], min(differences)/values[1]])
    return np.array(frequencies), eigenpairs, residuals, np.array(separations)


def gap(frequencies):
    return float(frequencies[:, 1].min()-frequencies[:, 0].max())


def error(estimate, reference, floor):
    difference = np.abs(estimate-reference)
    relative = difference/np.maximum(np.maximum(np.abs(estimate), np.abs(reference)), floor)
    return {"max_absolute": float(difference.max()), "max_relative": float(relative.max())}


def audit_case(config, multiplier, cutoff, grid, tolerance, progress):
    a = config["a_um"]
    sides = np.array(multiplier)*a/np.sqrt(3)
    ids = reciprocal_indices(cutoff)
    eye = np.eye(len(ids))
    g = ids@reciprocal_basis(a*1e-6)
    ks = path_coordinates(config["path_parameters"], a*1e-6)
    eta = eta_matrix(ids, a, sides, config["arc_order"])
    samples = normal_projector_grid(a, sides, grid)
    p = projector_matrices(ids, samples)
    b = np.linalg.solve(10*eye-9*eta, eye)
    base_tensor = energy_factorized_impermittivity(eta, 10*eye-9*eta, p)
    f0, eigenpairs, residuals, separation = modes(g, ks, base_tensor)
    del base_tensor
    smooth_modes = separation > config["near_degenerate_relative_separation"]
    base_edges = [int(np.argmax(f0[:, 0])), int(np.argmin(f0[:, 1]))]
    floor = config["derivative_absolute_floor_ghz_per_um"]
    result = {"sides_um": sides.tolist(), "cutoff": cutoff, "normal_grid": grid,
              "path_parameters": config["path_parameters"], "base_GHz": f0.tolist(),
              "base_gap_proxy_GHz": gap(f0), "base_edge_indices": base_edges,
              "base_relative_residuals": residuals, "relative_eigenvalue_separations": separation.tolist(),
              "simple_mode_mask": smooth_modes.tolist(), "parameters": []}
    for axis in (0, 1):
        de = eta_derivative(ids, a, sides, axis, config["arc_order"])
        branch = np.array([frequency_derivative(values, energy_eigenvalue_derivative(
            g, k, vectors, b, eta-b, p, de)) for k, (values, vectors) in zip(ks, eigenpairs)])
        branch_gap = float(branch[base_edges[1], 1]-branch[base_edges[0], 0])
        rows = []
        for h in config["steps_um"]:
            progress(f"L{axis+1}, step={h} um")
            perturbed = []
            # Frozen-projector control at all screen steps and smallest M20 step.
            control = cutoff == config["screen_cutoff"] or h == config["steps_um"][-1]
            for sign in (-1, 1):
                shifted = sides.copy()
                shifted[axis] += sign*h
                e = eta_matrix(ids, a, shifted, config["arc_order"])
                s = normal_projector_grid(a, shifted, grid)
                pp = projector_matrices(ids, s)
                t = energy_factorized_impermittivity(e, 10*eye-9*e, pp)
                ff, _, res, _ = modes(g, ks, t)
                del t
                frozen = None
                if control:
                    t = energy_factorized_impermittivity(e, 10*eye-9*e, p)
                    frozen = modes(g, ks, t)[0]
                    del t
                perturbed.append((e, s, pp, ff, frozen, res))
            minus, plus = perturbed
            material_fd = (plus[0]-minus[0])/(2*h)
            material_error = float(np.linalg.norm(de-material_fd)/max(np.linalg.norm(de), 1e-30))
            dp = tuple((xp-xm)/(2*h) for xp, xm in zip(plus[2], minus[2]))
            secant_chain = np.array([frequency_derivative(values, energy_eigenvalue_derivative(
                g, k, vectors, b, eta-b, p, de, dp))
                for k, (values, vectors) in zip(ks, eigenpairs)])
            direct_fd = (plus[3]-minus[3])/(2*h)
            changed = ((np.abs(plus[1][0]-minus[1][0]) > 1e-12)
                       | (np.abs(plus[1][1]-minus[1][1]) > 1e-12))
            branch_error = error(branch[smooth_modes], direct_fd[smooth_modes], floor) if smooth_modes.any() else None
            row = {"step_um": h, "material_relative_error": material_error,
                   "projector_changed_grid_points": int(changed.sum()),
                   "projector_secant_norm": float(np.sqrt(sum(np.linalg.norm(d)**2 for d in dp))),
                   "full_rebuild_fd_GHz_per_um": direct_fd.tolist(),
                   "projector_secant_chain_GHz_per_um": secant_chain.tolist(),
                   "branch_vs_full_fd": branch_error,
                   "branch_passes_simple_modes": branch_error["max_relative"] <= tolerance if branch_error else None,
                   "secant_chain_vs_full_fd": error(secant_chain, direct_fd, floor),
                   "gap_proxy_fd_GHz_per_um": (gap(plus[3])-gap(minus[3]))/(2*h),
                   "minus_edge_indices": [int(np.argmax(minus[3][:, 0])), int(np.argmin(minus[3][:, 1]))],
                   "plus_edge_indices": [int(np.argmax(plus[3][:, 0])), int(np.argmin(plus[3][:, 1]))],
                   "perturbed_max_eigenpair_residual": max(*plus[5], *minus[5])}
            if control:
                frozen_fd = (plus[4]-minus[4])/(2*h)
                row["frozen_projector_fd_GHz_per_um"] = frozen_fd.tolist()
                row["branch_vs_frozen_fd"] = error(branch, frozen_fd, floor)
            rows.append(row)
            print(f"  material rel={material_error:.3g}; projector switches={changed.sum()}; "
                  f"branch/full={branch_error}", flush=True)
        result["parameters"].append({"parameter": f"L{axis+1}",
                                     "analytic_branch_GHz_per_um": branch.tolist(),
                                     "analytic_branch_gap_proxy_GHz_per_um": branch_gap,
                                     "steps": rows})
    return result


def report(results, config, tolerance, preserved):
    summary = []
    for r in results:
        for p in r["parameters"]:
            smallest = p["steps"][-1]
            fd = np.array([s["full_rebuild_fd_GHz_per_um"] for s in p["steps"]])
            summary.append({"sides_um": r["sides_um"], "cutoff": r["cutoff"],
                            "normal_grid": r["normal_grid"], "parameter": p["parameter"],
                            "smallest_step_material_relative_error": smallest["material_relative_error"],
                            "smallest_step_frozen_relative_error": smallest["branch_vs_frozen_fd"]["max_relative"],
                            "smallest_step_full_relative_error": smallest["branch_vs_full_fd"]["max_relative"] if smallest["branch_vs_full_fd"] else None,
                            "smallest_step_projector_switches": smallest["projector_changed_grid_points"],
                            "step_sweep_max_derivative_spread_GHz_per_um": float(np.max(np.ptp(fd, axis=0))),
                            "full_fd_agrees_with_branch_at_every_step": all(s["branch_passes_simple_modes"] is True for s in p["steps"])})
    # Do not equate an infinitesimal fixed-grid branch match with physical convergence.
    payload = {"config": config, "existing_gradient_relative_tolerance": tolerance,
               "preserved_hashes_verified": preserved, "summary": summary, "cases": results,
               "ml_training": False, "comsol_spectra_used": False,
               "optimizer_ready_certified": False}
    atomic_json(OUT/"results.json", payload)
    lines = ["# Pure-PWE L1/L2 sensitivity audit", "",
             config["scope"], "",
             "Analytic material shape derivative uses boundary transport, not finite differences.",
             "Branch derivative differentiates the complete energy operator with fixed normal assignments.",
             "Full-rebuild differences recompute material AND normal projectors at perturbed lengths.",
             "Projector-secanted chain is a finite-step diagnostic, NOT an independently analytic projector derivative.",
             f"Unchanged criteria.yaml relative gradient threshold: {tolerance:g}.",
             f"Validated solver/results hashes preserved: {preserved}.",
             "No ML, COMSOL spectral data or fitting. No inverse optimizer enabled.", "",
             "|L1 um|L2 um|M|Grid|Derivative|Material rel error*|Frozen rel error*|Full rel error*|Switches*|FD step spread GHz/um|",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        full = s["smallest_step_full_relative_error"]
        lines.append(f"|{s['sides_um'][0]:.5f}|{s['sides_um'][1]:.5f}|{s['cutoff']}|{s['normal_grid']}|"
                     f"{s['parameter']}|{s['smallest_step_material_relative_error']:.3g}|"
                     f"{s['smallest_step_frozen_relative_error']:.3g}|"
                     f"{full if full is None else format(full, '.3g')}|"
                     f"{s['smallest_step_projector_switches']}|{s['step_sweep_max_derivative_spread_GHz_per_um']:.6g}|")
    lines += ["", "*At the smallest perturbation. Full relative error excludes near-degenerate base modes.",
              "Other rows, both signs, all mode sensitivities, eigenpair residuals and band-edge locations are in results.json.",
              "Finite-grid derivatives can match at tiny steps yet fail to converge under grid refinement.",
              "Gap values here are five-point proxies, not the full 281-point validated band gap.",
              "Analytic branches are not globally smooth at nearest-interface assignment switches."]
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-only", action="store_true")
    args = parser.parse_args()
    config_path = ROOT/"configs/pwe_sensitivity_audit.json"
    config = json.loads(config_path.read_text())
    tolerance = float(re.search(r"pwe_gradcheck:.*max_rel_err:\s*([0-9.eE+-]+)",
                                (ROOT/"configs/criteria.yaml").read_text()).group(1))
    frozen = json.loads((ROOT/"configs/energy_pwe_audit.json").read_text())
    if config["a_um"] != frozen["a_um"] or (frozen["eps_background"], frozen["eps_hole"]) != (9, 1):
        raise ValueError("Frozen geometry/material mismatch")
    protected = [ROOT/"src/invdes/interface_pwe.py", ROOT/"scripts/audit_pure_pwe.py",
                 ROOT/"scripts/validate_energy_pwe_full.py", ROOT/"scripts/benchmark_pwe_cuda.py",
                 ROOT/"configs/criteria.yaml", ROOT/"configs/geometry_spec.yaml"]
    for folder in ["energy_pwe_full_validation", "pure_pwe_inverse", "pure_pwe_inverse_lodo"]:
        protected.extend(p for p in (ROOT/"results"/folder).rglob("*") if p.is_file())
    hashes = {str(p): digest(p) for p in protected}
    protocol = {"config": config, "gradient_tolerance": tolerance,
                "code_hashes": {str(p): digest(p) for p in [
                    Path(__file__), ROOT/"src/invdes/pwe_sensitivity.py", config_path]},
                "protected_hashes": hashes}
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/"manifest.json").exists():
        if json.loads((OUT/"manifest.json").read_text()) != protocol:
            raise ValueError("Audit inputs changed; do not overwrite previous experiment")
    else:
        atomic_json(OUT/"manifest.json", protocol)
    run_id = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    cases = [(i, config["screen_cutoff"], grid) for i in range(len(config["side_multipliers"]))
             for grid in config["normal_grids"]]
    if not args.screen_only:
        cases += [(config["reference_design_index"], config["reference_cutoff"], grid)
                  for grid in config["normal_grids"]]
    with (OUT/"run.lock").open("x") as stream:
        stream.write(str(os.getpid()))
    results = []
    start = time.time()
    try:
        for i, cutoff, grid in cases:
            name = f"design{i}_M{cutoff}_grid{grid}"
            path = OUT/"cases"/f"{name}.json"
            def progress(stage):
                atomic_json(OUT/"status.json", {"status": "running", "pid": os.getpid(),
                            "case": name, "stage": stage, "completed_cases": len(results),
                            "total_cases": len(cases), "started_unix": start, "updated_unix": time.time()})
                print(f"{name}: {stage}", flush=True)
            progress("baseline")
            if path.exists():
                saved = json.loads(path.read_text())
                if saved["run_id"] != run_id:
                    raise ValueError("Audit checkpoint mismatch")
                result = saved["result"]
            else:
                result = audit_case(config, config["side_multipliers"][i], cutoff, grid, tolerance, progress)
                atomic_json(path, {"run_id": run_id, "result": result})
            results.append(result)
            preserved = all(digest(Path(p)) == h for p, h in hashes.items())
            if not preserved:
                raise ValueError("A protected input/result file changed")
            report(results, config, tolerance, preserved)
        atomic_json(OUT/"status.json", {"status": "complete", "completed_cases": len(results),
                    "total_cases": len(cases), "elapsed_seconds": time.time()-start,
                    "updated_unix": time.time(), "optimizer_ready_certified": False})
    except BaseException as exc:
        atomic_json(OUT/"status.json", {"status": "failed", "error": repr(exc),
                    "completed_cases": len(results), "updated_unix": time.time()})
        raise
    finally:
        (OUT/"run.lock").unlink()


if __name__ == "__main__":
    main()
