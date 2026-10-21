"""Checkpointed research harness for separate geometry-only PWE projectors.

Imports existing pure-physics helpers without running their data/report entry
points. Callers add the repository scripts directory to sys.path. No labels,
optimizer, training, or fitting are used here.
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from audit_pure_pwe import reciprocal_indices, reciprocal_basis, path_coordinates, eta_matrix
from audit_pwe_sensitivity import modes, error, gap
from validate_energy_pwe_full import atomic_json, lowest_two
from invdes.interface_pwe import (
    projector_matrices, energy_factorized_impermittivity, normal_projector_grid as legacy_grid)
from invdes.pwe_sensitivity import (
    eta_derivative, energy_eigenvalue_derivative, frequency_derivative)


class DeadlineReached(Exception):
    """A resumable checkpoint boundary reached the research time budget."""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def samples_for(method, a, sides, grid, tie_ulps):
    if method == "stable_nearest":
        from invdes.stable_projector import normal_projector_grid
    elif method == "homothetic_radial":
        from invdes.homothetic_projector import normal_projector_grid
    else:
        raise ValueError(f"Unknown experimental method: {method}")
    return normal_projector_grid(a, sides, grid, tie_ulps=tie_ulps)


def summarize_pairs(rows, key, values, floor):
    """Compare consecutive refinements only when all other parameters match."""
    result = []
    for left, right in zip(values[:-1], values[1:]):
        a, b = rows.get(left), rows.get(right)
        if a is None or b is None:
            continue
        result.append({f"{key}_from": left, f"{key}_to": right,
                       "frequency": error(np.array(a["frequencies_GHz"]),
                                          np.array(b["frequencies_GHz"]), floor),
                       "gap_difference_GHz": b["gap_GHz"]-a["gap_GHz"]})
    return result


class Experiment:
    def __init__(self, root, config, run_id, protected, device="cpu"):
        self.root, self.config, self.run_id = Path(root), config, run_id
        self.out = self.root/"results/projector_stabilization_v1"
        self.protected, self.device = protected, device
        self.deadline = datetime.fromisoformat(config["deadline_utc"]).timestamp()
        self.started = time.time()
        self.completed = 0
        self.total = 0
        self.current = None

    def verify_preserved(self):
        changed = [p for p, h in self.protected.items() if digest(p) != h]
        if changed:
            raise ValueError(f"Protected files changed: {changed}")

    def progress(self, stage):
        atomic_json(self.out/"status.json", {
            "status": "running", "case": self.current, "stage": stage,
            "completed_jobs": self.completed, "total_jobs": self.total,
            "started_unix": self.started, "updated_unix": time.time(),
            "pid": os.getpid(),
            "runtime_deadline_unix": self.deadline,
            "device_for_full_path": self.device,
        })
        print(f"{self.current}: {stage}", flush=True)

    def check_deadline(self):
        if time.time() >= self.deadline:
            raise DeadlineReached("Bounded research deadline reached; completed checkpoints retained")

    def get_or_compute(self, name, compute):
        path = self.out/"cases"/f"{name}.json"
        self.current = name
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["run_id"] != self.run_id:
                raise ValueError(f"Checkpoint fingerprint mismatch: {name}")
            self.completed += 1
            return saved["result"]
        self.check_deadline()
        self.progress("starting")
        start = time.perf_counter()
        result = compute()
        result["elapsed_seconds"] = time.perf_counter()-start
        result["execution"] = {"numpy_version": np.__version__,
                               "full_path_device": self.device,
                               "physics_dtype": "float64/complex128",
                               "sensitivity_device": "cpu"}
        self.verify_preserved()
        atomic_json(path, {"run_id": self.run_id, "result": result})
        self.completed += 1
        return result

    def geometry(self, index, cutoff, grid, method, sides=None):
        c = self.config
        a = c["a_um"]
        if sides is None:
            sides = np.array(c["side_multipliers"][index])*a/np.sqrt(3)
        ids = reciprocal_indices(cutoff)
        eta = eta_matrix(ids, a, sides, c["arc_order"])
        samples = samples_for(method, a, sides, grid, c["tie_ulps"])
        p = projector_matrices(ids, samples)
        identity = np.eye(len(ids))
        epsilon = 10*identity-9*eta
        tensor = energy_factorized_impermittivity(eta, epsilon, p)
        g = ids@reciprocal_basis(a*1e-6)
        return sides, ids, eta, p, tensor, g, samples

    def gradient(self, ids, sides, g, ks, eigenpairs, eta, p):
        identity = np.eye(len(ids))
        b = np.linalg.solve(10*identity-9*eta, identity)
        gradients = []
        for axis in (0, 1):
            de = eta_derivative(ids, self.config["a_um"], sides, axis, self.config["arc_order"])
            gradients.append(np.array([
                frequency_derivative(values, energy_eigenvalue_derivative(
                    g, k, vectors, b, eta-b, p, de))
                for k, (values, vectors) in zip(ks, eigenpairs)]))
        return gradients

    def sensitivity(self, method, index, cutoff, grid, tolerance):
        c = self.config
        sides, ids, eta, p, tensor, g, samples = self.geometry(index, cutoff, grid, method)
        ks = path_coordinates(c["path_parameters"], c["a_um"]*1e-6)
        f0, eigenpairs, residuals, separation = modes(g, ks, tensor)
        del tensor
        gradients = self.gradient(ids, sides, g, ks, eigenpairs, eta, p)
        identity = np.eye(len(ids))
        b = np.linalg.solve(10*identity-9*eta, identity)
        edges = [int(np.argmax(f0[:, 0])), int(np.argmin(f0[:, 1]))]
        params = []
        for axis in (0, 1):
            de = eta_derivative(ids, c["a_um"], sides, axis, c["arc_order"])
            rows = []
            for h in c["steps_um"]:
                self.check_deadline()
                self.progress(f"L{axis+1} finite difference {h} um")
                ff, pp, ss, ee, seps, rr = [], [], [], [], [], []
                for sign in (-1, 1):
                    shifted = sides.copy()
                    shifted[axis] += sign*h
                    e = eta_matrix(ids, c["a_um"], shifted, c["arc_order"])
                    # Rebuild even for radial projector to independently test
                    # the claimed length independence, not freeze by assumption.
                    s = samples_for(method, c["a_um"], shifted, grid, c["tie_ulps"])
                    proj = projector_matrices(ids, s)
                    t = energy_factorized_impermittivity(e, 10*identity-9*e, proj)
                    f, _, res, sep = modes(g, ks, t)
                    ff.append(f)
                    pp.append(proj)
                    ss.append(s)
                    ee.append(e)
                    seps.append(sep)
                    rr.extend(res)
                    del t
                fd = (ff[1]-ff[0])/(2*h)
                mask = ((separation > c["near_degenerate_relative_separation"])
                        & (seps[0] > c["near_degenerate_relative_separation"])
                        & (seps[1] > c["near_degenerate_relative_separation"]))
                discrepancy = error(gradients[axis][mask], fd[mask],
                                    c["derivative_absolute_floor_ghz_per_um"]) if mask.any() else None
                dp = tuple((right-left)/(2*h) for left, right in zip(pp[0], pp[1]))
                secant_chain = np.array([
                    frequency_derivative(values, energy_eigenvalue_derivative(
                        g, k, vectors, b, eta-b, p, de, dp))
                    for k, (values, vectors) in zip(ks, eigenpairs)])
                changed = (np.abs(ss[1][0]-ss[0][0]) > 1e-12) | (np.abs(ss[1][1]-ss[0][1]) > 1e-12)
                rows.append({
                    "step_um": h,
                    "analytic_vs_full_fd": discrepancy,
                    "passes_simple_sampled_modes": discrepancy["max_relative"] <= tolerance if discrepancy else None,
                    "simple_at_base_and_endpoints_mask": mask.tolist(),
                    "full_rebuild_fd_GHz_per_um": fd.tolist(),
                    "secant_chain_vs_full_fd": error(secant_chain, fd, c["derivative_absolute_floor_ghz_per_um"]),
                    "projector_changed_grid_points": int(changed.sum()),
                    "projector_secant_norm": float(np.sqrt(sum(np.linalg.norm(x)**2 for x in dp))),
                    "material_relative_error": float(np.linalg.norm(de-(ee[1]-ee[0])/(2*h))/np.linalg.norm(de)),
                    "gap_proxy_fd_GHz_per_um": (gap(ff[1])-gap(ff[0]))/(2*h),
                    "minus_edge_indices": [int(np.argmax(ff[0][:, 0])), int(np.argmin(ff[0][:, 1]))],
                    "plus_edge_indices": [int(np.argmax(ff[1][:, 0])), int(np.argmin(ff[1][:, 1]))],
                    "max_perturbed_eigenpair_residual": max(rr),
                })
                print(f"  changed={changed.sum()}, gradient_error={discrepancy}", flush=True)
            params.append({
                "parameter": f"L{axis+1}", "analytic_GHz_per_um": gradients[axis].tolist(),
                "analytic_gap_proxy_GHz_per_um": float(gradients[axis][edges[1], 1]-gradients[axis][edges[0], 0]),
                "steps": rows,
            })
        return {"kind": "sensitivity", "method": method, "design_index": index,
                "sides_um": sides.tolist(), "cutoff": cutoff, "grid": grid,
                "frequencies_GHz": f0.tolist(), "gap_GHz": gap(f0),
                "gap_scope": "five-point proxy", "base_edge_indices": edges,
                "base_max_eigenpair_residual": max(residuals),
                "parameters": params,
                "derivative_interpretation": (
                    "Analytic full length derivative of this fixed spatial discretization."
                    if method == "homothetic_radial" else
                    "Fixed nearest-assignment branch derivative; moving partitions not differentiated."),
                "continuum_gradient_certified": False}

    def spectrum(self, method, index, cutoff, grid, full=False, sides=None, checkpoint_tag=None):
        c = self.config
        sides, ids, eta, p, tensor, g, _ = self.geometry(index, cutoff, grid, method, sides)
        parameters = ((c["full_path_start"]+c["full_path_step"]*np.arange(c["full_path_count"])).tolist()
                      if full else c["path_parameters"])
        ks = path_coordinates(parameters, c["a_um"]*1e-6)
        if not full:
            f, pairs, residuals, separations = modes(g, ks, tensor)
            gradients = self.gradient(ids, sides, g, ks, pairs, eta, p)
            # The legacy predictor is only evaluated at matching test points;
            # no saved COMSOL spectra enter this comparison.
            old = projector_matrices(ids, legacy_grid(c["a_um"], sides, grid))
            old_t = energy_factorized_impermittivity(eta, 10*np.eye(len(ids))-9*eta, old)
            legacy_f = modes(g, ks, old_t)[0]
            checks = {"legacy_pwe_frequency_difference": error(f, legacy_f, 1e-30),
                      "legacy_pwe_gap_proxy_difference_GHz": gap(f)-gap(legacy_f)}
        else:
            partial_path = self.out/"partial"/f"{checkpoint_tag or self.current}.json"
            frequencies, diagnostics = [], []
            if partial_path.exists():
                saved = json.loads(partial_path.read_text())
                if (saved["run_id"] != self.run_id or saved["sides_um"] != sides.tolist()
                        or saved["method"] != method or saved["cutoff"] != cutoff
                        or saved["grid"] != grid):
                    raise ValueError("Full-path partial checkpoint mismatch")
                frequencies, diagnostics = saved["frequencies_GHz"], saved["diagnostics"]
                if (len(frequencies) != len(diagnostics) or len(frequencies) > len(ks)
                        or (frequencies and np.asarray(frequencies).shape != (len(frequencies), 2))
                        or not np.all(np.isfinite(frequencies))):
                    raise ValueError("Invalid full-path partial checkpoint contents")
            cuda = None
            if self.device == "cuda":
                from benchmark_pwe_cuda import CudaSolver
                import torch
                torch.set_num_threads(2)
                torch.set_grad_enabled(False)
                cuda = CudaSolver(g, tensor)
            for ik in range(len(frequencies), len(ks)):
                self.check_deadline()
                frow, diag = (cuda.solve(ks[ik]) if cuda is not None else lowest_two(g, ks[ik], tensor))
                frequencies.append(frow.tolist())
                diagnostics.append(diag)
                atomic_json(partial_path, {
                    "run_id": self.run_id, "sides_um": sides.tolist(),
                    "method": method, "cutoff": cutoff, "grid": grid,
                    "frequencies_GHz": frequencies, "diagnostics": diagnostics})
                if (ik+1) % 10 == 0 or ik+1 == len(ks):
                    self.progress(f"full path {ik+1}/{len(ks)}")
            f = np.array(frequencies)
            # Independent CPU/GPU check on the new tensor, not only old benchmark.
            cpu_difference = 0.
            for ik in (0, len(ks)//2, len(ks)-1):
                cpu_f, _ = lowest_two(g, ks[ik], tensor)
                cpu_difference = max(cpu_difference, float(np.max(abs(cpu_f-f[ik]))))
            if cpu_difference > 1e-5:
                raise ValueError(f"CPU/full-path device disagreement {cpu_difference} GHz")
            edge_ids = [int(np.argmax(f[:, 0])), int(np.argmin(f[:, 1]))]
            _, pairs, _, separations = modes(g, ks[edge_ids], tensor)
            gradients = self.gradient(ids, sides, g, ks[edge_ids], pairs, eta, p)
            residuals = [d["relative_eigenpair_residual"] for d in diagnostics]
            checks = {"cpu_device_max_difference_GHz": cpu_difference,
                      "edge_indices": edge_ids,
                      "edge_uniqueness_margins_GHz": [
                          float(np.sort(f[:, 0])[-1]-np.sort(f[:, 0])[-2]),
                          float(np.sort(f[:, 1])[1]-np.sort(f[:, 1])[0])],
                      "edge_path_parameters": [parameters[j] for j in edge_ids],
                      "edge_mode_relative_separations": separations.tolist(),
                      "gap_gradient_interpretation": "Derivative of selected active edges; inspect uniqueness margins and mode separations before treating as gap derivative.",
                      "analytic_edge_gap_gradient_GHz_per_um": [
                          float(x[1, 1]-x[0, 0]) for x in gradients]}
            if cuda is not None:
                del cuda
                torch.cuda.empty_cache()
        return {"kind": "full_spectrum" if full else "convergence_spectrum",
                "method": method, "design_index": index, "sides_um": sides.tolist(),
                "cutoff": cutoff, "grid": grid, "path_parameters": parameters,
                "frequencies_GHz": f.tolist(), "gap_GHz": gap(f),
                "gap_scope": "281-point path" if full else "five-point proxy",
                "max_eigenpair_residual": max(residuals),
                "analytic_gradients_GHz_per_um": [v.tolist() for v in gradients],
                **checks}

    def jobs(self, tolerance, stage):
        c = self.config
        result = []
        for method in c["methods"]:
            settings = [(i, c["screen_cutoff"], grid) for i in range(len(c["side_multipliers"]))
                        for grid in c["normal_grids"]]
            if stage != "screen":
                settings += [(c["reference_design_index"], c["reference_cutoff"], grid)
                             for grid in c["normal_grids"]]
            for i, cutoff, grid in settings:
                name = f"sens_{method}_d{i}_M{cutoff}_g{grid}"
                result.append((name, lambda m=method, j=i, n=cutoff, r=grid:
                               self.sensitivity(m, j, n, r, tolerance)))
        if stage == "screen":
            return result
        for method in c["methods"]:
            for i in range(len(c["side_multipliers"])):
                settings = [(cutoff, grid) for cutoff in c["convergence_cutoffs"]
                            for grid in c["normal_grids"]]
                settings += [(c["extra_grid_cutoff"], c["extra_normal_grid"])]
                for cutoff, grid in settings:
                    name = f"conv_{method}_d{i}_M{cutoff}_g{grid}"
                    result.append((name, lambda m=method, j=i, n=cutoff, r=grid:
                                   self.spectrum(m, j, n, r)))
        if stage == "all":
            for method in c["methods"]:
                for i in range(len(c["side_multipliers"])):
                    for cutoff in c["full_path_cutoffs"]:
                        grid = c["full_path_grid"]
                        name = f"full_{method}_d{i}_M{cutoff}_g{grid}"
                        result.append((name, lambda m=method, j=i, n=cutoff, r=grid:
                                       self.spectrum(m, j, n, r, full=True)))
            method, i, cutoff, grid = (c["full_path_sensitivity_method"],
                                      c["full_path_sensitivity_design_index"],
                                      c["full_path_sensitivity_cutoff"], c["full_path_grid"])
            for axis in (0, 1):
                for h in c["full_path_sensitivity_steps_um"]:
                    for sign in (-1, 1):
                        sides = np.array(c["side_multipliers"][i])*c["a_um"]/np.sqrt(3)
                        sides[axis] += sign*h
                        name = f"full_fd_{method}_d{i}_M{cutoff}_g{grid}_L{axis+1}_h{h}_s{sign}"
                        result.append((name, lambda m=method, j=i, n=cutoff, r=grid, s=sides, tag=name:
                                       self.spectrum(m, j, n, r, full=True, sides=s, checkpoint_tag=tag)))
        return result

    def report(self, tolerance, convergence_tolerance, reference_m):
        cases = {}
        for path in sorted((self.out/"cases").glob("*.json")):
            saved = json.loads(path.read_text())
            if saved["run_id"] != self.run_id:
                raise ValueError("Report contains mismatched experiment")
            cases[path.stem] = saved["result"]
        sensitivity = []
        for name, row in cases.items():
            if row["kind"] == "sensitivity":
                for p in row["parameters"]:
                    smallest = p["steps"][-1]
                    sensitivity.append({
                        "case": name, "method": row["method"], "design_index": row["design_index"],
                        "cutoff": row["cutoff"], "grid": row["grid"], "parameter": p["parameter"],
                        "smallest_step_error": smallest["analytic_vs_full_fd"],
                        "smallest_step_changed_points": smallest["projector_changed_grid_points"],
                        "passes_all_steps": all(x["passes_simple_sampled_modes"] is True for x in p["steps"]),
                        "analytic_gap_proxy_gradient": p["analytic_gap_proxy_GHz_per_um"],
                        "smallest_step_gap_proxy_fd": smallest["gap_proxy_fd_GHz_per_um"],
                    })
        refinements = []
        for method in self.config["methods"]:
            for i in range(len(self.config["side_multipliers"])):
                for prefix, full in (("conv", False), ("full", True)):
                    grids = [self.config["full_path_grid"]] if full else self.config["normal_grids"]
                    for grid in grids:
                        rows = {m: cases.get(f"{prefix}_{method}_d{i}_M{m}_g{grid}")
                                for m in (self.config["full_path_cutoffs"] if full else self.config["convergence_cutoffs"])}
                        diffs = summarize_pairs(rows, "cutoff", sorted(rows), 1e-30)
                        gradient_refinements = []
                        for left, right in zip(sorted(rows)[:-1], sorted(rows)[1:]):
                            if rows[left] and rows[right]:
                                field = ("analytic_edge_gap_gradient_GHz_per_um" if full
                                         else "analytic_gradients_GHz_per_um")
                                item = {
                                    "cutoff_from": left, "cutoff_to": right,
                                    "quantity": field,
                                    **error(np.array(rows[left][field]), np.array(rows[right][field]),
                                            self.config["derivative_absolute_floor_ghz_per_um"])}
                                if full:
                                    item["edge_indices_unchanged"] = rows[left]["edge_indices"] == rows[right]["edge_indices"]
                                gradient_refinements.append(item)
                        ref = rows.get(reference_m)
                        vs_reference = []
                        if ref:
                            for m, r in rows.items():
                                if r and m != reference_m:
                                    diff = error(np.array(r["frequencies_GHz"]), np.array(ref["frequencies_GHz"]), 1e-30)
                                    vs_reference.append({"cutoff": m, **diff,
                                                         "within_existing_tolerance": diff["max_relative"] <= convergence_tolerance})
                        refinements.append({"method": method, "design_index": i, "grid": grid,
                                            "full_path": full, "successive_refinements": diffs,
                                            "analytic_gradient_refinements": gradient_refinements,
                                            "against_existing_reference_M": vs_reference})
        grid_refinements = []
        for method in self.config["methods"]:
            for i in range(len(self.config["side_multipliers"])):
                for cutoff in self.config["convergence_cutoffs"]:
                    grids = self.config["normal_grids"] + ([self.config["extra_normal_grid"]]
                            if cutoff == self.config["extra_grid_cutoff"] else [])
                    rows = {g: cases.get(f"conv_{method}_d{i}_M{cutoff}_g{g}") for g in grids}
                    diffs = summarize_pairs(rows, "grid", grids, 1e-30)
                    gradient_differences = []
                    for left, right in zip(grids[:-1], grids[1:]):
                        if rows[left] and rows[right]:
                            gradient_differences.append({
                                "grid_from": left, "grid_to": right,
                                **error(np.array(rows[left]["analytic_gradients_GHz_per_um"]),
                                        np.array(rows[right]["analytic_gradients_GHz_per_um"]),
                                        self.config["derivative_absolute_floor_ghz_per_um"])})
                    grid_refinements.append({"method": method, "design_index": i, "cutoff": cutoff,
                                             "successive_refinements": diffs,
                                             "analytic_gradient_refinements": gradient_differences})
        full_gradients = []
        c = self.config
        method, i, cutoff, grid = (c["full_path_sensitivity_method"], c["full_path_sensitivity_design_index"],
                                  c["full_path_sensitivity_cutoff"], c["full_path_grid"])
        base = cases.get(f"full_{method}_d{i}_M{cutoff}_g{grid}")
        if base:
            for axis in (0, 1):
                for h in c["full_path_sensitivity_steps_um"]:
                    minus = cases.get(f"full_fd_{method}_d{i}_M{cutoff}_g{grid}_L{axis+1}_h{h}_s-1")
                    plus = cases.get(f"full_fd_{method}_d{i}_M{cutoff}_g{grid}_L{axis+1}_h{h}_s1")
                    if minus and plus:
                        fd = (plus["gap_GHz"]-minus["gap_GHz"])/(2*h)
                        analytic = base["analytic_edge_gap_gradient_GHz_per_um"][axis]
                        full_gradients.append({"parameter": f"L{axis+1}", "step_um": h,
                            "analytic_GHz_per_um": analytic, "full_path_fd_GHz_per_um": fd,
                            "error": error(np.array(analytic), np.array(fd), c["derivative_absolute_floor_ghz_per_um"]),
                            "edge_indices_unchanged": base["edge_indices"] == minus["edge_indices"] == plus["edge_indices"]})
        self.verify_preserved()
        payload = {"run_id": self.run_id, "config": c, "completed_jobs": len(cases),
                   "protected_hashes_verified": True, "protected_file_count": len(self.protected),
                   "existing_gradient_tolerance": tolerance, "existing_convergence_tolerance": convergence_tolerance,
                   "existing_reference_M": reference_m, "sensitivity_summary": sensitivity,
                   "cutoff_refinements": refinements, "grid_refinements": grid_refinements,
                   "full_path_gap_sensitivity": full_gradients,
                   "ml_training": False, "comsol_spectra_used": False,
                   "production_solver_replaced": False, "continuum_accuracy_certified": False,
                   "updated_unix": time.time()}
        atomic_json(self.out/"results.json", payload)
        lines = ["# Separate pure-PWE projector experiments", "", c["scope"], "",
                 f"Completed checkpointed jobs: {len(cases)}.",
                 f"Protected files unchanged: {len(self.protected)} (SHA-256 verified).",
                 f"Existing gradient tolerance: {tolerance:g}. Existing frequency-refinement tolerance: {convergence_tolerance:g}, reference M={reference_m}.",
                 "No ML, COMSOL spectra or fitting. The production solver and historical results are unchanged.",
                 "Stable-nearest derivatives describe fixed assignments only; radial derivatives include all length dependence of that discrete solver.",
                 "Base and perturbed endpoint separations are checked; an intermediate crossing is not ruled out by this alone.",
                 "A finite-M gradient check is not a continuum-error certificate.", "",
                 "|Method|Design|M|Grid|Parameter|Small-step relative error|Changed projector samples|Pass all tested steps|",
                 "|---|---|---|---|---|---|---|---|"]
        for r in sensitivity:
            err = r["smallest_step_error"]
            lines.append(f"|{r['method']}|{r['design_index']}|{r['cutoff']}|{r['grid']}|{r['parameter']}|"
                         f"{err['max_relative'] if err else 'excluded'}|{r['smallest_step_changed_points']}|{r['passes_all_steps']}|")
        lines += ["", "## Fourier-cutoff refinement", "",
                  "|Method|Design|Grid|Full path|M transition|Max frequency relative difference|Gap change GHz|",
                  "|---|---|---|---|---|---|---|"]
        for r in refinements:
            for d in r["successive_refinements"]:
                lines.append(f"|{r['method']}|{r['design_index']}|{r['grid']}|{r['full_path']}|"
                             f"{d['cutoff_from']} to {d['cutoff_to']}|{d['frequency']['max_relative']:.6g}|{d['gap_difference_GHz']:.6g}|")
        lines += ["", "## Normal-grid refinement", "",
                  "|Method|Design|M|Grid transition|Max frequency relative difference|Gap proxy change GHz|",
                  "|---|---|---|---|---|---|"]
        for r in grid_refinements:
            for d in r["successive_refinements"]:
                lines.append(f"|{r['method']}|{r['design_index']}|{r['cutoff']}|"
                             f"{d['grid_from']} to {d['grid_to']}|{d['frequency']['max_relative']:.6g}|{d['gap_difference_GHz']:.6g}|")
        lines += ["", "## Full-path gap sensitivity", "",
                  "|Parameter|Step um|Analytic GHz/um|FD GHz/um|Relative difference|Edges unchanged|",
                  "|---|---|---|---|---|---|"]
        for r in full_gradients:
            lines.append(f"|{r['parameter']}|{r['step_um']}|{r['analytic_GHz_per_um']:.9g}|"
                         f"{r['full_path_fd_GHz_per_um']:.9g}|{r['error']['max_relative']:.6g}|{r['edge_indices_unchanged']}|")
        (self.out/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
