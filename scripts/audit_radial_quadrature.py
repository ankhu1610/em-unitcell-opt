"""Isolated direct-projector audit; screen by default, full paths only with --stage all."""
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
from audit_pure_pwe import reciprocal_indices, reciprocal_basis, path_coordinates, eta_matrix
from audit_pwe_sensitivity import modes, error, gap
from validate_energy_pwe_full import atomic_json, lowest_two
from invdes.projector_experiment import Experiment, DeadlineReached, digest
from invdes.interface_pwe import energy_factorized_impermittivity
from invdes.radial_projector_quadrature import projector_coefficients, projector_matrices


def planned_jobs(config, stage):
    jobs = [{"name": "coefficient_refinement", "kind": "coefficients"},
            {"name": "material_refinement", "kind": "material"}]
    order = config["spectral_order"]
    for i in range(len(config["side_multipliers"])):
        for cutoff in config["cutoffs"]:
            jobs.append(dict(name=f"base_d{i}_M{cutoff}", kind="spectrum",
                             index=i, cutoff=cutoff, order=order, full=False, gradients=True))
        jobs.append(dict(name=f"angular_d{i}", kind="spectrum", index=i,
                         cutoff=config["angular_spectrum_cutoff"],
                         order=config["angular_spectrum_order"], full=False, gradients=True))
    if stage == "all":
        for i in range(len(config["side_multipliers"])):
            for cutoff in config["full_path_cutoffs"]:
                jobs.append(dict(name=f"full_d{i}_M{cutoff}", kind="spectrum",
                                 index=i, cutoff=cutoff, order=order, full=True, gradients=True))
    i = config["finite_difference_design_index"]
    for full, cutoffs, steps in (
            (False, config["finite_difference_cutoffs"], config["finite_difference_steps_um"]),
            (True, [config["full_fd_cutoff"]], config["full_fd_steps_um"])):
        if full and stage != "all":
            continue
        for cutoff in cutoffs:
            for axis in (0, 1):
                for step in steps:
                    for sign in (-1, 1):
                        prefix = "full_fd" if full else "fd"
                        jobs.append(dict(name=f"{prefix}_d{i}_M{cutoff}_L{axis+1}_h{step}_s{sign}",
                                         kind="spectrum", index=i, cutoff=cutoff, order=order,
                                         full=full, gradients=False, axis=axis, step=step, sign=sign))
    return jobs


class RadialAudit(Experiment):
    """Reuse generic checkpoint, progress and analytic energy-chain helpers."""
    def __init__(self, root, config, run_id, protected, device="cpu", deadline=float("inf")):
        self.root, self.config, self.run_id = Path(root), config, run_id
        self.out = self.root/"results/radial_quadrature_audit_v1"
        self.protected, self.device, self.deadline = protected, device, deadline
        self.started = time.time()
        self.completed = self.total = 0
        self.current = None

    def progress(self, stage):
        # Avoid nonstandard JSON Infinity for an unbounded runtime.
        atomic_json(self.out/"status.json", {
            "status": "running", "case": self.current, "stage": stage,
            "completed_jobs": self.completed, "total_jobs": self.total,
            "pid": os.getpid(), "updated_unix": time.time(),
            "runtime_deadline_unix": self.deadline if np.isfinite(self.deadline) else None})
        print(f"{self.current}: {stage}", flush=True)

    def coefficients(self):
        c = self.config
        differences = reciprocal_indices(2*c["coefficient_cutoff"])
        rows, previous = [], None
        for order in c["coefficient_orders"]:
            self.check_deadline()
            self.progress(f"full difference-range angular order {order}")
            values = projector_coefficients(differences, c["a_um"], order)
            reverse = projector_coefficients(-differences, c["a_um"], order)
            target = np.all(differences == 0, axis=1)
            row = {"order": order,
                   "raw_trace_max_absolute": float(np.max(abs(values[:, 0]+values[:, 2]-target))),
                   "conjugacy_max_absolute": float(np.max(abs(reverse-values.conj()))),
                   "zero_coefficient": values[target].real.tolist()}
            if previous is not None:
                row["previous_order_max_absolute_difference"] = float(np.max(abs(values-previous)))
                row["previous_order_frobenius_difference"] = float(np.linalg.norm(values-previous))
            rows.append(row)
            previous = values
        return {"kind": "coefficients", "basis_cutoff": c["coefficient_cutoff"],
                "difference_count": len(differences), "rows": rows}

    def material(self):
        c = self.config
        ids = reciprocal_indices(c["material_check_cutoff"])
        sides = np.array(c["material_check_side_multipliers"])*c["a_um"]/np.sqrt(3)
        left = eta_matrix(ids, c["a_um"], sides, c["material_check_orders"][0])
        self.check_deadline()
        right = eta_matrix(ids, c["a_um"], sides, c["material_check_orders"][1])
        delta = right-left
        return {"kind": "material", "cutoff": c["material_check_cutoff"],
                "sides_um": sides.tolist(), "orders": c["material_check_orders"],
                "max_absolute_eta_difference": float(np.max(abs(delta))),
                "frobenius_eta_difference": float(np.linalg.norm(delta)),
                "relative_frobenius_eta_difference": float(np.linalg.norm(delta)/np.linalg.norm(right))}

    def spectrum(self, job):
        c = self.config
        a = c["a_um"]
        sides = np.array(c["side_multipliers"][job["index"]])*a/np.sqrt(3)
        if "axis" in job:
            sides[job["axis"]] += job["sign"]*job["step"]
        ids = reciprocal_indices(job["cutoff"])
        self.progress("assembling independently rebuilt material and projector")
        eta = eta_matrix(ids, a, sides, c["arc_order"])
        p = projector_matrices(ids, a, job["order"])
        tensor = energy_factorized_impermittivity(eta, 10*np.eye(len(ids))-9*eta, p)
        g = ids@reciprocal_basis(a*1e-6)
        parameters = ((c["full_path_start"]+c["full_path_step"]*np.arange(c["full_path_count"])).tolist()
                      if job["full"] else c["path_parameters"])
        ks = path_coordinates(parameters, a*1e-6)
        checks = {}
        if not job["full"]:
            f, pairs, residuals, separations = modes(g, ks, tensor)
            gradient = self.gradient(ids, sides, g, ks, pairs, eta, p) if job["gradients"] else []
        else:
            partial = self.out/"partial"/f'{job["name"]}.json'
            identity = {"run_id": self.run_id, "job": job, "sides_um": sides.tolist(),
                        "path_parameters": parameters}
            frequencies, diagnostics = [], []
            if partial.exists():
                saved = json.loads(partial.read_text())
                if saved["identity"] != identity:
                    raise ValueError("Partial full-path fingerprint mismatch")
                frequencies, diagnostics = saved["frequencies_GHz"], saved["diagnostics"]
                if (len(frequencies) != len(diagnostics) or len(frequencies) > len(ks)
                        or (frequencies and np.asarray(frequencies).shape != (len(frequencies), 2))
                        or not np.all(np.isfinite(frequencies))):
                    raise ValueError("Invalid partial full-path contents")
            cuda = None
            if self.device == "cuda":
                from benchmark_pwe_cuda import CudaSolver
                import torch
                torch.set_num_threads(2)
                cuda = CudaSolver(g, tensor)
            try:
                for ik in range(len(frequencies), len(ks)):
                    self.check_deadline()
                    frow, diag = cuda.solve(ks[ik]) if cuda is not None else lowest_two(g, ks[ik], tensor)
                    frequencies.append(frow.tolist())
                    diagnostics.append({**diag, "device": self.device})
                    atomic_json(partial, {"identity": identity, "frequencies_GHz": frequencies,
                                          "diagnostics": diagnostics})
                    if (ik+1) % 10 == 0 or ik+1 == len(ks):
                        self.progress(f"full path {ik+1}/{len(ks)}")
                f = np.asarray(frequencies)
                cpu_difference = 0.
                for ik in (0, len(ks)//2, len(ks)-1):
                    cpu, _ = lowest_two(g, ks[ik], tensor)
                    cpu_difference = max(cpu_difference, float(np.max(abs(cpu-f[ik]))))
                if cpu_difference > 1e-5:
                    raise ValueError(f"New-tensor CPU/device disagreement {cpu_difference} GHz")
                checks["cpu_device_max_absolute_GHz"] = cpu_difference
            finally:
                if cuda is not None:
                    del cuda
                    torch.cuda.empty_cache()
            edges = [int(np.argmax(f[:, 0])), int(np.argmin(f[:, 1]))]
            _, pairs, _, separations = modes(g, ks[edges], tensor)
            gradient = self.gradient(ids, sides, g, ks[edges], pairs, eta, p) if job["gradients"] else []
            residuals = [d["relative_eigenpair_residual"] for d in diagnostics]
        edges = [int(np.argmax(f[:, 0])), int(np.argmin(f[:, 1]))]
        gradient = np.asarray(gradient)
        gap_grad = ([float(x[1, 1]-x[0, 0]) for x in gradient] if job["full"] else
                    [float(x[edges[1], 1]-x[edges[0], 0]) for x in gradient])
        return {"kind": "spectrum", "job": job, "sides_um": sides.tolist(),
                "path_parameters": parameters, "frequencies_GHz": f.tolist(),
                "analytic_gradients_GHz_per_um": gradient.tolist(),
                "gradient_scope": "selected two edges" if job["full"] else "five points",
                "analytic_gap_gradient_GHz_per_um": gap_grad, "edge_indices": edges,
                "edge_uniqueness_margins_GHz": [
                    float(np.sort(f[:, 0])[-1]-np.sort(f[:, 0])[-2]),
                    float(np.sort(f[:, 1])[1]-np.sort(f[:, 1])[0])],
                "mode_relative_separations": separations.tolist(), "gap_GHz": gap(f),
                "gap_scope": "sampled full path" if job["full"] else "five-point proxy",
                "max_eigenpair_residual": max(residuals), **checks}

    def compute(self, job):
        if job["kind"] == "coefficients":
            return self.coefficients()
        if job["kind"] == "material":
            return self.material()
        return self.spectrum(job)

    def report(self, tolerance):
        cases = {}
        for path in sorted((self.out/"cases").glob("*.json")):
            saved = json.loads(path.read_text())
            if saved["run_id"] != self.run_id:
                raise ValueError("Report checkpoint fingerprint mismatch")
            cases[path.stem] = saved["result"]
        c = self.config
        floor = c["derivative_absolute_floor_ghz_per_um"]
        refinements, angular, fdchecks = [], [], []

        def compare(left, right):
            return {"frequency": error(np.array(left["frequencies_GHz"]),
                                       np.array(right["frequencies_GHz"]), 1e-30),
                    "gap_change_GHz": right["gap_GHz"]-left["gap_GHz"],
                    "gradient": error(np.array(left["analytic_gradients_GHz_per_um"]),
                                      np.array(right["analytic_gradients_GHz_per_um"]), floor),
                    "gap_gradient": error(np.array(left["analytic_gap_gradient_GHz_per_um"]),
                                          np.array(right["analytic_gap_gradient_GHz_per_um"]), floor),
                    "same_edge_indices": left["edge_indices"] == right["edge_indices"]}

        for i in range(len(c["side_multipliers"])):
            for prefix, cutoffs in (("base", c["cutoffs"]), ("full", c["full_path_cutoffs"])):
                for low, high in zip(cutoffs[:-1], cutoffs[1:]):
                    left, right = cases.get(f"{prefix}_d{i}_M{low}"), cases.get(f"{prefix}_d{i}_M{high}")
                    if left and right:
                        refinements.append({"design": i, "scope": prefix, "M_from": low, "M_to": high,
                                            **compare(left, right)})
            left = cases.get(f'base_d{i}_M{c["angular_spectrum_cutoff"]}')
            right = cases.get(f"angular_d{i}")
            if left and right:
                angular.append({"design": i, "orders": [c["spectral_order"], c["angular_spectrum_order"]],
                                **compare(left, right)})
        i = c["finite_difference_design_index"]
        for full, cutoffs, steps in (
                (False, c["finite_difference_cutoffs"], c["finite_difference_steps_um"]),
                (True, [c["full_fd_cutoff"]], c["full_fd_steps_um"])):
            for cutoff in cutoffs:
                base = cases.get(f'{"full" if full else "base"}_d{i}_M{cutoff}')
                if base is None:
                    continue
                for axis in (0, 1):
                    for step in steps:
                        prefix = "full_fd" if full else "fd"
                        minus = cases.get(f"{prefix}_d{i}_M{cutoff}_L{axis+1}_h{step}_s-1")
                        plus = cases.get(f"{prefix}_d{i}_M{cutoff}_L{axis+1}_h{step}_s1")
                        if not minus or not plus:
                            continue
                        analytic_gap = base["analytic_gap_gradient_GHz_per_um"][axis]
                        gap_fd = (plus["gap_GHz"]-minus["gap_GHz"])/(2*step)
                        row = {"full_path": full, "cutoff": cutoff, "axis": axis, "step_um": step,
                               "gap_analytic_GHz_per_um": analytic_gap, "gap_fd_GHz_per_um": gap_fd,
                               "gap_error": error(np.array([analytic_gap]), np.array([gap_fd]), floor),
                               "edge_indices_unchanged":
                                   base["edge_indices"] == minus["edge_indices"] == plus["edge_indices"],
                               "edge_uniqueness_margins_GHz":
                                   [x["edge_uniqueness_margins_GHz"] for x in (base, minus, plus)],
                               "mode_relative_separations":
                                   [x["mode_relative_separations"] for x in (base, minus, plus)]}
                        if not full:
                            fd = (np.array(plus["frequencies_GHz"])-np.array(minus["frequencies_GHz"]))/(2*step)
                            analytic = np.array(base["analytic_gradients_GHz_per_um"])[axis]
                            mask = np.all(np.array(row["mode_relative_separations"])
                                          > c["near_degenerate_relative_separation"], axis=0)
                            discrepancy = error(analytic[mask], fd[mask], floor) if mask.any() else None
                            row.update(finite_difference_GHz_per_um=fd.tolist(),
                                       analytic_GHz_per_um=analytic.tolist(), simple_modes_mask=mask.tolist(),
                                       error=discrepancy, passes_existing_gradient_tolerance=(
                                           discrepancy["max_relative"] <= tolerance if discrepancy else None))
                        fdchecks.append(row)
        self.verify_preserved()
        result = {"run_id": self.run_id, "config": c, "completed_jobs": len(cases),
                  "coefficient_refinement": cases.get("coefficient_refinement"),
                  "material_refinement": cases.get("material_refinement"),
                  "cutoff_refinements": refinements, "angular_spectrum_refinements": angular,
                  "finite_difference_checks": fdchecks, "existing_gradient_tolerance": tolerance,
                  "protected_files_verified": len(self.protected),
                  "continuum_accuracy_certified": False, "ml_training": False,
                  "comsol_spectra_used": False, "production_replaced": False}
        atomic_json(self.out/"results.json", result)
        lines = ["# Direct radial-projector quadrature audit", "", c["scope"], "",
                 f"Completed jobs: {len(cases)}; protected files verified: {len(self.protected)}.",
                 f"Predeclared spectral angular order: {c['spectral_order']}.",
                 "Projector length derivatives vanish by construction. Finite-basis derivative checks are not continuum certificates.",
                 "Gap derivatives use active sampled edges; inspect uniqueness, mode separation and cutoff stability.", "",
                 "|Scope|Design|M change|Frequency relative change|Gap change GHz|Gradient relative change|",
                 "|---|---|---|---|---|---|"]
        for row in refinements:
            lines.append(f"|{row['scope']}|{row['design']}|{row['M_from']} to {row['M_to']}|"
                         f"{row['frequency']['max_relative']:.6g}|{row['gap_change_GHz']:.6g}|"
                         f"{row['gradient']['max_relative']:.6g}|")
        lines += ["", "Detailed quadrature, derivative and angular-frequency diagnostics are in results.json and cases/."]
        (self.out/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("screen", "all"), default="screen")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--until-utc", help="Timezone-aware runtime deadline, excluded from scientific fingerprint")
    args = parser.parse_args()
    deadline = float("inf")
    if args.until_utc:
        date = datetime.fromisoformat(args.until_utc.replace("Z", "+00:00"))
        if date.tzinfo is None:
            parser.error("--until-utc needs timezone information")
        deadline = date.timestamp()
    config_path = ROOT/"configs/radial_quadrature_audit_v1.json"
    config = json.loads(config_path.read_text())
    frozen = json.loads((ROOT/"configs/energy_pwe_audit.json").read_text())
    if any(config[k] != frozen[k] for k in ("a_um", "eps_background", "eps_hole")):
        raise ValueError("Frozen physics mismatch")
    upstream_path = ROOT/"results/projector_stabilization_v1/manifest.json"
    upstream = json.loads(upstream_path.read_text())
    protected = {**upstream["protected_hashes"], **upstream["code_hashes"],
                 str(upstream_path): digest(upstream_path)}
    tolerance = float(re.search(r"pwe_gradcheck:.*max_rel_err:\s*([0-9.eE+-]+)",
                                (ROOT/"configs/criteria.yaml").read_text()).group(1))
    paths = [Path(__file__), config_path, ROOT/"src/invdes/radial_projector_quadrature.py",
             ROOT/"tests/test_radial_projector_quadrature.py", ROOT/"tests/test_radial_quadrature_audit.py"]
    protocol = {"config": config, "protected_hashes": protected,
                "code_hashes": {str(p): digest(p) for p in paths},
                "gradient_tolerance": tolerance,
                "runtime": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                            "dtype": "float64/complex128", "allowed_full_path_backends": ["cpu", "cuda"]}}
    run_id = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    audit = RadialAudit(ROOT, config, run_id, protected, args.device, deadline)
    audit.verify_preserved()
    audit.out.mkdir(parents=True, exist_ok=True)
    manifest = audit.out/"manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != protocol:
            raise ValueError("Inputs changed; preserve old audit and use a new version")
    else:
        atomic_json(manifest, protocol)
    jobs = planned_jobs(config, args.stage)
    audit.total = len(jobs)
    lock = audit.out/"run.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    state = "complete"
    try:
        for job in jobs:
            audit.get_or_compute(job["name"], lambda job=job: audit.compute(job))
            audit.report(tolerance)
    except DeadlineReached:
        state = "deadline_checkpointed"
    except BaseException as exc:
        state = "failed"
        atomic_json(audit.out/"status.json", {"status": state, "error": repr(exc),
                    "case": audit.current, "completed_jobs": audit.completed, "updated_unix": time.time()})
        raise
    finally:
        lock.unlink()
        if state != "failed":
            audit.report(tolerance)
            atomic_json(audit.out/"status.json", {
                "status": state, "stage": args.stage, "completed_jobs": audit.completed,
                "total_jobs": len(jobs), "updated_unix": time.time(),
                "runtime_deadline_unix": deadline if np.isfinite(deadline) else None})


if __name__ == "__main__":
    main()
