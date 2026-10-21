"""Pure PWE refinement for the Hermitian energy-projector variant.

Reads only the JSON numerical configuration. Never loads spectral datasets.
Existing non-Hermitian audit/results remain untouched.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from invdes.interface_pwe import (
    normal_projector_grid, projector_matrices, energy_factorized_impermittivity,
    real_frequencies, hz_operator,
)
from audit_pure_pwe import reciprocal_indices, reciprocal_basis, path_coordinates, eta_matrix, solve
from audit_interface_pwe import layered_check


def change(a, b):
    # This audit fixes Gamma as the first k point. Exclude its first band by
    # index, not a floating threshold: sqrt(roundoff) is not a physical mode.
    mask = np.ones(a.shape, dtype=bool)
    mask[0, 0] = False
    return float(np.max(abs(a-b)[mask]/abs(a)[mask])*100)


def save(result):
    out = ROOT/"results/energy_pwe_audit"
    out.mkdir(parents=True, exist_ok=True)
    result["layered"] = layered_check(energy_factorized_impermittivity, real_frequencies)
    for d in result["designs"]:
        # Independent GENERAL eigensolver at the smallest basis: do not rely
        # on eigh returning a real dtype as evidence for physical realness.
        cfg = result["config"]
        a = cfg["a_um"]
        ids = reciprocal_indices(cfg["cutoffs"][0])
        sides = np.array(d["multipliers"])*a/np.sqrt(3)
        eta = eta_matrix(ids, a, sides, cfg["arc_order"])
        eps = 10*np.eye(len(ids))-9*eta
        p = projector_matrices(ids, normal_projector_grid(a, sides, cfg["normal_grids"][0]))
        tensor = energy_factorized_impermittivity(eta, eps, p)
        operator = hz_operator(ids@reciprocal_basis(a*1e-6),
                               path_coordinates([.37], a*1e-6)[0], tensor)
        eigenvalues = np.linalg.eigvals(operator)
        hermitian_values = np.linalg.eigvalsh(operator)
        scale = np.max(abs(eigenvalues))
        d["independent_general_eigensolver"] = {
            "cutoff": cfg["cutoffs"][0],
            "max_eigenvalue_imaginary_over_spectral_scale": float(np.max(abs(eigenvalues.imag))/scale),
            "general_vs_hermitian_eigenvalue_difference_over_scale":
                float(np.max(abs(np.sort(eigenvalues.real)-hermitian_values))/scale),
        }
        im = result["config"]["path_parameters"].index(1.)
        ik = result["config"]["path_parameters"].index(2.)
        for row in d["runs"]:
            f = np.array(row["energy_GHz"])
            row["direct_K_GHz"] = float(f[ik, 1]-f[ik, 0])
            row["indirect_M_K_GHz"] = float(f[im, 1]-f[ik, 0])
            row["direct_energy_disagreement_pct"] = change(f[:, :2], np.array(row["direct_GHz"])[:, :2])
        for old, new in zip(d["runs"][:-1], d["runs"][1:]):
            new["energy_change_pct"] = change(np.array(new["energy_GHz"])[:, :2], np.array(old["energy_GHz"])[:, :2])
            new["direct_change_pct"] = change(np.array(new["direct_GHz"])[:, :2], np.array(old["direct_GHz"])[:, :2])
        if "refinement_checks" in d:
            checks = d["refinement_checks"]
            checks["normal_grid_doubling_change_pct"] = change(
                np.array(checks["refined_grid_first6_GHz"])[:, :2],
                np.array(d["runs"][-1]["energy_GHz"])[:, :2])
    (out/"results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# Hermitian energy-projector PWE", "",
        "No training, fitting, reference-frequency datasets, or COMSOL calls.", "",
        "## Discretization", "",
        "The original ordered product has a nonzero commutator and is not Hermitian.",
        "This new variant uses B_block + P* Delta_block P, where B=inverse(T(epsilon)),",
        "Delta=T(1/epsilon)-B, and P=T(n n^T). It is the matrix of a real positive",
        "quadratic energy for positive scalar materials. It is NOT claimed to be the",
        "published Model B formula, nor a proven continuum error bound.",
        "No imaginary eigenvalue parts are discarded; no assembled operator is averaged with its adjoint.",
        "Hermiticity, material consistency, negative squared frequencies, and eigenpair residuals are checked.",
        "", "## Refinement", "",
        "First two bands, all sampled k points; Gamma zero excluded from relative changes.",
        "These are successive-basis changes, not errors against a known 2D solution.", "",
        "|Sides / (a/sqrt(3))|Cutoff|Energy change %|Direct PWE change %|Hermitian defect|Eigenpair residual|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for d in result["designs"]:
        for r in d["runs"]:
            lines.append(f'|{d["multipliers"]}|{r["cutoff"]}|{r.get("energy_change_pct", float("nan")):.6f}|'
                         f'{r.get("direct_change_pct", float("nan")):.6f}|'
                         f'{max(x["relative_nonhermiticity"] for x in r["checks"]):.3e}|'
                         f'{max(x["max_eigenpair_residual_over_operator_scale"] for x in r["checks"]):.3e}|')
    lines += ["", "## Selected-point gaps and cross-method check", "",
              "No full-zone band-gap claim. Cross-method disagreement is not an error bound.", "",
              "|Geometry|Final direct K GHz|Last change GHz|Final indirect M/K GHz|Last change GHz|Direct/energy frequency disagreement %|",
              "|---|---:|---:|---:|---:|---:|"]
    for d in result["designs"]:
        old, r = d["runs"][-2:]
        lines.append(f'|{d["multipliers"]}|{r["direct_K_GHz"]:.6f}|'
                     f'{abs(r["direct_K_GHz"]-old["direct_K_GHz"]):.6f}|'
                     f'{r["indirect_M_K_GHz"]:.6f}|'
                     f'{abs(r["indirect_M_K_GHz"]-old["indirect_M_K_GHz"]):.6f}|'
                     f'{r["direct_energy_disagreement_pct"]:.6f}|')
    lines += ["", "## Independent numerical checks", ""]
    for d in result["designs"]:
        lines += [f'### Geometry {d["multipliers"]}', "", "```json",
                  json.dumps(d.get("refinement_checks", {}), indent=2), "```", "",
                  "Independent general eigensolver (not constrained to return real eigenvalues):", "",
                  "```json", json.dumps(d["independent_general_eigensolver"], indent=2), "```", ""]
    lines += ["## Analytic layered test", "",
              "The energy variant is run against an independent analytic transfer-matrix solution.",
              "It also reduces algebraically to the flat-interface inverse rule:", "",
              "```json", json.dumps(result["layered"], indent=2), "```", "",
              "## Limits and reproduction", "",
              "The normal extension has medial-axis discontinuities and requires grid refinement.",
              "No full-zone gap or universal accuracy claim follows from selected-point convergence.",
              "Original production Torch code and earlier audit results are unchanged.",
              "No pre-existing acceptance gates are changed or claimed to pass.",
              "Run with NumPy from Tushar:", "",
              "```", "python tests/test_energy_pwe.py", "python scripts/audit_energy_pwe.py", "```"]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT/"configs/energy_pwe_audit.json")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        save(json.loads((ROOT/"results/energy_pwe_audit/results.json").read_text()))
        return
    cfg = json.loads(args.config.read_text())
    if cfg["path_parameters"][0] != 0:
        raise ValueError("Audit requires Gamma first for zero-mode exclusion")
    if (cfg["eps_background"], cfg["eps_hole"]) != (9., 1.):
        raise ValueError("Boundary helper supports eps={9,1}")
    result = {"config": cfg, "layered": layered_check(energy_factorized_impermittivity, real_frequencies), "designs": []}
    a = cfg["a_um"]
    kpoints = path_coordinates(cfg["path_parameters"], a*1e-6)
    for multipliers in cfg["side_multipliers"]:
        sides = np.array(multipliers)*a/np.sqrt(3)
        samples = normal_projector_grid(a, sides, cfg["normal_grids"][0])
        design = {"multipliers": multipliers, "runs": []}
        last = None
        for cutoff in cfg["cutoffs"]:
            start = time.perf_counter()
            ids = reciprocal_indices(cutoff)
            eta = eta_matrix(ids, a, sides, cfg["arc_order"])
            eps = 10*np.eye(len(ids))-9*eta
            projectors = projector_matrices(ids, samples)
            tensor = energy_factorized_impermittivity(eta, eps, projectors)
            g = ids@reciprocal_basis(a*1e-6)
            f, checks = real_frequencies(g, kpoints, tensor, cfg["bands"])
            direct = solve(ids, a*1e-6, eta, kpoints, cfg["bands"])
            row = {"cutoff": cutoff, "energy_GHz": f.tolist(), "direct_GHz": direct.tolist(),
                   "checks": checks, "seconds": time.perf_counter()-start}
            if last is not None:
                row["energy_change_pct"] = change(f[:, :2], last[0][:, :2])
                row["direct_change_pct"] = change(direct[:, :2], last[1][:, :2])
            last = f, direct
            design["runs"].append(row)
            print(multipliers, cutoff, {k: v for k, v in row.items() if k.endswith("pct") or k == "seconds"}, flush=True)
        refined_samples = normal_projector_grid(a, sides, cfg["normal_grids"][1])
        refined_tensor = energy_factorized_impermittivity(eta, eps, projector_matrices(ids, refined_samples))
        refined_f, _ = real_frequencies(g, kpoints, refined_tensor, cfg["bands"])
        # Time reversal at an off-symmetry point and separate quadrature check.
        minus_f, _ = real_frequencies(g, -kpoints[1:2], tensor, cfg["bands"])
        eta2 = eta_matrix(ids, a, sides, cfg["arc_order"]*2)
        design["refinement_checks"] = {
            "normal_grid_doubling_change_pct": change(refined_f[:, :2], f[:, :2]),
            "time_reversal_max_abs_GHz": float(np.max(abs(minus_f[0]-f[1]))),
            "arc_quadrature_doubling_max_abs_eta": float(np.max(abs(eta2-eta))),
            "refined_grid_first6_GHz": refined_f.tolist(),
        }
        result["designs"].append(design)
        save(result)
    print("Saved", ROOT/"results/energy_pwe_audit", flush=True)


if __name__ == "__main__":
    main()
