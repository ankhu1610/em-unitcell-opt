"""Generate a bounded-session evidence summary; never modify experiment inputs."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"results/pwe_projector_research_session"


def read(relative):
    return json.loads((ROOT/relative).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    primary = read("results/projector_stabilization_v1/results.json")
    quad = read("results/radial_quadrature_audit_v1/results.json")
    baseline = read("results/variational_pwe_audit_v1/results.json")
    manifest = read("results/radial_quadrature_audit_v1/manifest.json")
    hashes = {**manifest["protected_hashes"], **manifest["code_hashes"]}
    changed = [p for p, h in hashes.items() if sha(p) != h]
    if changed:
        raise ValueError(f"Protected source or historical results changed: {changed}")
    full = [r for r in quad["cutoff_refinements"] if r["scope"] == "full"]
    fd = quad["finite_difference_checks"]
    simple_fd = [r for r in fd if not r["full_path"]]
    gap_fd = [r for r in fd if r["full_path"]]
    sensitivities = primary["sensitivity_summary"]
    method_summary = {}
    for method in ("stable_nearest", "homothetic_radial"):
        rows = [r for r in sensitivities if r["method"] == method]
        # Preserve the complete source rows in the JSON for traceability.
        method_summary[method] = rows
    case_comparisons = []
    for design in range(len(quad["config"]["side_multipliers"])):
        q0 = read(f"results/radial_quadrature_audit_v1/cases/full_d{design}_M20.json")["result"]
        q1 = read(f"results/radial_quadrature_audit_v1/cases/full_d{design}_M25.json")["result"]
        raster = read(
            f"results/projector_stabilization_v1/cases/full_homothetic_radial_d{design}_M25_g512.json"
        )["result"]
        nearest = read(
            f"results/projector_stabilization_v1/cases/full_stable_nearest_d{design}_M25_g512.json"
        )["result"]
        fq = np.array(q1["frequencies_GHz"])
        fr = np.array(raster["frequencies_GHz"])
        fn = np.array(nearest["frequencies_GHz"])
        case_comparisons.append({
            "design": design, "sides_um": q1["sides_um"],
            "gap_M20_GHz": q0["gap_GHz"], "gap_M25_GHz": q1["gap_GHz"],
            "cutoff_gap_change_GHz": q1["gap_GHz"]-q0["gap_GHz"],
            "quadrature_minus_radial_grid512_gap_GHz": q1["gap_GHz"]-raster["gap_GHz"],
            "quadrature_vs_radial_grid512_max_frequency_GHz": float(np.max(abs(fq-fr))),
            "quadrature_minus_stable_nearest_gap_GHz": q1["gap_GHz"]-nearest["gap_GHz"],
            "quadrature_vs_stable_nearest_max_frequency_GHz": float(np.max(abs(fq-fn))),
        })
    sources = [
        ROOT/"results/projector_stabilization_v1/results.json",
        ROOT/"results/radial_quadrature_audit_v1/results.json",
        ROOT/"results/variational_pwe_audit_v1/results.json",
    ]
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "primary_status": read("results/projector_stabilization_v1/status.json"),
        "radial_status": read("results/radial_quadrature_audit_v1/status.json"),
        "primary_jobs": primary["completed_jobs"], "radial_jobs": quad["completed_jobs"],
        "variational_jobs": baseline["completed_jobs"],
        "historical_files_reverified": len(manifest["protected_hashes"]),
        "protected_and_source_files_reverified": len(hashes),
        "source_hashes": {str(p): sha(p) for p in sources},
        "projector_sensitivity_rows": method_summary,
        "projector_all_steps_counts": {
            name: {"passed": sum(bool(r["passes_all_steps"]) for r in rows),
                   "total": len(rows)} for name, rows in method_summary.items()},
        "point_fd_checks": len(simple_fd),
        "point_fd_all_pass": all(r["passes_existing_gradient_tolerance"] for r in simple_fd),
        "point_fd_worst_relative": max(r["error"]["max_relative"] for r in simple_fd),
        "full_gap_fd_checks": len(gap_fd),
        "full_gap_fd_worst_relative": max(r["gap_error"]["max_relative"] for r in gap_fd),
        "full_gap_fd_all_edges_unchanged": all(r["edge_indices_unchanged"] for r in gap_fd),
        "existing_gradient_tolerance": quad["existing_gradient_tolerance"],
        "full_path_refinements": full, "case_comparisons": case_comparisons,
        "coefficient_refinement": quad["coefficient_refinement"],
        "angular_point_refinements": quad["angular_spectrum_refinements"],
        "material_refinement": quad["material_refinement"],
        "ml_training": False, "comsol_fitting": False, "comsol_execution": False,
        "validated_solver_changed": False, "inverse_physics_run_launched": False,
        "continuum_accuracy_certified": False,
        "conclusion": "Fixed homothetic radial extension with direct Fourier quadrature is the leading differentiable research implementation. Stable-nearest still has assignment-switch nonsmoothness. Production remains unchanged.",
        "remaining": [
            "An anchor-free trust-region kernel, checkpointed evaluator and numerical diagnostics are prepared; no inverse physics experiment was launched in this session. The CLI is fail-closed pending angular-report fingerprint validation and comparison recomputation before future launch.",
            "Higher-cutoff and wider-geometry derivative convergence, genuinely independent off-grid inverse requests, and full-Brillouin-zone edge searches remain.",
            "Finite-basis finite-difference agreement verifies derivatives of that discretization, not continuum or COMSOL accuracy.",
        ],
    }
    supplement = ROOT/"results/quadrature_full_order_v1/results.json"
    if supplement.exists():
        summary["full_path_order_supplement"] = json.loads(supplement.read_text())
        summary["source_hashes"][str(supplement)] = sha(supplement)
    for label, path in (
            ("final_targeted_tests", OUT/"final_tests.json"),
            ("exploratory_launch_review", OUT/"exploratory_launch_review.json")):
        if path.exists():
            summary[label] = json.loads(path.read_text())
    tests_xml = OUT/"final_120_tests_sequential_mkl.xml"
    if tests_xml.exists():
        suites = ET.parse(tests_xml).getroot().findall("testsuite")
        summary["final_targeted_tests"] = {
            key: sum(int(s.get(key, "0")) for s in suites)
            for key in ("tests", "failures", "errors", "skipped")}
        summary["final_targeted_tests"].update(
            source=str(tests_xml), runtime="Anaconda Python; process-scoped MKL_THREADING_LAYER=SEQUENTIAL",
            environment_issue="Default combined import of NumPy/SciPy and PyTorch aborted with duplicate Intel OpenMP Error #15. Isolated minimal reproduction confirmed it; sequential MKL avoids the conflicting MKL OpenMP load. No persistent environment changes, dependencies, or KMP_DUPLICATE_LIB_OK bypass.",
            cuda_environment_note="The existing CUDA environment lacks pytest; no dependency was installed.",
            scope="Targeted pure-PWE and inverse regression suites, not the full historical ML test suite.")
    awake = OUT/"awake.log"
    summary["wake_log"] = awake.read_text() if awake.exists() else None
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"final_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    saved = json.loads((OUT/"final_summary.json").read_text())
    lines = [
        "# Bounded pure-PWE mathematical research", "", saved["conclusion"], "",
        f"Completed projector jobs: {saved['primary_jobs']}; direct-quadrature jobs: {saved['radial_jobs']}; variational-control jobs: {saved['variational_jobs']}.",
        f"Reverified unchanged protected historical/input files: {saved['historical_files_reverified']}.",
        "No ML, COMSOL execution, COMSOL fitting or production replacement.", "",
        "## Derivative evidence", "",
        f"Direct-quadrature pointwise checks: {saved['point_fd_checks']}; all pass: {saved['point_fd_all_pass']}; worst relative error: {saved['point_fd_worst_relative']:.9g}.",
        f"Full-path gap checks: {saved['full_gap_fd_checks']}; worst relative error: {saved['full_gap_fd_worst_relative']:.9g}; existing tolerance: {saved['existing_gradient_tolerance']:.9g}.",
        f"All tested full-gap active edges unchanged: {saved['full_gap_fd_all_edges_unchanged']}.",
        "Fixed radial projectors remove the moving assignment dependence. Stable-nearest tie handling alone does not remove genuine assignment changes.",
        "", "## Full-path Fourier-cutoff refinement", "",
        "|Geometry|L1 um|L2 um|M20 gap GHz|M25 gap GHz|Gap change GHz|Maximum frequency change GHz|",
        "|---|---|---|---|---|---|---|",
    ]
    for row, comparison in zip(saved["full_path_refinements"], saved["case_comparisons"]):
        lines.append(
            f"|{comparison['design']}|{comparison['sides_um'][0]:.9g}|{comparison['sides_um'][1]:.9g}|"
            f"{comparison['gap_M20_GHz']:.9g}|{comparison['gap_M25_GHz']:.9g}|"
            f"{comparison['cutoff_gap_change_GHz']:.9g}|{row['frequency']['max_absolute']:.9g}|")
    lines += [
        "", "These are convergence differences, NOT errors against COMSOL or certified continuum error bounds.",
        "Only the documented sampled path and representative geometries were checked.",
        "Frequency/gradient cutoff convergence must not be confused with fixed-cutoff finite-difference agreement.",
        "", "## Mathematical direction for the inverse", "",
        "The clipped triangles scale about fixed centers. A fixed nearest-center cell and radial interface-normal extension therefore give a projector independent of both side lengths: dP/dL = 0 within the documented size box.",
        "Material Fourier sensitivities still include the moving physical boundary: d eta_G/dL = (eta_air - eta_background)/A times the boundary integral of v_n exp(-i G dot r), where v_n = (r-c) dot n / L.",
        "Differentiate the same Hermitian energy operator used by the forward solver. For a simple normalized eigenpair, d lambda/dL = h* (dA/dL) h; convert this to frequency and subtract the active lower-edge sensitivity from the active upper-edge sensitivity.",
        "For target residual r = gap(L)-target and normalized-coordinate gradient j, the unconstrained minimum-norm linearized step is delta_x = -r j / (j dot j). The prototype enforces the geometry box and a trust region, verifies each trial by fresh PWE, and falls back to coordinate polling when active-edge derivatives are unreliable.",
        "This is numerical root finding through electromagnetic equations, not training. One requested gap generally admits multiple length pairs; the minimum-norm local step is a selection rule, not a uniqueness proof.",
        "", "## Quadrature and interface evidence", "",
        "Direct Fourier integration avoids a spatial normal raster. Angular convergence and Fourier cutoff remain separate controls.",
        "Detailed order-by-order coefficient, material and frequency diagnostics are retained in final_summary.json.",
    ]
    if "full_path_order_supplement" in saved:
        sup = saved["full_path_order_supplement"]
        m = sup["comparison"]["metrics"]
        lines += [
            f"Supplementary full-path order check passed: {sup['passes_exploratory_guard']}.",
            f"Maximum frequency change: {m['frequency_max_absolute_GHz']:.9g} GHz; gap change: {m['gap_change_GHz']:.9g} GHz.",
        ]
    lines += ["", "## Remaining work", ""] + [f"- {x}" for x in saved["remaining"]]
    lines += ["", "## Tests and limitations", ""]
    for method, counts in saved["projector_all_steps_counts"].items():
        lines.append(f"{method}: {counts['passed']}/{counts['total']} parameter checks passed all tested finite-difference steps.")
    if "final_targeted_tests" in saved:
        tests = saved["final_targeted_tests"]
        lines.append(f"Targeted tests: {tests['tests']}; failures: {tests['failures']}; errors: {tests['errors']}; skipped: {tests['skipped']}.")
        lines.append(tests["environment_issue"])
        lines.append(tests["scope"])
    lines += ["", "The bounded wake helper log and exact source hashes are saved in final_summary.json."]
    (OUT/"FINAL_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({k: saved[k] for k in (
        "primary_jobs", "radial_jobs", "variational_jobs", "historical_files_reverified",
        "point_fd_worst_relative", "full_gap_fd_worst_relative")}, indent=2))


if __name__ == "__main__":
    main()
