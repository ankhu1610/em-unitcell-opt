"""Read-only numerical guard for a separately reviewed exploratory inverse run.

This is neither launch authorization nor a continuum/Brillouin-zone certificate.
It recomputes comparisons from completed, fingerprinted audit cases. No solver,
optimizer, training, COMSOL data, or filesystem writes are invoked.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np


class AuditProvenanceError(ValueError):
    """Malformed audit data or changed scientific provenance; do not proceed."""


def _read(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AuditProvenanceError(f"Cannot read valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise AuditProvenanceError(f"Expected JSON object: {path}")
    return value


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array(value, shape, label):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise AuditProvenanceError(f"Malformed numeric field: {label}") from exc
    if result.shape != shape:
        raise AuditProvenanceError(f"Wrong shape for {label}: {result.shape}, expected {shape}")
    return result


def _relative(left, right, floor):
    left, right = np.asarray(left), np.asarray(right)
    delta = np.abs(left-right)
    return {"max_absolute": float(delta.max()),
            "max_relative": float((delta/np.maximum(np.maximum(abs(left), abs(right)), floor)).max())}


def check_quadrature_launch_gate(root):
    """Return a JSON-ready guard result; missing work fails without raising.

    ``root`` is the project directory containing ``configs/`` and ``results/``.
    Malformed schemas and changed hashes/fingerprints raise AuditProvenanceError.
    Nonfinite scientific results instead return a failed numerical guard.
    Even a passing result requires independent full-path Q256 evidence and an
    explicit human/root review before an exploratory inverse launch.
    """
    root = Path(root).resolve()
    out = root/"results/radial_quadrature_audit_v1"
    answer = {
        "passed": False, "reasons": [], "measurements": {}, "sources": {},
        "launch_authorized": False, "requires_manual_review": True,
        "requires_independent_full_path_order256_check": True,
        "continuum_accuracy_certified": False, "production_replacement_authorized": False,
    }
    reasons, measured, sources = answer["reasons"], answer["measurements"], answer["sources"]
    paths = {
        "manifest": out/"manifest.json", "status": out/"status.json",
        "report": out/"results.json",
        "audit_config": root/"configs/radial_quadrature_audit_v1.json",
        "inverse_config": root/"configs/pure_pwe_math_inverse_v1.json",
        "criteria": root/"configs/criteria.yaml",
    }
    sources.update({key: str(path) for key, path in paths.items()})
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        reasons.append("Incomplete audit: missing required files.")
        measured["missing_files"] = missing
        return answer
    try:
        return _check(root, out, paths, answer)
    except AuditProvenanceError:
        raise
    except (KeyError, TypeError, IndexError, ValueError, AttributeError) as exc:
        raise AuditProvenanceError(f"Malformed audit schema: {exc}") from exc


def _check(root, out, paths, answer):
    reasons, measured, sources = answer["reasons"], answer["measurements"], answer["sources"]
    manifest, status, report = (_read(paths[key]) for key in ("manifest", "status", "report"))
    config, inverse = (_read(paths[key]) for key in ("audit_config", "inverse_config"))
    if manifest["config"] != config:
        raise AuditProvenanceError("Current audit config differs from its manifest.")
    run_id = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    sources["run_id"] = run_id
    if report["run_id"] != run_id or report["config"] != config:
        raise AuditProvenanceError("Report fingerprint/config mismatch.")
    verified = {}
    for group in ("protected_hashes", "code_hashes"):
        hashes = manifest[group]
        if not isinstance(hashes, dict) or not hashes:
            raise AuditProvenanceError(f"Missing/non-object {group}.")
        for name, expected in hashes.items():
            path = Path(name)
            if not path.is_absolute():
                path = root/path
            if (not path.is_file() or not isinstance(expected, str)
                    or _digest(path) != expected):
                raise AuditProvenanceError(f"Changed/missing fingerprinted file: {path}")
            verified[str(path.resolve())] = expected
    measured["fresh_protected_hashes_verified"] = len(manifest["protected_hashes"])
    measured["fresh_code_hashes_verified"] = len(manifest["code_hashes"])
    sources["verified_hashes"] = verified
    for key in ("audit_config", "criteria"):
        if str(paths[key].resolve()) not in verified:
            raise AuditProvenanceError(f"Unprotected required threshold/config source: {key}")
    # The experiment's own declared plan, not a second handwritten job-name list.
    scripts = Path(__file__).resolve().parents[3]/"scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from audit_radial_quadrature import planned_jobs
    jobs = planned_jobs(config, "all")
    expected = {job["name"]: job for job in jobs}
    if len(expected) != len(jobs) or len(jobs) != 61:
        raise AuditProvenanceError("Expected the preregistered 61 unique all-stage jobs.")
    files = {p.stem: p for p in (out/"cases").glob("*.json")}
    measured.update(expected_case_count=len(expected), observed_case_count=len(files),
                    missing_cases=sorted(expected.keys()-files.keys()),
                    unexpected_cases=sorted(files.keys()-expected.keys()))
    complete = (status["status"] == "complete" and status.get("stage") == "all"
                and status.get("completed_jobs") == len(jobs)
                and status.get("total_jobs") == len(jobs))
    if not complete:
        reasons.append("Audit is not complete with stage=all and all 61 jobs.")
    if measured["missing_cases"] or measured["unexpected_cases"]:
        reasons.append("Case filenames do not exactly match the all-stage plan.")
    if report["completed_jobs"] != len(jobs):
        reasons.append("Report is not complete for all 61 jobs.")
    # Check available case identities even when work is incomplete.
    cases = {}
    for name in expected.keys() & files.keys():
        saved = _read(files[name])
        if saved["run_id"] != run_id:
            raise AuditProvenanceError(f"Case fingerprint mismatch: {name}")
        result = saved["result"]
        if not isinstance(result, dict) or result["kind"] != expected[name]["kind"]:
            raise AuditProvenanceError(f"Case kind mismatch: {name}")
        if expected[name]["kind"] == "spectrum" and result["job"] != expected[name]:
            raise AuditProvenanceError(f"Case job mismatch: {name}")
        cases[name] = result
    if reasons:
        return answer

    criteria_text = paths["criteria"].read_text(encoding="utf-8")
    def threshold(section, key):
        match = re.search(rf"{section}:.*{key}:\s*([0-9.eE+-]+)", criteria_text)
        if match is None:
            raise AuditProvenanceError(f"Missing criterion {section}.{key}")
        value = float(match.group(1))
        if not np.isfinite(value) or value <= 0:
            raise AuditProvenanceError(f"Invalid criterion {section}.{key}")
        return value
    grad_tol = threshold("pwe_gradcheck", "max_rel_err")
    freq_tol = threshold("pwe_convergence", "max_rel_diff_at_chosen_M")
    final_path = root/inverse["final_tolerance_source"]
    if not final_path.is_file() or str(final_path.resolve()) not in verified:
        raise AuditProvenanceError("Final tolerance source is absent or not protected.")
    gap_tol = float(_read(final_path)[inverse["final_tolerance_key"]])
    sources["final_tolerance"] = str(final_path)
    sources["inverse_config_sha256"] = _digest(paths["inverse_config"])
    if manifest["gradient_tolerance"] != grad_tol or report["existing_gradient_tolerance"] != grad_tol:
        raise AuditProvenanceError("Frozen/report gradient tolerance mismatch.")
    simple_tol = float(config["near_degenerate_relative_separation"])
    unique_tol = float(inverse["near_tied_edge_tolerance_GHz"])
    floor = float(config["derivative_absolute_floor_ghz_per_um"])
    guards = {"gradient_relative": grad_tol, "frequency_relative": freq_tol,
              "gap_change_GHz": gap_tol, "mode_relative_separation_strict": simple_tol,
              "edge_uniqueness_GHz_strict": unique_tol, "cpu_device_absolute_GHz": 1e-5,
              "eigenpair_residual": 1e-7, "coefficient_numerical_guard": 1e-12}
    if not all(np.isfinite(v) and v > 0 for v in (*guards.values(), floor)):
        raise AuditProvenanceError("Invalid finite positive guard threshold.")
    measured["thresholds"] = guards
    sources["coefficient_guard_scope"] = "New numerical quadrature guard, not a production acceptance criterion."
    sources["refinement_guard_scope"] = "Existing bounds reused for empirical M20-to-M25 checks, not a continuum error bound."
    for key in ("a_um", "eps_background", "eps_hole", "full_path_start",
                "full_path_step", "full_path_count"):
        if config[key] != inverse[key]:
            raise AuditProvenanceError(f"Inverse/audit physics or path mismatch: {key}")
    if (config["spectral_order"] != inverse["projector_quadrature_order"]
            or config["arc_order"] != inverse["material_arc_order"]
            or config["full_path_cutoffs"] != [inverse["search_cutoff"], inverse["independent_check_cutoff"]]
            or simple_tol != inverse["near_degenerate_relative_separation"]):
        raise AuditProvenanceError("Inverse/audit numerical protocol mismatch.")

    spectra = {}
    max_residual, max_cpu = 0., 0.
    for name, job in expected.items():
        if job["kind"] != "spectrum":
            continue
        result = cases[name]
        parameters = (config["full_path_start"]+config["full_path_step"]*np.arange(config["full_path_count"])
                      if job["full"] else np.asarray(config["path_parameters"]))
        sides = np.asarray(config["side_multipliers"][job["index"]])*config["a_um"]/np.sqrt(3)
        if "axis" in job:
            sides[job["axis"]] += job["sign"]*job["step"]
        if (result["path_parameters"] != parameters.tolist()
                or result["sides_um"] != sides.tolist()):
            raise AuditProvenanceError(f"Case geometry/path alignment mismatch: {name}")
        n = len(parameters)
        f = _array(result["frequencies_GHz"], (n, 2), name+":frequencies")
        sep = _array(result["mode_relative_separations"], (2 if job["full"] else n, 2), name+":separations")
        grad_shape = (2, 2 if job["full"] else n, 2) if job["gradients"] else (0,)
        grad = _array(result["analytic_gradients_GHz_per_um"], grad_shape, name+":gradients")
        stored_gapgrad = _array(result["analytic_gap_gradient_GHz_per_um"],
                                (2,) if job["gradients"] else (0,), name+":gap gradients")
        stored_edges = _array(result["edge_indices"], (2,), name+":edges")
        stored_margins = _array(result["edge_uniqueness_margins_GHz"], (2,), name+":margins")
        residual = float(result["max_eigenpair_residual"])
        cpu = float(result["cpu_device_max_absolute_GHz"]) if job["full"] else 0.
        numeric = (f, sep, grad, stored_gapgrad, stored_edges, stored_margins,
                   [result["gap_GHz"], residual, cpu])
        if not all(np.isfinite(item).all() for item in numeric):
            reasons.append(f"Nonfinite scientific values in {name}.")
            continue
        if np.any(f <= 0) or np.any(f[:, 1] < f[:, 0]):
            reasons.append(f"Nonpositive or unsorted frequencies in {name}.")
        edges = [int(np.argmax(f[:, 0])), int(np.argmin(f[:, 1]))]
        gap = float(f[edges[1], 1]-f[edges[0], 0])
        margins = [float(np.sort(f[:, 0])[-1]-np.sort(f[:, 0])[-2]),
                   float(np.sort(f[:, 1])[1]-np.sort(f[:, 1])[0])]
        if (edges != result["edge_indices"] or gap != result["gap_GHz"]
                or margins != result["edge_uniqueness_margins_GHz"]):
            raise AuditProvenanceError(f"Stored gap/edge metadata disagrees with frequencies: {name}")
        gapgrad = np.array([])
        if job["gradients"]:
            i, j = (0, 1) if job["full"] else edges
            gapgrad = grad[:, j, 1]-grad[:, i, 0]
            if not np.array_equal(gapgrad, stored_gapgrad):
                raise AuditProvenanceError(f"Stored gap derivative disagrees with selected edges: {name}")
        if residual < 0 or residual > 1e-7:
            reasons.append(f"Eigenpair residual exceeds guard in {name}.")
        if cpu < 0 or cpu > 1e-5:
            reasons.append(f"CPU/device difference exceeds guard in {name}.")
        max_residual, max_cpu = max(max_residual, residual), max(max_cpu, cpu)
        if job["full"]:
            if min(margins) <= unique_tol:
                reasons.append(f"Selected full-path edges are not unique enough in {name}.")
            if min(sep[0, 0], sep[1, 1]) <= simple_tol:
                reasons.append(f"Selected full-path edge modes are not simple enough in {name}.")
        spectra[name] = dict(f=f, grad=grad, sep=sep, gapgrad=gapgrad, gap=gap, edges=edges, margins=margins)
    measured["maximum_eigenpair_residual"] = max_residual
    measured["maximum_cpu_device_absolute_GHz"] = max_cpu
    # Avoid reductions of nonfinite data; it already fails without certifying a mask.
    if len(spectra) != sum(j["kind"] == "spectrum" for j in jobs):
        return answer

    fd_checks = []
    middle = config["finite_difference_design_index"]
    for full, cutoffs, steps in (
            (False, config["finite_difference_cutoffs"], config["finite_difference_steps_um"]),
            (True, [config["full_fd_cutoff"]], config["full_fd_steps_um"])):
        for cutoff in cutoffs:
            base = spectra[f'{"full" if full else "base"}_d{middle}_M{cutoff}']
            for axis in (0, 1):
                for step in steps:
                    prefix = f'{"full_fd" if full else "fd"}_d{middle}_M{cutoff}_L{axis+1}_h{step}'
                    minus, plus = spectra[prefix+"_s-1"], spectra[prefix+"_s1"]
                    same_edges = base["edges"] == minus["edges"] == plus["edges"]
                    if full:
                        analytic = base["gapgrad"][axis:axis+1]
                        finite_difference = np.array([(plus["gap"]-minus["gap"])/(2*step)])
                        if not same_edges:
                            reasons.append(f"Active full-gap edges changed for {prefix}.")
                        mask = np.array([min(v["sep"][0, 0], v["sep"][1, 1]) > simple_tol
                                         for v in (base, minus, plus)])
                    else:
                        analytic = base["grad"][axis]
                        finite_difference = (plus["f"]-minus["f"])/(2*step)
                        mask = np.all(np.array([v["sep"] for v in (base, minus, plus)]) > simple_tol, axis=0)
                    # All requested modes must be checked: silently dropping a degenerate
                    # mode would not certify the preregistered entire pointwise test.
                    if not mask.all():
                        reasons.append(f"Not all requested modes have a reliable simplicity mask for {prefix}.")
                    error = _relative(analytic, finite_difference, floor)
                    if not np.isfinite(list(error.values())).all() or error["max_relative"] > grad_tol:
                        reasons.append(f"Finite-difference derivative guard failed for {prefix}.")
                    fd_checks.append(dict(full_path=full, cutoff=cutoff, axis=axis, step_um=step,
                                          error=error, all_modes_simple=bool(mask.all()),
                                          edge_indices_unchanged=same_edges))
    measured["finite_difference_checks"] = fd_checks
    counts = [sum(row["full_path"] == full for row in fd_checks) for full in (False, True)]
    measured["point_fd_count"], measured["full_gap_fd_count"] = counts
    if counts != [12, 4]:
        raise AuditProvenanceError("Wrong finite-difference comparison count.")

    refinements, angular = [], []
    for i in range(3):
        left, right = spectra[f"full_d{i}_M20"], spectra[f"full_d{i}_M25"]
        error = _relative(left["f"], right["f"], 1e-30)
        gap_delta = right["gap"]-left["gap"]
        if error["max_relative"] > freq_tol or abs(gap_delta) > gap_tol:
            reasons.append(f"Full-path M20-to-M25 refinement guard failed for design {i}.")
        refinements.append(dict(design=i, frequency=error, gap_change_GHz=gap_delta,
                                same_edge_indices=left["edges"] == right["edges"],
                                gap_gradient=_relative(left["gapgrad"], right["gapgrad"], floor)))
        left, right = spectra[f"base_d{i}_M20"], spectra[f"angular_d{i}"]
        angular.append(dict(design=i, frequency=_relative(left["f"], right["f"], 1e-30),
                            gap_change_GHz=right["gap"]-left["gap"]))
    measured["full_path_cutoff_refinements"] = refinements
    measured["five_point_angular_refinements"] = angular

    coefficient = cases["coefficient_refinement"]
    if (coefficient["basis_cutoff"] != config["coefficient_cutoff"]
            or coefficient["difference_count"] != (4*config["coefficient_cutoff"]+1)**2
            or [r["order"] for r in coefficient["rows"]] != config["coefficient_orders"]
            or config["coefficient_orders"][-2:] != [128, 256]):
        raise AuditProvenanceError("Coefficient quadrature plan mismatch.")
    rows = {r["order"]: r for r in coefficient["rows"]}
    quad_checks = {
        "coefficient_128_to_256_max_absolute": float(rows[256]["previous_order_max_absolute_difference"]),
        "raw_trace_128_max_absolute": float(rows[128]["raw_trace_max_absolute"]),
        "raw_trace_256_max_absolute": float(rows[256]["raw_trace_max_absolute"]),
        "conjugacy_128_max_absolute": float(rows[128]["conjugacy_max_absolute"]),
        "conjugacy_256_max_absolute": float(rows[256]["conjugacy_max_absolute"]),
    }
    measured["quadrature_numerical_checks"] = quad_checks
    for key, value in quad_checks.items():
        if not np.isfinite(value) or value < 0 or value > 1e-12:
            reasons.append(f"Quadrature numerical guard failed: {key}.")
    material = cases["material_refinement"]
    if (material["cutoff"] != config["material_check_cutoff"]
            or material["orders"] != config["material_check_orders"]
            or material["sides_um"] != (np.asarray(config["material_check_side_multipliers"])
                                        *config["a_um"]/np.sqrt(3)).tolist()):
        raise AuditProvenanceError("Material quadrature plan mismatch.")
    material_checks = {key: float(material[key]) for key in (
        "max_absolute_eta_difference", "frobenius_eta_difference", "relative_frobenius_eta_difference")}
    measured["material_agreement_diagnostics"] = material_checks
    measured["material_agreement_scope"] = "Finite diagnostics only; no newly fitted material tolerance."
    if not all(np.isfinite(v) and v >= 0 for v in material_checks.values()):
        reasons.append("Nonfinite or negative material agreement diagnostics.")
    answer["passed"] = not reasons
    return answer
