"""Isolated full-path angular-order check; an explicit bounded deadline is required."""
import os
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "2"
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from invdes.inverse.quadrature_forward import QuadratureForward
from invdes.projector_experiment import DeadlineReached, digest
from invdes.utils import set_seed
from validate_energy_pwe_full import atomic_json


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def verify_hashes(hashes):
    for name, expected in hashes.items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Protected input changed: {name}")


def parse_deadline(text, maximum):
    """No invocation may silently extend the user-authorized research window."""
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    limit = datetime.fromisoformat(maximum.replace("Z", "+00:00"))
    if value.tzinfo is None or limit.tzinfo is None:
        raise ValueError("Deadline must include its timezone")
    if value > limit:
        raise ValueError("Deadline exceeds the authorized session")
    return value.timestamp()


@contextmanager
def exclusive_lock(path):
    """A stale lock needs human/root inspection; never delete another run's lock."""
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"pid": os.getpid(), "created_unix": time.time()}))
    try:
        yield
    finally:
        Path(path).unlink()


def validate_baseline(saved, upstream, config, forward):
    if saved.get("run_id") != content_hash(upstream):
        raise ValueError("Baseline case does not match its immutable manifest")
    result = saved["result"]
    job = result["job"]
    expected = {"name": config["baseline_job"], "kind": "spectrum", "index": 1,
                "cutoff": config["cutoff"], "order": config["baseline_order"],
                "full": True, "gradients": True}
    if job != expected:
        raise ValueError("Unexpected baseline job identity")
    source = upstream["config"]
    for key in ("a_um", "eps_background", "eps_hole",
                "full_path_start", "full_path_step", "full_path_count"):
        if source[key] != forward[key]:
            raise ValueError(f"Baseline physical/path identity differs: {key}")
    if (source["arc_order"] != forward["material_arc_order"]
            or source["spectral_order"] != config["baseline_order"]
            or source["side_multipliers"][1] != config["side_multipliers"]):
        raise ValueError("Baseline quadrature or geometry identity differs")
    expected_sides = np.asarray(config["side_multipliers"])*forward["a_um"]/np.sqrt(3)
    tolerance = config["exploratory_readiness_guards"]["side_identity_absolute_um"]
    if not np.allclose(result["sides_um"], expected_sides, rtol=0., atol=tolerance):
        raise ValueError("Baseline side lengths differ")
    parameters = forward["full_path_start"]+forward["full_path_step"]*np.arange(
        forward["full_path_count"])
    if not np.array_equal(result["path_parameters"], parameters):
        raise ValueError("Baseline path is not the required full sampled path")
    validate_spectrum(result["frequencies_GHz"], result["gap_GHz"], result["edge_indices"],
                      forward["full_path_count"])
    gradient = np.asarray(result["analytic_gap_gradient_GHz_per_um"])
    if gradient.shape != (2,) or not np.all(np.isfinite(gradient)):
        raise ValueError("Baseline gap gradient is invalid")
    return result


def validate_spectrum(frequencies, gap, edges, count):
    f = np.asarray(frequencies, dtype=float)
    if f.shape != (count, 2) or not np.all(np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("Expected a finite positive two-band full path")
    if np.any(f[:, 1] < f[:, 0]):
        raise ValueError("Frequency bands are not sorted")
    expected_edges = [int(np.argmax(f[:, 0])), int(np.argmin(f[:, 1]))]
    if list(edges) != expected_edges:
        raise ValueError("Stored edge indices disagree with the spectrum")
    expected_gap = float(f[expected_edges[1], 1]-f[expected_edges[0], 0])
    if not np.isclose(gap, expected_gap, rtol=0, atol=1e-12):
        raise ValueError("Stored gap disagrees with the spectrum")
    return f


def prepare_protocol(root, config):
    root = Path(root)
    forward_path = root/config["forward_config_source"]
    forward = json.loads(forward_path.read_text())
    if (forward["projector_quadrature_order"] != config["baseline_order"]
            or forward["search_cutoff"] != config["cutoff"]
            or forward["full_path_count"] != 281):
        raise ValueError("Forward configuration no longer matches the preregistered supplement")
    forward["projector_quadrature_order"] = config["comparison_order"]
    upstream_path = root/config["baseline_manifest"]
    baseline_path = root/config["baseline_case"]
    upstream = json.loads(upstream_path.read_text())
    saved = json.loads(baseline_path.read_text())
    baseline = validate_baseline(saved, upstream, config, forward)
    protected = {**upstream["protected_hashes"], **upstream["code_hashes"],
                 str(upstream_path): digest(upstream_path),
                 str(baseline_path): digest(baseline_path)}
    paths = [
        root/"scripts/audit_quadrature_full_order.py",
        root/"configs/quadrature_full_order_v1.json",
        root/"tests/test_quadrature_full_order.py",
        root/"src/invdes/inverse/quadrature_forward.py",
        root/"tests/test_quadrature_forward.py",
        root/"src/invdes/utils.py",
        forward_path,
    ]
    protocol = {"config": config, "effective_forward_config": forward,
                "protected_hashes": protected,
                "code_hashes": {str(path): digest(path) for path in paths},
                "baseline_run_id": saved["run_id"],
                "runtime": {"python": sys.version, "numpy": np.__version__,
                            "scipy": scipy.__version__, "dtype": "float64/complex128",
                            "allowed_backends": ["cpu", "cuda"]}}
    verify_hashes({**protected, **protocol["code_hashes"]})
    return protocol, forward, baseline


def compare_results(baseline, comparison, config):
    """Comparison is reported even if a new exploratory readiness guard fails."""
    guards = config["exploratory_readiness_guards"]
    f0 = validate_spectrum(baseline["frequencies_GHz"], baseline["gap_GHz"],
                           baseline["edge_indices"], len(baseline["path_parameters"]))
    checks = comparison["checks"]
    f1 = validate_spectrum(comparison["frequencies_GHz"], comparison["gap_GHz"],
                           checks["edge_indices"], len(baseline["path_parameters"]))
    if not np.array_equal(baseline["path_parameters"], comparison["path_parameters"]):
        raise ValueError("Comparison path differs from the baseline")
    if not np.allclose(baseline["sides_um"], comparison["sides_um"], rtol=0.,
                       atol=guards["side_identity_absolute_um"]):
        raise ValueError("Comparison geometry differs from the baseline")
    if comparison["cutoff"] != config["cutoff"]:
        raise ValueError("Comparison cutoff differs")
    g0 = np.asarray(baseline["analytic_gap_gradient_GHz_per_um"], dtype=float)
    g1 = np.asarray(comparison["gradient_GHz_per_um"], dtype=float)
    if g0.shape != (2,) or g1.shape != (2,) or not np.all(np.isfinite([g0, g1])):
        raise ValueError("Finite two-length gap gradients are required")
    frequency_difference = np.abs(f1-f0)
    gradient_difference = np.abs(g1-g0)
    gradient_relative = gradient_difference/np.maximum(
        np.maximum(abs(g0), abs(g1)), guards["derivative_absolute_floor_GHz_per_um"])
    metrics = {
        "frequency_max_absolute_GHz": float(np.max(frequency_difference)),
        "frequency_max_relative": float(np.max(
            frequency_difference/np.maximum(abs(f0), abs(f1)))),
        "gap_change_GHz": float(comparison["gap_GHz"]-baseline["gap_GHz"]),
        "gap_max_absolute_GHz": float(abs(comparison["gap_GHz"]-baseline["gap_GHz"])),
        "gap_gradient_absolute_GHz_per_um": gradient_difference.tolist(),
        "gap_gradient_relative": gradient_relative.tolist(),
        "gap_gradient_max_relative": float(np.max(gradient_relative)),
        "same_edge_indices": baseline["edge_indices"] == checks["edge_indices"],
    }
    guard_results = {name: metrics[name] <= guards[name] for name in
                     ("frequency_max_absolute_GHz", "gap_max_absolute_GHz",
                      "gap_gradient_max_relative")}
    guard_results.update(
        same_edge_indices=metrics["same_edge_indices"],
        cpu_device_agreement=all(np.isfinite(v) and v <= guards["cpu_device_max_absolute_GHz"]
                                for v in [baseline["cpu_device_max_absolute_GHz"],
                                          checks["cpu_device_max_absolute_GHz"],
                                          checks["edge_three_mode_max_absolute_GHz"]]),
        eigenpair_residuals=all(np.isfinite(v) and v <= guards["relative_eigenpair_residual"]
                               for v in [baseline["max_eigenpair_residual"],
                                         checks["max_relative_eigenpair_residual"]]))
    return {
        "orders": [config["baseline_order"], config["comparison_order"]],
        "cutoff": config["cutoff"], "sides_um": baseline["sides_um"],
        "sampled_path_count": len(f0), "metrics": metrics,
        "gap_GHz": [baseline["gap_GHz"], comparison["gap_GHz"]],
        "analytic_gap_gradients_GHz_per_um": [g0.tolist(), g1.tolist()],
        "edge_indices": [baseline["edge_indices"], checks["edge_indices"]],
        "edge_uniqueness_margins_GHz": [
            baseline["edge_uniqueness_margins_GHz"], checks["edge_uniqueness_margins_GHz"]],
        "baseline_mode_relative_separations": baseline["mode_relative_separations"],
        "comparison_active_mode_relative_separations": checks["active_mode_relative_separations"],
        "comparison_gradient_reliable": comparison["gradient_reliable"],
        "comparison_gradient_reliability_reasons": checks["gradient_reliability_reasons"],
        "baseline_cpu_device_max_absolute_GHz": baseline["cpu_device_max_absolute_GHz"],
        "comparison_cpu_device_max_absolute_GHz": checks["cpu_device_max_absolute_GHz"],
        "baseline_max_eigenpair_residual": baseline["max_eigenpair_residual"],
        "comparison_max_eigenpair_residual": checks["max_relative_eigenpair_residual"],
        "exploratory_readiness_guard_results": guard_results,
        "exploratory_readiness_passed": all(guard_results.values()),
        "comparison_evaluation_key": comparison["evaluation_key"],
        "comparison_evaluation_path": comparison["evaluation_path"],
        "continuum_accuracy_certified": False,
        "production_criteria_changed": False,
        "ml_training": False, "comsol_spectra_used": False, "comsol_fitting": False,
    }


def write_report(output, result):
    """All measured report values come from the machine-readable result."""
    atomic_json(output/"results.json", result)
    saved = json.loads((output/"results.json").read_text())
    metrics = saved["comparison"]["metrics"]
    lines = [
        "# Full-path angular-quadrature supplement", "",
        saved["config"]["scope"], "",
        f"Orders: {saved['comparison']['orders']}; cutoff: {saved['comparison']['cutoff']}.",
        f"Protected and source files verified: {saved['verified_file_count']}.",
        f"Exploratory readiness guards passed: {saved['comparison']['exploratory_readiness_passed']}.",
        "", "| Quantity | Measured change |", "|---|---:|",
        f"| Maximum absolute frequency difference, GHz | {metrics['frequency_max_absolute_GHz']:.12g} |",
        f"| Maximum relative frequency difference | {metrics['frequency_max_relative']:.12g} |",
        f"| Signed gap difference, GHz | {metrics['gap_change_GHz']:.12g} |",
        f"| Maximum relative gap-gradient difference | {metrics['gap_gradient_max_relative']:.12g} |",
        f"| Same active edge indices | {metrics['same_edge_indices']} |", "",
        "This single-geometry check is supplementary evidence for the sampled path; it does not certify",
        "continuum accuracy, a complete Brillouin-zone gap, global inverse reachability, or production promotion.",
        "No ML, COMSOL execution, COMSOL spectral fitting, or historical result modifications were used.",
    ]
    (output/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--until-utc", required=True,
                        help="Timezone-aware deadline no later than the authorized session limit")
    args = parser.parse_args()
    config = json.loads((ROOT/"configs/quadrature_full_order_v1.json").read_text())
    deadline = parse_deadline(args.until_utc, config["deadline_utc"])
    if time.time() >= deadline:
        parser.error("The bounded deadline has already passed; no experiment launched")
    set_seed(0)
    protocol, forward, baseline = prepare_protocol(ROOT, config)
    run_id = content_hash(protocol)
    protected = {**protocol["protected_hashes"], **protocol["code_hashes"]}
    output = ROOT/config["output_directory"]
    output.mkdir(parents=True, exist_ok=True)

    def check_deadline():
        if time.time() >= deadline:
            raise DeadlineReached("Authorized runtime expired; preserve evaluation checkpoint")

    with exclusive_lock(output/"run.lock"):
        manifest = output/"manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != protocol:
                raise ValueError("Inputs changed; keep this output and use a new version")
        else:
            atomic_json(manifest, protocol)
        state = "running"
        atomic_json(output/"status.json", {
            "status": state, "run_id": run_id, "pid": os.getpid(),
            "updated_unix": time.time(), "runtime_deadline_unix": deadline})
        try:
            check_deadline()
            evaluator = QuadratureForward(forward, output, run_id, device=args.device,
                                          deadline_check=check_deadline)
            bounds = np.asarray(forward["side_multiplier_bounds"])
            x = (np.asarray(config["side_multipliers"])-bounds[:, 0])/np.diff(
                bounds, axis=1).ravel()
            comparison = evaluator.evaluate(x, cutoff=config["cutoff"])
            verify_hashes(protected)
            comparison_result = compare_results(baseline, comparison, config)
            result = {"status": "complete", "run_id": run_id, "config": config,
                      "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                      "verified_file_count": len(protected),
                      "passes_exploratory_guard": comparison_result["exploratory_readiness_passed"],
                      "comparison": comparison_result}
            write_report(output, result)
            state = "complete"
        except DeadlineReached:
            verify_hashes(protected)
            state = "deadline_checkpointed"
        except BaseException as exc:
            atomic_json(output/"status.json", {
                "status": "failed", "run_id": run_id, "error": repr(exc),
                "updated_unix": time.time(), "runtime_deadline_unix": deadline})
            raise
        atomic_json(output/"status.json", {
            "status": state, "run_id": run_id, "updated_unix": time.time(),
            "runtime_deadline_unix": deadline})
        print(json.dumps({"status": state, "output": str(output), "run_id": run_id}), flush=True)


if __name__ == "__main__":
    main()
