"""Checkpointed frozen-solver full-path validation. No ML or spectral fitting.

Requires the already installed NumPy/SciPy. Source spectra are used ONLY by
the reporting function, never by the prediction function or eigensolver.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.linalg import eigh
import scipy

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT.parent/"ML files/COMSOL_ML_Handoff_20261008"
OUT = ROOT/"results/energy_pwe_full_validation"
sys.path.insert(0, str(ROOT/"src"))
from invdes.interface_pwe import (
    C, energy_factorized_impermittivity, normal_projector_grid, projector_matrices,
    hz_operator, _require_hermitian,
)
from invdes.metrics import compute_metrics
from audit_pure_pwe import reciprocal_indices, reciprocal_basis, path_coordinates, eta_matrix


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def inputs():
    # Start from the processed data contract; only k coordinates are used
    # during prediction. F is not passed to the prediction function.
    with np.load(ROOT/"data/processed/bands.npz") as data:
        ks = data["k"].copy()
    if len(ks) != 281 or not np.allclose(ks, np.arange(281)*.01+.1):
        raise ValueError("Unexpected path")
    frozen = json.loads((ROOT/"configs/energy_pwe_audit.json").read_text())
    cfg = {"a_um": frozen["a_um"], "cutoff": frozen["cutoffs"][-1],
           "normal_grid": frozen["normal_grids"][0], "arc_order": frozen["arc_order"],
           "eps_background": frozen["eps_background"], "eps_hole": frozen["eps_hole"],
           "n_bands": 2, "k": ks.tolist(), "eigensolver": "scipy.linalg.eigh/evr/subset",
           "numpy_version": np.__version__, "scipy_version": scipy.__version__}
    if (cfg["eps_background"], cfg["eps_hole"]) != (9, 1):
        raise ValueError("Geometry helper supports eps={9,1}")
    paths = [ROOT/"configs/energy_pwe_audit.json", ROOT/"src/invdes/interface_pwe.py",
             ROOT/"scripts/audit_pure_pwe.py", Path(__file__),
             HANDOFF/"design_inventory.csv"]
    cfg["input_hashes"] = {str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    with (HANDOFF/"design_inventory.csv").open(encoding="utf-8-sig", newline="") as f:
        designs = list(csv.DictReader(f))
    if len(designs) != 64 or len(set(d["design_id"] for d in designs)) != 64:
        raise ValueError("Inventory must contain 64 unique designs")
    return cfg, fingerprint, designs


def lowest_two(g, k, tensor):
    operator = hz_operator(g, k, tensor)
    defect = _require_hermitian(operator, "Hz operator")
    # Gershgorin/infinity norm is an upper bound for the Hermitian spectral norm.
    scale = float(np.max(np.sum(abs(operator), axis=1)))
    values, vectors = eigh(operator, subset_by_index=[0, 1], driver="evr", check_finite=False)
    if values[0] <= 0:
        raise ValueError("Nonpositive squared frequency away from Gamma")
    residual = np.linalg.norm(operator@vectors-vectors*values, axis=0)
    relative_residual = float(np.max(residual/np.maximum(abs(values), 1.)))
    if relative_residual > 1e-7:
        raise ValueError(f"Eigenpair residual too large: {relative_residual}")
    f = C*np.sqrt(values)/(2*np.pi*1e9)
    return f, {"hermitian_defect": defect, "relative_eigenpair_residual": relative_residual,
               "backward_residual": float(np.max(residual)/scale)}


def tensor_for(design, cfg):
    a = cfg["a_um"]
    sides = [float(design["L1_um"]), float(design["L2_um"])]
    ids = reciprocal_indices(cfg["cutoff"])
    eta = eta_matrix(ids, a, sides, cfg["arc_order"])
    eps = 10*np.eye(len(ids))-9*eta
    projectors = projector_matrices(ids, normal_projector_grid(a, sides, cfg["normal_grid"]))
    tensor = energy_factorized_impermittivity(eta, eps, projectors)
    return ids@reciprocal_basis(a*1e-6), tensor


def benchmark(cfg, fingerprint, designs):
    """Same frozen operator, compare subset solve with earlier dense results."""
    start = time.perf_counter()
    g, tensor = tensor_for(designs[0], cfg)
    assembly = time.perf_counter()-start
    saved = json.loads((ROOT/"results/energy_pwe_audit/results.json").read_text())
    expected = np.array(saved["designs"][0]["runs"][-1]["energy_GHz"])
    samples, times = [], []
    for index in [1, 2, 4]:
        k = saved["config"]["path_parameters"][index]
        start = time.perf_counter()
        f, diagnostic = lowest_two(g, path_coordinates([k], cfg["a_um"]*1e-6)[0], tensor)
        seconds = time.perf_counter()-start
        times.append(seconds)
        difference = float(np.max(abs(f-expected[index, :2])))
        if difference > 1e-5:
            raise ValueError(f"Subset/dense solver disagreement: {difference} GHz")
        samples.append({"k": k, "seconds": seconds, "max_dense_difference_GHz": difference,
                        "diagnostic": diagnostic})
    result = {"fingerprint": fingerprint, "config": cfg, "assembly_seconds": assembly,
              "samples": samples,
              "estimated_total_hours": (assembly*64+np.mean(times)*281*64)/3600}
    atomic_json(OUT/"benchmark.json", result)
    print(json.dumps(result, indent=2), flush=True)


def predict(cfg, fingerprint, designs, max_designs=None):
    # NO saved eigenfrequency table is read in this function.
    kpoints = path_coordinates(cfg["k"], cfg["a_um"]*1e-6)
    completed_this_run = 0
    for design in designs:
        name = design["design_id"]
        path = OUT/"predictions"/f"{name}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record["fingerprint"] != fingerprint:
                raise ValueError(f"Checkpoint configuration/code changed: {path}")
        else:
            record = {"fingerprint": fingerprint, "design_id": name,
                      "sides_um": [float(design["L1_um"]), float(design["L2_um"])],
                      "frequencies_GHz": [], "diagnostics": [], "elapsed_compute_seconds": 0.}
        offset = len(record["frequencies_GHz"])
        if offset == len(kpoints):
            continue
        print(f"{name}: preparing tensor, resume at k index {offset}", flush=True)
        start = time.perf_counter()
        g, tensor = tensor_for(design, cfg)
        previous = time.perf_counter()
        record["elapsed_compute_seconds"] += previous-start
        for ik in range(offset, len(kpoints)):
            f, diagnostic = lowest_two(g, kpoints[ik], tensor)
            record["frequencies_GHz"].append(f.tolist())
            record["diagnostics"].append(diagnostic)
            if (ik+1) % 10 == 0 or ik+1 == len(kpoints):
                now = time.perf_counter()
                record["elapsed_compute_seconds"] += now-previous
                previous = now
                atomic_json(path, record)
                atomic_json(OUT/"status.json", {"status": "running", "pid": os.getpid(),
                            "design_id": name, "k_points_finished_in_design": ik+1,
                            "k_points_per_design": len(kpoints), "updated_unix": time.time()})
                print(f"{name}: {ik+1}/{len(kpoints)} k points", flush=True)
        report(cfg, fingerprint, designs)
        completed_this_run += 1
        if max_designs is not None and completed_this_run >= max_designs:
            break
    report(cfg, fingerprint, designs)


def report(cfg, fingerprint, designs):
    # Only this reporting stage reads COMSOL frequencies; it cannot change
    # predictions, the operator, configuration, or eigensolver settings.
    source_path = HANDOFF/"saved_eigenfrequencies.csv"
    with source_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    lookup = {}
    for row in rows:
        key = row["design_id"], round(float(row["k"]), 8), int(row["stored_mode_index"])
        if key in lookup:
            raise ValueError(f"Duplicate source record {key}")
        lookup[key] = float(row["frequency_real_GHz"])
    data = np.load(ROOT/"data/processed/bands.npz")
    items, truth_all, pred_all, processed_all, r1_all = [], [], [], [], []
    finished_points = 0
    for design in designs:
        name = design["design_id"]
        path = OUT/"predictions"/f"{name}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        if record["fingerprint"] != fingerprint:
            raise ValueError("Report/checkpoint fingerprint mismatch")
        finished_points += len(record["frequencies_GHz"])
        if len(record["frequencies_GHz"]) != 281:
            continue
        pred = np.array(record["frequencies_GHz"])
        truth = np.array([sorted(lookup[name, round(k, 8), band] for band in [1, 2]) for k in cfg["k"]])
        i = int(np.argmin(abs(data["L1_um"]-float(design["L1_um"]))))
        j = int(np.argmin(abs(data["L2_um"]-float(design["L2_um"]))))
        truth_all.append(truth)
        pred_all.append(pred)
        processed_all.append(data["F"][0, i, j, :, :2])
        r1_all.append(data["F"][1, i, j, :, :2])
        contract = compute_metrics(truth[None], pred[None])
        def features(f):
            lo, hi = int(np.argmax(f[:, 0])), int(np.argmin(f[:, 1]))
            return {"f1_max_GHz": float(f[lo, 0]), "f2_min_GHz": float(f[hi, 1]),
                    "lower_edge_k": cfg["k"][lo], "upper_edge_k": cfg["k"][hi],
                    "signed_path_gap_GHz": float(f[hi, 1]-f[lo, 0]),
                    "mid_GHz": float((f[hi, 1]+f[lo, 0])/2),
                    "direct_K_GHz": float(f[190, 1]-f[190, 0]),
                    "indirect_M_K_GHz": float(f[90, 1]-f[190, 0])}
        items.append({"design_id": name, "metrics": contract,
                      "max_relative_frequency_error_pct": float(np.max(abs(pred-truth)/truth)*100),
                      "COMSOL_features": features(truth), "PWE_features": features(pred),
                      "max_hermitian_defect": max(d["hermitian_defect"] for d in record["diagnostics"]),
                      "max_relative_eigenpair_residual": max(d["relative_eigenpair_residual"] for d in record["diagnostics"])})
    result = {"status": "complete" if len(items) == len(designs) else "partial",
              "verified_structure": "para", "unverified_structure": "r1",
              "completed_geometries": len(items), "requested_verified_geometries": len(designs),
              "computed_k_points": finished_points, "total_k_points": 281*len(designs),
              "config": cfg, "fingerprint": fingerprint,
              "source_spectra_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
              "designs": items}
    if items:
        result["overall_source"] = compute_metrics(np.array(truth_all), np.array(pred_all))
        result["overall_processed_para"] = compute_metrics(np.array(processed_all), np.array(pred_all))
        result["r1_diagnostic_NOT_VALIDATION"] = compute_metrics(np.array(r1_all), np.array(pred_all))
        result["source_processed_max_disagreement_GHz"] = float(np.max(abs(np.array(truth_all)-np.array(processed_all))))
        result["max_relative_frequency_error_pct"] = max(d["max_relative_frequency_error_pct"] for d in items)
        result["gap_sign_disagreements"] = sum((d["COMSOL_features"]["signed_path_gap_GHz"] > 0) !=
                                               (d["PWE_features"]["signed_path_gap_GHz"] > 0) for d in items)
    atomic_json(OUT/"results.json", result)
    lines = ["# Full-path pure-PWE validation", "",
             f'Status: **{result["status"]}**. Completed {len(items)} / {len(designs)} verified geometries.',
             f'Checkpointed k points: {finished_points} / {281*len(designs)}.', "",
             "No ML, training, spectral fitting, or COMSOL execution. Solver settings are frozen.",
             "All existing sampled k points are used; missing Gamma endpoints are not invented.",
             "A sampled-path gap is not a certified full-Brillouin-zone gap.",
             "r1 geometry provenance is unresolved; its diagnostic comparison is NOT physical validation.",
             "Frequency slope is df/dk in GHz per path parameter, not a physical group velocity in m/s.",
             "No COMSOL mesh-convergence study has been performed.", "",
             "## Metrics on completed geometries only", "", "```json",
             json.dumps(result.get("overall_source", {}), indent=2), "```", "",
             "|Design|Frequency MAE GHz|Max relative frequency error %|COMSOL path gap GHz|PWE path gap GHz|",
             "|---|---:|---:|---:|---:|"]
    for d in items:
        lines.append(f'|{d["design_id"]}|{d["metrics"]["pointwise"]["mae_ghz"]:.6f}|'
                     f'{d["max_relative_frequency_error_pct"]:.6f}|'
                     f'{d["COMSOL_features"]["signed_path_gap_GHz"]:.6f}|'
                     f'{d["PWE_features"]["signed_path_gap_GHz"]:.6f}|')
    lines += ["", "Per-geometry band edges, their k locations, midpoint, direct K and indirect M/K gaps,",
              "per-band errors and slope errors are retained in results.json.",
              "Incomplete geometries are excluded from aggregate accuracy metrics.",
              "The full-project tests and acceptance gates are not certified by this report."]
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--max-designs", type=int)
    args = parser.parse_args()
    cfg, fingerprint, designs = inputs()
    if args.benchmark:
        benchmark(cfg, fingerprint, designs)
        return
    if args.report_only:
        report(cfg, fingerprint, designs)
        return
    bench = json.loads((OUT/"benchmark.json").read_text())
    if bench["fingerprint"] != fingerprint:
        raise ValueError("Run benchmark for this exact code/configuration before prediction")
    lock = OUT/"run.lock"
    # Exclusive creation prevents two writers; a stale lock requires inspection.
    with lock.open("x") as f:
        f.write(str(os.getpid()))
    try:
        predict(cfg, fingerprint, designs, args.max_designs)
        status = report(cfg, fingerprint, designs)
        atomic_json(OUT/"status.json", {"status": status["status"], "pid": os.getpid(),
                    "completed_geometries": status["completed_geometries"], "updated_unix": time.time()})
    except Exception as exc:
        atomic_json(OUT/"status.json", {"status": "failed", "error": repr(exc), "updated_unix": time.time()})
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
