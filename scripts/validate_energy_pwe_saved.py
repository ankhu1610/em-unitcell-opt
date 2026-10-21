"""Read-only validation of frozen PWE predictions against existing COMSOL.

No solver imports, solver executions, parameter selection, fitting, or training.
The primary comparison uses the final pre-existing basis and original normal
grid. The pre-existing doubled-grid result is disclosed separately.
"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT.parent/"ML files/COMSOL_ML_Handoff_20261008"


def metrics(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape or not np.all(np.isfinite(prediction)):
        raise ValueError("Invalid predictions")
    error = prediction-truth
    relative = error/np.maximum(abs(truth), 1e-6)*100
    return {
        "count": int(error.size), "mae_GHz": float(np.mean(abs(error))),
        "rmse_GHz": float(np.sqrt(np.mean(error**2))),
        "max_abs_GHz": float(np.max(abs(error))),
        "mean_abs_relative_pct": float(np.mean(abs(relative))),
        "rms_relative_pct": float(np.sqrt(np.mean(relative**2))),
        "max_abs_relative_pct": float(np.max(abs(relative))),
        "signed_mean_GHz": float(np.mean(error)),
    }


def gap_comparison(truth, prediction, ks):
    im, ik = list(ks).index(1.), list(ks).index(2.)
    answer = {}
    for name, low, high in [
        ("direct_K", (ik, 0), (ik, 1)),
        ("indirect_M_K", (ik, 0), (im, 1)),
    ]:
        actual, pred = float(truth[high]-truth[low]), float(prediction[high]-prediction[low])
        answer[name] = {
            "COMSOL_GHz": actual, "PWE_GHz": pred,
            "absolute_error_GHz": abs(pred-actual),
            "absolute_relative_error_pct": abs(pred-actual)/max(abs(actual), 1e-6)*100,
            "mid_frequency_abs_error_GHz":
                float(abs((prediction[high]+prediction[low]-truth[high]-truth[low])/2)),
        }
    return answer


def main():
    # Start from the processed data contract; cross-check recovered source
    # because the processed geometry labels are rounded.
    data_path = ROOT/"data/processed/bands.npz"
    data = np.load(data_path)
    pwe_path = ROOT/"results/energy_pwe_audit/results.json"
    pwe = json.loads(pwe_path.read_text())
    inventory_path = HANDOFF/"design_inventory.csv"
    source_path = HANDOFF/"saved_eigenfrequencies.csv"
    with inventory_path.open(encoding="utf-8-sig", newline="") as stream:
        inventory = list(csv.DictReader(stream))
    with source_path.open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    lookup = {}
    for row in records:
        key = row["design_id"], round(float(row["k"]), 8), int(row["stored_mode_index"])
        if key in lookup:
            raise ValueError(f"Duplicate source sample {key}")
        lookup[key] = float(row["frequency_real_GHz"])
    cfg = pwe["config"]
    available = np.array(cfg["path_parameters"])
    mask = available != 0  # source begins at k=.1; no invented Gamma target
    ks = available[mask]
    result = {
        "scope": "Frozen predictions, three para geometries, first two bands, five shared k points",
        "primary_settings": {"cutoff": cfg["cutoffs"][-1], "normal_grid": cfg["normal_grids"][0]},
        "shared_k": ks.tolist(), "excluded_k": [0.],
        "input_sha256": {str(path.relative_to(ROOT.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in [data_path, pwe_path, inventory_path, source_path]},
        "designs": [],
        "not_evaluated": ["full 281-point path gap", "full Brillouin-zone gap",
                          "group velocity (sparse samples)", "remaining geometries", "r1 provenance",
                          "independent COMSOL mesh convergence"],
    }
    all_truth, all_pred, all_old, all_refined = [], [], [], []
    for design in pwe["designs"]:
        sides = np.array(design["multipliers"])*cfg["a_um"]/np.sqrt(3)
        matches = [d for d in inventory if np.allclose(
            [float(d["L1_um"]), float(d["L2_um"])], sides, rtol=0, atol=1e-8)]
        if len(matches) != 1:
            raise ValueError("Geometry mapping is not unique/exact")
        source_design = matches[0]
        name = source_design["design_id"]
        truth = np.array([sorted(lookup[name, round(float(k), 8), band] for band in [1, 2]) for k in ks])
        i = int(np.argmin(abs(data["L1_um"]-sides[0])))
        j = int(np.argmin(abs(data["L2_um"]-sides[1])))
        indices = [int(np.argmin(abs(data["k"]-k))) for k in ks]
        if not np.allclose(data["k"][indices], ks, rtol=0, atol=1e-8):
            raise ValueError("Processed k alignment mismatch")
        processed = data["F"][0, i, j, indices, :2]
        run = design["runs"][-1]
        if run["cutoff"] != cfg["cutoffs"][-1]:
            raise ValueError("Saved final cutoff disagrees with frozen config")
        pred = np.array(run["energy_GHz"])[mask, :2]
        old = np.array(run["direct_GHz"])[mask, :2]
        refined = np.array(design["refinement_checks"]["refined_grid_first6_GHz"])[mask, :2]
        item = {
            "design_id": name, "sides_um": sides.tolist(),
            "COMSOL_GHz": truth.tolist(), "PWE_GHz": pred.tolist(),
            "metrics": metrics(truth, pred),
            "per_band": [metrics(truth[:, b], pred[:, b]) for b in range(2)],
            "direct_PWE_same_cutoff_metrics": metrics(truth, old),
            "refined_normal_grid_metrics": metrics(truth, refined),
            "source_processed_disagreement": metrics(truth, processed),
            "versus_processed_para_metrics": metrics(processed, pred),
            "gaps": gap_comparison(truth, pred, ks),
            "refined_normal_grid_gaps": gap_comparison(truth, refined, ks),
            "basis_history": [
                {"cutoff": r["cutoff"], "metrics": metrics(truth, np.array(r["energy_GHz"])[mask, :2])}
                for r in design["runs"]],
        }
        result["designs"].append(item)
        all_truth.append(truth)
        all_pred.append(pred)
        all_old.append(old)
        all_refined.append(refined)
    result["overall"] = metrics(all_truth, all_pred)
    result["overall_direct_PWE_same_cutoff"] = metrics(all_truth, all_old)
    result["overall_refined_normal_grid"] = metrics(all_truth, all_refined)
    result["overall_per_band"] = [metrics(np.array(all_truth)[..., b], np.array(all_pred)[..., b]) for b in range(2)]
    out = ROOT/"results/energy_pwe_comsol_validation"
    out.mkdir(parents=True, exist_ok=True)
    (out/"results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Frozen pure-PWE validation against existing COMSOL", "",
             "Validation only: no solver execution, training, fitting, material adjustment, or cutoff selection.",
             f'Primary settings: cutoff {cfg["cutoffs"][-1]}, normal grid {cfg["normal_grids"][0]}.',
             f'Shared k parameters: {ks.tolist()}. Gamma is excluded because COMSOL has no saved Gamma target.',
             "Recovered source geometries are matched by unrounded dimensions; source spectra align with processed para.",
             "Only the first two frequency-sorted bands are compared; no eigenfield-based mode tracking.", "",
             "## Overall frequency agreement", "", "```json", json.dumps(result["overall"], indent=2), "```", "",
             "## Frequency agreement by geometry", "",
             "|Design|MAE GHz|RMSE GHz|Max error GHz|Mean relative error %|Max relative error %|Direct PWE MAE at same cutoff GHz|",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for d in result["designs"]:
        m = d["metrics"]
        lines.append(f'|{d["design_id"]}|{m["mae_GHz"]:.6f}|{m["rmse_GHz"]:.6f}|{m["max_abs_GHz"]:.6f}|'
                     f'{m["mean_abs_relative_pct"]:.6f}|{m["max_abs_relative_pct"]:.6f}|'
                     f'{d["direct_PWE_same_cutoff_metrics"]["mae_GHz"]:.6f}|')
    lines += ["", "## Local band-gap comparisons", "",
              "Direct = f2(K)-f1(K). Indirect M/K = f2(M)-f1(K). These are NOT verified full-path/full-zone gaps.", "",
              "|Design|Gap definition|COMSOL GHz|PWE GHz|Absolute error GHz|Relative error %|",
              "|---|---|---:|---:|---:|---:|"]
    for d in result["designs"]:
        for name, gap in d["gaps"].items():
            lines.append(f'|{d["design_id"]}|{name}|{gap["COMSOL_GHz"]:.6f}|{gap["PWE_GHz"]:.6f}|'
                         f'{gap["absolute_error_GHz"]:.6f}|{gap["absolute_relative_error_pct"]:.6f}|')
    lines += ["", "## Individual frequencies", "",
              "|Design|k|Band|COMSOL GHz|PWE GHz|Signed difference GHz|", "|---|---:|---:|---:|---:|---:|"]
    for d in result["designs"]:
        for k, truth, pred in zip(ks, d["COMSOL_GHz"], d["PWE_GHz"]):
            for b in range(2):
                lines.append(f'|{d["design_id"]}|{k:.2f}|{b+1}|{truth[b]:.6f}|{pred[b]:.6f}|{pred[b]-truth[b]:.6f}|')
    lines += ["", "## Grid sensitivity and baseline", "",
              "Doubled-grid predictions were also generated before this comparison; they are disclosed, not selected by COMSOL error.",
              "```json", json.dumps({"direct_PWE_same_cutoff": result["overall_direct_PWE_same_cutoff"],
                                    "refined_normal_grid": result["overall_refined_normal_grid"]}, indent=2), "```", "",
              "## Limitations", "",
              "Agreement is with saved COMSOL results, not an independently certified continuum solution.",
              "COMSOL mesh error and remaining PWE truncation error may both contribute.",
              "These geometries appeared in earlier diagnostic work: this is not a blind holdout.",
              "No claim is made for all geometries or all k points. Sparse samples cannot certify extrema or group velocity.",
              "The full-path metrics contract is intentionally not evaluated on a five-point subset.",
              "Per-band metrics, processed/source differences, basis history and input hashes are in results.json.",
              "No existing acceptance thresholds or solver files were changed."]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"overall": result["overall"], "designs": [
        {"id": d["design_id"], "metrics": d["metrics"], "gaps": d["gaps"]}
        for d in result["designs"]]}, indent=2))


if __name__ == "__main__":
    main()
