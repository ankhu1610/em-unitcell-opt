"""Offline model comparison on existing atlas evidence; no physics or retuning.

The running candidate selector stays unchanged. These descriptive diagnostics
can inform a later separately authorized proposal-model choice.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator
from scipy.spatial import Delaunay

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from audit_atlas_validation_v1 import coordinate_groups, error_metrics, file_hash, verify_hashes

MODELS = ("frozen_RBF_no_polynomial", "RBF_linear_polynomial", "piecewise_linear")


def fit(name,points,values):
    if name == "piecewise_linear":
        return LinearNDInterpolator(points,values,fill_value=np.nan)
    if name not in MODELS:
        raise ValueError("Unknown proposal model")
    return RBFInterpolator(points,values,kernel="thin_plate_spline",smoothing=.02,
                           degree=-1 if name == "frozen_RBF_no_polynomial" else 1)


def compare(summary):
    rows = summary["source_observations"]
    points = np.asarray([r["normalized_x"] for r in rows])
    values = np.asarray([r["gap_GHz"] for r in rows])
    groups = coordinate_groups(points)
    axis = np.linspace(0.,1.,81)
    grid = np.asarray([(x,y) for x in axis for y in axis])
    results = []
    for name in MODELS:
        errors, boundary, details = [], [], []
        for group in groups:
            keep = [i for i in range(len(points)) if i not in group]
            query = points[group[0]]
            inside = bool(Delaunay(points[keep]).find_simplex([query])[0] >= 0)
            predicted = float(fit(name,points[keep],values[keep])([query])[0])
            err = predicted-values[group[0]] if np.isfinite(predicted) else None
            if inside:
                if err is None:
                    raise ValueError("Model failed inside the shared training hull")
                errors.append(err)
            elif err is not None:
                boundary.append(err)
            details.append({"normalized_x":query.tolist(),"inside_remaining_hull":inside,
                            "held_out_record_count":len(group),"error_GHz":err})
        model = fit(name,points,values)
        pred = np.asarray(model(grid))
        valid = np.isfinite(pred)
        starts = [float(v) for v in model(summary["atlas_starts_normalized"])]
        if not np.all(np.isfinite(starts)):
            raise ValueError("Nonfinite start prediction")
        results.append({"name":name,"grouped_CV_inside_hull":error_metrics(errors),
                        "grouped_CV_boundary":error_metrics(boundary) if boundary else None,
                        "grouped_CV_rows":details,
                        "grid_finite_count":int(valid.sum()),
                        "grid_range_GHz":[float(pred[valid].min()),float(pred[valid].max())],
                        "grid_count_outside_observed_range":int(((pred[valid]<values.min()-1e-10) | (pred[valid]>values.max()+1e-10)).sum()),
                        "predictions_at_frozen_starts_GHz":starts})
    counts = {r["grouped_CV_inside_hull"]["count"] for r in results}
    if len(counts) != 1:
        raise ValueError("Models were not compared on the same interior held-out groups")
    return {"scope":"Post-launch offline comparison only. No changes to the running worker, frozen proposal surface, physics or acceptance tolerance.",
            "source_M20_records":len(points),"coordinate_groups":len(groups),
            "grouping_tolerance_normalized":1e-8,"models":results,
            "atlas_starts_normalized":summary["atlas_starts_normalized"],
            "COMSOL_used":False,"physics_executed":False,"hyperparameter_search_performed":False,
            "limitations":"Nonuniform adaptive data; grouped leave-one-out is not an independent spatial test set. Boundary extrapolation is reported separately, and piecewise-linear interpolation intentionally returns no prediction outside its hull. Better CV does not certify the 0.05 GHz tolerance. Alternative fits are not deployed by this comparison.",
            "source_snapshot_hashes":summary["source_snapshot_hashes"]}


def report(result):
    lines = ["# Offline atlas interpolant comparison","",result["scope"],"",
             "|Model|Interior held-out groups|MAE GHz|RMSE GHz|Worst absolute GHz|81×81 range GHz|Range excursions|",
             "|---|---|---|---|---|---|---|"]
    for r in result["models"]:
        m=r["grouped_CV_inside_hull"]
        lines.append(f"|{r['name']}|{m['count']}|{m['MAE_GHz']:.6f}|{m['RMSE_GHz']:.6f}|{m['max_absolute_GHz']:.6f}|{r['grid_range_GHz'][0]:.5f} to {r['grid_range_GHz'][1]:.5f}|{r['grid_count_outside_observed_range']}|")
    lines += ["",result["limitations"],"",
              "All fits use the same saved pure-PWE M20 observations. Numerically equivalent coordinates are removed together in each holdout. "
              "Both RBF variants use the existing smoothing value, with no tuning search; only the polynomial tail differs. "
              "No COMSOL values are used. The existing four atlas starts and all validation outcomes are untouched.","",
              "|Frozen start|Existing RBF prediction GHz|Linear-tail RBF prediction GHz|Piecewise-linear prediction GHz|",
              "|---|---|---|---|"]
    for i,start in enumerate(result["atlas_starts_normalized"]):
        v=[r["predictions_at_frozen_starts_GHz"][i] for r in result["models"]]
        lines.append(f"|{start}|{v[0]:.7f}|{v[1]:.7f}|{v[2]:.7f}|")
    return "\n".join(lines)+"\n"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary",type=Path,required=True)
    parser.add_argument("--outdir",type=Path,required=True)
    args=parser.parse_args()
    source=json.loads(args.summary.read_text())
    verify_hashes(source["source_snapshot_hashes"])
    result=compare(source)
    result["implementation_sha256"]=file_hash(__file__)
    verify_hashes(source["source_snapshot_hashes"])
    if args.outdir.exists():
        raise ValueError("Refusing to overwrite an existing comparison")
    args.outdir.mkdir()
    (args.outdir/"comparison.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (args.outdir/"COMPARISON.md").write_text(report(result),encoding="utf-8")
    print(json.dumps([{k:v for k,v in r.items() if k in ("name","grouped_CV_inside_hull","grid_range_GHz")} for r in result["models"]],indent=2))


if __name__ == "__main__":
    main()
