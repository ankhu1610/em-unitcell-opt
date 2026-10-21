"""Generate a labelled scientific contour map and Markdown from audited JSON."""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from audit_atlas_validation_v1 import file_hash, verify_hashes
from invdes.inverse.response_atlas_v1 import fit_pwe_surface, load_comsol_reference, ResponseAtlasExperiment

TARGET_COLOURS = {7.5:"#ffea00",15.:"#ff5a38",22.5:"#00efff",27.5:"#ff87ff"}


def contour_segments(values, level):
    """Piecewise-linear contours on a fixed triangulated grid, including saddles.

    Output coordinates are (column, row); row increases with physical L2.
    Each quad uses the bottom-left to top-right diagonal. This is a plotting
    approximation only and is never consulted by the candidate selector.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or min(values.shape) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Finite two-dimensional contour grid required")
    segments = []
    for iy in range(values.shape[0]-1):
        for ix in range(values.shape[1]-1):
            p = [(ix,iy),(ix+1,iy),(ix+1,iy+1),(ix,iy+1)]
            for triangle in ((0,1,2),(0,2,3)):
                crossing = []
                for a,b in zip(triangle,triangle[1:]+triangle[:1]):
                    va, vb = values[p[a][1],p[a][0]]-level, values[p[b][1],p[b][0]]-level
                    if (va < 0 <= vb) or (vb < 0 <= va):
                        t = -va/(vb-va)
                        crossing.append((p[a][0]+t*(p[b][0]-p[a][0]),
                                         p[a][1]+t*(p[b][1]-p[a][1])))
                if len(crossing) == 2 and crossing[0] != crossing[1]:
                    segments.append(tuple(crossing))
    return segments


def pixel(x, left, top=100., width=440., height=330.):
    return left+width*x[0], top+height*(1-x[1])


def report(summary):
    d = summary["surface_diagnostics"]
    lines = ["# Atlas-derived inverse-design validation", "",
             f"Audit status: **{summary['status']}**.", "",
             f"The frozen proposal surface uses {summary['source_M20_count']} integrity-checked, float-distinct M20 sample records. "
             f"There are {summary['prior_M25_validated_count']} earlier accepted M25 cases. "
             "The new validation results below are independent of the interpolation fit.", "",
             f"For leakage-aware diagnostics, near-identical coordinates form {d.get('coordinate_group_count', summary['source_M20_count'])} geometry groups "
             "at normalized distance tolerance 1e-8. The frozen fit itself is not changed by this grouping.", "",
             "## New atlas-derived designs", "",
             "|Target GHz|Final L1 µm|Final L2 µm|M20 gap GHz|M25 gap GHz|M25 error GHz|M20 error + cutoff change GHz|Outcome|",
             "|---|---|---|---|---|---|---|---|"]
    for r in summary["cases"]:
        lines.append(f"|{r['target_gap_GHz']:g}|{r['sides_um'][0]:.7f}|{r['sides_um'][1]:.7f}|{r['gap_M20_GHz']:.7f}|{r['gap_M25_GHz']:.7f}|{r['error_M25_GHz']:.7f}|{r['error_plus_refinement_GHz']:.7f}|{r['outcome']}|")
    if not summary["cases"]:
        lines += ["", "**New candidate validation is still pending. No new design is accepted in this preview.**"]
    else:
        lines += ["", f"Combined successes: {summary['successful_cases']}/{len(summary['cases'])}. "
                  f"Distinct new full-path evaluations: {summary['distinct_evaluations']} "
                  f"({summary['evaluations_by_cutoff']}). Saved evaluation work: {summary['evaluation_seconds']/60:.2f} minutes "
                  "(not an elapsed-wall-time measurement).", "",
                  "|Target GHz|Map prediction at start GHz|Actual M20 at start GHz|Prediction error GHz|M20 search calls|",
                  "|---|---|---|---|---|"]
        for r in summary["cases"]:
            lines.append(f"|{r['target_gap_GHz']:g}|{r['atlas_prediction_at_start_GHz']:.6f}|{r['actual_M20_at_start_GHz']:.6f}|{r['prediction_error_at_start_GHz']:.6f}|{r['search_calls']}|")
        lines += ["",f"Each physical evaluation covers {summary['sampled_k_count_per_evaluation']} sampled k-points. "
                  f"Recorded path backends: {summary['point_backends']}. "
                  f"Worst CPU/device frequency difference: {summary['worst_cpu_device_difference_GHz']:.6g} GHz; "
                  f"worst relative eigenpair residual: {summary['worst_relative_eigenpair_residual']:.6g}; "
                  f"worst active-edge three-mode difference: {summary['worst_edge_three_mode_difference_GHz']:.6g} GHz."]
    lines += ["", "## What the contour map does and does not establish", "",
              "The frozen M20 RBF surface proposes starting geometries. Contour lines are **not** validated families of designs. "
              "Only the plotted individual M25 markers passed the recorded same-geometry acceptance checks. "
              "A negative signed path gap means band overlap; it is not a positive band gap.", "",
              f"Observed M20 range: {d['observed_M20_range_GHz'][0]:.5f} to {d['observed_M20_range_GHz'][1]:.5f} GHz. "
              f"81×81 interpolation range: {d['interpolated_grid_range_GHz'][0]:.5f} to {d['interpolated_grid_range_GHz'][1]:.5f} GHz. "
              f"{100*d['grid_fraction_outside_observed_value_range']:.3f}% of grid predictions lie outside the observed value range. "
              "This range excursion is not, by itself, a measured interpolation error.", "",
              "|Diagnostic (gap-only)|Count|MAE GHz|RMSE GHz|Worst absolute GHz|",
              "|---|---|---|---|---|"]
    diagnostics = [("Leave-one-record-out, all",d["leave_one_design_out_error"])]
    if "leave_one_design_out_inside_hull_error" in d:
        diagnostics.append(("Leave-one-record-out, inside remaining hull",d["leave_one_design_out_inside_hull_error"]))
    if "grouped_leave_one_geometry_out_error" in d:
        diagnostics += [("Grouped leave-one-geometry-out, all",d["grouped_leave_one_geometry_out_error"]),
                        ("Grouped leave-one-geometry-out, inside remaining hull",d["grouped_leave_one_geometry_out_inside_hull_error"])]
    diagnostics += [("PWE interpolation minus COMSOL reference",d["PWE_interpolant_minus_COMSOL_gap"])]
    for name,m in diagnostics:
        if m:
            lines.append(f"|{name}|{m['count']}|{m['MAE_GHz']:.6f}|{m['RMSE_GHz']:.6f}|{m['max_absolute_GHz']:.6f}|")
    lines += ["",d["loo_note"],d["convex_hull_note"], "",d["comparison_note"], "",
              "COMSOL is a reference-only layer. No COMSOL execution, calibration, or fitting of the PWE surface was performed. "
              "COMSOL's rounded length grid does not exactly share the PWE box endpoints; comparisons exclude points outside the common box/hull.", "",
              "## Adaptive-batch prospective errors", "",
              "These predictions are reconstructed using only samples available before each batch point was evaluated. "
              "The batches probe different, adaptively chosen locations, so their errors are not directly comparable test-set scores.", "",
              "|Batch|New points|Prospective MAE GHz|Prospective worst absolute GHz|", "|---|---|---|---|"]
    for r in summary["source_batches"]:
        m = r["prospective_error"]
        lines.append(f"|{r['version']}|{m['count']}|{m['MAE_GHz']:.6f}|{m['max_absolute_GHz']:.6f}|")
    lines += ["", "## Evidence and limitations", "",
              "The audit checks checkpoint identities and payload hashes, all saved path diagnostics, recomputed signed gaps, "
              "the two adaptive sampling sequences, and the atlas-start selection. A completed validation audit additionally "
              "replays every new optimizer decision using saved physical values and independently recomputes the same-geometry M20/M25 acceptance guard. "
              "The audit launches no physical simulation.", "", summary["audit_limitation"], "",
              "All gaps refer to the existing sampled k-path. Two-cutoff agreement is empirical: no continuum accuracy, "
              "full-Brillouin-zone gap, unique inverse, global feasibility, or fabrication robustness is certified.", "",
              "The geometry family remains the fixed para disk-clipped triangular-hole model. The inverse variables are only "
              "L1 and L2; this experiment is not an unrestricted topology search.", "",
              "## Next decision", "",
              "Use the atlas for fast initialization and keep full-path M20 plus unchanged-geometry M25 as the acceptance gate. "
              "Do not accept interpolated contours at the 0.05 GHz tolerance. If further work is authorized, prioritize "
              "locations with measured prospective error and held-out spatial coverage, rather than indiscriminately filling a dense grid."]
    if "focused_tests" in summary:
        t = summary["focused_tests"]
        lines += ["", "## Focused verification tests", "",
                  f"{t['tests_run']} tests run; {t['failures']} failures; {t['errors']} errors; {t['skipped']} skipped. "
                  "These stdlib-unittest CPU checks include checkpoint corruption, unchanged-geometry refinement, "
                  "deterministic trace replay, proposal selection, near-duplicate grouping and contour-axis orientation. "
                  "The complete project pytest suite was not run: pytest is absent from the available physics environment. "
                  "No dependency was installed.", "",
                  "The static SVG figure was inspected offline. Browser-layout QA is unavailable because browser security rejects local file URLs."]
    if "proposal_selection_benchmark" in summary:
        b = summary["proposal_selection_benchmark"]
        lines += ["", "## Where time is spent", "",
                  f"Selecting all four atlas starts took a median {1000*b['median_seconds']:.3f} ms over {b['repeats']} local repeats. "
                  +b["scope"], "",
                  "|Target GHz|Earlier three starts: M20-call min–max|New atlas start: M20 calls|New M25 check minutes|",
                  "|---|---|---|---|"]
        for r in summary["cases"]:
            counts = [p["total_evaluation_calls"] for p in summary["prior_validated_cases"] if p["target_gap_GHz"] == r["target_gap_GHz"]]
            lines.append(f"|{r['target_gap_GHz']:g}|{min(counts)}–{max(counts)}|{r['search_calls']}|{r['M25_elapsed_seconds']/60:.3f}|")
        lines += ["", "This is a descriptive comparison of different starting geometries, not a controlled speedup trial. "
                  "The expensive stage remains the independent physics check. The atlas amortizes its construction cost over future requests; "
                  "fast interpolation does not remove the acceptance computation."]
    if "interpolant_comparison" in summary:
        c = summary["interpolant_comparison"]
        lines += ["", "## Lower-data-cost alternative explored", "",c["scope"],"",
                  "|Interpolant|Grouped interior holdouts|MAE GHz|Worst absolute GHz|",
                  "|---|---|---|---|"]
        for r in c["models"]:
            m=r["grouped_CV_inside_hull"]
            lines.append(f"|{r['name']}|{m['count']}|{m['MAE_GHz']:.6f}|{m['max_absolute_GHz']:.6f}|")
        lines += ["",c["limitations"],"",
                  "The linear-polynomial RBF is a candidate for the next proposal-model version, not a replacement deployed in this run. "
                  "A separately held-out spatial test is still needed before relying on its improved average error. "
                  "See [the detailed comparison](INTERPOLANT_COMPARISON.md)."]
    return "\n".join(lines)+"\n"


def render_html(summary, forward, reference):
    axis = np.linspace(0.,1.,81)
    xx,yy = np.meshgrid(axis,axis,indexing="xy")
    queries = np.column_stack((xx.ravel(),yy.ravel()))
    pwe = fit_pwe_surface(summary["source_observations"],.02)(queries).reshape(xx.shape)
    l1,l2 = reference["L1_um"],reference["L2_um"]
    physical = np.column_stack((l1[0]+queries[:,0]*(l1[-1]-l1[0]),l2[0]+queries[:,1]*(l2[-1]-l2[0])))
    comsol = RegularGridInterpolator((l1,l2),reference["signed_path_gap_GHz"])(physical).reshape(xx.shape)
    minimum = float(np.floor(min(pwe.min(),comsol.min())/5)*5)
    maximum = float(np.ceil(max(pwe.max(),comsol.max())/5)*5)
    bounds = np.asarray(forward["side_multiplier_bounds"])*forward["a_um"]/np.sqrt(3.)

    def panel(values,left,title,xrange,yrange):
        svg = [f'<text x="{left}" y="65" class="panel-title">{title}</text>']
        n = values.shape[0]-1
        for j in range(n):
            for i in range(n):
                val = float(np.mean(values[j:j+2,i:i+2]))
                col = ResponseAtlasExperiment._colour(val,minimum,maximum)
                x,y = pixel((i/n,(j+1)/n),left)
                svg.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{440/n+.05:.2f}" height="{330/n+.05:.2f}" fill="{col}"/>')
        for target,col in TARGET_COLOURS.items():
            paths = []
            for a,b in contour_segments(values,target):
                x1,y1 = pixel((a[0]/n,a[1]/n),left)
                x2,y2 = pixel((b[0]/n,b[1]/n),left)
                paths.append(f'M{x1:.2f},{y1:.2f}L{x2:.2f},{y2:.2f}')
            data = ''.join(paths)
            svg.append(f'<path d="{data}" stroke="#101820" stroke-width="3.5" fill="none"/>')
            svg.append(f'<path d="{data}" stroke="{col}" stroke-width="1.6" fill="none"><title>{target:g} GHz proposal/reference contour</title></path>')
        svg.append(f'<rect x="{left}" y="100" width="440" height="330" fill="none" stroke="#23303c"/>')
        for t in np.linspace(0.,1.,5):
            x,y = pixel((t,t),left)
            svg.append(f'<text x="{x}" y="452" text-anchor="middle" class="tick">{xrange[0]+t*(xrange[1]-xrange[0]):.2f}</text>')
            svg.append(f'<text x="{left-9}" y="{y+4}" text-anchor="end" class="tick">{yrange[0]+t*(yrange[1]-yrange[0]):.2f}</text>')
        svg.append(f'<text x="{left+220}" y="479" text-anchor="middle" class="axis">L1 (µm)</text>')
        svg.append(f'<text x="{left-57}" y="265" transform="rotate(-90 {left-57} 265)" text-anchor="middle" class="axis">L2 (µm)</text>')
        return ''.join(svg)

    content = ''
    for i,(target,col) in enumerate(TARGET_COLOURS.items()):
        lx = 315+160*i
        content += f'<path d="M{lx},22h30" stroke="#101820" stroke-width="6"/><path d="M{lx},22h30" stroke="{col}" stroke-width="3"/><text x="{lx+40}" y="27" class="axis">{target:g} GHz</text>'
    content += panel(pwe,85,"Pure-PWE M20 proposal surface",bounds[0],bounds[1])
    content += panel(comsol,675,"COMSOL reference — para",[l1[0],l1[-1]],[l2[0],l2[-1]])
    for row in summary["source_observations"]:
        x,y = pixel(row["normalized_x"],85)
        content += f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.3" fill="#fff" stroke="#111" stroke-width=".7"><title>M20 sample: {row["gap_GHz"]:.6f} GHz</title></circle>'
    for a in l1:
        for b in l2:
            x,y = pixel(((a-l1[0])/(l1[-1]-l1[0]),(b-l2[0])/(l2[-1]-l2[0])),675)
            content += f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2" fill="#fff" stroke="#111" stroke-width=".6"/>'
    for row in summary["prior_validated_cases"]:
        if row["success"]:
            x,y = pixel(row["normalized_x"],85)
            content += f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="none" stroke="white" stroke-width="2"><title>Earlier accepted M25 case: target {row["target_gap_GHz"]:g}, actual {row["gap_M25_GHz"]:.6f} GHz</title></circle>'
    for i,start in enumerate(summary["atlas_starts_normalized"]):
        x,y = pixel(start,85)
        content += f'<path d="M{x-4:.2f},{y-4:.2f}l8,8m-8,0l8,-8" stroke="#fff" stroke-width="2"><title>Atlas start {i+1}</title></path>'
    for row in summary["cases"]:
        x,y = pixel(row["normalized_x"],85)
        colour = "#49ff81" if row["success"] else "#ff3838"
        content += f'<path d="M{x:.2f},{y-7:.2f}l7,7l-7,7l-7,-7Z" fill="{colour}" stroke="#111" stroke-width="1.4"><title>New case: target {row["target_gap_GHz"]:g} GHz, M25 {row["gap_M25_GHz"]:.7f} GHz, {row["outcome"]}</title></path>'
    gradient = ''.join(f'<stop offset="{i}%" stop-color="{ResponseAtlasExperiment._colour(i,0,100)}"/>' for i in range(0,101,10))
    content += f'<defs><linearGradient id="scale">{gradient}</linearGradient></defs><rect x="370" y="525" width="460" height="17" fill="url(#scale)"/>'
    for t in np.linspace(0.,1.,6):
        content += f'<text x="{370+460*t}" y="560" text-anchor="middle" class="tick">{minimum+t*(maximum-minimum):g}</text>'
    content += '<text x="600" y="585" text-anchor="middle" class="axis">Shared signed sampled-path gap (GHz); negative = band overlap</text>'
    content += '<text x="600" y="616" text-anchor="middle" class="tick">Left: small dots = M20 samples; rings = earlier accepted M25 cases; × = new atlas starts.</text>'
    content += '<text x="600" y="637" text-anchor="middle" class="tick">Diamonds: green = new accepted case, red = unsuccessful case. Right dots = COMSOL reference designs.</text>'
    legend = ''.join(f'<span><i style="background:{col}"></i>{target:g} GHz</span>' for target,col in TARGET_COLOURS.items())
    rows = ''.join(f'<tr><td>{r["target_gap_GHz"]:g}</td><td>{r["sides_um"][0]:.7f}</td><td>{r["sides_um"][1]:.7f}</td><td>{r["gap_M25_GHz"]:.7f}</td><td>{r["error_M25_GHz"]:.7f}</td><td>{escape(r["outcome"])}</td></tr>' for r in summary["cases"])
    state = (f'{summary["successful_cases"]}/{len(summary["cases"])} new cases passed the unchanged acceptance checks.'
             if summary["status"] == "verified_complete" else 'PREVIEW: new candidate checks are still pending; no new design accepted here.')
    d = summary["surface_diagnostics"]
    discrepancy = d["PWE_interpolant_minus_COMSOL_gap"]
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Audited inverse-response atlas</title>
<style>body{{margin:0;background:#eef2f5;color:#142333;font:16px/1.5 system-ui,Arial,sans-serif}}main{{max-width:1240px;margin:auto;padding:28px}}h1{{font-size:30px;margin:0 0 8px}}h2{{font-size:21px}}.lead{{max-width:1050px}}.badge{{padding:12px 16px;background:#fff2cc;border-left:4px solid #946200}}.card{{background:white;border:1px solid #d7dfe5;border-radius:10px;padding:18px;margin-top:18px}}svg{{width:100%;height:auto;display:block}}.panel-title{{font:bold 18px system-ui}}.tick{{font:12px system-ui}}.axis{{font:14px system-ui}}.legend{{display:flex;justify-content:center;gap:24px;flex-wrap:wrap}}.legend span{{display:flex;align-items:center;gap:7px}}.legend i{{width:28px;height:5px;border:1px solid #23303c}}.notes{{font-size:14px;color:#3a4b5b}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border-bottom:1px solid #dde3e9;padding:9px;text-align:left}}.scroll{{overflow:auto}}a{{color:#145ca4}}</style></head><body><main>
<h1>Inverse-response atlas: proposal map + checked designs</h1><p class="lead">A fast starting-point map for the four target gaps. The PWE and COMSOL layers remain separate. Smooth contours are interpolation—not accepted families of designs.</p><p class="badge">{state}</p>
<section class="card"><svg viewBox="0 0 1200 660" role="img" aria-label="Two separately labelled signed band-gap contour maps with physical axes, shared numeric color scale and individual checked design markers">{content}</svg>
<p class="notes">Left markers: small white dots = {summary['source_M20_count']} M20 samples; white rings = {summary['prior_M25_validated_count']} earlier accepted M25 cases; × = new atlas starts; green diamonds = new accepted M25 cases; red diamonds = new unsuccessful cases. Overlapping markers may hide individual cases. Right dots = the 64 processed COMSOL designs; the background between them is bilinear interpolation.</p>
<p class="notes">The M20 proposal surface is frozen at atlas v2; new validation points are not refitted into it. Hover over a marker for its recorded gap. Both panels share the color scale but retain their actual, slightly different physical axis bounds.</p></section>
<section class="card"><h2>New independently checked candidates</h2><div class="scroll"><table><thead><tr><th>Target GHz</th><th>L1 µm</th><th>L2 µm</th><th>M25 gap GHz</th><th>Absolute error GHz</th><th>Outcome</th></tr></thead><tbody>{rows or '<tr><td colspan="6">Pending — physics worker remains active.</td></tr>'}</tbody></table></div></section>
<section class="card"><h2>How much should we trust the contours?</h2><p>At {d['common_supported_COMSOL_grid_count']} common supported COMSOL coordinates, the PWE interpolation/reference gap difference has MAE {discrepancy['MAE_GHz']:.3f} GHz and maximum {discrepancy['max_absolute_GHz']:.3f} GHz. This combines interpolation and physical-model differences; it is not pure solver error.</p><p>Use contours to propose a geometry, then run full-path M20 and unchanged-geometry M25. No contour-wide 0.05 GHz guarantee is claimed. A negative signed gap means band overlap.</p><p class="notes">No full-Brillouin-zone, continuum, unique-inverse or fabrication-robustness certificate. <a href="VALIDATION_REPORT.md">Detailed report</a> · <a href="audit_summary.json">Audited data and provenance</a></p></section></main></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary",type=Path,required=True)
    parser.add_argument("--outdir",type=Path,required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    verify_hashes(summary["source_snapshot_hashes"])
    forward = json.loads((ROOT/"configs/pure_pwe_math_inverse_v1.json").read_text())
    ref = load_comsol_reference(ROOT/"data/processed/bands.npz",0)
    artifacts = {"validated_response_atlas.html":render_html(summary,forward,ref),
                 "VALIDATION_REPORT.md":report(summary),
                 "audit_summary.json":json.dumps(summary,indent=2,allow_nan=False)+"\n"}
    if any((args.outdir/name).exists() for name in artifacts):
        raise ValueError("Refusing to overwrite existing report artifacts")
    args.outdir.mkdir(parents=True,exist_ok=True)
    for name,text in artifacts.items():
        (args.outdir/name).write_text(text,encoding="utf-8")
    print(json.dumps({"output":str(args.outdir.resolve()),"files":list(artifacts)}))


if __name__ == "__main__":
    main()
