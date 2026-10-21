# Atlas-derived inverse-design validation

Audit status: **verified_complete**.

The frozen proposal surface uses 58 integrity-checked, float-distinct M20 sample records. There are 12 earlier accepted M25 cases. The new validation results below are independent of the interpolation fit.

For leakage-aware diagnostics, near-identical coordinates form 56 geometry groups at normalized distance tolerance 1e-8. The frozen fit itself is not changed by this grouping.

## New atlas-derived designs

|Target GHz|Final L1 µm|Final L2 µm|M20 gap GHz|M25 gap GHz|M25 error GHz|M20 error + cutoff change GHz|Outcome|
|---|---|---|---|---|---|---|---|
|7.5|150.6302438|116.0718831|7.5001776|7.4992081|0.0007919|0.0011471|success|
|15|164.5578661|128.3568872|15.0000110|15.0013845|0.0013845|0.0013845|success|
|22.5|166.3879618|97.1196987|22.5000061|22.5068172|0.0068172|0.0068172|success|
|27.5|178.5152240|119.8166304|27.5000248|27.4920886|0.0079114|0.0079611|success|

Combined successes: 4/4. Distinct new full-path evaluations: 12 ({'20': 8, '25': 4}). Saved evaluation work: 34.93 minutes (not an elapsed-wall-time measurement).

|Target GHz|Map prediction at start GHz|Actual M20 at start GHz|Prediction error GHz|M20 search calls|
|---|---|---|---|---|
|7.5|7.501496|7.355397|0.146099|2|
|15|14.992199|14.963142|0.029057|2|
|22.5|22.506260|22.464356|0.041904|2|
|27.5|27.511182|27.573027|-0.061845|2|

Each physical evaluation covers 281 sampled k-points. Recorded path backends: ['cuda']. Worst CPU/device frequency difference: 8.55045e-09 GHz; worst relative eigenpair residual: 5.9043e-09; worst active-edge three-mode difference: 5.23812e-10 GHz.

## What the contour map does and does not establish

The frozen M20 RBF surface proposes starting geometries. Contour lines are **not** validated families of designs. Only the plotted individual M25 markers passed the recorded same-geometry acceptance checks. A negative signed path gap means band overlap; it is not a positive band gap.

Observed M20 range: -0.95585 to 43.62523 GHz. 81×81 interpolation range: -2.41231 to 43.83279 GHz. 0.229% of grid predictions lie outside the observed value range. This range excursion is not, by itself, a measured interpolation error.

|Diagnostic (gap-only)|Count|MAE GHz|RMSE GHz|Worst absolute GHz|
|---|---|---|---|---|
|Leave-one-record-out, all|58|0.716444|2.256587|14.267257|
|Leave-one-record-out, inside remaining hull|54|0.293732|0.565431|2.192636|
|Grouped leave-one-geometry-out, all|56|0.741808|2.296530|14.267257|
|Grouped leave-one-geometry-out, inside remaining hull|52|0.304789|0.576205|2.192636|
|PWE interpolation minus COMSOL reference|49|0.196890|0.296344|0.800277|

Fixed RBF hyperparameters; spatially clustered samples. The legacy leave_one_design_out fields hold out one float-distinct record; grouped metrics remove all coordinates within 1e-8 normalized distance (connected groups) to avoid numerical near-duplicate leakage. Neither is a random/generalization guarantee or confidence interval.
Inside the hull is geometric support, not a guarantee of interpolation accuracy.

Scalar signed path-gap only, at COMSOL grid coordinates inside the PWE box and sample convex hull. PWE is interpolated: differences mix interpolation error and model/geometry/discretization differences. No pointwise-band, midband, or group-velocity validation is claimed.

COMSOL is a reference-only layer. No COMSOL execution, calibration, or fitting of the PWE surface was performed. COMSOL's rounded length grid does not exactly share the PWE box endpoints; comparisons exclude points outside the common box/hull.

## Adaptive-batch prospective errors

These predictions are reconstructed using only samples available before each batch point was evaluated. The batches probe different, adaptively chosen locations, so their errors are not directly comparable test-set scores.

|Batch|New points|Prospective MAE GHz|Prospective worst absolute GHz|
|---|---|---|---|
|pure_pwe_response_atlas_v1|12|10.033691|72.101000|
|pure_pwe_response_atlas_v2|12|2.411755|16.312948|

## Evidence and limitations

The audit checks checkpoint identities and payload hashes, all saved path diagnostics, recomputed signed gaps, the two adaptive sampling sequences, and the atlas-start selection. A completed validation audit additionally replays every new optimizer decision using saved physical values and independently recomputes the same-geometry M20/M25 acceptance guard. The audit launches no physical simulation.

The new driver manifest did not freeze implementation hashes or include its own reviewed gate before launch. This is a post-launch independent audit, not retrospective preregistration. The unchanged inherited physics files are checked against the earlier multistart manifest.

All gaps refer to the existing sampled k-path. Two-cutoff agreement is empirical: no continuum accuracy, full-Brillouin-zone gap, unique inverse, global feasibility, or fabrication robustness is certified.

The geometry family remains the fixed para disk-clipped triangular-hole model. The inverse variables are only L1 and L2; this experiment is not an unrestricted topology search.

## Next decision

Use the atlas for fast initialization and keep full-path M20 plus unchanged-geometry M25 as the acceptance gate. Do not accept interpolated contours at the 0.05 GHz tolerance. If further work is authorized, prioritize locations with measured prospective error and held-out spatial coverage, rather than indiscriminately filling a dense grid.

## Focused verification tests

26 tests run; 0 failures; 0 errors; 0 skipped. These stdlib-unittest CPU checks include checkpoint corruption, unchanged-geometry refinement, deterministic trace replay, proposal selection, near-duplicate grouping and contour-axis orientation. The complete project pytest suite was not run: pytest is absent from the available physics environment. No dependency was installed.

The static SVG figure was inspected offline. Browser-layout QA is unavailable because browser security rejects local file URLs.

## Where time is spent

Selecting all four atlas starts took a median 10.762 ms over 20 local repeats. Fit 58-record frozen RBF and choose all four starts on the 81x81 grid; excludes physics and source-audit I/O. One host under current load, not a controlled hardware benchmark.

|Target GHz|Earlier three starts: M20-call min–max|New atlas start: M20 calls|New M25 check minutes|
|---|---|---|---|
|7.5|4–6|2|5.395|
|15|3–5|2|5.237|
|22.5|3–5|2|5.160|
|27.5|4–6|2|5.166|

This is a descriptive comparison of different starting geometries, not a controlled speedup trial. The expensive stage remains the independent physics check. The atlas amortizes its construction cost over future requests; fast interpolation does not remove the acceptance computation.

## Lower-data-cost alternative explored

Post-launch offline comparison only. No changes to the running worker, frozen proposal surface, physics or acceptance tolerance.

|Interpolant|Grouped interior holdouts|MAE GHz|Worst absolute GHz|
|---|---|---|---|
|frozen_RBF_no_polynomial|52|0.304789|2.192636|
|RBF_linear_polynomial|52|0.040663|0.276783|
|piecewise_linear|52|0.122331|1.368737|

Nonuniform adaptive data; grouped leave-one-out is not an independent spatial test set. Boundary extrapolation is reported separately, and piecewise-linear interpolation intentionally returns no prediction outside its hull. Better CV does not certify the 0.05 GHz tolerance. Alternative fits are not deployed by this comparison.

The linear-polynomial RBF is a candidate for the next proposal-model version, not a replacement deployed in this run. A separately held-out spatial test is still needed before relying on its improved average error. See [the detailed comparison](INTERPOLANT_COMPARISON.md).
