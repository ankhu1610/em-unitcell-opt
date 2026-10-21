# Gap-only pure-PWE inverse design

Only design input: positive target band-gap width in GHz.
Output: one pair of parent-equilateral-triangle side lengths L1 and L2, in micrometres,
plus the achieved gap, band edges, error and search status.
The validated disk-clipped `para` construction, materials, lattice and k-path remain fixed.

From the `Tushar` directory, CPU:

```powershell
python scripts/inverse_pwe.py 20
```

Optional existing CUDA environment:

```powershell
& '..\.venvs\pwe-cuda\Scripts\python.exe' scripts/inverse_pwe.py 20 --device cuda
```

Saved-result and held-out tests:

```powershell
& '..\.venvs\pwe-cuda\Scripts\python.exe' scripts/inverse_pwe.py --test-existing --device cuda
```

Results are written under `results/pure_pwe_inverse/`. Full new spectra and
eigenpair diagnostics are stored separately in `evaluations/`. The original
64 validation checkpoints are read-only.

## Algorithm and limitations

- The library contains full-path frozen-PWE outputs only. No COMSOL spectral
  tables, learned surrogates, training, calibration or fitted physics are used.
- A saved candidate within tolerance may be returned immediately.
- Otherwise select a short geometry segment whose endpoint PWE gaps bracket
  the target. Safeguarded secant steps choose intermediate side lengths.
  Every intermediate candidate receives an actual full-path PWE calculation.
  Interpolation is used to choose geometry, not to predict a spectrum.
- The method finds one solution on a segment in two-dimensional parameter
  space. It does not find every solution or guarantee global optimality.
- Bounds are derived from the precise saved geometry dimensions, not rounded
  processed-dataset labels. All new candidates stay within those bounds.
- If no saved endpoints bracket a target, return the closest saved candidate
  with `no_saved_bracket_not_proven_unreachable`. This is not an impossibility
  certificate, nor an exhaustive search of targets beyond the sampled range.
- Success requires a positive achieved gap within the fixed tolerance.
  A zero or negative requested width is rejected: overlap is not a positive gap.
- The signed gap is `min(f2) - max(f1)` over the same 281 sampled k-points.
  Gamma endpoints and the Brillouin-zone interior are not added or certified.
- The matching tolerance is an optimization tolerance, not an error bound
  against experiment or COMSOL. Further convergence studies are separate.
- Search controls and the experiment protocol are in
  `configs/pure_pwe_inverse.json`, fixed before the initial experiments.
- Retrieving saved targets is a sanity check only. The held-out experiment
  excludes the target design from the search library; only its scalar PWE gap
  is supplied. Exact side-length recovery is neither expected nor required.
- Existing GP-based inverse-engine APIs are preserved but not used by this path.

## All-positive-case leave-one-design-out benchmark

```powershell
& '..\.venvs\pwe-cuda\Scripts\python.exe' scripts/benchmark_inverse_pwe_lodo.py --device cuda
```

This evaluates all 61 positive-gap targets using the repository's design-level
LODO splits. Each search receives the other 63 designs, including the three
overlap cases as possible bracketing endpoints. Only the held-out scalar PWE
gap is supplied as a target. Each fold has its own evaluation cache, and no
generated candidates are added to other folds' libraries.

Outputs are in `results/pure_pwe_inverse_lodo/`: `status.json`, `results.json`,
`REPORT.md`, and one checkpoint per completed fold. Unchanged completed fold
files are reused on restart. The manifest rejects changes to benchmark code,
settings or input checkpoints; a run lock prevents concurrent writers.
Execution-error checkpoints are retained rather than silently retried.

The fixed matching tolerance and evaluation budget are unchanged from the
initial experiment. Nonmatches, including targets lacking a saved bracket,
are reported in the denominator. Numerical failures are counted separately;
gap-error aggregates cover all returned candidates, including nonmatches.
Success rates while running use completed cases only and are not final.
This evaluates scalar response recovery, not unique geometry recovery, and
does not certify the older learned-engine parameter-recovery gates.
