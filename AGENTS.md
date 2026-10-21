# AGENTS.md — Pure-PWE electromagnetic forward and inverse design

## Current scope
The user retired the ML workflow on 2026-10-21. Maintain only pure plane-wave
expansion (PWE), geometry/material calculations, analytic sensitivities, bounded
inverse search, response-atlas proposals, and their validation/audit tools.
No GP/PCA, neural-network, learned residual, or ML training workflow belongs in
this package. PyTorch is a numerical CPU/GPU backend, not a training requirement.
The original ML plan and removed files have a recoverable external archive.

## Scientific rules
- Never call COMSOL or edit raw exports. Use saved processed reference data.
- Geometry/material authority is configs/geometry_spec.yaml. Do not invent
  physical parameters or silently replace unresolved facts. The later
  validation verifies para; r1 provenance is not established.
- Preserve experiment configurations, thresholds, source hashes and historical
  evidence. Never retune a gate after observing its result.
- Existing manifests include absolute paths and exact-byte hashes. Do not
  normalize or rewrite protected files; new experiments need separate outputs.
- Use float64/complex128 for physics. CPU tests must work without a GPU.
- Report signed sampled-path gaps, not claims of full-zone or continuum gaps.
- Numerical summaries must be generated from saved JSON by scripts. Preserve
  failed PWE experiments; do not present them as successful or omit their role.
- Inverse response matching does not establish unique geometry, unrestricted
  topology optimization, global reachability or fabrication robustness.
- The RBF atlas interpolates PWE samples only and proposes starts; fresh physics
  accepts candidates. Keep COMSOL reference comparisons separate.
- Splits, where needed for saved-library/atlas checks, are by whole geometry,
  never individual k rows.
- Stop for unexplained existing-test failures, failed new scientific gates,
  missing physical specifications or new dependencies requiring approval.

## Engineering
- Package: src/invdes; launchers/audits: scripts; complete maintained suite: tests.
- Declared runtime: NumPy, SciPy, PyTorch and PyYAML. pytest is the test extra.
  Do not add scikit-learn, pandas, polars or ML frameworks.
- Preserve deterministic seed conventions via invdes.utils.set_seed.
- Run make test (or its identical python -m pytest tests/ -v command) before
  every commit. The complete maintained physics suite must pass, not a subset.
- Keep source and acceptance results separate. Use new paths for new runs.
- User changes and untracked research evidence must be preserved. Cleanup
  removals need a recoverable copy and explicit inventory.
- Commit messages use a ticket prefix; T6.2 covers this publication preparation.

## Reference data and metrics
data/processed/bands.npz stores F with shape (2,8,8,281,2), GHz; load as float64.
L1 labels are 140..189 um, L2 labels 91..140 um. The sampled path is
0.10..2.90 at 0.01 increments, with two frequency-sorted bands.
The precise PWE bounds use the physical multipliers in the experiment config,
not silently rounded reference labels.

gap = min_k(f2) - max_k(f1), signed; mid = (min_k(f2) + max_k(f1))/2.
Report pointwise MAE/RMSE/max, gap/mid errors and path-frequency slope errors.
Exclude two neighbors either side of path kinks at indices 90 and 190 for
slope metrics. A path derivative is not automatically SI group velocity.
