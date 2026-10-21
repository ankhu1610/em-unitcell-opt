# Two-hour atlas validation execution plan

## Purpose and guardrails

Turn the exploratory PWE response atlas into a validated proposal tool without
altering the existing multistart experiment, the PWE evaluator, thresholds, or
the separation between PWE and COMSOL data. COMSOL remains a labelled reference
layer and is not used to fit or accept a pure-PWE inverse candidate.

Every map-derived geometry is first evaluated on the unchanged M20 full path,
then evaluated at the same geometry with M25. The existing empirical acceptance
guard is reported unchanged. No contour interpolation itself is treated as an
accepted design.

## Time-boxed work

| Window | Work | Completion evidence |
|---|---|---|
| 0–15 min | Freeze v2 atlas inputs; specify deterministic contour-candidate selection and output identity. | New config, source hashes, unit tests. |
| 15–30 min | Implement a checkpointed validation driver using the unchanged M20/M25 evaluator. | Dry-run candidate plan and preflight checks. |
| 30–90 min | Run four representative candidates, one for each requested target; checkpoint every M20/M25 calculation. | Live status and complete physics records. |
| 90–105 min | Verify candidate records, compute M20/M25 errors and the unchanged refinement guard. | Machine-readable validation summary. |
| 105–115 min | Regenerate the atlas report/map annotations to distinguish proposal-only from M25-validated points. | Updated HTML map and generated report. |
| 115–120 min | Run focused tests, record limitations and make the next decision explicit. | Test output, report links, and a concise handoff. |

## Candidate-selection protocol

1. Fit the existing thin-plate PWE interpolant to the integrity-checked atlas-v2
   samples only.
2. For each target, choose the lowest predicted-residual point on a fixed
   normalized grid that is sufficiently separated from already M25-validated
   multistart solutions and previously selected atlas candidates.
3. Use that point only as the deterministic start for the unchanged bounded
   trust-region PWE solve; do not accept the interpolated value.
4. Perform the independent M25 calculation on the unchanged final M20 geometry.
5. Classify every candidate as success, unresolved local search, or independent
   refinement failure. These are local outcomes, not global-feasibility claims.

## Decision rules after the time box

- If each map-derived candidate survives M25, use the atlas as a proposal tool
  while retaining M25 as the final acceptance step.
- If any candidate misses, preserve it as a negative result and refine samples
  around that contour before trusting the local interpolation.
- If the PWE and COMSOL layers visibly disagree in a decision-relevant region,
  report the disagreement separately; do not calibrate it away in this task.

## Detailed completion checklist

This is a bounded research work window, not permission to keep launching new
physics indefinitely. Independent reporting work can overlap the background
solver. If all deliverables finish early, report the actual elapsed time rather
than using up the budget with redundant calculations.

### A. Preserve and verify existing evidence

- Keep earlier multistart and atlas-v1/v2 result files unchanged.
- Verify frozen source hashes and every complete M20/M25 checkpoint identity.
- Check all saved eigenvalue diagnostics and recompute signed path gaps.
- Replay the two adaptive sampling sequences against the samples available at
  each step, including their original deterministic tie breaks.
- Treat floating-point-distinct records separately in the frozen interpolant,
  but group numerically equivalent coordinates for leakage-aware diagnostics.

### B. Four new inverse candidates

- Select one separated atlas start for each of 7.5, 15, 22.5 and 27.5 GHz.
- Preserve the original trust-region controls and internal tolerance.
- Evaluate the full saved k-path, not an abbreviated trial path.
- Compute M25 at exactly the geometry returned by M20; no target-dependent
  correction or COMSOL calibration is allowed.
- Record search failures and refinement failures without changing thresholds.
- Enforce the physics-worker deadline and preserve partial checkpoints if the
  window is insufficient. A checkpointed timeout is not completion.

### C. Independent acceptance audit

- Replay optimizer arithmetic against saved physical results without rerunning
  PWE, checking the proposed points, accepted/rejected steps and final result.
- Independently recompute M20 error, M25 error, cutoff change and the empirical
  error-plus-refinement guard.
- Check all four prescribed cases, inventory counts, success classifications
  and source/result hashes before publishing a final summary.
- State that the new driver did not freeze implementation hashes or its own
  reviewed gate in its launch manifest. A later audit does not retroactively
  turn that into preregistration; inherited physics is checked against the
  earlier frozen multistart manifest.

### D. Interpolation and COMSOL diagnostics

- Reconstruct pre-evaluation prediction errors for the adaptive atlas batches.
- Report leave-one-record-out and near-duplicate-grouped leave-one-geometry-out
  errors separately, with interior-hull and boundary-extrapolation distinctions.
- Compare signed scalar gaps only on common physical coordinates supported by
  the PWE box and sample hull. Never silently snap the rounded COMSOL endpoints
  onto the slightly different PWE bounds.
- Explain that PWE-interpolant/COMSOL differences combine interpolation and
  model/discretization effects; they are not pure solver error measurements.
- Keep the new candidate checks out of the frozen proposal fit, so their start
  prediction errors remain prospective checks.

### E. Deliverables and quality checks

- Machine-readable audited summary with input hashes and numerical diagnostics.
- Generated Markdown report with all candidate lengths, gaps, errors and caveats.
- New HTML contour map with labelled physical axes, common numeric color scale,
  target-specific contour colors and distinct sample/validation markers.
- Offline static-figure inspection. Browser-layout review is conditional on
  browser policy allowing local artifacts; do not bypass a security denial.
- Focused CPU tests for tampering, same-geometry refinement, optimizer replay,
  deterministic selection, near-duplicate grouping and contour orientation.
- Explicit handoff: completed work, negative findings, remaining limitations
  and the smallest evidence-supported next experiment.

## Awake window and operational limits

The requested Windows system/display wake assertion is bounded to
2026-10-21 18:03:28 IST. It releases automatically and does not change persistent
power settings, lock/security policies or lid-close behavior. Keep the computer
connected to power; shutdown, loss of power or closing the lid can still
interrupt the job. The numerical worker has an earlier deadline to leave room
for audit and reporting. No external uploads, new dependencies or COMSOL runs
are authorized by this plan.
