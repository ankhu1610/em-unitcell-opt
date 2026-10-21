# Physics-only maintenance and publication plan

The user explicitly retired the old ML/hybrid plan on 2026-10-21.
This file describes the maintained scope, not a claim of completed future work.
The original plan is preserved in the external cleanup archive.

## T6.2 — Physics-only publication
- Remove ML models, training scripts, old web demos, their tests and redundant
  derived data. Preserve a verified recoverable archive outside the checkout.
- Keep PWE forward, analytic sensitivities, trust-region inverse, saved-library
  inverse, adaptive atlas and independent validation/audit evidence.
- Declare only the numerical physics dependencies plus a pytest test extra.
- Check the entire retained suite in an environment without scikit-learn.
- Verify that protected scientific files retain their original bytes.
- Publish a clear README, equations/workflow guide, JSON-generated results
  summary, reproduction instructions and cleanup/test record.
- Review exact Git contents, secrets and file sizes before commit/push.
- Do not weaken OS security or change acceptance thresholds.

## Future research (requires a new scope decision)
- Independent spatial holdout for the proposed improved atlas interpolant.
- Higher-cutoff convergence and full-Brillouin-zone coverage.
- Fabrication-tolerance sensitivity and unresolved r1 geometry provenance.
- Portable mapping of historical absolute-path manifests without rewriting
  the original records.

These are not prerequisites for accurately publishing the completed, bounded
PWE experiments. Their limitations remain explicit in the documentation.
