# Physics-only repository preparation

## User-requested scope change

On 2026-10-21 the user explicitly retired ML training and learned inverse
models. The maintained package is now pure PWE: forward eigensolves,
sensitivities, saved-library and trust-region inverse algorithms, adaptive
atlas proposals, and their audits. This is an intentional product-scope change,
not a claim that the old ML suite passed.

## Removed and preserved

Removed from the checkout: GP/PCA and polynomial/NN baselines, PINN/MLP and
hybrid residual models, their training and learned-inverse code/tests, old
web demos, duplicate legacy tree, obsolete raster demo code, old ML reports,
redundant Parquet/NPY arrays, and unrelated COMSOL automation scripts.

A hash-verified recoverable archive is outside Git at:

`../repository_archives/physics_only_20261021/`

It contains `removed/`, `before_edit/`, and `removed_manifest.json`.
The manifest records original relative paths, sizes and SHA-256 hashes.
Do not blindly copy it over newer work; selectively restore an item to its
original relative path after reviewing conflicts. Committed old files also
remain recoverable from Git history. No raw exports were edited.

Retained: `data/processed/bands.npz`, complete PWE checkpoints, negative PWE
investigations, numerical configs, code/test files required by provenance
manifests, and final visualizations. The shared split helpers remain because
the saved-PWE LODO benchmark uses them; they do not train a model.
`criteria.yaml` retains legacy sections because its original bytes are frozen
in physics manifests. Those sections do not enable any ML code.

Ignored local-only files such as raw CSVs, COMSOL models, virtual environments,
scratch caches and logs are not part of the Git publication candidate.

## Dependency and test boundary

Runtime dependencies are NumPy, SciPy, PyTorch and PyYAML; pytest is the test
extra. There is no scikit-learn requirement or import in maintained source.
PyTorch is used as a numerical backend, not for network training.

The whole `tests/` directory is the maintained physics suite. New regression
checks enforce the no-ML source/dependency boundary. The two earlier stale
geometry tests were migrated to independent clipped-triangle area integration
and analytic energy-gradient checks; their original text is in the external
archive. Production physics and frozen acceptance thresholds were not edited.

The complete maintained suite passed: 304 tests and 5 subtests, with no failures,
errors or skips. The environment has no scikit-learn, pandas, polars or matplotlib;
`pip check` found no broken requirements. A fresh saved-evidence atlas audit
verified all four cases and 793 source paths without launching new physics.
See the [review summary](../results/physics_only_preparation_v1/review_summary.json),
[full-suite JUnit result](../results/physics_only_preparation_v1/pytest.xml),
[atlas audit](../results/physics_only_preparation_v1/atlas_reaudit.json), and
[cleanup inventory](../results/physics_only_preparation_v1/cleanup_inventory.json).

The initial full run found one stale-package-metadata failure, now preserved in
`initial_stale_metadata.xml`. Archiving the ignored `src/invdes.egg-info` resolved
it; no assertion or solver was weakened. The installed package metadata already
declared the physics-only requirements. No earlier subset pass is substituted
for the final full-suite result.

## Scientific integrity and publication

Cleanup first checked removals against recorded SHA-256 path maps and rejected
any overlap with protected evidence. Historical absolute-path limitations and
post-launch atlas-audit limitations remain documented; they are not retroactively
fixed by cleanup. The current README makes no neural-model or continuum claim.

The Windows scikit-learn trust block was not bypassed or disabled. The retired
ML test failure and support note were archived as history; the new
package has no reason to import that blocked component.

All retained tests and the saved-evidence audit passed. The publication candidate
was scanned for common credential/private-key patterns with no matches; this
is a limited pattern scan, not a comprehensive security audit. No individual
candidate file exceeded the Git hosting size limit. Existing whitespace in
the frozen geometry specification is preserved rather than changing its hash.
Review the staged Git contents, then commit and push normally.
The current remote is `ankhu1610/em-unitcell-opt`; no force push is authorized.
