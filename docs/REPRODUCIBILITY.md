# Physics-only installation and reproduction

## Scope

There is no ML training in the maintained forward or inverse workflow.
scikit-learn, pandas, polars, matplotlib, Streamlit and the old ML demo stack
are not required. NumPy/SciPy implement numerical physics and interpolation;
PyTorch is retained for numerical CPU/GPU support. PyYAML reads the geometry
specification; pytest runs checks.

## Standard installation

Run from the root containing pyproject.toml:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m pytest tests/ -v
```

Linux/macOS activation: `source .venv/bin/activate`.
Python 3.10+ is declared; the reviewed installation uses Python 3.12.
The recorded direct versions may require a newer Python than the minimum.
For CUDA, select a compatible official PyTorch build separately; the historical
GPU environment is recorded in `configs/pwe_cuda_requirements.txt`.
Neither a GPU nor a COMSOL licence is required for the test suite.

`requirements.lock` is a direct-dependency snapshot, not a complete
cross-platform/transitive lock. The local review environment reuses the original
physics environment's NumPy/SciPy/PyTorch through a read-only site-packages path;
pytest/PyYAML are installed separately. It has no scikit-learn. This saves a
second large PyTorch download, but is not evidence of a fully isolated
clean-download installation. Review records are under
`results/physics_only_preparation_v1/`.

On this Windows machine, use the established thread settings to avoid competing
Intel OpenMP runtimes:

```powershell
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python -m pytest tests/ -v
```

These settings do not override Windows application-control policies. The old
scikit-learn block is irrelevant to this deliberately physics-only package;
no security settings or blocked binary were changed.

## Read results without rerunning physics

View the final atlas HTML/PNG and audit report under
`results/pure_pwe_atlas_validation_v1/final/`.

```powershell
python scripts/build_repository_snapshot.py
python scripts/build_repository_snapshot.py --check
```

This generator reads saved JSON and records its hashes. It does not rerun or
independently certify the experiments.

For a fresh saved-evidence audit in the original workspace:

```powershell
python scripts/audit_atlas_validation_v1.py --output results/atlas_reaudit.json
```

The output must not exist. The audit replays optimizer decisions and validates
saved checkpoints; it does not calculate new spectra.

## Saved-library inverse

```powershell
python scripts/inverse_pwe.py 20 --device cpu
```

The command uses saved candidates/brackets, then fresh PWE as necessary. It
writes `results/pure_pwe_inverse/target_20_GHz.json` and cache entries.
The older request CLI may overwrite an existing target record. Use an isolated
working copy when repeating a published target and read
[the saved-PWE guide](pure_pwe_inverse.md).

## Anchor-free and atlas inverse

```powershell
python scripts/run_math_pwe_inverse.py --help
python scripts/run_pwe_multistart_v1.py --help
python scripts/run_pwe_atlas_validation_v1.py --help
```

The first two require `--run --acknowledge-exploratory-review`; the atlas uses
`--run --acknowledge-exploratory-validation`. Each requires `--device cpu` or
`cuda` and a future, timezone-aware `--until-utc` deadline.
Review the saved readiness evidence/configuration before acknowledging a run.

These launchers checkpoint/resume fixed experiment directories. They are not
generic commands that silently create new independent experiments. Preserve
published outputs and preregister a separate configuration for a new study.

## Historical manifests and portability

Historical manifests contain absolute Windows paths and exact-byte source
hashes. Some auditors dereference those paths directly. A clone elsewhere
therefore cannot automatically replay every historical audit. Do not rewrite
frozen manifests to hide mismatches. A future portable mapping layer must
validate original bytes and preserve original evidence.

Some research modules import helpers from `scripts/`; supplied root launchers
set their paths. The installed Python package alone is not the complete
research artifact: configs, scripts, processed data and checkpoints are also
needed. Older forward source-data audits may still need the original handoff
outside the repository. The saved spectra remain available for inspection.

Raw CSVs and COMSOL model binaries are local-only. `bands.npz` is retained
as reference data, not a training set for this workflow. No COMSOL execution
was performed as part of cleanup or testing.

## Limits

There is no all-in-one `make reproduce` claim. A full unit/regression test pass,
a saved-evidence audit, fresh M20/M25 evaluations, and a clean-clone reproduction
are distinct achievements. The documentation reports each separately.
