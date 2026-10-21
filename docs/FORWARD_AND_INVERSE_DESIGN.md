# Forward and inverse electromagnetic design

## Purpose and current interpretation

The project asks two related questions. The forward problem predicts a band
diagram from geometric lengths. The inverse problem finds lengths that produce
a requested response. The newest experiments solve a scalar signed band-gap
request within a fixed geometry family. They do not generate arbitrary material
layouts or recover a unique topology from one number.

The maintained workflow contains PWE forward solvers, analytic sensitivities,
and physics-accepted inverse searches. The older learned models and training
pipelines were retired at the user's request. See the [generated results
snapshot](RESULTS_SNAPSHOT.md) for measured outcomes and the [cleanup record](REPOSITORY_PREPARATION.md) for the recovery archive.

## Data and physical conventions

### Processed reference data

`data/processed/bands.npz` provides `F`, `L1_um`, `L2_um` and `k`.
The frequency tensor axes are `(structure, L1 index, L2 index, path index, band)`.
The structure labels are `para` and `r1`; the nominal design grid has eight
values per length, with 281 path samples and two frequency-ordered bands.
Frequencies are in GHz and geometric lengths in micrometres.

The project contract requests float64; loaders should promote numerical work
to float64 rather than infer the stored archive dtype from the contract alone.
Raw CSV exports are read-only. Redundant Parquet/NPY arrays and ML training
pipelines were removed; bands.npz remains for independent reference comparisons.

Validation splits separate whole geometries. The shared split helpers support
the saved-PWE LODO benchmark without training any model.

### Geometry and materials

The current specification gives a hexagonal primitive lattice with
`a = 242.5 µm`, background relative permittivity 9, and air inclusions with
relative permittivity 1. The two inclusions are oppositely oriented equilateral
triangles clipped by concentric disks, with fixed centres. L1 and L2 describe
the parent triangle sides—not the radii, area fractions or arbitrary widths.

The field convention is TE/H_z in the PWE code: in-plane electric field and
out-of-plane magnetic field. COMSOL's naming for the same two-dimensional field
configuration differs; use the fields, not only the TE/TM label, to compare models.
The current specification assumes real, constant, nonmagnetic dielectric data
without loss or PML. The code is not a dispersive-metal or lossy-mode solver.

For the quadrature inverse route, let `b = a / sqrt(3)`. The box is
`L1/b ∈ [1, 1.35]`, `L2/b ∈ [0.65, 1]`. The optimizer uses `x ∈ [0,1]^2` and
`L = lower + width*x`. These precise PWE bounds differ slightly from the rounded
COMSOL grid labels `[140,189] × [91,140] µm`; the audit does not silently snap them.

The later full-path PWE report verifies `para` against its source spectra.
It explicitly calls the `r1` comparison diagnostic, because physical provenance
is unresolved. Identical entries currently written for both structures in the
geometry YAML do not remove that limitation.

### Reciprocal path and reported quantities

The documented path is Γ–M–K–Γ, parameterized from 0 to 3; saved observations
cover 0.10 to 2.90 in steps of 0.01. The missing Γ endpoints and interior of the
Brillouin zone are not invented. Kinks occur at path parameters 1 and 2.

For a calculated spectrum:

```text
lower edge = max_k f1(k)
upper edge = min_k f2(k)
signed path gap = upper edge - lower edge
midpoint = (upper edge + lower edge) / 2
```

Do not replace the signed gap with the minimum direct separation `min(f2-f1)`.
The first quantity can be negative even though the two bands are sorted at each k.
Frequency slope with respect to the path parameter is not automatically group
velocity in metres per second; physical velocity requires the wavevector mapping
and angular-frequency conversion. The numerical metric excludes kink neighbours.

## Forward models

### Scalar H_z plane-wave expansion

For periodic inverse permittivity `eta = 1/epsilon`, the scalar H_z operator has
Fourier matrix entries

```text
A[G,G'] = eta_hat[G-G'] * dot(k+G, k+G')
A h = lambda h
f_GHz = c * sqrt(lambda) / (2*pi*1e9)   # SI wavevectors
```

The dot product is essential; it is not the product of two vector magnitudes.
Reciprocal indices are truncated to `[-M,M]^2`, giving `(2M+1)^2` basis functions.
Changing M changes the discretization, not the number of k samples. Matrix
assembly, units, material coefficients, eigensolves and band selection are all
part of the forward model.

`src/invdes/variational_pwe.py` retains the conforming scalar H_z formulation as
a separate control. Its derivative checks and convergence evidence are not
interchangeable with those of the interface-energy formulation.

### Interface-aware energy formulation

The accepted numerical research also examines interface-normal Fourier
factorization. `src/invdes/interface_pwe.py` preserves both an earlier ordering
and a distinct Hermitian energy construction. The latter should not be described
as an algebraically identical implementation of the earlier non-Hermitian form.

With Toeplitz material matrices, define

```text
B = inverse(T(epsilon))
Delta = T(1/epsilon) - B
P = T(n n^T)
E = blockdiag(B,B) + P* blockdiag(Delta,Delta) P
```

Here `*` denotes conjugate transpose. Contracting this tensor with the Fourier
wavevector components gives the H_z eigen-operator. The implementation checks
material positivity and Hermiticity rather than clipping eigenvalues or fitting
a frequency correction. For homogeneous material, Delta vanishes. At finite
cutoff the truncated projector need not satisfy `P^2=P`, so positivity does not
by itself prove spectral accuracy; refinement remains necessary.

The frozen full-grid forward comparison uses its recorded sampled-normal
projector implementation. Its direct COMSOL frequency/gap errors belong to that
specific configuration. The later fixed radial projector is a separate research
evaluator whose quadrature, derivatives and cutoff refinements were examined
independently.

### Why the fixed radial projector was developed

Nearest-interface normal assignments can switch as side lengths change.
Stabilizing ties alone did not remove the measured derivative nonsmoothness.
The documented geometry scales about fixed centres, permitting a fixed
homothetic radial extension within the specified box. Its projector is independent
of L1/L2, so `dP/dL = 0` for this construction; the material boundaries still move.

`src/invdes/radial_projector_quadrature.py` integrates the radial factor
analytically and angle with Gauss–Legendre quadrature, splitting intervals at
sector and chord/arc transitions. Material-interface integration has its own
arc order. Projector quadrature order, material quadrature order, Fourier cutoff
and k-path sampling are separate convergence controls.

The material sensitivity has the boundary-integral structure

```text
d eta_hat[G]/dL = contrast / cell_area
                 * integral_boundary(v_n * exp(-i G·r) ds)
v_n = (r-centre)·n / L
```

For a simple normalized eigenpair, the discrete sensitivity is
`d lambda/dL = h* (dA/dL) h`. Frequency derivatives follow by the square-root
chain rule. Gap derivatives subtract the sensitivity at the active lower-band
maximum from that at the active upper-band minimum.

Near-degenerate modes or tied extrema can invalidate a single active-edge
gradient. The evaluator records mode separation, edge margins and derivative
reliability. Finite-difference agreement verifies the derivative of a particular
discretization; it is not a continuum accuracy or COMSOL agreement certificate.

## Inverse methods and their distinct evidence

One scalar gap can correspond to many length pairs. Matching a response does
not identify a unique geometry. A nearest observed-grid residual is not a
rigorous lower bound over all continuous geometries.

### Saved-PWE bracket search

`src/invdes/inverse/pure_pwe.py` and `scripts/inverse_pwe.py` use a library of
previously calculated PWE spectra. A saved design may already meet tolerance.
Otherwise the method selects a geometry segment whose endpoint gaps bracket the
target and proposes safeguarded secant steps. Intermediate candidates receive
actual full-path PWE evaluations; the interpolated geometry does not replace
the electromagnetic calculation.

The positive-gap LODO benchmark excludes each target design from its own search
library. Generated candidates are not shared across folds. The recorded
unbracketed case remains a nonmatch in the denominator, not a claim that the
requested gap is physically impossible. This is PWE response recovery, not
independent COMSOL validation of the inverse outputs.

### Anchor-free trust-region inverse

The mathematical inverse route removes the saved-design initialization library.
`physics_trust_region.py` operates on normalized lengths. With residual
`r = gap(x)-target` and reliable gap gradient `j`, its local minimum-norm model is

```text
delta_x = -r*j / dot(j,j)
```

Bounds and a trust radius limit the proposed step. A fresh M20 path calculation
measures the actual residual reduction; the algorithm accepts or rejects the
trial and updates the trust radius. Unreliable gradients or repeatedly rejected
models trigger bounded coordinate polling. Budget exhaustion and unresolved
local searches are retained, not silently relabelled as success or infeasibility.

`QuadratureForward` checkpoints each exact normalized geometry and cutoff under
a scientific fingerprint. The cache key uses exact float identities and the
forward configuration. Saved path eigenpairs, diagnostics, gap edges and
derivative checks are validated before reuse. A run lock protects each experiment
from simultaneous writers; bounded deadlines preserve completed k-point work.

The initial experiment and the multistart experiment have separate directories
and launch reviews. Multistart tests several prescribed starting geometries for
each target. That demonstrates finite-sample starting-point behaviour, not
convergence from every possible start.

### M20 search and unchanged-geometry M25 check

The current research protocol uses M20 for search and M25 for an independent
cutoff check at the same final geometry. It reports all three quantities:

```text
e20   = abs(gap_M20 - target)
e25   = abs(gap_M25 - target)
delta = abs(gap_M25 - gap_M20)
```

The existing empirical guard requires `e20 + delta <= final_tolerance` in addition
to a successful internal search and the M25 target match. The final tolerance
comes from the unchanged config; it is not adjusted after an outcome is known.
M25 is not allowed to optimize a different geometry and then call that an
independent check of the M20 candidate.

The triangle-inequality-style guard is a useful recorded acceptance rule, but
two Fourier cutoffs do not produce a certified bound to the continuum solution.

## Adaptive inverse-response atlas

### Construction and use

The atlas interpolates a scalar map `(L1,L2) -> signed PWE path gap` from saved M20
observations. Its frozen thin-plate RBF uses only PWE samples. The processed
COMSOL field is displayed separately as a reference, never mixed into this fit.

The first adaptive batch expands coverage and emphasizes requested target
contours. The second additionally uses an observed-range-excursion priority on
every third selection. That score is a heuristic: a value outside the sampled
range is not automatically a physical error, and a value inside it is not
automatically accurate. Both sampling sequences are replayed in the audit.

For validation, a deterministic grid search selects an atlas start near each
target while enforcing separation from earlier accepted solutions and other
selected starts. The frozen surface only chooses the initial point. M20 then
corrects the geometry and M25 checks the returned design. New validation points
are not refit into the frozen surface when reporting its prospective start errors.

### What has been checked

The latest audit verifies source/checkpoint hashes, the original sampling
sequences, deterministic start selection, every optimizer decision, and the
unchanged-geometry refinement assessment. It reuses saved physical results;
the audit itself does not run the eigensolver.

The new atlas-validation launch manifest did not freeze its own implementation
hashes or include its own reviewed launch gate. The subsequent audit explicitly
records this limitation. It does not retrospectively claim preregistration.
Inherited physics files are checked against the earlier frozen multistart evidence.

### Interpolation accuracy and the next candidate model

The audit reports prospective errors, leave-one-record-out errors and grouped
leave-one-geometry-out errors. Nearly identical coordinates must be held out
together; otherwise numerical duplicates leak into a holdout. Boundary points
that fall outside the remaining sample hull are distinguished from interior tests.

An offline comparison examined the frozen no-polynomial RBF, a linear-polynomial
RBF at the same smoothing value, and piecewise-linear interpolation. The linear
polynomial improved average grouped interior error on these saved observations,
but its worst error still exceeded the inverse tolerance. It remains a candidate
for a later proposal-model version, not a deployed replacement or acceptance rule.

COMSOL comparisons are made at common physical coordinates inside the PWE box
and sample hull. The resulting discrepancy mixes interpolation and model effects.
It is not the direct forward-solver error from the separate full-path validation.

## Evidence directory map

| Directory under `results/` | Role |
|---|---|
| `energy_pwe_full_validation/` | Frozen full-path `para` PWE versus source COMSOL spectra |
| `pure_pwe_inverse/` | Initial saved-PWE inverse requests |
| `pure_pwe_inverse_lodo/` | Held-out positive-gap saved-library benchmark |
| `pwe_sensitivity_audit/`, `projector_stabilization_v1/` | Derivative/projector investigations, including failures |
| `radial_quadrature_audit_v1/`, `quadrature_full_order_v1/` | Fixed radial Fourier/quadrature checks |
| `variational_pwe_audit_v1/` | Separate scalar variational control |
| `pwe_projector_research_session/` | Consolidated research conclusions and provenance |
| `pure_pwe_math_inverse_v2/` | Initial anchor-free inverse experiment |
| `pure_pwe_multistart_v1/` | Prescribed multistart experiment and independent verification |
| `pure_pwe_response_atlas_v1/`, `pure_pwe_response_atlas_v2/` | Adaptive proposal maps and their M20 samples |
| `pure_pwe_atlas_validation_v1/final/` | Latest candidate audit, map, tests and interpolant comparison |

Keep reports with their JSON and checkpoints. Removing negative experiments or
rewriting old manifests would erase the trail needed to understand the numerical
choices. Later success does not revise an earlier failed gate.

## Practical conclusions

The atlas makes candidate selection inexpensive, but independent physics still
dominates accepted-design latency. Saved-library search, atlas proposal selection and
fresh PWE validation therefore have different runtime meanings. Performance
numbers in the generated snapshot are descriptive measurements on the recorded
machine, not a universal GPU speedup claim.

The next scientifically useful step is an independently held-out spatial test
of the improved proposal model, followed by selective physics checks. Full-zone
gap certification, higher-cutoff convergence, `r1` provenance resolution and
fabrication tolerance studies remain separate work requiring explicit scope.
