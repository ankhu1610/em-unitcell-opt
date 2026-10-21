# L1/L2 sensitivity audit

The audit is separate from the validated forward solver and the inverse-design
benchmarks. It uses no learned model, COMSOL spectrum, spectral fit or optimizer.
It hashes the protected solver files and prior results and checks those hashes
after every completed case. All numerical output is under
`results/pwe_sensitivity_audit/`.

Run from `Tushar`:

```powershell
python scripts/audit_pwe_sensitivity.py
python scripts/audit_projector_ties.py
```

The main audit is checkpointed per geometry/cutoff/grid case. The input manifest
prevents mixing results after code, settings or protected files have changed.
`--screen-only` runs only the lower-cutoff geometry checks; the default also
performs the reference-cutoff checks. No new packages are required.

## What is differentiated

With fixed centres and orientations, each disk-clipped triangle is a homothetic
domain: `Omega(L) = centre + L*S`. Its boundary moves with normal velocity
`(r dot n)/L`. Reynolds transport therefore gives

`d/dL integral_Omega exp(-i q.r) dA = integral_boundary exp(-i q.r) (r dot n)/L ds`.

For a straight chord, `r dot n` is the triangle inradius; for a circular arc it
is the clipping-circle radius. Chords are integrated analytically, and arcs
use the same Gauss-Legendre order as the forward material integrals.
The resulting derivative has no finite-difference step.

For the existing permittivities, `E = 10I - 9 eta`, `B = inverse(E)`,
`Delta = eta - B`, and `B' = 9 B eta' B`. The energy matrix is

`W = diag(B,B) + P* diag(Delta,Delta) P`.

The audit differentiates all material factors and uses the Hellmann-Feynman
identity on normalized, simple eigenmodes. Frequencies are in GHz and side
length derivatives in GHz per micrometre.

## Three distinct comparisons

1. Analytic material derivative versus central differences of the independent
   forward boundary-integral formula.
2. Analytic energy derivative versus PWE frequency differences while holding
   the interface projectors fixed. This verifies the material/eigenvalue chain
   on that branch, not the full geometry-dependent solver.
3. The same derivative versus frequency differences after fully rebuilding
   both material and normal projectors at each perturbed geometry.

An additional diagnostic uses central projector secants in the complete energy
chain. It is explicitly not an independent analytic projector derivative.

## Why projector switching matters

For this self-similar geometry, each individual chord/arc normal candidate at
a fixed spatial sample is independent of side length. The selected candidate,
however, depends on which primitive and periodic image is nearest.

Nearest-interface assignments may change across medial regions. At tied
distances, floating-point ordering can also select different normals under
tiny geometry perturbations. Thus, a fixed-assignment derivative does not
automatically provide a numerically stable derivative of the actual solver.
`audit_projector_ties.py` records one reproducible periodic-image example.
It does not claim to classify every changed grid point.

## Interpretation limits

- The protocol is fixed in `configs/pwe_sensitivity_audit.json`; the existing
  relative gradient tolerance is read without modification from `criteria.yaml`.
- Frequency checks cover selected k-points; gaps computed from those points are
  explicitly gap proxies, not the full 281-point gap used in validation.
- The reference cutoff is tested on the middle geometry, not every geometry.
- The relative separation screen detects near-degenerate base modes. It does
  not certify absence of crossings throughout a finite perturbation interval.
- Small-step agreement, finite-step accuracy, and convergence with normal-grid
  or Fourier-basis resolution are different questions.
- A failed fully rebuilt derivative check is not proof that continuum Maxwell
  sensitivities fail. Numerical interface representations must be audited.
- No gradient-based optimizer should be enabled on the strength of the
  fixed-projector control alone. Any change to projector handling would require
  a separate implementation and renewed forward validation.
