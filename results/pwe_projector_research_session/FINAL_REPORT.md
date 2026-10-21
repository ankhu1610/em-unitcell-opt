# Bounded pure-PWE mathematical research

Fixed homothetic radial extension with direct Fourier quadrature is the leading differentiable research implementation. Stable-nearest still has assignment-switch nonsmoothness. Production remains unchanged.

Completed projector jobs: 120; direct-quadrature jobs: 61; variational-control jobs: 34.
Reverified unchanged protected historical/input files: 326.
No ML, COMSOL execution, COMSOL fitting or production replacement.

## Derivative evidence

Direct-quadrature pointwise checks: 12; all pass: True; worst relative error: 2.99582558e-05.
Full-path gap checks: 4; worst relative error: 1.81342642e-07; existing tolerance: 0.0001.
All tested full-gap active edges unchanged: True.
Fixed radial projectors remove the moving assignment dependence. Stable-nearest tie handling alone does not remove genuine assignment changes.

## Full-path Fourier-cutoff refinement

|Geometry|L1 um|L2 um|M20 gap GHz|M25 gap GHz|Gap change GHz|Maximum frequency change GHz|
|---|---|---|---|---|---|---|
|0|147.007812|98.0052082|7.86507951|7.86381813|-0.00126137605|0.0316652484|
|1|161.008556|112.005952|15.4197916|15.4217312|0.00193953953|0.0196853409|
|2|182.009672|133.007068|27.2553751|27.2440837|-0.0112914213|0.0535872315|

These are convergence differences, NOT errors against COMSOL or certified continuum error bounds.
Only the documented sampled path and representative geometries were checked.
Frequency/gradient cutoff convergence must not be confused with fixed-cutoff finite-difference agreement.

## Mathematical direction for the inverse

The clipped triangles scale about fixed centers. A fixed nearest-center cell and radial interface-normal extension therefore give a projector independent of both side lengths: dP/dL = 0 within the documented size box.
Material Fourier sensitivities still include the moving physical boundary: d eta_G/dL = (eta_air - eta_background)/A times the boundary integral of v_n exp(-i G dot r), where v_n = (r-c) dot n / L.
Differentiate the same Hermitian energy operator used by the forward solver. For a simple normalized eigenpair, d lambda/dL = h* (dA/dL) h; convert this to frequency and subtract the active lower-edge sensitivity from the active upper-edge sensitivity.
For target residual r = gap(L)-target and normalized-coordinate gradient j, the unconstrained minimum-norm linearized step is delta_x = -r j / (j dot j). The prototype enforces the geometry box and a trust region, verifies each trial by fresh PWE, and falls back to coordinate polling when active-edge derivatives are unreliable.
This is numerical root finding through electromagnetic equations, not training. One requested gap generally admits multiple length pairs; the minimum-norm local step is a selection rule, not a uniqueness proof.

## Quadrature and interface evidence

Direct Fourier integration avoids a spatial normal raster. Angular convergence and Fourier cutoff remain separate controls.
Detailed order-by-order coefficient, material and frequency diagnostics are retained in final_summary.json.
Supplementary full-path order check passed: True.
Maximum frequency change: 1.16519061e-09 GHz; gap change: -1.28466127e-11 GHz.

## Remaining work

- An anchor-free trust-region kernel, checkpointed evaluator and numerical diagnostics are prepared; no inverse physics experiment was launched in this session. The CLI is fail-closed pending angular-report fingerprint validation and comparison recomputation before future launch.
- Higher-cutoff and wider-geometry derivative convergence, genuinely independent off-grid inverse requests, and full-Brillouin-zone edge searches remain.
- Finite-basis finite-difference agreement verifies derivatives of that discretization, not continuum or COMSOL accuracy.

## Tests and limitations

stable_nearest: 0/16 parameter checks passed all tested finite-difference steps.
homothetic_radial: 16/16 parameter checks passed all tested finite-difference steps.
Targeted tests: 120; failures: 0; errors: 0; skipped: 0.
Default combined import of NumPy/SciPy and PyTorch aborted with duplicate Intel OpenMP Error #15. Isolated minimal reproduction confirmed it; sequential MKL avoids the conflicting MKL OpenMP load. No persistent environment changes, dependencies, or KMP_DUPLICATE_LIB_OK bypass.
Targeted pure-PWE and inverse regression suites, not the full historical ML test suite.

The bounded wake helper log and exact source hashes are saved in final_summary.json.
