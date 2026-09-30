# PLAN.md — Executable plan for the coding agent (Google Antigravity)

Companion to `AGENTS.md` (rules) and `PHYSICS_INFORMED_INVERSE_DESIGN_PROPOSAL.md` (background). Work **one ticket at a time, in order**. Each ticket has inputs, steps, outputs and acceptance checks. **Gates (G) and human steps (H) are hard stops.**

## 0. Kickoff prompt (paste into Antigravity's Agent Manager, Planning mode)

> Read `AGENTS.md`, `PLAN.md` and `PHYSICS_INFORMED_INVERSE_DESIGN_PROPOSAL.md`. Do ticket T0.1 only. Produce an implementation-plan artifact listing the files you will create or change, then wait for my review before writing code. After each ticket, stop, summarise results from `results/*.json`, and wait. Never call COMSOL. Never invent geometry or material values.

Suggested setup: `AGENTS.md` in the repo root (Antigravity reads it, and `GEMINI.md` overrides it if present), `PLAN.md` beside it, review policy set so the agent asks before moving past each gate.

---

## 1. Goal, scope, success

**Goal.** A fast, differentiable forward model of the band diagram and an inverse-design engine that returns (L1, L2) for a target response, with no COMSOL in the loop, reproducible without a COMSOL licence.

**Model to build (Method 5).** `ω(L1,L2,k) = ω_PWE(L1,L2,k; θ_phys) + Δ(L1,L2,k)` where `ω_PWE` is a differentiable 2-D plane-wave-expansion eigensolver, `θ_phys` is a handful of fitted physical constants, and `Δ` is a small learned residual. Method 2 (soft physics losses on the MLP) is built first as a cheap baseline and ablation.

**Out of scope.** FNO/DeepONet (no field data), INN/flow models (2-parameter design space, grid search is exact), adjoint FDTD/Meep (wrong problem type: eigenvalue, not driven), any new COMSOL sweeps beyond the small blind set in H3.

## 2. What this plan changes relative to the proposal (and why)

These are corrections or open checks; each is handled by a ticket. Do not silently "fix" them elsewhere.

| # | Proposal statement | Issue | Handled in |
|---|---|---|---|
| 1 | TE equation uses `\|k+G\|\|k+G'\|` | For H-polarisation the operator is `η(G−G')·(k+G)·(k+G')` (dot product). The two differ unless k+G ∥ k+G'. E-polarisation is a different form (below). | T3.2 |
| 2 | Method 4 is "independent, no training data" | PWE needs ε of each region, the exact inclusion shape, polarisation, cell and k-path. None of these are verified. | H1, T3.4, T3.5 |
| 3 | Band range 250–530 GHz | The earlier full-pass report gave ~28–531 GHz. Re-check. | T0.2 |
| 4 | Zero group velocity at k = 1.0, 2.0 | True only for the direction perpendicular to a zone-boundary face; at a **path corner** the along-path slope can be non-zero. | T2.1 |
| 5 | Monotonicity `∂ω/∂L ≤ 0` ("high-index filling") | Assumes inclusions are high-index. Unverified. | T2.1 |
| 6 | The 6.26 GHz OOD residual is a surrogate/physics failure | For an *arbitrary* target curve, part of the residual may be irreducible: the target may not lie on the 2-parameter design family. No model fixes that. | T0.3 |
| 7 | `eigh` autograd "exact" | Backward through `eigh` produces NaN at (near-)degenerate eigenvalues (they occur at high-symmetry k). Use `eigvalsh` for eigenvalue-only losses and test. | T3.3 |
| 8 | PWE solves 281 k in 5–15 ms | Plausible on GPU, unverified; on CPU expect ~seconds. | T3.6, T5.2 |
| 9 | R1 needs 8–12 off-grid verified designs; also "3–5 COMSOL runs only" | These conflict. Resolved in H3 with a tiered default. | H3 |
| 10 | `r1` LODO MAE 11.66 GHz (3.7%) vs `para` 0.69 GHz | A 17× gap suggests a data or indexing problem (band swaps, mode-count anomalies), not just model weakness. | T0.4 |

## 3. Repository layout (target; if the repo already differs, keep existing paths and map them in T0.1)

```
AGENTS.md  PLAN.md  Makefile  pyproject.toml  requirements.lock
configs/   geometry_spec.yaml (HUMAN)  criteria.yaml  exp_*.yaml
data/      raw/ (RO, ignored)  processed/bands.npz  blind/
src/invdes/  utils.py data.py splits.py metrics.py kpath.py
             baselines/{nn.py,poly.py,gp_pca.py}
             pinn/{mlp.py,losses.py}
             physics/{geometry.py,pwe.py,calibrate.py}
             multifidelity/{delta_gp.py,model.py}
             inverse/{spec.py,screen.py,refine.py,engine.py}
scripts/  tests/  results/ (json)  reports/ (md, generated)  figures/  artifacts/ (tables, weights)
```

`make` targets: `setup`, `test`, `aggregate` (needs raw data, optional), `baselines`, `priors`, `pinn`, `pwe-verify`, `calibrate`, `multifidelity`, `inverse-test`, `benchmark`, `figures`, `reproduce` (= everything from `data/processed/` on CPU).

---

## Phase 0 — Ground truth and diagnostics (no modelling yet)

### T0.1 Repo inventory and environment
- **Do:** list existing modules for aggregation, GP-on-PCA, MLP, inverse search, splits, metrics. Write `reports/T0.1_inventory.md`: path, function, what it does, what you will reuse, what is missing vs §3. Create `pyproject.toml`, `Makefile` skeleton, `src/invdes/utils.py` (`set_seed`), a passing smoke test.
- **Accept:** `make setup && make test` green; inventory lists the existing GP-on-PCA and inverse engine entry points. **Stop for review.**

### T0.2 Data contract and re-verification of proposal claims
- **Do:** make/validate `data/processed/bands.npz` per the contract in `AGENTS.md`. Script `scripts/verify_claims.py` writes `results/T0.2.json` with: shape and NaN count; f min/max per structure/band; min/max of `f2−f1` (must be ≥ 0); frequency range per band; second-difference of f(k) averaged over designs with the indices of its top peaks (expect ik = 90, 190, i.e. k = 1.0, 2.0); slope of f(k) just left/right of each kink relative to the median slope.
- **Accept:** file matches contract; `f2 ≥ f1` everywhere (else list violations); kink indices confirmed or contradicted; the 250–530 vs 28–531 GHz discrepancy is resolved in the report with actual numbers.
- **Gate G0:** if shape, ordering or kink location contradict the contract, stop.

### T0.3 Reachability-floor diagnostic (is the 6.26 GHz real model error?)
- **Do:** with the **existing** GP-on-PCA and inverse engine, build three target families per structure: (A) true curves of held-out designs (reachable by construction), (B) convex blends `0.5·T_i + 0.5·T_j` of two designs' true curves, (C) analytic user-style targets (e.g. "gap centred at X GHz, width W") fitted to a plausible band shape. For each target record residual RMSE at the engine's optimum. Estimate a **floor** by the minimum RMSE between the target and the *true COMSOL curves of all 64 grid designs* (nearest real design) and, where available, the same on a finer surrogate table.
- **Accept:** `results/T0.3.json` and a short table: residual vs floor for A/B/C. Write one of two conclusions in the report: "residual ≈ floor → target mostly unreachable; physics cannot fix it, redefine the OOD metric" or "residual ≫ floor → model error; continue".
- **H-note:** show the human this table before Phase 4; it decides how success is measured.

### T0.4 Diagnose the `r1` error
- **Do:** with the existing GP-on-PCA LODO, produce an error heatmap over (design × k) per structure and per band; list designs/k where |error| > 5× median; list any (design, k) with a mode count ≠ 2 or a frequency jump > 5× median neighbour step; check for band swaps (f1 of k+1 closer to f2 of k).
- **Accept:** `reports/T0.4.md` states whether `r1` error is concentrated (specific designs/k → data/indexing issue) or diffuse (model limitation). If data issues are found, list exact indices and **stop**.

### H1 — Human intake: physics specification (BLOCKING for Phase 3)
Human fills `configs/geometry_spec.yaml` from the COMSOL `.mph`. The agent must not proceed to T3.x while any field is `TODO`.

```yaml
a_um: 210.01                 # lattice constant used for normalisation
cell: {x_um: 363.75, y_um: 210.01, n_lattice_points_in_cell: TODO}  # 1 or 2 (hex in rectangular supercell?)
lattice: TODO                # hexagonal | rectangular | other
polarization: TODO           # TE (H_z) | TM (E_z) | both-mixed   (what does the COMSOL study solve?)
structures:
  para:
    background: {eps_r: TODO}
    inclusions:              # list; coordinates relative to cell origin, in um
      - {shape: TODO, eps_r: TODO, center_um: [TODO, TODO], size_params: {L1: TODO, L2: TODO}}
  r1:
    background: {eps_r: TODO}
    inclusions: TODO
k_path:                      # how k in [0.1, 2.9] maps to (kx, ky)
  segments: TODO             # e.g. [[G,X],[X,M],[M,G]] with coordinates in units of 2π/cell
  kink_indices: [90, 190]
solver: {n_eigs_requested: TODO, search_around_GHz: TODO, has_PML_or_loss: TODO}
materials_dispersive_or_metal: TODO   # if true, PWE is invalid: report and stop
```
Also provide: which COMSOL sketch dimension is L1 and L2 for each structure.

---

## Phase 1 — Benchmark harness and baselines (reproduce the known numbers first)

### T1.1 Splits, metrics, pre-registered criteria
- **Do:** implement `splits.py` (LODO per structure; checkerboard by parity of `i1+i2`; leave-one-edge-out for each of the 4 edges; hooks for the blind set), `metrics.py` per `AGENTS.md`, and commit `configs/criteria.yaml` (below). Unit-test splits (no design in both train and test; all 64 covered) and metrics (hand-computable synthetic case).
- **Accept:** tests pass; `criteria.yaml` committed **before** any Phase 2–4 run.

```yaml
# configs/criteria.yaml  (human may edit BEFORE T2.x; frozen afterwards)
priors: {min_fraction_holding: 0.95}
pwe_gate_H2: {proceed_if_rel_mae_below: 0.05, marginal_if_below: 0.15}
pwe_gradcheck: {dtype: float64, max_rel_err: 1.0e-4}
pwe_convergence: {ref_M: 15, max_rel_diff_at_chosen_M: 0.005}
multifidelity:
  r1_lodo_mae_ratio_vs_gp_pca_max: 0.5
  para_lodo_mae_ratio_vs_gp_pca_max: 1.10
  edge_out_mae_ratio_vs_gp_pca_max: 0.8
inverse:
  lodo_roundtrip_param_err_um_max: 0.5
  reachability_factor: 3.0
blind:
  mae_ratio_vs_lodo_max: 1.5
  gap_edge_err_ghz_max: 2.0
```

### T1.2 Baselines on identical splits
- **Do:** wrap existing code as `baselines/{nn.py, poly.py, gp_pca.py}` with one interface `fit(train_idx) → predict(L1, L2) → (281, 2) GHz`. Nearest-neighbour (in normalised L1, L2), polynomial degree 2 and 3 per (k, band), GP-on-PCA.
- **Accept:** `make baselines` writes `results/T1.2.json` for LODO, checkerboard, edge-out, all metrics. LODO MAE for GP-on-PCA reproduces the stated 0.69 GHz (para) and 11.66 GHz (r1) within 5%; otherwise stop and report the discrepancy.

---

## Phase 2 — Method 2: soft physics priors on the MLP (quick baseline and ablation)

### T2.1 Verify each prior against the data before using it
- **Do:** `scripts/check_physics_priors.py` → `results/T2.1.json`: (P1) ordering `f2 ≥ f1`; (P2) |∂f/∂k| at each kink, left side and right side separately, relative to median |∂f/∂k|; (P3) sign of `∂f/∂L1` and `∂f/∂L2` per (k, band) from grid finite differences: fraction negative, fraction positive.
- **Rule:** a prior is enabled only if it holds for ≥ `priors.min_fraction_holding` of entries. For P2, enable per side. For P3 use the sign the data supports; if neither sign reaches the threshold, disable.
- **Accept:** a table saying which priors are enabled and why.

### T2.2 Physics-regularised MLP
- **Do:** `pinn/mlp.py`: input `(structure one-hot, L1n, L2n, Fourier(k), segment features)`, output `(f1, gap12)` with `f2 = f1 + softplus(gap12)`. `pinn/losses.py`: Huber data loss; derivative-matching loss on k (excluding kink neighbours); optional enabled priors from T2.1. Choose λ from `{0, 0.01, 0.1, 1}` using the checkerboard split only, then report LODO. 3 seeds.
- **Accept:** `results/T2.2.json` with: MLP (no priors), MLP + priors, for both structures. Report honestly whether priors help. No gate; this is a baseline and an ablation row.

---

## Phase 3 — Differentiable PWE (Method 4)

**Precondition: H1 complete.**

### T3.1 Geometry → Fourier coefficients
- **Do:** `physics/geometry.py` with `fourier_eps(G, params)` and `fourier_inv_eps(G, params)` returning complex tensors over the set of difference vectors `G−G'`. Closed forms: rectangle (sinc product), ellipse/circle (Bessel `J1` via `torch.special.bessel_j1`), union of shapes by inclusion–exclusion if overlapping. For `η = FT(1/ε)` with piecewise-constant ε use `1/ε` as the piecewise-constant field (same formulas, different contrast).
- **Test:** compare to a brute-force FFT of a 1024×1024 rasterised cell (tolerance 1e-3 relative on the lowest 100 coefficients); compare derivatives w.r.t. L1, L2 to central finite differences.
- **Accept:** tests pass for every shape appearing in `geometry_spec.yaml`. If a shape is not covered, **stop** and ask rather than approximate.

### T3.2 PWE solver
- **Do:** `physics/pwe.py`, float64, batched over k. Plane waves `G = (2π m/ax, 2π n/ay)`, `m, n ∈ [−M, M]`, `N = (2M+1)²`. Work in dimensionless `ω a / (2π c)` internally; convert to GHz with `a_um`.
  - **H_z (TE):** `Σ_G' η(G−G') (k+G)·(k+G') H_G' = (ω/c)² H_G`, Hermitian standard problem.
  - **E_z (TM):** `|k+G|² E_G = (ω/c)² Σ_G' ε(G−G') E_G'`. Generalised problem `A x = λ B x` with `B = ε(G−G')` positive definite: Cholesky `B = L Lᴴ`, solve `L⁻¹ A L⁻ᴴ y = λ y`.
  - Return the lowest `n_bands` (default 2, configurable to 6 for mapping in T3.4) via `torch.linalg.eigvalsh`, sorted ascending.
  - Use the k-path from `kpath.py` built from `geometry_spec.yaml`.
- **Tests (all must pass):** (1) homogeneous medium, both polarisations: `ω = c|k+G|/√ε` for the lowest bands (error < 1e-10); (2) `ω(k) = ω(−k)`; (3) convergence: relative difference vs `M = 15` at M ∈ {5, 7, 10, 12}, recorded in `results/T3.2.json`; (4) Hermiticity of the matrices (max asymmetry < 1e-12).
- **Accept:** tests pass; chosen `M` is the smallest meeting `pwe_convergence`.

### T3.3 Gradient verification
- **Do:** `tests/test_pwe_grad.py`: autograd of `ω_n(k)` w.r.t. L1, L2 (and each ε) vs central finite differences (step 1e-4 of the parameter range) at 20 random designs × 10 random k, float64.
- **Accept:** max relative error ≤ `pwe_gradcheck.max_rel_err` away from degeneracies. Separately, evaluate at a k with a known degeneracy (e.g. Γ for a symmetric cell if present): record whether gradients are finite. If `eigvalsh` yields NaN/inf anywhere, add a documented fix (e.g. epsilon-shifted degenerate handling) and test it. Never use `eigh` when only eigenvalues are needed.

### T3.4 Band mapping and k-path check (hypothesis testing, not truth)
- **Do:** with initial `θ_phys` from `geometry_spec.yaml`, compute PWE bands 1–6 at all 281 k for 4 designs. Determine which PWE band indices match COMSOL's two modes (most likely 1 and 2; report otherwise). If the k-path in the spec is uncertain, test the candidate paths listed in the spec and report error for each.
- **Accept:** `results/T3.4.json` with the mapping and path error table. If no mapping gives a plausible match (rel error < 0.30), **stop** (likely wrong polarisation, cell or shape).

### T3.5 Calibrate physical constants and **Gate H2**
- **Do:** `physics/calibrate.py` fits a small `θ_phys` (only ε values the spec marks unknown, plus an optional global frequency scale) by least squares on training designs (Adam or L-BFGS on `ω_PWE − ω_COMSOL`). Inside every LODO fold refit on that fold's training designs. Report PWE-only (no Δ) LODO error, per band, with an error heatmap over (design, k), and compare `∂ω/∂L` from PWE vs finite differences of the COMSOL grid.
- **Gate H2 (human decides):**
  - rel MAE < `proceed_if_rel_mae_below` → go to Phase 4 as planned.
  - between that and `marginal_if_below` → go, but report that Δ will be large and the physics backbone contributes mainly gradients.
  - above → stop. Likely wrong shape/ε/polarisation. Fall back to Phase 2 + GP as the forward model and revisit H1.

### T3.6 Speed benchmark
- **Do:** time PWE forward (and forward + backward) for one design × 281 k at chosen `M`, CPU and (if present) GPU, batch sizes 1, 16, 256 designs.
- **Accept:** `results/T3.6.json`. No target; numbers feed T5.2 design choices.

---

## Phase 4 — Multi-fidelity model (Method 5)

### T4.1 Residual model
- **Do:** residual `R = ω_COMSOL − ω_PWE(θ_phys)` per training design (same k grid). Fit the **existing GP-on-PCA** to `R` (reuse code; keep the learned part small and smooth). `multifidelity/delta_gp.py`: a torch re-implementation of the **posterior mean only** (fixed hyperparameters from the fit) so it is differentiable w.r.t. `(L1n, L2n)`: `Δ(x) = (k(x, X) α) Pᵀ + mean`, where `α` and PCA basis `P` come from the fit. Keep `σ` for uncertainty.
- **Test:** torch mean equals sklearn mean to 1e-8 on random inputs; autograd gradient matches finite differences.

### T4.2 Full model and evaluation
- **Do:** `multifidelity/model.py`: `predict(structure, L1, L2) → (281, 2) GHz` with gradients; nested fitting (θ_phys and Δ refit on train designs of each fold). Evaluate on LODO, checkerboard, edge-out with the full metric set. Ablation rows: PWE only; PWE + Δ (M5); M2 (T2.2); GP-on-PCA; polynomial; nearest neighbour. 3 seeds where stochastic.
- **Accept (pre-registered, from `criteria.yaml`):** M5 LODO MAE ratio vs GP-on-PCA ≤ 0.5 on `r1`, ≤ 1.10 on `para`; edge-out ratio ≤ 0.8. Also record |Δ| amplitude relative to f. **Report pass/fail as-is.** If a criterion fails, do not retune; write the analysis and stop for the human.

---

## Phase 5 — Inverse-design engine

### T5.1 Specification types (`inverse/spec.py`)
- `TargetCurve(ω1(k), ω2(k), weights)` — objective = weighted RMSE over k and bands.
- `GapSpec(center_ghz, min_width_ghz, tolerance)` — objective on `mid` and `path_gap` (definitions in `AGENTS.md`), with hinge penalties.
- Weights default to 0 on the 2 samples either side of each kink.

### T5.2 Cached table screening (`inverse/screen.py`)
- **Do:** `scripts/build_table.py` evaluates the multi-fidelity model on a `G×G` grid per structure (default 101×101) once, saves `artifacts/table_{structure}.npz` (F values, L grids, model hash). Screening scores every table entry against the spec (vectorised), returns the top-`M` distinct local minima (non-maximum suppression in L1, L2; default M = 10).
- **Note:** this is the answer to PWE being too slow for 10⁴–10⁵ live solves. Live PWE is only used in refinement.

### T5.3 Gradient refinement (`inverse/refine.py`)
- **Do:** from each candidate, L-BFGS-B (SciPy) using torch gradients from the live M5 model, box-bounded to the sweep range (L1 ∈ [140, 189], L2 ∈ [91, 140] µm; never extrapolate outside).
- **Accept:** refinement never increases the objective; failures return the screened candidate.

### T5.4 Engine and reachability flag (`inverse/engine.py`)
- `design(spec, structure=None, top_k=5) → list[{structure, L1, L2, predicted_curve, objective, sigma, reachable}]`; both structures searched if `structure=None`.
- `reachable = objective ≤ reachability_factor × (LODO RMSE of the model)`. When false, the result is labelled "not achievable in this design family; closest found".

### T5.5 Inverse evaluation (`make inverse-test`)
1. **Realistic round-trip:** for each of 128 designs, refit the model **excluding** that design, use its true COMSOL curve as the target, run the engine. Report `|ΔL1|, |ΔL2|` (µm) and residual RMSE. Accept: median `|ΔL|` ≤ `lodo_roundtrip_param_err_um_max`.
2. **OOD families A/B/C from T0.3**, comparing engines: (a) GP-on-PCA + grid + L-BFGS-B (existing), (b) M2 MLP, (c) M5. Report residual vs the T0.3 floor. The expected-and-honest outcome is that M5 improves only the model-error part of the residual.
3. **Runtime:** p50/p95 for screening and refinement separately, CPU and GPU.
- **Accept:** `results/T5.5.json` and `reports/T5.5.md`. No speed target, report measured values.

---

## Phase 6 — Blind validation and reproducibility package

### H3 — Human: COMSOL blind set (the only new COMSOL work)
The agent generates `data/blind/blind_design_list.csv` **before** COMSOL is run: committed and hashed (SHA-256 in the file header).

Tiered default, human chooses:
- **Minimum (8 runs):** per structure, 2 Sobol designs inside the L1, L2 box (quasi-random, off-grid) + 2 designs chosen by the inverse engine from two pre-written specs.
- **Recommended (12 runs):** 3 + 3 per structure.
Sobol designs give an unbiased generalisation estimate; inverse-chosen designs test the optimiser where it can exploit model error. Human runs these in COMSOL with the **same study settings and export format** as the original sweep and drops the CSVs in `data/blind/raw/`.

### T6.1 Blind evaluation
- **Do:** aggregate blind CSVs with the existing aggregator into `data/blind/blind_bands.npz`. Evaluate all models (frozen, no refit) on it with the full metric set. For inverse-chosen designs also report predicted vs COMSOL path-gap and mid-gap.
- **Accept:** blind MAE ≤ `blind.mae_ratio_vs_lodo_max` × LODO MAE and gap-edge error ≤ `gap_edge_err_ghz_max`. Report pass/fail.

### T6.2 Reproducibility package (proposal §5)
- `make reproduce` runs, from `data/processed/` only, on CPU: tests → baselines → priors → PINN → PWE verify → calibrate → multifidelity → inverse-test → benchmark → figures → auto-generated `reports/RESULTS.md`.
- Pinned `requirements.lock`; all seeds fixed; trained weights and tables in `artifacts/` with hashes; `README.md` states clone-to-result commands and the expected wall time measured on the reference machine; raw 8 GB CSVs are not in the repo (document how to obtain them, subject to the data owners' permission).
- Figures (script-generated): parity plots (per structure, per model), dispersion overlays (best/median/worst held-out designs), parameter-recovery map over the (L1, L2) plane, error heatmaps, ablation bar chart with seed std.
- **Accept:** a clean clone runs `make reproduce` green with no COMSOL and no GPU.

### H4 — Human: freeze
Human reviews `reports/RESULTS.md`, decides claims and wording. No claims are written by the agent beyond measured numbers.

---

## 4. Decision flow

```
T0.2 fails -> fix data contract, stop
T0.3: residual ≈ floor -> redefine OOD metric (report reachability), keep going
T0.4 finds data issues -> fix indexing before any modelling
H1 incomplete -> Phases 0-2 only
T3.5 above marginal -> drop M5; ship GP (+M2 if it helps); revisit H1
T4.2 criteria fail -> report, do not retune; human decides
T6.1 fails -> add the blind designs to training as a new round; re-run; human decides
```

## 5. Time and effort (rough; agent + reviews)
Phase 0: 1–2 days (mostly waiting on H1). Phase 1–2: 2–3 days. Phase 3: 4–6 days (the risky part). Phase 4: 2–3 days. Phase 5: 3–4 days. Phase 6: 3–4 days plus COMSOL time for H3. Human review gates add latency; the critical path is H1 and H3.
