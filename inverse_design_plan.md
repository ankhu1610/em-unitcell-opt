# Inverse Design of EM Unit Cells — Full Build Plan

**Goal:** a model that, given design parameters, produces the electromagnetic response (band diagram), and the reverse: given a target response, returns the parameters that achieve it.

Data: COMSOL 2D eigenfrequency sweeps, two structures (`para.csv`, `r1.csv`), 64 designs each (8 × L1 × 8 × L2), 281 k-points per design.

---

## 0. Read this first: corrections to the two reports

I cross-checked the analysis report against the JSON/MD full-pass numbers. Several conclusions in the report do not hold up, and they change the plan.

| # | Report says | What the data actually shows | Consequence |
|---|---|---|---|
| 1 | L1, L2 are **nm** (140–189 nm); "sub-wavelength metamaterial" | Raw values are `1.40E-4 … 1.89E-4` while X/Y extents are 363.75 × 210.01 µm. Raw L is in **metres → L1 = 140–189 µm, L2 = 91–140 µm**. The unit cell is ~364 × 210 µm, and λ₀ at 300 GHz is 1 mm. | Off by 1000×. The cell is wavelength-scale (a/λ ≈ 0.02–0.40), i.e. photonic-crystal regime, not deep-subwavelength. Use µm everywhere. |
| 2 | L1 "likely the lattice constant" | X/Y extents are **identical for all designs** (max 363.75 × 210.01). The cell does not change with L1/L2. 363.75 / 210.011 = 1.7320 = **√3**. | Fixed lattice, likely a triangular/hexagonal lattice drawn as a √3a × a rectangular cell, a ≈ 210 µm (hypothesis, confirm in .mph). L1, L2 are **inclusion dimensions**. Normalise frequency by the fixed `a`, not by L1. |
| 3 | λ is the eigenvalue of ∇×∇×E = λE, f = λ/2π | Re(λ) = f × 10⁹ exactly (28.03 GHz ↔ 2.80E10, r = 1.0). λ is just **the eigenfrequency in Hz**. | Drop the 2π relation. `lambda` adds no information beyond `f` and Im(λ). |
| 4 | "60% of modes are lossy", loss is a key design parameter | Im(λ)/Re(λ) ≤ ~7×10⁻⁶ (Im ≈ 10⁶ Hz vs Re ≈ 3×10¹¹ Hz), median Im = 0, sign flips between files. The 60% figure is counted **per mesh node**, not per mode. | Implied Q ≳ 10⁵. Effectively lossless. Treat Im(λ) as solver residue unless the .mph has PML or lossy materials. **Do not build a loss objective** until E0 shows Im(λ) is smooth in k. |
| 5 | "Flat-band spikes" at 310–320 GHz (para) and 330–340 GHz (r1); different loss profiles between files | Histograms and Im statistics come from the **first 2M rows** of a design-ordered file (2M ≈ 4 of 64 designs, since ~461K rows per design) and are **node-weighted**. Max f in the sample is 494 GHz vs 531 GHz in the full pass. | A mode with more mesh nodes counts more. The spikes may be mesh artefacts. Recompute at mode level over all designs before claiming flat bands or loss differences. |
| 6 | "2–4 bands per k, average 2.5" | Unique frequencies = 35,965 (para), 35,967 (r1) ≈ **2 × 17,984**. That points to exactly ~2 eigenmodes per (design, k). | The 2.5 average is probably from the biased sample. E0 must verify. Plan for **2 bands as the baseline**, with bands 3–4 optional. |
| 7 | "Band gaps, band-gap-to-midgap ratio" as headline features | With ~2 bands on a 1-D k path you can only see the gap **between band 1 and band 2 along the path**, not a complete gap. | Define "path gap" honestly. Higher gaps (2–3, …) are not observable in this data. |
| 8 | Details MD: "L1 constant, L2 has 3 values" | This came from a 500K-row head sample; the JSON full pass says 8 × 8. | Trust the full pass (8 × 8), confirm in E0. |
| 9 | k is "normalised to the Brillouin zone" | k runs 0.10 → 2.90 in 0.01 steps. That is not a BZ-normalised range. It looks like a **path coordinate** (e.g. Γ→X→M→Γ or Γ→M→K→Γ, three segments of length 1, endpoints excluded). | If true, f(k) has **kinks at k = 1 and k = 2**, and the model needs a segment-aware k feature. Check in E0 (second-difference peaks at k = 1, 2). |
| 10 | "Two topologies" (para vs r1) | Only evidence is a different row count/mesh and different frequency range. No geometry was inspected. | Two *structures* is the safe wording. Confirm by plotting mesh nodes (mesh refines around inclusion edges). |

**Also true and useful:** there is **no field data** in the files. Columns are only X, Y (mesh coordinates), L1, L2, k, λ, f. The only spatial information is the mesh, which is useful as a geometry diagnostic, not as an ML input.

---

## 1. What we are building (scope)

**M1: Forward surrogate (given parameters → response).**
Input `(structure, L1, L2, k)` or `(structure, L1, L2)` for the whole curve. Output: f₁(k), f₂(k) (+ optional bands 3–4), with uncertainty. Also derived: path gap, mid-gap, gap ratio, group velocity.

**M2: Inverse design (given target → parameters).**
Input: a spec (e.g. "gap centred at 300 GHz, width ≥ 20 GHz", or a target curve). Output: ranked candidate `(structure, L1, L2)` with predicted response and uncertainty, verified by COMSOL for the top few.

If by "produce" you meant something else (e.g. generating the geometry shape or the field maps), that is not possible with this data: there are no fields and only two scalar geometric parameters.

**Key design decision:** the design space is 2-D (per structure). Inverse design here is a **search problem**, not a generative-modelling problem. A fast forward surrogate + exhaustive search over a fine (L1, L2) grid returns *all* solutions (the one-to-many issue disappears) and is exact and cheap. Tandem nets and cVAEs are deferred to the day the design space has ≥ 3–4 parameters or a shape parameterisation.

---

## 2. Blocking questions (answer from the .mph file, not from guesses)

Open `parametric_sweep_UC.mph` and record:

1. **Geometry:** what are L1 and L2 (hole diameter? pillar width? slot length?), what is the lattice (hexagonal a = 210.01 µm?), materials, and what differs in `r1`?
2. **Study settings:** "Number of eigenfrequencies" and "Search for eigenfrequencies around". If the solver returns the N modes nearest a shift, then band index ≠ physical band number and "bands 3–4" may be an artefact.
3. **k definition:** expressions for kx, ky in terms of `k`, units, and which path k = 0.1 … 2.9 traces. Are k = 1 and k = 2 high-symmetry points?
4. **Loss:** any PML, scattering boundary, or lossy material? This decides whether Im(λ) is physics or noise.
5. **Polarisation / physics:** TE, TM, or mixed? 2D EMW with out-of-plane E or H?
6. **Application and target spec** (gap centre, width, or a specific curve), **accuracy needed** (± GHz), and **fabrication limits** (min feature size, tolerance).
7. **Can more COMSOL runs be launched** (off-grid validation designs, finer grid)? This is the single biggest lever on model quality.

Until answered, defaults used below: 2 bands, path gap between band 1 and 2, target accuracy = **0.5% of f** (≈ 1.5 GHz at 300 GHz) and gap-edge error ≤ 2 GHz, batch + interactive use, 2D is sufficient.

---

## 3. Roadmap

Assumes 1–2 people at 15–20 h/week. Total ≈ **5 weeks**; the critical dependency is COMSOL access in weeks 3–5.

| Phase | Time | Objective | Deliverable | Gate |
|---|---|---|---|---|
| P0 Ground truth | Days 1–2 | Answer §2, fix units, define spec language | `assumptions.md` | Geometry + k-path + solver settings known |
| P1 Aggregate | Days 2–4 | 61M rows → mode table + band tensors | `modes_{para,r1}.parquet`, `F.npy` | **G0:** each file gives 17,984 (design, k) points, mode counts reconcile |
| P2 Validate physics (E0) | Days 4–6 | Test the 10 corrections above on the mode-level data | EDA notebook, band-diagram plots for all 128 designs | Units, k-path, mode counts, Im(λ) status settled |
| P3 Baselines (E1) | Week 2 | Response-surface + interpolation baselines with honest CV | LODO error table | **G1:** baseline error known; defines the bar |
| P4 Surrogates (E2–E4) | Weeks 2–3 | GP-on-PCA and MLP surrogates, ablations | best forward model + calibrated uncertainty | Beats baseline on LODO by ≥ 20% or matches it with uncertainty |
| P5 Off-grid validation (E5) | Weeks 3–4 | 12–16 new COMSOL designs per structure (Sobol in the box) | validation set + active-learning round | **G2:** meets accuracy target off-grid; otherwise add data |
| P6 Inverse (E6–E7) | Weeks 4–5 | Spec → grid search → local refine → verify | `inverse.py`, 3 verified example designs | **G3:** round-trip recovers held-out designs; COMSOL agrees with prediction within tolerance |
| P7 Package | Week 5 | API + small UI + docs | Docker image, README, demo | Forward < 50 ms, inverse < 2 s |

**Resources.** Data pipeline needs a machine with a fast SSD and 16 GB RAM (streaming aggregation of each 4 GB file ≈ 10–30 min). All models are tiny (thousands of parameters, ≤ 20k training rows), so **no GPU is needed**; a laptop is enough. COMSOL licence access is required for P5–P6.

---

## 4. Data preprocessing pipeline

### 4.1 Levels and shapes

| Level | Content | Shape (per file) |
|---|---|---|
| L0 raw | one row per (mode, mesh node) | ~30M × 7 |
| L1 modes | one row per (design, k, eigenmode) | ~36K × 12 (≈ 2 per point, up to 4) |
| L2 band tensor | `F[i1, i2, ik, band]` in GHz, NaN = missing | 8 × 8 × 281 × 4 |
| L3 designs | curve vector + derived features per design | 64 × (562 + ~30) |
| ML tensors | `X = (structure, L1n, L2n)`, `Y = normalised curves` | 128 × 3, 128 × 562 |

### 4.2 Aggregation (tested on a synthetic COMSOL-style file)

```python
# src/aggregate.py
import numpy as np, polars as pl

COLS = ["X", "Y", "L1", "L2", "k", "lam", "f_ghz"]
NUM = r"[+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?"
LAM_RE = rf"^({NUM})(?:({NUM})i)?$"          # COMSOL writes "a" or "a+bi"

def aggregate(path: str, structure: int) -> pl.DataFrame:
    """L0 (~30M node rows) -> L1 (one row per eigenmode). Streams the file."""
    lf = (pl.scan_csv(path, comment_prefix="%", has_header=False, new_columns=COLS,
                      schema_overrides={"lam": pl.Utf8})
          .group_by(["L1", "L2", "k", "lam", "f_ghz"])       # key includes lam string, not just f
          .agg(pl.len().alias("n_nodes")))
    df = lf.collect(engine="streaming")
    return (df.with_columns(
                lam_re=pl.col("lam").str.extract(LAM_RE, 1).cast(pl.Float64),
                lam_im=pl.col("lam").str.extract(LAM_RE, 2).cast(pl.Float64).fill_null(0.0),
                structure=pl.lit(structure, dtype=pl.Int8),
                # raw L is in metres -> um, then snap to the nominal 7 um grid (removes the 1.0000532 offset)
                i1=((pl.col("L1") * 1e6 - 140) / 7).round().cast(pl.Int16),
                i2=((pl.col("L2") * 1e6 - 91) / 7).round().cast(pl.Int16),
                ik=(pl.col("k") * 100).round().cast(pl.Int16) - 10)
            .sort(["structure", "i1", "i2", "ik", "f_ghz"])
            .with_columns(band=pl.col("f_ghz").cum_count()
                              .over(["structure", "i1", "i2", "ik"]).cast(pl.Int8))
            .drop("lam"))

def to_tensor(modes: pl.DataFrame, n_bands=4) -> np.ndarray:
    """L1 -> L2: F[i1, i2, ik, band] (GHz), NaN where the band is absent."""
    F = np.full((8, 8, 281, n_bands), np.nan)
    d = modes.filter(pl.col("band") <= n_bands)
    F[d["i1"].to_numpy(), d["i2"].to_numpy(), d["ik"].to_numpy(),
      d["band"].to_numpy() - 1] = d["f_ghz"].to_numpy()
    return F
```

Notes: check the first data lines for the delimiter before running (COMSOL is normally comma). Run `aggregate` once per file, save Parquet, never touch the raw CSVs again. The `n_nodes` column is kept only for diagnostics.

### 4.3 Band indexing and checks

Sorting by frequency gives a representation that stays continuous at band crossings (avoids label swaps). Then validate:

- Every (design, k) has ≥ 2 modes; log any with ≠ 2 and where they occur (k, design).
- Continuity: flag |f_b(k+δ) − f_b(k)| jumps > 5 × the median step (missed mode or swap).
- Bands 3–4 are kept only if present at a consistent fraction of k-points; otherwise mask them and ignore in v1.
- Model the ordered parametrisation `f₁, gap₁₂ = softplus(·)`, `f₂ = f₁ + gap₁₂` so f₂ ≥ f₁ always.

### 4.4 Features (per design, from `F[:, :, :2]`)

```python
def design_features(f1, f2, dk=0.01):
    """f1, f2: (281,) GHz for one design."""
    lo, hi = np.nanmax(f1), np.nanmin(f2)             # band-1 top, band-2 bottom along the path
    gap = max(hi - lo, 0.0); mid = 0.5 * (hi + lo)
    return dict(f1_max=lo, f2_min=hi, path_gap=gap, mid=mid,
                gap_ratio=gap / mid if gap > 0 else 0.0,
                vg1_max=np.nanmax(np.abs(np.gradient(f1, dk))),
                vg2_max=np.nanmax(np.abs(np.gradient(f2, dk))),
                flat1=np.mean(np.abs(np.gradient(f1, dk)) < 0.05 * np.nanmax(np.abs(np.gradient(f1, dk)))))
```

Add per-segment versions if E0 confirms k = 1, 2 are high-symmetry points (band edges at those k values). Skip loss features until E0 clears Im(λ).

### 4.5 Normalisation

- **Frequency:** ν = f · a / c with the fixed lattice constant a (≈ 210.01 µm, confirm), giving values 0.02–0.40. For the network, additionally standardise per band.
- **Design:** `L1n = (L1 − 140)/49`, `L2n = (L2 − 91)/49` (µm), structure as one-hot.
- **k:** `s = (k − 0.1)/2.8`, plus segment features (`seg = floor(k)`, `t = k − seg`) if the path hypothesis holds; add Fourier features `sin/cos(2π m s)`, m = 1…8.
- Fit all scalers on training designs only (inside each CV fold).

---

## 5. ML architecture

### 5.1 Forward surrogate (M1)

Two complementary formulations, compared head-to-head (they see the same 64 designs per structure):

**A. Curve-wise (data-efficient default).**
Stack the band curves of each design into a vector (2 × 281 = 562), PCA/POD to ~8–20 components (keep 99.9% variance), then regress the components on `(L1n, L2n)` with a **Gaussian Process** (Matern-5/2, ARD length-scales, one GP per component). Gives smooth interpolation and calibrated uncertainty in 2-D.

**B. Point-wise neural.**
MLP `(structure, L1n, L2n, k-features) → (f₁, gap₁₂)`; 4 × 128 units, SiLU, residual connections, dropout 0.05 or a 5-member ensemble for uncertainty. Loss:

```
L = Huber(f̂, f)  +  λ_d · MSE(∂f̂/∂k, ∂f/∂k)  +  λ_o · relu(-gap̂)²
```

The derivative term (finite difference in k) matters because group velocity and kinks drive the gap features. Train with AdamW, cosine schedule, early stopping on the **design-level** validation fold.

**C. Baselines that must be beaten:** (i) per-k polynomial response surface of degree 2–3 in (L1, L2); (ii) bilinear/cubic interpolation on the 8 × 8 grid; (iii) nearest design.

### 5.2 Inverse design (M2)

1. Evaluate the forward model on a dense grid (e.g. 400 × 400 = 160K designs per structure, vectorised; the GP/PCA path makes this milliseconds).
2. Compute the spec metrics for every grid design (gap edges, mid, width, ratio, group velocity, flatness, or distance to a target curve).
3. Score `J = weighted distance to spec + penalty · uncertainty + penalty · manufacturability`, keep the feasible set, cluster it in (L1, L2) so different *families* of solutions are returned separately.
4. Refine each cluster with a local optimiser (L-BFGS or CMA-ES) on the continuous surrogate.
5. Return top-K with predicted curve ± uncertainty; **verify the top 3 in COMSOL**, and feed the results back as new training data.

Why not tandem/cVAE now: with 2 parameters the exhaustive search is global, exact and interpretable; a generative model would only add error. Revisit only if the design space grows.

### 5.3 Evaluation protocol

**Never split by k-points or rows.** Adjacent k-points are near-duplicates, so any random split leaks. Split by **design**.

| Test | What it measures |
|---|---|
| Leave-one-design-out (64 folds/structure) | interior interpolation |
| Checkerboard hold-out (drop every other grid cell) | reduced data density |
| Leave-one-edge-out (drop an L1 or L2 row/column) | extrapolation, reported separately |
| **Off-grid COMSOL set (12–16 Sobol designs)** | the real generalisation test |

Metrics: MAE and max error in GHz and in % of f, per band; error on **derived** quantities (gap edges, gap ratio, v_g); uncertainty calibration (coverage of 90% intervals, or conformal wrap); inverse round-trip error `|(L1,L2)_recovered − (L1,L2)_true|` and `|f_COMSOL − f_pred|` on the verified designs. Estimate the **solver noise floor** from second differences along k and from the mesh-convergence spread; targets below that floor are meaningless.

---

## 6. Experiment plan

| ID | Hypothesis | Method | Expected | Success criterion |
|---|---|---|---|---|
| **E0** Data & physics validation | Units are µm; ~2 modes per point; k is a 3-segment path with kinks at 1, 2; Im(λ) is noise; cell is hexagonal | Mode-level counts per (design, k); `Re(λ)/1e9 − f`; second-difference of f(k); Im/Re by mode; scatter mesh nodes of one mode per structure and per design; read the .mph | Confirms items 1–10 in §0 | All checks documented; counts reconcile with 17,984 points/file |
| **E1** Baselines | The 64-point response is smooth enough that per-k polynomials already do well | Degree 1–3 per-k response surface, bilinear/cubic grid interpolation; LODO | MAE ≈ 0.5–2 GHz interior | Baseline table exists; sets the bar |
| **E2** GP on PCA | Curves live on a low-dim manifold; 8–20 components suffice | PCA + ARD-Matern GP; LODO + checkerboard | Beats E1 with calibrated σ | ≥ 20% lower MAE than E1, or equal MAE with usable σ |
| **E3** Point-wise MLP | Neural surrogate captures kinks better than PCA | MLP + Fourier k-features + ordering; 5-seed ensemble | Comparable to E2 interior, better near kinks | Within 10% of E2 on MAE, better on v_g/gap-edge error |
| **E4** Ablations | Derivative loss, ν normalisation, segment-aware k, and joint-structure embedding each help | One factor at a time, same folds, 3 seeds | Derivative loss and segment k help; joint model helps only if structures are similar | Keep only changes with consistent gain |
| **E5** Off-grid + active learning | Interior grid error underestimates off-grid error | 12–16 new COMSOL runs per structure; then add points where GP σ is largest and retrain | Off-grid error ≈ 1–2× LODO | Meets accuracy target; otherwise one more active-learning round |
| **E6** Inverse round-trip | A design's own predicted spec is recoverable | For held-out designs, extract spec from the surrogate, run inverse, compare (L1, L2) | Recovered within grid spacing; multiple solutions when physics allows | ≥ 90% recovered within 1 grid step; non-recovered cases explained by degeneracy |
| **E7** Real specs + COMSOL verification | Surrogate optimum survives the full solver | 3 specs (e.g. gap at 300 GHz; flat band near 270 GHz; widest gap), verify top 3 each | Error ≈ surrogate σ | COMSOL gap edges within tolerance (default 2 GHz) |
| **E8** (stretch) Uncertainty & robustness | Fabrication tolerance ±x µm moves the gap out of spec | Monte-Carlo on the surrogate over (L1, L2) ± tolerance | Robust-design map | Robustness score added to inverse objective |
| **E9** (stretch) Extra parameters/structures | Model generalises across structures | Only if new geometries or ≥ 3 parameters appear; then tandem/cVAE | n/a now | Not run in v1 |

### Decision tree

```
E0 fails (units/k-path/mode counts differ)  -> fix assumptions, redo P1 with corrected schema, then continue
E0 ok
 └─ E1 baseline error already ≤ target?
     ├─ YES -> ship E1/E2 (simplest), spend time on E5–E7
     └─ NO  -> E2, E3
         ├─ E2 or E3 ≤ target on LODO
         │    └─ E5 off-grid ≤ target?
         │         ├─ YES -> P6 inverse
         │         └─ NO  -> active learning (add runs at max σ), repeat E5 (max 2 rounds)
         │                   still NO -> narrow the design box or spec range, or request finer-grid sweep
         └─ both miss target on LODO
              ├─ errors concentrated at kinks/crossings -> segment-aware k, derivative loss, per-band models
              └─ errors spread out -> data-limited: more COMSOL designs before more modelling
E6 fails (not recovered) -> check degeneracy vs surrogate error; if surrogate error, back to E5
E7 fails (COMSOL disagrees) -> add verified points to training set, retrain, re-rank
```

---

## 7. Production architecture

**Serving:** FastAPI service, model artifacts versioned (PCA + GP as joblib, MLP ensemble as TorchScript or ONNX), Docker image, tiny CPU footprint. A Streamlit/Gradio front end plots the band diagram with uncertainty bands and highlights the gap.

**API sketch**

```
POST /forward   {structure, L1_um, L2_um, k_grid?}
  -> {bands: [[f1(k)],[f2(k)]], sigma: [[...],[...]], features: {path_gap, mid, gap_ratio, vg_max}}

POST /inverse   {structure?, spec: {gap_center_GHz, gap_min_width_GHz, ...} | target_curve,
                 constraints: {L1_um:[lo,hi], L2_um:[lo,hi], tol_um}, top_k}
  -> [{structure, L1_um, L2_um, predicted_features, sigma, robustness, verify: "recommended"}]
```

**Guards:** reject inputs outside the sweep box (L1 140–189 µm, L2 91–140 µm); flag any result whose GP σ exceeds a threshold as "verify in COMSOL"; never present extrapolated results as validated.

**Monitoring and retraining:** log every query; store every COMSOL verification as a new labelled design; retrain when (a) verification error exceeds tolerance twice in a row, (b) ≥ 10 new designs accumulate, or (c) the structure/parameter set changes. Keep the off-grid test set frozen.

---

## 8. Risks and mitigations (revised)

| Risk | Real severity | Mitigation / contingency |
|---|---|---|
| Sparse design grid (8 × 8, 7 µm step) | **High.** This is the real bottleneck. | GP uncertainty + active learning + off-grid COMSOL set (E5); if still short, request a finer sweep |
| Wrong physical assumptions from the reports (units, λ meaning, loss, bands) | **High** until E0 | E0 first; nothing downstream starts before G0 |
| Solver mode selection (N nearest a shift) makes band index unreliable | Medium | Read study settings; continuity checks; use sorted-frequency parametrisation; mask bands > 2 |
| Kinks/crossings at path corners | Medium | Segment-aware k features, derivative loss, per-segment error reporting |
| Split leakage (adjacent k, neighbouring designs) | High if ignored | Design-level splits only; off-grid COMSOL set as final test |
| Only two structures | Medium | Separate models per structure in v1; joint model only tested in E4; no claims about topology generalisation |
| Loss modelling built on noise | Medium | Skip until E0 shows Im(λ) smooth in k and consistent with a physical loss source |
| Node-weighted statistics misleading EDA | Medium | Always aggregate to mode level first; never histogram raw rows |
| Surrogate optimum fails in COMSOL | Medium | Uncertainty penalty, verification step, feedback into training data |
| 2D model not enough for the real device | Unknown | Confirm with the team; state the 2D limitation in every result |

---

## 9. Suggested repo layout and first-week checklist

```
inverse-design-em/
  data/{raw,interim,processed}/         # raw CSVs untouched; parquet + npy in interim/processed
  src/aggregate.py  bands.py  features.py  dataset.py
  src/models/{baseline.py, gp_pca.py, mlp.py}
  src/{train.py, evaluate.py, inverse.py, spec.py}
  serve/{app.py, Dockerfile}
  notebooks/{E0_validation, E1_baselines, ...}
  docs/{assumptions.md, results.md}
```

**Week 1**
1. Fill `assumptions.md` from the .mph (§2).
2. Run `aggregate.py` on both files → Parquet; check 17,984 points/file and mode-count histogram.
3. Plot 128 band diagrams (8 × 8 grid per structure); look for kinks at k = 1, 2, crossings, and jumps.
4. Run E0 checks; write down which of the 10 corrections were confirmed.
5. Set the accuracy target and write 3 example target specs for E6–E7.
6. Ask for / schedule the 12–16 off-grid COMSOL designs per structure.
