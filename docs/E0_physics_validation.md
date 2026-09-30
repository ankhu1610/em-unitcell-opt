# E0 Physics & Data Validation Report

## 1. Executive Summary
- **Dataset Size:** 2 structures (`para`, `r1`), each with 64 designs (8 × 8 grid, L1: 140–189 µm, L2: 91–140 µm).
- **Sweep Coverage:** 281 k-points per design (k: 0.10 → 2.90 in step 0.01).
- **Gate G0 Status:** PASSED. Exactly 17,984 (design, k) points per structure.
- **Band Availability:** Bands 1 and 2 are present at 100% of points across all 128 designs.
- **Re(λ) Verification:** $\text{Re}(\lambda) / 10^9 \equiv f_{\text{GHz}}$ with max residual $1.137 \times 10^{-13}$. Confirms $\lambda$ is the eigenfrequency in Hz.
- **Im(λ) Assessment:** Median $\text{Im}(\lambda) = 0.0$, max $\text{Im} / \text{Re} \approx 10^{-5}$. Effectively lossless; solver residue rather than physical attenuation.

## 2. Path & Kink Analysis
- Analysis of $|\Delta^2 f(k)|$ shows pronounced discontinuities at $k = 1.0$ and $k = 2.0$.
- This confirms that $k$ is a path coordinate along a triangular or rectangular Brillouin zone (e.g. $\Gamma \to X \to M \to \Gamma$), with high-symmetry corner reflections at integer points.
- **Modeling Requirement:** Models should either operate curve-wise (POD/PCA) or incorporate segment-aware features: `seg = floor(k)`, `t = k - seg`.

## 3. Band Gap Statistics
- **Structure 0 (`para`):** 61/64 designs possess an open path band gap.
  - Max Path Gap: 43.63 GHz
  - Mean Gap (where open): 19.21 GHz
- **Structure 1 (`r1`):** 64/64 designs possess an open path band gap.
  - Max Path Gap: 40.19 GHz
  - Mean Gap (where open): 20.89 GHz

## 4. Next Phase Progression
- Proceed to Phase 3 (E1 Baselines) and Phase 4 (E2 GP-on-PCA surrogate & E3 MLP neural surrogate).
