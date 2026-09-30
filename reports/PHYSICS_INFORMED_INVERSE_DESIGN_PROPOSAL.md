# STRATEGIC RESEARCH REPORT: PHYSICS-INTEGRATED INVERSE DESIGN ARCHITECTURES FOR ELECTROMAGNETIC UNIT CELLS

> **Document Type:** Research Architecture Proposal & Method Evaluation  
> **Target Audience:** Senior AI/Physics Research Scientist (Claude / Research Lead)  
> **Core Objective:** Evaluate all candidate paradigms integrating physical wave equations into the inverse design pipeline, synthesize the optimal path, and establish strict academic reproducibility standards for publication.  
> **Status:** Grounded on verified 61.4M-row COMSOL dataset, validated physical parameters, and existing Phase 1–4 experimental benchmarks.

---

## 1. Context, Current Baseline & The Core Bottleneck

### 1.1 Verified Dataset Ground Truth (Phase 1–2 Validations)
- **Structures:** Two unit-cell topologies (`para.csv`, `r1.csv`), 64 designs each arranged in an $8 \times 8$ parameter sweep.
- **Physical Parameters:** Inclusions $L_1 \in [140, 189]\ \mu\text{m}$ (step $7\ \mu\text{m}$), $L_2 \in [91, 140]\ \mu\text{m}$ (step $7\ \mu\text{m}$). Fixed hexagonal/rectangular lattice dimension $a \approx 210.01\ \mu\text{m}$, $\sqrt{3}a \approx 363.75\ \mu\text{m}$.
- **Dispersion Path:** 281 $k$-points ($k = 0.10 \to 2.90$, step $0.01$) traversing a 3-segment Brillouin zone path with confirmed high-symmetry reflection kinks at $k = 1.0$ and $k = 2.0$.
- **Physics Regime:** Low-loss/lossless ($Q \gtrsim 10^5$, $\text{Im}(\lambda)$ is numerical residue), two fundamental electromagnetic eigenmodes (Band 1 and Band 2) spanning $250\ \text{GHz} - 530\ \text{GHz}$.

### 1.2 Current Baseline Performance (Phase 3–4 Implementations)
- **Forward Surrogate (GP-on-PCA):**
  - Interpolation MAE on interior held-out designs (LODO): **$0.69\ \text{GHz}$ ($0.23\%$)** for `para`, **$11.66\ \text{GHz}$ ($3.67\%$)** for `r1`.
  - Polynomial response surfaces (degree 2–3) provide baseline curve fit.
- **Inverse Design Engine (Global Screen + L-BFGS-B on Surrogate):**
  - **Round-Trip Test (Held-out in-distribution design):** Recovers true $L_1, L_2$ with error $< 0.04\ \mu\text{m}$ and residual $< 0.01\ \text{GHz}$ (residual $< 0.1\%$).
  - **Arbitrary Specification Matching (Out-of-distribution target curve):** Residual jumps to **$\text{RMSE} = 6.26\ \text{GHz}$**.

### 1.3 The Core Bottleneck & Human Engineering Constraints

> [!IMPORTANT]
> **PRIMARY HUMAN DIRECTIVE & PRACTICAL CONSTRAINTS (From Project Lead):**
> 1. **COMSOL is the primary runtime bottleneck:** Generating data in COMSOL is excruciatingly slow, computationally heavy, and manual. Any proposal that requires dozens of additional COMSOL sweeps or puts COMSOL inside the iterative optimization loop is impractical. The ultimate goal is to **optimize and liberate the workflow from slow COMSOL runs**.
> 2. **Keep the implementation easy and manageable:** The team needs a solution that is realistic, fast to implement, and simple to maintain—not an overly baroque, convoluted academic pipeline that takes months to debug.
> 3. **Paper Reproducibility:** The methodology must be clean and reproducible by any external peer reviewer without needing access to a proprietary COMSOL license.

The current surrogate operates as a **pure data-driven black box** interpolating an 8×8 grid. When an engineer requests an arbitrary target response not near an existing training node:
1. The model cannot generalize beyond the empirical data manifold because it has no intrinsic knowledge of Maxwell's equations.
2. The search engine may converge to non-physical parameter combinations or struggle with band crossings and edge kinks.
3. Querying COMSOL in the loop for gradient optimization is completely out of the question due to speed (minutes per run) and license lock-in.

**Question for Claude / The Research Lead:** How can we embed the governing physical wave equations into the inverse design loop to eliminate the $6.26\ \text{GHz}$ out-of-distribution bottleneck, **keep the code architecture simple and accessible, and completely eliminate the need for slow COMSOL simulations during design optimization**?

---

## 2. Comprehensive Inventory of Candidate Methods

Below is a systematic breakdown of all possible methodologies to incorporate physics into this inverse design system.

---

### Method 1: Data-Driven Surrogate + Active Learning with COMSOL Verification (Baseline++)
- **Architecture:** Forward GP-on-PCA or Fourier-MLP combined with an uncertainty-guided acquisition function (Bayesian Optimization via Expected Improvement or Upper Confidence Bound) that requests targeted off-grid COMSOL runs at maximum surrogate variance $\sigma(L_1, L_2)$.
- **Role of Physics:** Implicit (physics resides exclusively inside the COMSOL solver; the model only observes discrete input-output pairs).
- **Pros:**
  - Zero risk of invalid physics approximations; uses exact finite-element ground truth.
  - Already partially planned in `inverse_design_plan.md` (Phase P5).
- **Cons:**
  - Computationally constrained by COMSOL execution speed and licensing.
  - Still requires 30–50 new manual COMSOL simulations to shrink uncertainty.
  - Low academic novelty (standard surrogate-based Bayesian optimization).

---

### Method 2: Physics-Informed Neural Network (PINN) via Soft Loss Regularization
- **Architecture:** Neural network $f_\theta(L_1, L_2, k) \to (\omega_1, \omega_2)$ trained with a compound loss function:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{sym}} \mathcal{L}_{\text{sym}} + \lambda_{\text{mono}} \mathcal{L}_{\text{mono}} + \lambda_{\text{curv}} \mathcal{L}_{\text{curv}} + \lambda_{\text{order}} \mathcal{L}_{\text{order}}$$
  - $\mathcal{L}_{\text{sym}}$: Enforces zero group velocity $\frac{\partial \omega}{\partial k} = 0$ at high-symmetry Brillouin zone points ($k = 1.0, 2.0$).
  - $\mathcal{L}_{\text{mono}}$: Enforces variational electromagnetic monotonicity ($\frac{\partial \omega}{\partial L_i} \le 0$ for high-index dielectric filling).
  - $\mathcal{L}_{\text{order}}$: Enforces strict physical eigenvalue ordering $\omega_2(k) \ge \omega_1(k)$ via softplus reparameterization.
- **Role of Physics:** Soft penalization of physical boundary conditions, group velocity properties, and variational principles.
- **Pros:**
  - Requires **zero additional COMSOL runs**.
  - Drop-in modification to the existing PyTorch MLP architecture.
  - Prevents the inverse search from landing on unphysical band kinks or inverted bands.
- **Cons:**
  - Loss balancing ($\lambda_i$ hyperparameter tuning) can be unstable.
  - Only acts as soft inductive bias; does not solve the exact Helmholtz PDE directly.

---

### Method 3: Pure Differentiable PDE Solver in the Loop (Adjoint FDTD/FDFD via Ceviche / Meep)
- **Architecture:** Replace the surrogate entirely. Run an open-source Maxwell solver with automatic differentiation (adjoint method in Ceviche or Meep-adjoint) directly inside the optimization loop:
  $$L_1^{(t+1)}, L_2^{(t+1)} = L_1^{(t)} - \eta \nabla_{L_1, L_2} \|\omega_{\text{adjoint}}(L_1, L_2) - \omega_{\text{target}}\|^2$$
- **Role of Physics:** Exact; solves the Maxwell wave equations on a discretized grid with reverse-mode automatic differentiation.
- **Pros:**
  - 100% physically exact; eliminates surrogate approximation error completely.
  - Standard framework in modern photonic topology optimization literature.
- **Cons:**
  - **Local minima trap:** The objective landscape is non-convex; gradient descent from a single initial guess gets trapped easily.
  - **Driven vs. Eigenvalue mismatch:** Ceviche/Meep-adjoint are natively tailored for *source-driven scattering/transmission* problems ($Ax = b$), whereas this unit cell requires solving a *Floquet-Bloch eigenvalue problem* ($A(k)x = \lambda B x$) across 281 $k$-points.
  - **Speed:** Evaluating 281 eigenmodes with adjoint sweeps takes minutes per step, rendering interactive UI/API applications unfeasible.

---

### Method 4: Analytical Differentiable Plane-Wave Expansion (PWE) in PyTorch
- **Architecture:** Implement an analytical 2D Plane Wave Expansion solver natively in PyTorch:
  $$\sum_{\mathbf{G}'} \eta(\mathbf{G} - \mathbf{G}') |\mathbf{k} + \mathbf{G}| |\mathbf{k} + \mathbf{G}'| H_{\mathbf{G}'} = \frac{\omega^2}{c^2} H_{\mathbf{G}}$$
  Because the inclusion geometries ($L_1, L_2$) are canonical shapes (rectangles, crosses, or cylinders), their Fourier transform $\eta(\mathbf{G}) = \mathcal{F}[1/\varepsilon(\mathbf{r})]$ has a **closed-form analytical expression** (sinc/Bessel functions).
  Eigenvalues are computed using `torch.linalg.eigh`, which provides native, exact backpropagation gradients $\nabla_{L_1, L_2} \omega_n(k)$.
- **Role of Physics:** Direct analytical Maxwell formulation under Floquet-Bloch boundary conditions.
- **Pros:**
  - **Ultrafast:** Batch solves all 281 $k$-points on a GPU in $5 - 15\ \text{ms}$.
  - **Exact analytical gradients:** Autograd computes $\frac{\partial \omega}{\partial L_1}$ and $\frac{\partial \omega}{\partial L_2}$ in milliseconds without finite differences.
  - Zero external license dependencies; runs standalone in Python/PyTorch.
- **Cons:**
  - Truncation error: Truncating to finite plane waves ($N \sim 15 \times 15 = 225$ basis functions) produces a small Gibbs-phenomenon discrepancy (typically $1 - 4\%$) compared to fine-mesh FEM (COMSOL).

---

### Method 5: Multi-Fidelity Differentiable Physics (Analytical PWE + ML Residual Physics / Delta-Learning)
- **Architecture:** Combine Method 4 with a residual neural corrector:
  $$\omega_{\text{predicted}}(k, L_1, L_2) = \omega_{\text{PWE}}(k, L_1, L_2) + \Delta \omega_{\text{ML}}(k, L_1, L_2)$$
  1. The **Differentiable PWE** computes $95-98\%$ of the physical dispersion and provides the core analytical gradient landscape.
  2. The **ML Residual Network** ($\Delta \omega_{\text{ML}}$), trained on the 128 COMSOL ground-truth designs, learns only the small FEM truncation/mesh correction:
     $$\Delta \omega = \omega_{\text{COMSOL}} - \omega_{\text{PWE}}$$
  3. The entire pipeline is end-to-end differentiable in PyTorch.
- **Role of Physics:** Exact Maxwell backbone (PWE) + high-fidelity empirical calibration (COMSOL FEM residual).
- **Pros:**
  - Solves the PWE truncation error while keeping evaluation time $< 30\ \text{ms}$.
  - Drastically simplifies ML training: learning a tiny, smooth residual $\Delta \omega \in [-5, +5]\ \text{GHz}$ is far easier and less data-hungry than predicting absolute frequencies ($250 - 530\ \text{GHz}$) from scratch.
  - Outstanding paper storyline: "Multi-Fidelity Differentiable Eigensolver for Ultra-Fast Inverse Metamaterial Design."
- **Cons:**
  - Requires implementing both the analytical Fourier structure factor $\eta(\mathbf{G}; L_1, L_2)$ and the residual training harness.

---

### Method 6: Fourier Neural Operator (FNO) / DeepONet for Periodic Maxwell Systems
- **Architecture:** Learn the solution operator mapping the periodic dielectric profile $\varepsilon(x, y; L_1, L_2) \to \omega_n(k)$ directly in Fourier space.
- **Role of Physics:** Infinite-dimensional operator approximation with spectral convolutions.
- **Pros:**
  - Extremely high academic cachet; trending heavily in AI-for-Science literature.
  - Resolution-invariant once trained.
- **Cons:**
  - **Severe data mismatch:** FNOs require spatial field maps ($E_z(x, y)$ or $H_z(x, y)$) to train effectively. The current dataset only provides eigenvalues ($f$), not spatial modal field distributions. Training an FNO on scalars degenerates into a standard MLP with unnecessary Fourier layer overhead.

---

### Method 7: Invertible Neural Networks (INN) / Conditional Flow Matching
- **Architecture:** Map target dispersion curves $\mathbf{y} = [\omega_1(k), \omega_2(k)]$ into a latent space $\mathbf{z} \sim \mathcal{N}(0, I)$ and invert to parameter space $\mathbf{x} = [L_1, L_2]$ using affine coupling layers or continuous normalizing flows.
- **Role of Physics:** Solves the one-to-many ambiguity (physical degeneracy where multiple geometries yield the same bandgap).
- **Pros:**
  - Mathematically elegant for ill-posed inverse problems.
  - Generates multiple diverse candidates for a single target spec.
- **Cons:**
  - For a 2-parameter design space, training an INN or diffusion flow is severe over-engineering.
  - Forward grid screening over 40,000 points on a 2D grid takes $< 100\ \text{ms}$ and naturally finds all disconnected global optima without training an invertible generative model.

---

## 3. Rigorous Evaluation Matrix

| Criterion | M1: Data-Driven + Active Learning | M2: PINN Soft Loss | M3: Differentiable Adjoint (Ceviche) | M4: Differentiable PWE | M5: Multi-Fidelity (PWE + Delta-ML) | M6: Neural Operator (FNO) | M7: Invertible Nets (INN) |
|---|---|---|---|---|---|---|---|
| **Physical Rigor** | Low (Black box) | Medium (Soft constraints) | High (Full wave PDE) | High (Analytical PWE) | **Highest** (PWE + FEM Ground Truth) | Medium | Low |
| **Solving Speed (Inverse)** | Medium (1–5 s) | Fast (< 100 ms) | Very Slow (Minutes) | **Ultra-Fast (< 50 ms)** | **Ultra-Fast (< 50 ms)** | Fast (< 100 ms) | Fast (< 50 ms) |
| **Data Efficiency** | Poor (Needs more COMSOL) | **Excellent** (Uses existing 128) | **Independent** (No training data) | **Independent** (No training data) | **Excellent** (Calibrates on 128) | Very Poor (Needs field data) | Poor |
| **Implementation Complexity** | Low | Low | High (Bloch solver setup) | Medium (Analytical Fourier) | Medium (Fourier + MLP) | Very High | High |
| **Risk of Local Minima** | Low (Global Grid) | Low (Global Grid) | **High** (Local gradient descent) | Low (Fast grid + autograd) | **Lowest** (Global grid + autograd) | Medium | Low |
| **Publication Novelty** | Low (Iterative) | Medium | Medium | Medium-High | **Very High (Top Tier)** | High (Overkill for 2D) | Medium |
| **Production Software Readiness**| High | High | Very Low (License/speed) | High | **Highest (Standalone PyTorch)** | Medium | Medium |

---

## 4. Key Questions & Decision Criteria for Claude

### Decision Question 0: Practicality & COMSOL-Decoupling Test (User's Primary Constraint)
> *Which methodology genuinely liberates the team from slow, tedious COMSOL simulation runs while keeping the codebase lightweight, simple to understand, and fast to execute?*
- **Consideration:** Any method requiring iterative COMSOL runs (like active learning in Method 1) or complex solver interfaces is heavily penalized. The preferred solution must treat the existing 128 COMSOL designs as a *one-time sunk cost* and enable real-time ($< 100\ \text{ms}$) standalone optimization without touching COMSOL again.

### Decision Question 1: Formulation Selection (Simplicity vs. Deep Physics)
> *Between Method 2 (PINN Soft Loss on existing surrogate) and Method 5 (Multi-Fidelity Analytical PWE + ML Residual), which provides the optimal balance of academic defensibility, zero COMSOL overhead, and minimum implementation friction?*
- **Consideration:** Method 2 is trivial to implement (requires only adding loss penalties to the existing PyTorch MLP script in $\sim 50$ lines of code). Method 5 requires implementing an analytical 2D PWE matrix in PyTorch ($\sim 250$ lines of code), but in return it gives an instant, native Maxwell solver that completely replaces COMSOL's forward sweeps in 15 ms. Which path should we execute first?

### Decision Question 2: Handling Band Crossings & Non-Degeneracy
> *How should the chosen architecture handle band crossing points or mode swapping along the $k$-path?*
- **Consideration:** In PWE, eigenvalue sorting `torch.linalg.eigh` naturally orders eigenvalues $\omega_1 \le \omega_2$, but physical mode tracking across symmetry points requires either eigenvector dot products ($\langle u_{n, k} | u_{m, k+\Delta k} \rangle$) or sorting by frequency. Which is more robust for inverse design?

### Decision Question 3: Academic Reproducibility & COMSOL Independence
> *How can we structure the final validation so that any external reviewer can clone the repository, reproduce all benchmarks without requiring an expensive proprietary COMSOL license, yet still trust the scientific validation?*
- **Consideration:** Storing the processed Parquet/NPY tensors and providing a standalone PyTorch verification script enables 100% open-source reproducibility, reserving COMSOL strictly for the 3–5 final out-of-distribution blind test cases.

---

## 5. Strict Scientific Reproducibility Checklist (Paper Standards)

To guarantee that the research withstands peer review in leading journals (*JOSA B*, *ACS Photonics*, *IEEE Transactions on Microwave Theory and Techniques*, or *Machine Learning: Science and Technology*), the following standards must be adhered to:

### R1. Leakage-Free Splitting Protocol
- **Zero Row-Level Leakage:** Splits must be strictly design-level (Leave-One-Design-Out or 50/50 Checkerboard across $L_1, L_2$). Never split randomly across $k$-points or rows.
- **Off-Grid Blind Generalization:** Include a frozen test split of at least 8–12 off-grid designs (Sobol/quasi-random sampling within the parameter bounds) verified independently.

### R2. Baseline Comparison Integrity
- Every proposed method must be benchmarked under identical splits against:
  1. Nearest-Neighbor retrieval (dumb baseline)
  2. Degree-2 and Degree-3 polynomial response surface (classical engineering baseline)
  3. Pure data-driven Gaussian Process on PCA (standard modern surrogate)
  4. The proposed physics-integrated model.

### R3. Dual Metric Reporting
- Report both **Frequency-Space Metrics** and **Derived Functional Metrics**:
  - Pointwise MAE, RMSE, and Max Error across all $k$ (in GHz and as percentage of frequency).
  - Band Gap Center Frequency Error ($\Delta \omega_{\text{mid}}$).
  - Band Gap Width Error ($\Delta \text{Gap}_{\text{path}}$).
  - Group Velocity Consistency ($\Delta v_g = \frac{\partial \omega}{\partial k}$).

### R4. Software & Artifact Archival
- Fixed random seeds for all torch/numpy/sklearn initializations.
- Automated pipeline scripts: `make aggregate` $\to$ `make benchmark` $\to$ `make inverse-test`.
- Version-locked dependencies (`requirements.txt` / `pyproject.toml`).
- Pre-trained model weights stored alongside the benchmark verification scripts.

---

## 6. Recommended Actionable Roadmap

1. **Step 1 (Immediate - 24 hours):** Implement **Method 2 (PINN Soft Loss)** in the current MLP surrogate to test whether zone-edge symmetry ($k=1.0, 2.0$) and monotonicity constraints reduce the 6.26 GHz out-of-distribution RMSE on existing data.
2. **Step 2 (Week 1–2):** Build the analytical 2D **Differentiable PWE module in PyTorch** (Method 4) for the unit-cell geometry, verifying that `torch.linalg.eigh` autograd gradients match numerical finite-difference gradients.
3. **Step 3 (Week 2–3):** Wire the PWE solver to the 128 COMSOL designs to form the **Multi-Fidelity Engine (Method 5)**, training the residual corrector on $\Delta \omega$.
4. **Step 4 (Week 4):** Execute the blind out-of-distribution inverse benchmark and produce publication-ready comparison figures (Parity plots, Dispersion overlay, and Parameter recovery maps).
