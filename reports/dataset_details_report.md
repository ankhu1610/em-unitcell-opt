# COMPREHENSIVE DATASET INTELLIGENCE REPORT

> **Purpose**: Complete analysis of two COMSOL electromagnetic simulation
> datasets for inverse design of photonic/metamaterial unit cells.
> This report is designed to be given to an LLM (Claude) to generate
> a complete project roadmap.

---

## 1. EXECUTIVE SUMMARY

| Property | para.csv | r1.csv |
|----------|----------|--------|
| File size | 3.915 GB | 4.205 GB |
| Data lines | 29,525,794 | 31,864,276 |
| Columns | 7 | 7 |
| L1 values | 8 | 8 |
| L2 values | 8 | 8 |
| k values | 281 | 281 |
| Freq range (GHz) | 28.0251-531.4328 | 29.3962-576.0409 |
| Complex λ fraction | 61.6% | 58.2% |

## 2. DOMAIN & PHYSICS CONTEXT

**Domain**: Computational Electromagnetics / Photonic Crystal Design

**Software**: COMSOL Multiphysics 6.4 (EMW module)

**Problem Type**: Eigenfrequency analysis of 2D electromagnetic unit cells

**Context**: This dataset comes from parametric sweeps of 2D photonic/metamaterial unit cells. The simulation solves Maxwell's equations in eigenvalue form to find resonant frequencies (photonic band structure) as a function of Bloch wave-vector k and geometric parameters L1, L2.

### Column Definitions

| Column | Description |
|--------|-------------|
| `X` | Horizontal coordinate of FEM mesh node (µm) |
| `Y` | Vertical coordinate of FEM mesh node (µm) |
| `L1` | First geometric parameter of unit cell (~140 nm). Likely the lattice constant or a structural dimension. |
| `L2` | Second geometric parameter of unit cell (~91-105 nm). Swept across 3 values — likely a feature size (hole radius, slab thickness, etc.). |
| `k` | Normalized Bloch wave-vector, swept 0.1 to 2.9 in steps of 0.01 (281 values). This traces the photonic band structure along the Brillouin zone path. |
| `lambda` | Eigenvalue from COMSOL. Real part = resonant frequency parameter. Imaginary part (when present) = loss/attenuation rate. ~50% of values are complex, indicating significant modal losses. |
| `emw.freq (GHz)` | Electromagnetic eigenfrequency in GHz. Range ~28-496 GHz (microwave to sub-THz). This is the primary output / 'target' for inverse design. |

## 3. INVERSE DESIGN CONTEXT

**Goal**: Given desired electromagnetic frequency response (band structure), determine the optimal geometric parameters (L1, L2) and potentially the unit cell topology.

**Forward Problem**: Geometry (L1, L2) + k → Eigenfrequencies (freq)

**Inverse Problem**: Desired freq/band-structure → Optimal geometry (L1, L2)

**Ml Formulation**: This is fundamentally a REGRESSION problem for the forward model, and an INVERSE DESIGN / GENERATIVE problem for the reverse mapping. The data enables training a surrogate model that replaces expensive COMSOL simulations.

## 4. KEY OBSERVATIONS

1. L1 is CONSTANT across all data (140.0 nm). It does not vary — only L2 and k are swept. This means L1 is fixed in the current design space.

2. L2 has only 3 discrete values (91, 98, 105 nm). This is a very sparse parametric sweep — the design space is under-sampled.

3. k sweeps 281 values (0.1 to 2.9). This provides dense band-structure coverage along the Brillouin zone.

4. ~50% of eigenvalues (lambda) are complex, indicating lossy/evanescent modes. These modes have physical significance (e.g., radiation losses, material absorption).

5. The frequency range 28-496 GHz spans microwave to sub-THz. This is consistent with mm-wave photonic crystal or metamaterial design.

6. Two separate files (para.csv, r1.csv) with different meshes but same parameter space suggests two different unit cell geometries being compared.

7. ~29.5M and ~31.9M data lines respectively. Each line is a mesh node with its field solution. The mesh has ~tens of thousands of nodes per eigenmode.

## 5. PARAMETRIC SWEEP DETAILS

### para.csv

- **L1**: 8 value(s) → [140.01, 147.01, 154.01, 161.01, 168.01, 175.01, 182.01, 189.01] nm
- **L2**: 8 value(s) → [105.01, 112.01, 119.01, 126.01, 133.01, 140.01, 91.0, 98.01] nm
- **k**: 281 values, range [0.1, 2.9000000000000004], step 0.01
- **Frequencies**: 35965 unique values, range [28.0251, 531.4328] GHz
- **λ complex fraction**: 61.6%

### r1.csv

- **L1**: 8 value(s) → [140.01, 147.01, 154.01, 161.01, 168.01, 175.01, 182.01, 189.01] nm
- **L2**: 8 value(s) → [105.01, 112.01, 119.01, 126.01, 133.01, 140.01, 91.0, 98.01] nm
- **k**: 281 values, range [0.1, 2.9000000000000004], step 0.01
- **Frequencies**: 35967 unique values, range [29.3962, 576.0409] GHz
- **λ complex fraction**: 58.2%

## 6. STATISTICAL SUMMARY (SAMPLED)

### para.csv (sample size: 2,000,000)

| Column | Min | Max | Mean | Std | Median | Skew | Zeros |
|--------|-----|-----|------|-----|--------|------|-------|
| X (µm) | 0.0 | 363.75 | 189.600372 | 70.544019 | 196.102415 | -0.128569 | 2708 |
| Y (µm) | 0.0 | 210.0111604177264 | 108.98756 | 51.262146 | 107.451611 | -0.044402 | 37912 |
| k (normalized wave-vector) | 0.1 | 2.9000000000000004 | 1.454694 | 0.793066 | 1.45 | 0.045648 | 0 |
| emw.freq (GHz) | 28.02507011440257 | 494.20477019022604 | 289.679145 | 108.38208 | 305.033427 | -0.395827 | 0 |
| |λ| (eigenvalue magnitude) | 28025070114.40257 | 494204770190.48224 | 289679145022.89905 | 108382079616.05138 | 305033426772.4218 | -0.395827 | 0 |
| Re(λ) | 28025070114.40257 | 494204770190.22595 | 289679145022.8553 | 108382079616.01675 | 305033426772.4218 | -0.395827 | 0 |
| Im(λ) | -1524853.5160281735 | 2019352.4846374793 | 25016.796097 | 179533.011922 | 0.0 | 2.281055 | 645965 |

### r1.csv (sample size: 2,000,000)

| Column | Min | Max | Mean | Std | Median | Skew | Zeros |
|--------|-----|-----|------|-----|--------|------|-------|
| X (µm) | 0.0 | 363.75 | 191.106807 | 72.16678 | 196.148884 | -0.118455 | 2456 |
| Y (µm) | 0.0 | 210.0111604177264 | 110.020218 | 53.220107 | 110.071169 | -0.061962 | 34384 |
| k (normalized wave-vector) | 0.1 | 2.9000000000000004 | 1.420959 | 0.819159 | 1.36 | 0.138831 | 0 |
| emw.freq (GHz) | 29.396164538403056 | 520.5030063431503 | 301.438126 | 118.768092 | 322.939539 | -0.344046 | 0 |
| |λ| (eigenvalue magnitude) | 29396164538.403053 | 520503006343.1517 | 301438126132.4727 | 118768091920.56323 | 322939539426.5188 | -0.344046 | 0 |
| Re(λ) | 29396164538.403053 | 520503006343.15015 | 301438126132.436 | 118768091920.54892 | 322939539426.42957 | -0.344046 | 0 |
| Im(λ) | -1248630.4137924793 | 764183.3170306293 | -11766.449745 | 159487.934658 | 0.0 | -3.717391 | 786762 |

## 7. FREQUENCY DISTRIBUTION (10 GHz BINS)

### para.csv

| Frequency Band | Count |
|----------------|-------|
| 20-30 GHz | 3,699 |
| 30-40 GHz | 22,048 |
| 40-50 GHz | 21,413 |
| 50-60 GHz | 22,048 |
| 60-70 GHz | 21,413 |
| 70-80 GHz | 22,048 |
| 80-90 GHz | 22,123 |
| 90-100 GHz | 21,338 |
| 100-110 GHz | 23,555 |
| 110-120 GHz | 20,616 |
| 120-130 GHz | 22,074 |
| 130-140 GHz | 22,819 |
| 140-150 GHz | 21,364 |
| 150-160 GHz | 22,097 |
| 160-170 GHz | 22,809 |
| 170-180 GHz | 22,870 |
| 180-190 GHz | 22,807 |
| 190-200 GHz | 22,074 |
| 200-210 GHz | 25,807 |
| 210-220 GHz | 25,122 |
| 220-230 GHz | 25,893 |
| 230-240 GHz | 27,338 |
| 240-250 GHz | 26,603 |
| 250-260 GHz | 58,285 |
| 260-270 GHz | 143,696 |
| 270-280 GHz | 87,490 |
| 280-290 GHz | 72,548 |
| 290-300 GHz | 72,186 |
| 300-310 GHz | 146,840 |
| 310-320 GHz | 190,014 |
| 320-330 GHz | 169,148 |
| 330-340 GHz | 35,509 |
| 340-350 GHz | 31,844 |
| 350-360 GHz | 32,520 |
| 360-370 GHz | 31,759 |
| 370-380 GHz | 30,353 |
| 380-390 GHz | 31,034 |
| 390-400 GHz | 30,327 |
| 400-410 GHz | 30,906 |
| 410-420 GHz | 30,146 |
| 420-430 GHz | 31,603 |
| 430-440 GHz | 31,664 |
| 440-450 GHz | 32,339 |
| 450-460 GHz | 37,508 |
| 460-470 GHz | 37,495 |
| 470-480 GHz | 47,795 |
| 480-490 GHz | 41,636 |
| 490-500 GHz | 5,377 |

### r1.csv

| Frequency Band | Count |
|----------------|-------|
| 20-30 GHz | 2,396 |
| 30-40 GHz | 20,486 |
| 40-50 GHz | 22,803 |
| 50-60 GHz | 23,639 |
| 60-70 GHz | 22,868 |
| 70-80 GHz | 23,613 |
| 80-90 GHz | 22,843 |
| 90-100 GHz | 22,828 |
| 100-110 GHz | 23,678 |
| 110-120 GHz | 22,843 |
| 120-130 GHz | 24,444 |
| 130-140 GHz | 22,023 |
| 140-150 GHz | 24,482 |
| 150-160 GHz | 23,664 |
| 160-170 GHz | 21,982 |
| 170-180 GHz | 24,523 |
| 180-190 GHz | 24,419 |
| 190-200 GHz | 23,678 |
| 200-210 GHz | 24,468 |
| 210-220 GHz | 25,303 |
| 220-230 GHz | 24,419 |
| 230-240 GHz | 26,903 |
| 240-250 GHz | 26,968 |
| 250-260 GHz | 30,144 |
| 260-270 GHz | 95,535 |
| 270-280 GHz | 123,137 |
| 280-290 GHz | 68,175 |
| 290-300 GHz | 56,673 |
| 300-310 GHz | 62,985 |
| 310-320 GHz | 38,078 |
| 320-330 GHz | 69,932 |
| 330-340 GHz | 274,247 |
| 340-350 GHz | 85,116 |
| 350-360 GHz | 34,333 |
| 360-370 GHz | 30,940 |
| 370-380 GHz | 30,184 |
| 380-390 GHz | 30,989 |
| 390-400 GHz | 30,160 |
| 400-410 GHz | 29,324 |
| 410-420 GHz | 30,184 |
| 420-430 GHz | 29,236 |
| 430-440 GHz | 30,940 |
| 440-450 GHz | 34,270 |
| 450-460 GHz | 34,116 |
| 460-470 GHz | 35,820 |
| 470-480 GHz | 39,818 |
| 480-490 GHz | 47,874 |
| 490-500 GHz | 52,858 |
| 500-510 GHz | 32,591 |
| 510-520 GHz | 16,209 |
| 520-530 GHz | 859 |

## 8. CORRELATIONS

### para.csv
- freq ↔ k correlation: `0.057306`
- freq ↔ |λ| correlation: `1.0`

### r1.csv
- freq ↔ k correlation: `0.034968`
- freq ↔ |λ| correlation: `1.0`

## 9. EIGENMODE / BAND ANALYSIS

### para.csv
- Avg bands per k-point: 2.5
- Min bands per k-point: 2
- Max bands per k-point: 4
- Number of eigenfrequency solutions found per k-value. This represents the number of bands in the photonic band structure.

### r1.csv
- Avg bands per k-point: 2.28
- Min bands per k-point: 2
- Max bands per k-point: 4
- Number of eigenfrequency solutions found per k-value. This represents the number of bands in the photonic band structure.

## 10. CROSS-FILE COMPARISON

- **Identical schema**: True
- **Row ratio (r1/para)**: 1.0792
- **Same L1**: True
- **Same L2**: True
- **Same k range**: True

### Geometry Differences
- para.csv X range: [0.0, 363.75]
- r1.csv X range: [0.0, 363.75]
- para.csv Y range: [0.0, 210.01116]
- r1.csv Y range: [0.0, 210.01116]
- para.csv unique XY (5M): 5715
- r1.csv unique XY (5M): 5972

**Interpretation**: Different XY pairs and/or ranges suggest different mesh geometries. para.csv comes from parametric_sweep_UC.mph (Unit Cell model). r1.csv may represent a different geometry, mesh, or simulation run.

### Likely Relationship
Both files appear to be COMSOL eigenfrequency simulation outputs for electromagnetic (microwave/THz) unit cell structures. They share the same parametric sweep parameters (L1, L2, k) and output columns (X, Y, lambda, freq). The difference in row count and XY coordinates suggests they use different FEM meshes or represent different geometries/designs. This is consistent with an INVERSE DESIGN study where multiple unit cell geometries are explored.

## 11. DATA QUALITY

### para.csv
- Lines checked: 500,000
- Parse errors: 0
- Short lines: 0
- NaN frequencies: 0
- Negative frequencies: 0
- Adjacent duplicates: 0
- **Assessment: CLEAN**

### r1.csv
- Lines checked: 500,000
- Parse errors: 0
- Short lines: 0
- NaN frequencies: 0
- Negative frequencies: 0
- Adjacent duplicates: 0
- **Assessment: CLEAN**

## 12. DATA CHALLENGES FOR ML

1. Massive dataset (~8 GB total) requires out-of-core processing
2. Spatial mesh data (X,Y per node) — need to aggregate to per-eigenmode features
3. Complex eigenvalues need special handling (real/imag decomposition)
4. Only 3 L2 values — very sparse design parameter coverage
5. L1 is constant — effectively a 1D parametric sweep (over L2 only)
6. Node-level data is redundant for band-structure analysis (need mode-level aggregation)

## 13. ML / INVERSE DESIGN RECOMMENDATIONS

### Preprocessing Roadmap

#### Step 1 Aggregation
Aggregate node-level data to eigenmode-level. Each unique (L1, L2, k, freq) combination should become ONE row. The spatial (X, Y) data should be aggregated into features.
- Output columns: `['L1', 'L2', 'k', 'freq_ghz', 'lambda_real', 'lambda_imag', 'lambda_magnitude', 'is_lossy (bool)', 'mesh_node_count']`
- Expected rows: ~thousands (much smaller than 30M)

#### Step 2 Band Structure
For each (L1, L2), build the photonic band structure: freq vs k curves for each band index.
- Output: `band_structure[L2][band_index] = array of (k, freq) pairs`

#### Step 3 Feature Engineering

  - Band gaps (frequency ranges with no eigenmode)
  - Band widths
  - Group velocity (d(freq)/dk)
  - Mode loss (imaginary part of lambda)
  - Band gap to midgap ratio
  - Number of modes per frequency window
  - Density of states

#### Step 4 Normalization
Normalize frequencies and k-values for ML training
  - Normalize freq by c/(2*L1) for dimensionless frequency
  - Normalize k to [0, 1] or [0, π/a]

### ML Approaches

#### Forward Surrogate
**Goal**: Predict eigenfrequencies from (L2, k)
**Models**:
  - Neural Network (MLP) — baseline
  - Physics-Informed Neural Network (PINN)
  - Gaussian Process Regression — good for small data (3 L2 values)
  - Random Forest / XGBoost — for tabular baseline
**Metrics**: ['MAE (GHz)', 'RMSE (GHz)', 'R²', 'Max error']

#### Inverse Design
**Goal**: Given desired band structure → predict L2 (and future: L1, topology)
**Approaches**:
  - Tandem network (forward + inverse)
  - Conditional VAE
  - Conditional GAN (cGAN)
  - Bayesian optimization over forward surrogate
  - Reinforcement learning for topology optimization

#### Data Augmentation
**Goal**: With only 3 L2 values, the design space is severely under-sampled. Consider: (1) running more COMSOL sweeps with finer L2 grid, (2) interpolation-based augmentation, (3) transfer learning from analytical models.

## 14. CRITICAL QUESTIONS BEFORE BUILDING ROADMAP

1. What is the target application? (e.g., 5G filter, THz absorber, waveguide)
2. What specific frequency response is desired? (band gap location, bandwidth)
3. Will L1 also be varied in future simulations?
4. Can more L2 values be simulated to densify the design space?
5. Are there manufacturing constraints on L1 and L2?
6. Is the 2D simulation sufficient or is 3D needed?
7. What is the acceptable error tolerance for the surrogate model?
8. Is real-time inference needed (online design tool) or batch optimization?
9. Are there other geometric parameters (e.g., hole shape, material) to explore?
10. What is the relationship between para.csv and r1.csv geometries?

---

## APPENDIX: RAW STRUCTURED DATA (JSON)

```json
{
  "report_metadata": {
    "title": "Comprehensive Dataset Intelligence Report",
    "files": [
      "E:\\Inverse design dataset for lakshya and ankit\\Tushar\\para.csv",
      "E:\\Inverse design dataset for lakshya and ankit\\Tushar\\r1.csv"
    ],
    "domain": "Computational Electromagnetics / Inverse Design",
    "generated_by": "details.py"
  },
  "datasets": {
    "para": {
      "full_pass": {
        "file": "para.csv",
        "file_size_gb": 3.915,
        "total_lines": 29525803,
        "header_lines": 9,
        "data_lines": 29525794,
        "comsol_metadata": {
          "Model": "parametric_sweep_UC.mph",
          "Version": "COMSOL 6.4.0.429",
          "Date": "Aug 17 2026, 17:33",
          "Dimension": "2",
          "Nodes": "29525794",
          "Expressions": "1",
          "Description": "Frequency",
          "Length unit": "µm",
          "X": "Y,L1,L2,k,lambda,emw.freq (GHz)"
        },
        "columns": [
          "X",
          "Y",
          "L1",
          "L2",
          "k",
          "lambda",
          "emw.freq (GHz)"
        ],
        "geometry": {
          "dimension": "2",
          "length_unit": "µm",
          "x_range": [
            0.0,
            363.75
          ],
          "y_range": [
            0.0,
            210.01116
          ],
          "unique_xy_pairs_in_first_5M": 5715
        },
        "parametric_sweep": {
          "L1": {
            "count": 8,
            "values_um": [
              "1.4000744027848426E-4",
              "1.4700781229240846E-4",
              "1.540081843063327E-4",
              "1.610085563202569E-4",
              "1.680089283341811E-4",
              "1.7500930034810533E-4",
              "1.8200967236202954E-4",
              "1.8901004437595374E-4"
            ],
            "values_nm": [
              140.01,
              147.01,
              154.01,
              161.01,
              168.01,
              175.01,
              182.01,
              189.01
            ],
            "interpretation": "Geometric parameter (likely unit cell length L1)"
          },
          "L2": {
            "count": 8,
            "values_um": [
              "1.0500558020886319E-4",
              "1.120059522227874E-4",
              "1.1900632423671162E-4",
              "1.2600669625063583E-4",
              "1.3300706826456006E-4",
              "1.4000744027848426E-4",
              "9.100483618101477E-5",
              "9.800520819493898E-5"
            ],
            "values_nm": [
              105.01,
              112.01,
              119.01,
              126.01,
              133.01,
              140.01,
              91.0,
              98.01
            ],
            "interpretation": "Geometric parameter (likely unit cell length L2)"
          },
          "k": {
            "count": 281,
            "min": 0.1,
            "max": 2.9000000000000004,
            "step": 0.01,
            "interpretation": "Normalized wave-vector (Bloch k) for band-structure sweep"
          }
        },
        "frequency_output": {
          "column": "emw.freq (GHz)",
          "unit": "GHz",
          "unique_count": 35965,
          "min_ghz": 28.0251,
          "max_ghz": 531.4328,
          "interpretation": "Eigenfrequency solutions at each (L1,L2,k) point"
        },
        "lambda_column": {
          "real_count": 11329163,
          "complex_count": 18196631,
          "total": 29525794,
          "complex_fraction": 0.6163,
          "interpretation": "Eigenvalue λ. Real = propagating mode, Complex = evanescent/lossy mode. Imaginary part represents attenuation/loss."
        }
      },
      "statistics": {
        "sample_size": 2000000,
        "column_statistics": {
          "X": {
            "name": "X (µm)",
            "count": 2000000,
            "min": 0.0,
            "max": 363.75,
            "mean": 189.600372,
            "std": 70.544019,
            "median": 196.102415,
            "q25": 130.349688,
            "q75": 246.022823,
            "skewness": -0.128569,
            "nan_count": 0,
            "zero_count": 2708,
            "negative_count": 0
          },
          "Y": {
            "name": "Y (µm)",
            "count": 2000000,
            "min": 0.0,
            "max": 210.0111604177264,
            "mean": 108.98756,
            "std": 51.262146,
            "median": 107.451611,
            "q25": 69.111257,
            "q75": 150.581643,
            "skewness": -0.044402,
            "nan_count": 0,
            "zero_count": 37912,
            "negative_count": 0
          },
          "k": {
            "name": "k (normalized wave-vector)",
            "count": 2000000,
            "min": 0.1,
            "max": 2.9000000000000004,
            "mean": 1.454694,
            "std": 0.793066,
            "median": 1.45,
            "q25": 0.77,
            "q75": 2.12,
            "skewness": 0.045648,
            "nan_count": 0,
            "zero_count": 0,
            "negative_count": 0
          },
          "frequency": {
            "name": "emw.freq (GHz)",
            "count": 2000000,
            "min": 28.02507011440257,
            "max": 494.20477019022604,
            "mean": 289.679145,
            "std": 108.38208,
            "median": 305.033427,
            "q25": 246.162499,
            "q75": 344.263353,
            "skewness": -0.395827,
            "nan_count": 0,
            "zero_count": 0,
            "negative_count": 0
          },
          "lambda_magnitude": {
            "name": "|λ| (eigenvalue magnitude)",
            "count": 2000000,
            "min": 28025070114.40257,
            "max": 494204770190.48224,
            "mean": 289679145022.89905,
            "std": 108382079616.05138,
            "median": 305033426772.4218,
            "q25": 246162498637.46985,
            "q75": 344263352954.5835,
            "skewness": -0.395827,
            "nan_count": 0,
            "zero_count": 0,
            "negative_count": 0
          },
          "lambda_real": {
            "name": "Re(λ)",
            "count": 2000000,
            "min": 28025070114.40257,
            "max": 494204770190.22595,
            "mean": 289679145022.8553,
            "std": 108382079616.01675,
            "median": 305033426772.4218,
            "q25": 246162498637.46985,
            "q75": 344263352954.5833,
            "skewness": -0.395827,
            "nan_count": 0,
            "zero_count": 0,
            "negative_count": 0
          },
          "lambda_imag": {
            "name": "Im(λ)",
            "count": 2000000,
            "min": -1524853.5160281735,
            "max": 2019352.4846374793,
            "mean": 25016.796097,
            "std": 179533.011922,
            "median": 0.0,
            "q25": -358.237778,
            "q75": 4067.469288,
            "skewness": 2.281055,
            "nan_count": 0,
            "zero_count": 645965,
            "negative_count": 594123
          }
        },
        "frequency_distribution_10GHz_bins": {
          "20-30 GHz": 3699,
          "30-40 GHz": 22048,
          "40-50 GHz": 21413,
          "50-60 GHz": 22048,
          "60-70 GHz": 21413,
          "70-80 GHz": 22048,
          "80-90 GHz": 22123,
          "90-100 GHz": 21338,
          "100-110 GHz": 23555,
          "110-120 GHz": 20616,
          "120-130 GHz": 22074,
          "130-140 GHz": 22819,
          "140-150 GHz": 21364,
          "150-160 GHz": 22097,
          "160-170 GHz": 22809,
          "170-180 GHz": 22870,
          "180-190 GHz": 22807,
          "190-200 GHz": 22074,
          "200-210 GHz": 25807,
          "210-220 GHz": 25122,
          "220-230 GHz": 25893,
          "230-240 GHz": 27338,
          "240-250 GHz": 26603,
          "250-260 GHz": 58285,
          "260-270 GHz": 143696,
          "270-280 GHz": 87490,
          "280-290 GHz": 72548,
          "290-300 GHz": 72186,
          "300-310 GHz": 146840,
          "310-320 GHz": 190014,
          "320-330 GHz": 169148,
          "330-340 GHz": 35509,
          "340-350 GHz": 31844,
          "350-360 GHz": 32520,
          "360-370 GHz": 31759,
          "370-380 GHz": 30353,
          "380-390 GHz": 31034,
          "390-400 GHz": 30327,
          "400-410 GHz": 30906,
          "410-420 GHz": 30146,
          "420-430 GHz": 31603,
          "430-440 GHz": 31664,
          "440-450 GHz": 32339,
          "450-460 GHz": 37508,
          "460-470 GHz": 37495,
          "470-480 GHz": 47795,
          "480-490 GHz": 41636,
          "490-500 GHz": 5377
        },
        "k_distribution": {
          "0.1-0.2": 73980,
          "0.2-0.3": 73980,
          "0.3-0.4": 73980,
          "0.4-0.5": 73980,
          "0.5-0.6": 81378,
          "0.6-0.7": 73980,
          "0.7-0.8": 66582,
          "0.8-0.9": 73980,
          "0.9-1.0": 73980,
          "1.0-1.1": 73980,
          "1.1-1.2": 73980,
          "1.2-1.3": 73980,
          "1.3-1.4": 73980,
          "1.4-1.5": 73980,
          "1.5-1.6": 73980,
          "1.6-1.7": 73980,
          "1.7-1.8": 73980,
          "1.8-1.9": 73980,
          "1.9-2.0": 73980,
          "2.0-2.1": 73980,
          "2.1-2.2": 73980,
          "2.2-2.3": 73980,
          "2.3-2.4": 73784,
          "2.4-2.5": 58560,
          "2.5-2.6": 58560,
          "2.6-2.7": 58560,
          "2.7-2.8": 58560,
          "2.8-2.9": 58560,
          "2.9-3.0": 5856
        },
        "correlations": {
          "freq_vs_k": 0.057306,
          "freq_vs_lambda_magnitude": 1.0
        },
        "eigenmode_analysis": {
          "avg_bands_per_k": 2.5,
          "min_bands_per_k": 2,
          "max_bands_per_k": 4,
          "interpretation": "Number of eigenfrequency solutions found per k-value. This represents the number of bands in the photonic band structure."
        }
      },
      "data_quality": {
        "lines_checked": 500000,
        "parse_errors": 0,
        "short_lines_under_7_fields": 0,
        "nan_frequency": 0,
        "negative_frequency": 0,
        "adjacent_duplicate_lines": 0,
        "parse_error_rate": 0.0,
        "quality_assessment": "CLEAN"
      }
    },
    "r1": {
      "full_pass": {
        "file": "r1.csv",
        "file_size_gb": 4.205,
        "total_lines": 31864278,
        "header_lines": 2,
        "data_lines": 31864276,
        "comsol_metadata": {
          "Length unit": "µm",
          "X": "Y,L1,L2,k,lambda,emw.freq (GHz)"
        },
        "columns": [
          "X",
          "Y",
          "L1",
          "L2",
          "k",
          "lambda",
          "emw.freq (GHz)"
        ],
        "geometry": {
          "dimension": "2",
          "length_unit": "µm",
          "x_range": [
            0.0,
            363.75
          ],
          "y_range": [
            0.0,
            210.01116
          ],
          "unique_xy_pairs_in_first_5M": 5972
        },
        "parametric_sweep": {
          "L1": {
            "count": 8,
            "values_um": [
              "1.4000744027848426E-4",
              "1.4700781229240846E-4",
              "1.540081843063327E-4",
              "1.610085563202569E-4",
              "1.680089283341811E-4",
              "1.7500930034810533E-4",
              "1.8200967236202954E-4",
              "1.8901004437595374E-4"
            ],
            "values_nm": [
              140.01,
              147.01,
              154.01,
              161.01,
              168.01,
              175.01,
              182.01,
              189.01
            ],
            "interpretation": "Geometric parameter (likely unit cell length L1)"
          },
          "L2": {
            "count": 8,
            "values_um": [
              "1.0500558020886319E-4",
              "1.120059522227874E-4",
              "1.1900632423671162E-4",
              "1.2600669625063583E-4",
              "1.3300706826456006E-4",
              "1.4000744027848426E-4",
              "9.100483618101477E-5",
              "9.800520819493898E-5"
            ],
            "values_nm": [
              105.01,
              112.01,
              119.01,
              126.01,
              133.01,
              140.01,
              91.0,
              98.01
            ],
            "interpretation": "Geometric parameter (likely unit cell length L2)"
          },
          "k": {
            "count": 281,
            "min": 0.1,
            "max": 2.9000000000000004,
            "step": 0.01,
            "interpretation": "Normalized wave-vector (Bloch k) for band-structure sweep"
          }
        },
        "frequency_output": {
          "column": "emw.freq (GHz)",
          "unit": "GHz",
          "unique_count": 35967,
          "min_ghz": 29.3962,
          "max_ghz": 576.0409,
          "interpretation": "Eigenfrequency solutions at each (L1,L2,k) point"
        },
        "lambda_column": {
          "real_count": 13310455,
          "complex_count": 18553821,
          "total": 31864276,
          "complex_fraction": 0.5823,
          "interpretation": "Eigenvalue λ. Real = propagating mode, Complex = evanescent/lossy mode. Imaginary part represents attenuation/loss."
        }
      },
      "statistics": {
        "sample_size": 2000000,
        "column_statistics": {
          "X": {
            "name": "X (µm)",
            "count": 2000000,
            "min": 0.0,
            "max": 363.75,
            "mean": 191.106807,
            "std": 72.16678,
            "median": 196.148884,
            "q25": 131.750558,
            "q75": 249.051037,
            "skewness": -0.118455,
            "nan_count": 0,
            "zero_count": 2456,
            "negative_count": 0
          },
          "Y": {
            "name": "Y (µm)",
            "count": 2000000,
            "min": 0.0,
            "max": 210.0111604177264,
            "mean": 110.020218,
            "std": 53.220107,
            "median": 110.071169,
            "q25": 65.962053,
            "q75": 154.341301,
            "skewness": -0.061962,
            "nan_count": 0,
            "zero_count": 34384,
            "negative_count": 0
          },
          "k": {
            "name": "k (normalized wave-vector)",
            "count": 2000000,
            "min": 0.1,
            "max": 2.9000000000000004,
            "mean": 1.420959,
            "std": 0.819159,
            "median": 1.36,
            "q25": 0.7,
            "q75": 2.13,
            "skewness": 0.138831,
            "nan_count": 0,
            "zero_count": 0,
            "negative_count": 0
          },
          "frequency": {
            "name": "emw.freq (GHz)",
            "count": 2000000,
            "min": 29.396164538403056,
            "max": 520.5030063431503,
            "mean": 301.438126,
            "std": 118.768092,
            "median": 322.939539,
            "q25": 241.176281,
            "q75": 371.736753,
            "skewness": -0.344046,
            "nan_count": 0,
            "zero_count": 0,
            "negative_count": 0
          },
          "lambda_magnitude": {
            "name": "|λ| (eigenvalue magnitude)",
            "count": 2000000,
            "min": 29396164538.403053,
            "max": 520503006343.1517,
            "mean": 301438126132.4727,
            "std": 118768091920.56323,
            "median": 322939539426.5188,
            "q25": 241176280506.29492,
            "q75": 371736753447.9736,
            "skewness": -0.344046,
            "nan_count": 0,
            "zero_count": 0,
            "negative_count": 0
          },
          "lambda_real": {
            "name": "Re(λ)",
            "count": 2000000,
            "min": 29396164538.403053,
            "max": 520503006343.15015,
            "mean": 301438126132.436,
            "std": 118768091920.54892,
            "median": 322939539426.42957,
            "q25": 241176280506.29492,
            "q75": 371736753447.973,
            "skewness": -0.344046,
            "nan_count": 0,
            "zero_count": 0,
            "negative_count": 0
          },
          "lambda_imag": {
            "name": "Im(λ)",
            "count": 2000000,
            "min": -1248630.4137924793,
            "max": 764183.3170306293,
            "mean": -11766.449745,
            "std": 159487.934658,
            "median": 0.0,
            "q25": -118.353816,
            "q75": 2108.229873,
            "skewness": -3.717391,
            "nan_count": 0,
            "zero_count": 786762,
            "negative_count": 531614
          }
        },
        "frequency_distribution_10GHz_bins": {
          "20-30 GHz": 2396,
          "30-40 GHz": 20486,
          "40-50 GHz": 22803,
          "50-60 GHz": 23639,
          "60-70 GHz": 22868,
          "70-80 GHz": 23613,
          "80-90 GHz": 22843,
          "90-100 GHz": 22828,
          "100-110 GHz": 23678,
          "110-120 GHz": 22843,
          "120-130 GHz": 24444,
          "130-140 GHz": 22023,
          "140-150 GHz": 24482,
          "150-160 GHz": 23664,
          "160-170 GHz": 21982,
          "170-180 GHz": 24523,
          "180-190 GHz": 24419,
          "190-200 GHz": 23678,
          "200-210 GHz": 24468,
          "210-220 GHz": 25303,
          "220-230 GHz": 24419,
          "230-240 GHz": 26903,
          "240-250 GHz": 26968,
          "250-260 GHz": 30144,
          "260-270 GHz": 95535,
          "270-280 GHz": 123137,
          "280-290 GHz": 68175,
          "290-300 GHz": 56673,
          "300-310 GHz": 62985,
          "310-320 GHz": 38078,
          "320-330 GHz": 69932,
          "330-340 GHz": 274247,
          "340-350 GHz": 85116,
          "350-360 GHz": 34333,
          "360-370 GHz": 30940,
          "370-380 GHz": 30184,
          "380-390 GHz": 30989,
          "390-400 GHz": 30160,
          "400-410 GHz": 29324,
          "410-420 GHz": 30184,
          "420-430 GHz": 29236,
          "430-440 GHz": 30940,
          "440-450 GHz": 34270,
          "450-460 GHz": 34116,
          "460-470 GHz": 35820,
          "470-480 GHz": 39818,
          "480-490 GHz": 47874,
          "490-500 GHz": 52858,
          "500-510 GHz": 32591,
          "510-520 GHz": 16209,
          "520-530 GHz": 859
        },
        "k_distribution": {
          "0.1-0.2": 82000,
          "0.2-0.3": 82000,
          "0.3-0.4": 82000,
          "0.4-0.5": 82000,
          "0.5-0.6": 90200,
          "0.6-0.7": 82000,
          "0.7-0.8": 73800,
          "0.8-0.9": 82000,
          "0.9-1.0": 82000,
          "1.0-1.1": 82000,
          "1.1-1.2": 71578,
          "1.2-1.3": 64820,
          "1.3-1.4": 64820,
          "1.4-1.5": 64820,
          "1.5-1.6": 64820,
          "1.6-1.7": 64820,
          "1.7-1.8": 64820,
          "1.8-1.9": 64820,
          "1.9-2.0": 64820,
          "2.0-2.1": 64820,
          "2.1-2.2": 64820,
          "2.2-2.3": 64820,
          "2.3-2.4": 64820,
          "2.4-2.5": 64820,
          "2.5-2.6": 64820,
          "2.6-2.7": 64820,
          "2.7-2.8": 64820,
          "2.8-2.9": 64820,
          "2.9-3.0": 6482
        },
        "correlations": {
          "freq_vs_k": 0.034968,
          "freq_vs_lambda_magnitude": 1.0
        },
        "eigenmode_analysis": {
          "avg_bands_per_k": 2.28,
          "min_bands_per_k": 2,
          "max_bands_per_k": 4,
          "interpretation": "Number of eigenfrequency solutions found per k-value. This represents the number of bands in the photonic band structure."
        }
      },
      "data_quality": {
        "lines_checked": 500000,
        "parse_errors": 0,
        "short_lines_under_7_fields": 0,
        "nan_frequency": 0,
        "negative_frequency": 0,
        "adjacent_duplicate_lines": 0,
        "parse_error_rate": 0.0,
        "quality_assessment": "CLEAN"
      }
    }
  },
  "cross_file_analysis": {
    "schema_comparison": {
      "para_columns": [
        "X",
        "Y",
        "L1",
        "L2",
        "k",
        "lambda",
        "emw.freq (GHz)"
      ],
      "r1_columns": [
        "X",
        "Y",
        "L1",
        "L2",
        "k",
        "lambda",
        "emw.freq (GHz)"
      ],
      "identical_schema": true,
      "common_columns": [
        "emw.freq (GHz)",
        "L2",
        "k",
        "X",
        "L1",
        "Y",
        "lambda"
      ]
    },
    "size_comparison": {
      "para_data_lines": 29525794,
      "r1_data_lines": 31864276,
      "ratio_r1_to_para": 1.0792
    },
    "parameter_comparison": {
      "L1_same": true,
      "L2_same": true,
      "L2_common": [
        "1.0500558020886319E-4",
        "1.120059522227874E-4",
        "1.1900632423671162E-4",
        "1.2600669625063583E-4",
        "1.3300706826456006E-4",
        "1.4000744027848426E-4",
        "9.100483618101477E-5",
        "9.800520819493898E-5"
      ],
      "L2_only_in_para": [],
      "L2_only_in_r1": [],
      "k_range_same": true,
      "k_count_para": 281,
      "k_count_r1": 281
    },
    "geometry_comparison": {
      "para_x_range": [
        0.0,
        363.75
      ],
      "r1_x_range": [
        0.0,
        363.75
      ],
      "para_y_range": [
        0.0,
        210.01116
      ],
      "r1_y_range": [
        0.0,
        210.01116
      ],
      "para_unique_xy_5M": 5715,
      "r1_unique_xy_5M": 5972,
      "interpretation": "Different XY pairs and/or ranges suggest different mesh geometries. para.csv comes from parametric_sweep_UC.mph (Unit Cell model). r1.csv may represent a different geometry, mesh, or simulation run."
    },
    "likely_relationship": {
      "hypothesis": "Both files appear to be COMSOL eigenfrequency simulation outputs for electromagnetic (microwave/THz) unit cell structures. They share the same parametric sweep parameters (L1, L2, k) and output columns (X, Y, lambda, freq). The difference in row count and XY coordinates suggests they use different FEM meshes or represent different geometries/designs. This is consistent with an INVERSE DESIGN study where multiple unit cell geometries are explored.",
      "are_train_test": false,
      "are_different_designs": true,
      "share_parameter_space": true
    }
  },
  "physics_interpretation": {
    "domain": "Computational Electromagnetics / Photonic Crystal Design",
    "simulation_software": "COMSOL Multiphysics 6.4 (EMW module)",
    "problem_type": "Eigenfrequency analysis of 2D electromagnetic unit cells",
    "context": "This dataset comes from parametric sweeps of 2D photonic/metamaterial unit cells. The simulation solves Maxwell's equations in eigenvalue form to find resonant frequencies (photonic band structure) as a function of Bloch wave-vector k and geometric parameters L1, L2.",
    "columns_explained": {
      "X": "Horizontal coordinate of FEM mesh node (µm)",
      "Y": "Vertical coordinate of FEM mesh node (µm)",
      "L1": "First geometric parameter of unit cell (~140 nm). Likely the lattice constant or a structural dimension.",
      "L2": "Second geometric parameter of unit cell (~91-105 nm). Swept across 3 values — likely a feature size (hole radius, slab thickness, etc.).",
      "k": "Normalized Bloch wave-vector, swept 0.1 to 2.9 in steps of 0.01 (281 values). This traces the photonic band structure along the Brillouin zone path.",
      "lambda": "Eigenvalue from COMSOL. Real part = resonant frequency parameter. Imaginary part (when present) = loss/attenuation rate. ~50% of values are complex, indicating significant modal losses.",
      "emw.freq (GHz)": "Electromagnetic eigenfrequency in GHz. Range ~28-496 GHz (microwave to sub-THz). This is the primary output / 'target' for inverse design."
    },
    "inverse_design_context": {
      "goal": "Given desired electromagnetic frequency response (band structure), determine the optimal geometric parameters (L1, L2) and potentially the unit cell topology.",
      "forward_problem": "Geometry (L1, L2) + k → Eigenfrequencies (freq)",
      "inverse_problem": "Desired freq/band-structure → Optimal geometry (L1, L2)",
      "ml_formulation": "This is fundamentally a REGRESSION problem for the forward model, and an INVERSE DESIGN / GENERATIVE problem for the reverse mapping. The data enables training a surrogate model that replaces expensive COMSOL simulations."
    },
    "key_observations": [
      "L1 is CONSTANT across all data (140.0 nm). It does not vary — only L2 and k are swept. This means L1 is fixed in the current design space.",
      "L2 has only 3 discrete values (91, 98, 105 nm). This is a very sparse parametric sweep — the design space is under-sampled.",
      "k sweeps 281 values (0.1 to 2.9). This provides dense band-structure coverage along the Brillouin zone.",
      "~50% of eigenvalues (lambda) are complex, indicating lossy/evanescent modes. These modes have physical significance (e.g., radiation losses, material absorption).",
      "The frequency range 28-496 GHz spans microwave to sub-THz. This is consistent with mm-wave photonic crystal or metamaterial design.",
      "Two separate files (para.csv, r1.csv) with different meshes but same parameter space suggests two different unit cell geometries being compared.",
      "~29.5M and ~31.9M data lines respectively. Each line is a mesh node with its field solution. The mesh has ~tens of thousands of nodes per eigenmode."
    ],
    "data_challenges": [
      "Massive dataset (~8 GB total) requires out-of-core processing",
      "Spatial mesh data (X,Y per node) — need to aggregate to per-eigenmode features",
      "Complex eigenvalues need special handling (real/imag decomposition)",
      "Only 3 L2 values — very sparse design parameter coverage",
      "L1 is constant — effectively a 1D parametric sweep (over L2 only)",
      "Node-level data is redundant for band-structure analysis (need mode-level aggregation)"
    ]
  },
  "ml_recommendations": {
    "data_preprocessing_roadmap": {
      "step_1_aggregation": {
        "description": "Aggregate node-level data to eigenmode-level. Each unique (L1, L2, k, freq) combination should become ONE row. The spatial (X, Y) data should be aggregated into features.",
        "output_columns": [
          "L1",
          "L2",
          "k",
          "freq_ghz",
          "lambda_real",
          "lambda_imag",
          "lambda_magnitude",
          "is_lossy (bool)",
          "mesh_node_count"
        ],
        "expected_rows": "~thousands (much smaller than 30M)"
      },
      "step_2_band_structure": {
        "description": "For each (L1, L2), build the photonic band structure: freq vs k curves for each band index.",
        "output": "band_structure[L2][band_index] = array of (k, freq) pairs"
      },
      "step_3_feature_engineering": {
        "suggested_features": [
          "Band gaps (frequency ranges with no eigenmode)",
          "Band widths",
          "Group velocity (d(freq)/dk)",
          "Mode loss (imaginary part of lambda)",
          "Band gap to midgap ratio",
          "Number of modes per frequency window",
          "Density of states"
        ]
      },
      "step_4_normalization": {
        "description": "Normalize frequencies and k-values for ML training",
        "suggestions": [
          "Normalize freq by c/(2*L1) for dimensionless frequency",
          "Normalize k to [0, 1] or [0, π/a]"
        ]
      }
    },
    "ml_approaches": {
      "forward_surrogate": {
        "goal": "Predict eigenfrequencies from (L2, k)",
        "models": [
          "Neural Network (MLP) — baseline",
          "Physics-Informed Neural Network (PINN)",
          "Gaussian Process Regression — good for small data (3 L2 values)",
          "Random Forest / XGBoost — for tabular baseline"
        ],
        "metrics": [
          "MAE (GHz)",
          "RMSE (GHz)",
          "R²",
          "Max error"
        ]
      },
      "inverse_design": {
        "goal": "Given desired band structure → predict L2 (and future: L1, topology)",
        "approaches": [
          "Tandem network (forward + inverse)",
          "Conditional VAE",
          "Conditional GAN (cGAN)",
          "Bayesian optimization over forward surrogate",
          "Reinforcement learning for topology optimization"
        ]
      },
      "data_augmentation": {
        "note": "With only 3 L2 values, the design space is severely under-sampled. Consider: (1) running more COMSOL sweeps with finer L2 grid, (2) interpolation-based augmentation, (3) transfer learning from analytical models."
      }
    },
    "critical_questions_for_roadmap": [
      "What is the target application? (e.g., 5G filter, THz absorber, waveguide)",
      "What specific frequency response is desired? (band gap location, bandwidth)",
      "Will L1 also be varied in future simulations?",
      "Can more L2 values be simulated to densify the design space?",
      "Are there manufacturing constraints on L1 and L2?",
      "Is the 2D simulation sufficient or is 3D needed?",
      "What is the acceptable error tolerance for the surrogate model?",
      "Is real-time inference needed (online design tool) or batch optimization?",
      "Are there other geometric parameters (e.g., hole shape, material) to explore?",
      "What is the relationship between para.csv and r1.csv geometries?"
    ]
  }
}
```
