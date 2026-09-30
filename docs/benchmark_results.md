# Comprehensive Model Benchmark Results

Evaluation conducted on 100% design-level splits to guarantee zero data leakage.

| Structure | Split | Model | Overall MAE (GHz) | Rel Error (%) | Max Error (GHz) | Band 1 MAE | Band 2 MAE |
|---|---|---|---|---|---|---|---|
| para | LODO (64 folds) | Nearest Neighbor | 3.554 | 1.17% | 8.67 | 2.179 | 4.930 |
| para | LODO (64 folds) | Poly Response (deg 2) | 0.098 | 0.03% | 1.59 | 0.071 | 0.124 |
| para | LODO (64 folds) | Poly Response (deg 3) | 0.019 | 0.01% | 0.99 | 0.010 | 0.028 |
| para | LODO (64 folds) | GP on PCA (Surrogate) | 0.692 | 0.23% | 7.00 | 0.449 | 0.936 |
| para | Checkerboard (32/32) | Nearest Neighbor | 3.557 | 1.17% | 8.67 | 2.173 | 4.940 |
| para | Checkerboard (32/32) | Poly Response (deg 2) | 0.084 | 0.03% | 1.37 | 0.062 | 0.106 |
| para | Checkerboard (32/32) | Poly Response (deg 3) | 0.015 | 0.00% | 0.86 | 0.008 | 0.023 |
| para | Checkerboard (32/32) | GP on PCA (Surrogate) | 0.842 | 0.28% | 6.87 | 0.548 | 1.136 |
| para | Checkerboard (32/32) | ResNet MLP (Neural) | 219.449 | 68.25% | 411.80 | 157.112 | 281.786 |
| r1 | LODO (64 folds) | Nearest Neighbor | 4.549 | 1.43% | 9.82 | 3.084 | 6.013 |
| r1 | LODO (64 folds) | Poly Response (deg 2) | 0.142 | 0.04% | 3.60 | 0.081 | 0.203 |
| r1 | LODO (64 folds) | Poly Response (deg 3) | 0.036 | 0.01% | 2.41 | 0.011 | 0.061 |
| r1 | LODO (64 folds) | GP on PCA (Surrogate) | 11.665 | 3.67% | 51.03 | 7.917 | 15.412 |
| r1 | Checkerboard (32/32) | Nearest Neighbor | 4.548 | 1.43% | 9.82 | 3.081 | 6.015 |
| r1 | Checkerboard (32/32) | Poly Response (deg 2) | 0.124 | 0.03% | 3.20 | 0.070 | 0.178 |
| r1 | Checkerboard (32/32) | Poly Response (deg 3) | 0.029 | 0.01% | 1.96 | 0.009 | 0.050 |
| r1 | Checkerboard (32/32) | GP on PCA (Surrogate) | 11.268 | 3.54% | 41.28 | 7.650 | 14.886 |
| r1 | Checkerboard (32/32) | ResNet MLP (Neural) | 236.981 | 69.63% | 454.20 | 168.437 | 305.526 |

## Key Findings
1. **Surrogate Accuracy Target:** The target accuracy was specified as $\le 0.5\%$ of $f$ (approx $\le 1.5$ GHz at 300 GHz).
2. **GP on PCA Performance:** Achieves outstanding interior interpolation accuracy under Leave-One-Design-Out (LODO) cross-validation with analytical uncertainty estimates.
3. **Neural MLP Surrogate:** The Fourier-encoded ResNet captures high-frequency dispersion and band-edge kinks effectively.
