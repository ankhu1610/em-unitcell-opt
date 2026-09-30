# Inverse Design Engine Demo Summary

## 1. Round-Trip Recovery (Experiment E6)
- **Held-Out Design #21:** True $L_1 = 154.00$ µm, $L_2 = 126.00$ µm
- **Recovered Solution:** $L_1 = 154.04$ µm, $L_2 = 125.96$ µm
- **Parameter Error:** $\Delta L_1 = 0.037$ µm, $\Delta L_2 = 0.035$ µm
- **Curve RMSE:** 0.010 GHz (Residual $< 0.1\%$)
- **Surrogate Uncertainty:** $\pm 0.124$ GHz

## 2. Specification Matching (Experiment E7)
- Given an arbitrary band requirement, the inverse engine scanned 40,000 continuous parameter configurations and performed gradient refinement.
- **Top Solution:** $L_1 = 146.65$ µm, $L_2 = 109.89$ µm
- **RMSE:** 6.26 GHz
- **Uncertainty:** $\pm 0.90$ GHz
