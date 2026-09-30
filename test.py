import sys
sys.path.append("inverse-design-em/src")

from inverse import InverseDesignEngine

# 1. Initialize engine for 'para' (structure_id=0) or 'r1' (structure_id=1)
engine = InverseDesignEngine(structure_id=0)

# 2. Forward Prediction:
# Pass any L1 in [140, 189] um and L2 in [91, 140] um
pred = engine.predict_design(l1_um=162.5, l2_um=110.0, return_std=True)

print("Band 1 frequencies f1(k):", pred["f1"].shape)       # (281,) values in GHz
print("Band 2 frequencies f2(k):", pred["f2"].shape)       # (281,) values in GHz
print("Path Band Gap (GHz):", pred["features"]["path_gap_ghz"])
print("Surrogate Uncertainty (GHz): ±", pred["mean_uncertainty_ghz"])

# 3. Inverse Design:
# Pass your desired target curve(s) to find optimal (L1, L2)
target_f1 = pred["f1"] + 3.0  # example target
candidates = engine.inverse_target_curve(target_f1=target_f1, top_k=3)

for c in candidates:
    print(f"Rank #{c['rank']}: L1 = {c['L1_um']:.2f} µm, L2 = {c['L2_um']:.2f} µm | RMSE: {c['rmse_ghz']:.2f} GHz")
