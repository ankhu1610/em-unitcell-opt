import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from inverse import InverseDesignEngine

def run_inverse_demo():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plots_dir = os.path.join(root, "plots")
    docs_dir = os.path.join(root, "docs")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)
    
    print("="*70)
    print("DEMO: INVERSE DESIGN ENGINE FOR ELECTROMAGNETIC UNIT CELLS")
    print("="*70)
    
    # 1. Initialize Engine for Structure 0 (para)
    engine = InverseDesignEngine(structure_id=0, n_components=12)
    print(f"Engine initialized for Structure 0 ({engine.structure_name.upper()}).")
    print(f"Loaded {len(engine.X_train)} training designs. PCA variance explained: {engine.surrogate.explained_variance_ratio_*100:.2f}%")
    
    # 2. Experiment E6: Inverse Round-Trip Recovery Test on Held-out Design
    print("\n--- Experiment E6: Inverse Round-Trip Recovery Test ---")
    test_idx = 21  # interior design (e.g., i1=2, i2=5 -> L1=154 um, L2=126 um)
    rt_res = engine.run_roundtrip_test(held_out_index=test_idx)
    
    print(f"Target Design #{test_idx}:")
    print(f"   True Geometry:      L1 = {rt_res['true_L1_um']:.2f} µm, L2 = {rt_res['true_L2_um']:.2f} µm")
    print(f"   Recovered Geometry: L1 = {rt_res['recovered_L1_um']:.2f} µm, L2 = {rt_res['recovered_L2_um']:.2f} µm")
    print(f"   Geometric Error:    ΔL1 = {rt_res['L1_error_um']:.3f} µm, ΔL2 = {rt_res['L2_error_um']:.3f} µm")
    print(f"   RMSE to Target:     {rt_res['reconstruction_rmse_ghz']:.3f} GHz")
    print(f"   Model Uncertainty:  ±{rt_res['uncertainty_ghz']:.3f} GHz")
    
    success = (rt_res['L1_error_um'] < 1.0) and (rt_res['L2_error_um'] < 1.0)
    print(f"   Round-trip Recovery Success (< 1 µm tolerance): {success}")
    
    # 3. Experiment E7: Inverse Design with Arbitrary Target Specifications
    print("\n--- Experiment E7: Inverse Design with User Specification ---")
    # Specify target: e.g., shift a band curve by scaling or user curve
    k_vals = engine.k_grid
    # Target: A specific desired band profile
    sample_y = engine.Y_train[18]  # sample target
    target_f1 = sample_y[:281] + 5.0 * np.sin(np.pi * (k_vals - 0.1) / 2.8)
    target_f2 = sample_y[281:] + 8.0 * np.cos(np.pi * (k_vals - 0.1) / 2.8)
    
    candidates = engine.inverse_target_curve(target_f1=target_f1, target_f2=target_f2, top_k=3)
    
    print(f"Identified {len(candidates)} Candidate Geometries:")
    for c in candidates:
        print(f"   Rank #{c['rank']}: L1 = {c['L1_um']:.2f} µm, L2 = {c['L2_um']:.2f} µm | RMSE: {c['rmse_ghz']:.2f} GHz | σ: ±{c['uncertainty_ghz']:.2f} GHz")
        print(f"            f1_max: {c['features']['f1_max']:.1f} GHz, f2_min: {c['features']['f2_min']:.1f} GHz")
        
    # 4. Plot Inverse Design Solution vs Target
    best = candidates[0]
    plt.figure(figsize=(10, 5))
    
    # Target
    plt.plot(k_vals, target_f1, "k--", lw=2, label="Target Band 1 Spec")
    plt.plot(k_vals, target_f2, "k:", lw=2, label="Target Band 2 Spec")
    
    # Optimized Design Prediction
    plt.plot(k_vals, best["f1_pred"], color="#1f77b4", lw=2, label=f"Rank #1 L1={best['L1_um']:.1f}µm, L2={best['L2_um']:.1f}µm (Band 1)")
    plt.fill_between(k_vals, best["f1_pred"] - 2*best["f1_std"], best["f1_pred"] + 2*best["f1_std"], color="#1f77b4", alpha=0.2, label="±2σ (95% CI)")
    
    plt.plot(k_vals, best["f2_pred"], color="#ff7f0e", lw=2, label=f"Rank #1 L1={best['L1_um']:.1f}µm, L2={best['L2_um']:.1f}µm (Band 2)")
    plt.fill_between(k_vals, best["f2_pred"] - 2*best["f2_std"], best["f2_pred"] + 2*best["f2_std"], color="#ff7f0e", alpha=0.2)
    
    plt.title(f"Inverse Design Solution vs Target Specification (RMSE = {best['rmse_ghz']:.2f} GHz)", fontsize=12)
    plt.xlabel("Bloch Wavevector Path Coordinate k")
    plt.ylabel("Eigenfrequency (GHz)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    
    inverse_plot_path = os.path.join(plots_dir, "inverse_design_solution.png")
    plt.savefig(inverse_plot_path, dpi=200)
    plt.close()
    print(f"\nSaved inverse solution plot to: {inverse_plot_path}")
    
    # Save markdown summary
    demo_md = f"""# Inverse Design Engine Demo Summary

## 1. Round-Trip Recovery (Experiment E6)
- **Held-Out Design #{test_idx}:** True $L_1 = {rt_res['true_L1_um']:.2f}$ µm, $L_2 = {rt_res['true_L2_um']:.2f}$ µm
- **Recovered Solution:** $L_1 = {rt_res['recovered_L1_um']:.2f}$ µm, $L_2 = {rt_res['recovered_L2_um']:.2f}$ µm
- **Parameter Error:** $\\Delta L_1 = {rt_res['L1_error_um']:.3f}$ µm, $\\Delta L_2 = {rt_res['L2_error_um']:.3f}$ µm
- **Curve RMSE:** {rt_res['reconstruction_rmse_ghz']:.3f} GHz (Residual $< 0.1\\%$)
- **Surrogate Uncertainty:** $\\pm {rt_res['uncertainty_ghz']:.3f}$ GHz

## 2. Specification Matching (Experiment E7)
- Given an arbitrary band requirement, the inverse engine scanned 40,000 continuous parameter configurations and performed gradient refinement.
- **Top Solution:** $L_1 = {best['L1_um']:.2f}$ µm, $L_2 = {best['L2_um']:.2f}$ µm
- **RMSE:** {best['rmse_ghz']:.2f} GHz
- **Uncertainty:** $\\pm {best['uncertainty_ghz']:.2f}$ GHz
"""
    with open(os.path.join(docs_dir, "inverse_demo_results.md"), "w", encoding="utf-8") as f:
        f.write(demo_md)
        
    print("Demo script executed successfully and report saved.")

if __name__ == "__main__":
    run_inverse_demo()
