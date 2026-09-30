import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

# Add src to sys.path
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from inverse import InverseDesignEngine

def main():
    parser = argparse.ArgumentParser(description="EM Unit Cell Forward Prediction & Inverse Design CLI")
    subparsers = parser.add_subparsers(dest="command", help="Mode: 'forward' or 'inverse'")

    # --- Forward command ---
    fwd_parser = subparsers.add_parser("forward", help="Given (L1, L2), predict frequency bands f1(k), f2(k) and band gap")
    fwd_parser.add_argument("--l1", type=float, required=True, help="L1 geometric dimension in micrometers (140 to 189 um)")
    fwd_parser.add_argument("--l2", type=float, required=True, help="L2 geometric dimension in micrometers (91 to 140 um)")
    fwd_parser.add_argument("--structure", type=str, default="para", choices=["para", "r1"], help="Structure topology (default: para)")

    # --- Inverse command ---
    inv_parser = subparsers.add_parser("inverse", help="Given target frequencies, solve for optimal (L1, L2)")
    inv_parser.add_argument("--f1_target", type=float, required=True, help="Target frequency for Band 1 at k=1.0 in GHz")
    inv_parser.add_argument("--f2_target", type=float, required=True, help="Target frequency for Band 2 at k=1.0 in GHz")
    inv_parser.add_argument("--structure", type=str, default="para", choices=["para", "r1"], help="Structure topology (default: para)")
    inv_parser.add_argument("--top_k", type=int, default=3, help="Number of candidate solutions to return (default: 3)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    struct_id = 0 if args.structure == "para" else 1
    engine = InverseDesignEngine(structure_id=struct_id, n_components=12)

    if args.command == "forward":
        # Validate ranges
        if not (140.0 <= args.l1 <= 189.0):
            print(f"[Warning] L1={args.l1} um is outside the sweep range [140, 189] um.")
        if not (91.0 <= args.l2 <= 140.0):
            print(f"[Warning] L2={args.l2} um is outside the sweep range [91, 140] um.")

        res = engine.predict_design(args.l1, args.l2, return_std=True)
        feats = res["features"]

        print("\n" + "="*60)
        print(f"FORWARD PREDICTION RESULTS ({args.structure.upper()})")
        print("="*60)
        print(f"Input Parameters:  L1 = {args.l1:.2f} µm, L2 = {args.l2:.2f} µm")
        print(f"Model Uncertainty: ±{res['mean_uncertainty_ghz']:.3f} GHz (95% CI)")
        print("\n--- Physical Band Metrics ---")
        print(f"Band 1 Top:       {feats['f1_max']:.2f} GHz")
        print(f"Band 2 Bottom:    {feats['f2_min']:.2f} GHz")
        print(f"Path Band Gap:    {feats['path_gap_ghz']:.2f} GHz (Ratio: {feats['gap_ratio']*100:.1f}%)")
        print(f"Has Open Gap:     {'YES' if feats['has_gap'] else 'NO'}")
        print(f"Max v_g (Band 1): {feats['vg1_max']:.2f} GHz/unit-k")
        print(f"Max v_g (Band 2): {feats['vg2_max']:.2f} GHz/unit-k")
        print("\n--- Sample Frequencies along k-path ---")
        k_indices = [0, 90, 190, 280]  # k = 0.1, 1.0, 2.0, 2.9
        for idx in k_indices:
            k_val = engine.k_grid[idx]
            f1_val = res["f1"][idx]
            f2_val = res["f2"][idx]
            s1_val = res["f1_std"][idx]
            s2_val = res["f2_std"][idx]
            print(f"k = {k_val:.2f} | Band 1: {f1_val:6.2f} ± {2*s1_val:.2f} GHz | Band 2: {f2_val:6.2f} ± {2*s2_val:.2f} GHz")
        print("="*60 + "\n")

    elif args.command == "inverse":
        print("\n" + "="*60)
        print(f"INVERSE DESIGN SYNTHESIS ({args.structure.upper()})")
        print("="*60)
        print(f"Target Spec at k=1.0: Band 1 = {args.f1_target:.1f} GHz, Band 2 = {args.f2_target:.1f} GHz")
        print("Searching 40,000 continuous parameter configurations & refining...")

        # Target synthesis
        k_vals = engine.k_grid
        k1_idx = np.argmin(np.abs(k_vals - 1.0))
        ref_y = engine.Y_train[16]
        delta1 = args.f1_target - ref_y[k1_idx]
        delta2 = args.f2_target - ref_y[281 + k1_idx]
        target_f1 = ref_y[:281] + delta1
        target_f2 = ref_y[281:] + delta2

        candidates = engine.inverse_target_curve(target_f1=target_f1, target_f2=target_f2, top_k=args.top_k)

        print(f"\nTop {len(candidates)} Recommended Physical Geometries:")
        print(f"{'Rank':<6}{'L1 (µm)':<12}{'L2 (µm)':<12}{'RMSE (GHz)':<14}{'Uncertainty σ':<16}{'Path Gap':<12}")
        print("-" * 72)
        for c in candidates:
            gap_str = f"{c['features']['path_gap_ghz']:.1f} GHz" if c['features']['has_gap'] else "None"
            print(f"#{c['rank']:<5}{c['L1_um']:<12.2f}{c['L2_um']:<12.2f}{c['rmse_ghz']:<14.2f}±{c['uncertainty_ghz']:<15.2f}{gap_str:<12}")
        print("="*72 + "\n")

if __name__ == "__main__":
    main()
