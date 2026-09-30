"""
Electromagnetic Unit Cell Inverse Design & Forward AI Platform
==============================================================
Live Interactive Web Application powered by Gaussian Process POD Surrogates & PyTorch PINN.
Provides:
  - Real-time unit cell geometry visualizer (L1, L2, lattice period a = 200 um)
  - Full Brillouin zone band dispersion diagram (Gamma - X - M - Gamma)
  - Confidence uncertainty intervals (±2σ, 95% Bayesian credible interval)
  - Optional live overlay with ground-truth COMSOL simulation
  - Instantaneous inverse design synthesis (<1 sec for 40,000 design candidates)
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gradio as gr

# Ensure src is on sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(root_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from inverse import InverseDesignEngine

# Load ground-truth dataset for optional live verification overlay
data_path = os.path.join(root_dir, "data", "processed", "bands.npz")
has_gt = os.path.exists(data_path)
if has_gt:
    gt_data = np.load(data_path)
    gt_F = gt_data["F"]            # (2, 8, 8, 281, 2)
    gt_L1 = gt_data["L1_um"]        # (8,)
    gt_L2 = gt_data["L2_um"]        # (8,)
    gt_k = gt_data["k"]            # (281,)
else:
    gt_F, gt_L1, gt_L2, gt_k = None, None, None, None

# Pre-initialize surrogate engines for both topologies
print("Initializing EM Inverse Design Engines...")
engines = {
    "para (Structure 0)": InverseDesignEngine(structure_id=0, n_components=12),
    "r1 (Structure 1)": InverseDesignEngine(structure_id=1, n_components=12)
}
print("Engines initialized successfully.")


def render_unit_cell_and_bands(struct_choice: str, l1_um: float, l2_um: float, show_gt: bool = True):
    """
    Renders a unified 2-panel figure:
      Panel 1: Physical Unit Cell Geometry with dynamic dimensions L1, L2
      Panel 2: Band Dispersion Diagram across Gamma -> X -> M -> Gamma with uncertainty and gap
    """
    t_start = time.perf_counter()
    engine = engines[struct_choice]
    struct_id = 0 if "para" in struct_choice else 1
    
    # 1. Evaluate Forward Surrogate
    res = engine.predict_design(l1_um, l2_um, return_std=True)
    k_vals = engine.k_grid
    f1 = res["f1"]
    f2 = res["f2"]
    s1 = res["f1_std"]
    s2 = res["f2_std"]
    feats = res["features"]
    eval_ms = (time.perf_counter() - t_start) * 1000

    # 2. Check for matching or nearest ground truth
    gt_match = None
    if has_gt and show_gt:
        i1 = np.argmin(np.abs(gt_L1 - l1_um))
        i2 = np.argmin(np.abs(gt_L2 - l2_um))
        # If within 0.1 um, it's an exact grid point
        is_exact = (abs(gt_L1[i1] - l1_um) < 0.1) and (abs(gt_L2[i2] - l2_um) < 0.1)
        gt_f1 = gt_F[struct_id, i1, i2, :, 0]
        gt_f2 = gt_F[struct_id, i1, i2, :, 1]
        gt_match = {
            "exact": is_exact,
            "l1": gt_L1[i1],
            "l2": gt_L2[i2],
            "f1": gt_f1,
            "f2": gt_f2
        }

    # 3. Create Figure with custom aesthetic
    plt.style.use("fast")
    fig = plt.figure(figsize=(13, 5), dpi=120)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.7], wspace=0.25)
    
    # --- PANEL 1: Unit Cell Physical Geometry ---
    ax_cell = fig.add_subplot(gs[0, 0])
    a = 200.0  # Unit cell pitch in um
    
    # Unit cell boundary (dielectric background)
    rect_bg = patches.Rectangle((-a/2, -a/2), a, a, linewidth=2, edgecolor="#2c3e50", facecolor="#ecf0f1", alpha=0.9, linestyle="--")
    ax_cell.add_patch(rect_bg)
    
    # Inclusion (L1 x L2)
    # In para structure: rectangle centered at origin
    rect_inc = patches.Rectangle((-l1_um/2, -l2_um/2), l1_um, l2_um, linewidth=2, edgecolor="#16a085", facecolor="#1abc9c", alpha=0.85)
    ax_cell.add_patch(rect_inc)
    
    # Dimension arrows & annotations
    ax_cell.annotate(f"L1 = {l1_um:.1f} µm", xy=(-l1_um/2, -l2_um/2 - 12), xytext=(l1_um/2, -l2_um/2 - 12),
                     arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.5),
                     ha="center", va="top", color="#c0392b", fontweight="bold", fontsize=9)
    
    ax_cell.annotate(f"L2 = {l2_um:.1f} µm", xy=(l1_um/2 + 10, -l2_um/2), xytext=(l1_um/2 + 10, l2_um/2),
                     arrowprops=dict(arrowstyle="<->", color="#2980b9", lw=1.5),
                     ha="left", va="center", color="#2980b9", fontweight="bold", fontsize=9)

    ax_cell.text(0, 0, f"Inclusion\n({struct_choice.split()[0].upper()})", ha="center", va="center", color="#ffffff", fontweight="bold", fontsize=10)
    ax_cell.text(-a/2 + 5, a/2 - 10, "Unit Cell: 200 × 200 µm", ha="left", va="top", color="#7f8c8d", fontsize=8, fontstyle="italic")
    
    ax_cell.set_xlim(-a/2 - 25, a/2 + 35)
    ax_cell.set_ylim(-a/2 - 25, a/2 + 25)
    ax_cell.set_aspect("equal", adjustable="box")
    ax_cell.set_title(f"Physical Unit Cell Geometry\n(Pitch a = 200 µm)", fontsize=11, fontweight="bold", pad=10)
    ax_cell.set_xlabel("x (µm)", fontsize=9)
    ax_cell.set_ylabel("y (µm)", fontsize=9)
    ax_cell.grid(True, linestyle=":", alpha=0.4)

    # --- PANEL 2: Band Dispersion Diagram ---
    ax_disp = fig.add_subplot(gs[0, 1])
    
    # High-symmetry path dividers: k=0.1 (Gamma), k=1.0 (X), k=2.0 (M), k=2.9 (Gamma)
    ax_disp.axvline(1.0, color="#7f8c8d", linestyle="--", lw=1.2, alpha=0.7)
    ax_disp.axvline(2.0, color="#7f8c8d", linestyle="--", lw=1.2, alpha=0.7)
    
    # Band gap region shading
    if feats["has_gap"]:
        ax_disp.axhspan(feats["f1_max"], feats["f2_min"], color="#f39c12", alpha=0.22,
                        label=f"Path Gap: {feats['path_gap_ghz']:.1f} GHz ({feats['gap_ratio']*100:.1f}%)")
    
    # Ground truth overlay if enabled
    if gt_match is not None and show_gt:
        lbl_gt = f"COMSOL Truth (L1={gt_match['l1']:.0f}, L2={gt_match['l2']:.0f})"
        ax_disp.plot(gt_k, gt_match["f1"], "k--", lw=1.8, alpha=0.85, label=lbl_gt)
        ax_disp.plot(gt_k, gt_match["f2"], "k--", lw=1.8, alpha=0.85)

    # Predicted bands with confidence envelopes
    line1, = ax_disp.plot(k_vals, f1, color="#2980b9", lw=2.4, label="Band 1 (Surrogate)")
    ax_disp.fill_between(k_vals, f1 - 2*s1, f1 + 2*s1, color="#2980b9", alpha=0.25, label="±2σ (95% CI)")
    
    line2, = ax_disp.plot(k_vals, f2, color="#d35400", lw=2.4, label="Band 2 (Surrogate)")
    ax_disp.fill_between(k_vals, f2 - 2*s2, f2 + 2*s2, color="#d35400", alpha=0.25)

    # High-symmetry tick labels on X-axis
    ax_disp.set_xticks([0.1, 1.0, 2.0, 2.9])
    ax_disp.set_xticklabels([r"$\Gamma$ (0.1)", r"$\mathrm{X}$ (1.0)", r"$\mathrm{M}$ (2.0)", r"$\Gamma$ (2.9)"], fontsize=10, fontweight="bold")
    
    ax_disp.set_title(f"EM Dispersion Diagram across Brillouin Zone\n({struct_choice} | Eval: {eval_ms:.2f} ms)", fontsize=11, fontweight="bold", pad=10)
    ax_disp.set_xlabel("Bloch Wavevector Path Coordinate k", fontsize=10)
    ax_disp.set_ylabel("Eigenfrequency (GHz)", fontsize=10)
    ax_disp.grid(True, linestyle=":", alpha=0.5)
    ax_disp.legend(loc="upper left", framealpha=0.9, fontsize=8)
    
    plt.tight_layout()

    # Markdown Summary Cards
    gap_badge = f"<span style='color: #27ae60; font-weight: bold;'>OPEN GAP ({feats['path_gap_ghz']:.2f} GHz)</span>" if feats["has_gap"] else "<span style='color: #c0392b; font-weight: bold;'>CLOSED GAP</span>"
    
    summary_md = fr"""
### 📊 Physical Response & Model Diagnostics
| Metric | Value | Reference / Notes |
|:---|:---|:---|
| **Band Gap Status** | {gap_badge} | Gap-to-Mid Ratio: `{feats['gap_ratio']*100:.1f}%` |
| **Band 1 Peak ($f_{{1,\max}}$)** | `{feats['f1_max']:.2f} GHz` | Upper boundary of acoustic branch |
| **Band 2 Valley ($f_{{2,\min}}$)** | `{feats['f2_min']:.2f} GHz` | Lower boundary of optical branch |
| **Max Group Velocity ($v_g$)** | Band 1: `{feats['vg1_max']:.1f}` \| Band 2: `{feats['vg2_max']:.1f}` GHz/unit-k | Central-difference derivative $\partial\omega/\partial k$ |
| **Surrogate Uncertainty** | `±{res['mean_uncertainty_ghz']:.3f} GHz` | 95% Bayesian Confidence (2σ) |
| **Inference Time** | `{eval_ms:.2f} ms` | **~{300000 / max(eval_ms, 0.1):,.0f}× faster** than COMSOL FEA (~5 min) |
"""
    return fig, summary_md


def run_inverse_synthesis(struct_choice: str, target_f1_k1: float, target_f2_k1: float):
    """
    Inverse synthesis: solves for optimal L1, L2 geometry given target eigenfrequencies at X-point (k=1.0).
    """
    t_start = time.perf_counter()
    engine = engines[struct_choice]
    k_vals = engine.k_grid
    
    # Synthesize target dispersion curve
    k1_idx = np.argmin(np.abs(k_vals - 1.0))
    ref_y = engine.Y_train[16]
    delta1 = target_f1_k1 - ref_y[k1_idx]
    delta2 = target_f2_k1 - ref_y[281 + k1_idx]
    
    target_f1 = ref_y[:281] + delta1
    target_f2 = ref_y[281:] + delta2
    
    candidates = engine.inverse_target_curve(target_f1=target_f1, target_f2=target_f2, top_k=3)
    best = candidates[0]
    solve_ms = (time.perf_counter() - t_start) * 1000
    
    # Plot candidate vs target
    fig = plt.figure(figsize=(13, 5), dpi=120)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.7], wspace=0.25)
    
    # Left: Synthesized unit cell geometry
    ax_cell = fig.add_subplot(gs[0, 0])
    a = 200.0
    rect_bg = patches.Rectangle((-a/2, -a/2), a, a, linewidth=2, edgecolor="#2c3e50", facecolor="#ecf0f1", linestyle="--")
    ax_cell.add_patch(rect_bg)
    rect_inc = patches.Rectangle((-best["L1_um"]/2, -best["L2_um"]/2), best["L1_um"], best["L2_um"],
                                 linewidth=2, edgecolor="#27ae60", facecolor="#2ecc71", alpha=0.85)
    ax_cell.add_patch(rect_inc)
    
    ax_cell.annotate(f"L1 = {best['L1_um']:.1f} µm", xy=(-best["L1_um"]/2, -best["L2_um"]/2 - 12), xytext=(best["L1_um"]/2, -best["L2_um"]/2 - 12),
                     arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.5),
                     ha="center", va="top", color="#c0392b", fontweight="bold", fontsize=9)
    ax_cell.annotate(f"L2 = {best['L2_um']:.1f} µm", xy=(best["L1_um"]/2 + 10, -best["L2_um"]/2), xytext=(best["L1_um"]/2 + 10, best["L2_um"]/2),
                     arrowprops=dict(arrowstyle="<->", color="#2980b9", lw=1.5),
                     ha="left", va="center", color="#2980b9", fontweight="bold", fontsize=9)
    ax_cell.text(0, 0, f"Synthesized\nUnit Cell", ha="center", va="center", color="#ffffff", fontweight="bold", fontsize=10)
    ax_cell.set_xlim(-a/2 - 25, a/2 + 35)
    ax_cell.set_ylim(-a/2 - 25, a/2 + 25)
    ax_cell.set_aspect("equal", adjustable="box")
    ax_cell.set_title("Synthesized Physical Geometry", fontsize=11, fontweight="bold", pad=10)
    ax_cell.set_xlabel("x (µm)", fontsize=9)
    ax_cell.set_ylabel("y (µm)", fontsize=9)
    ax_cell.grid(True, linestyle=":", alpha=0.4)
    
    # Right: Target vs Synthesized curve
    ax_disp = fig.add_subplot(gs[0, 1])
    ax_disp.axvline(1.0, color="#7f8c8d", linestyle="--", lw=1.2, alpha=0.7)
    ax_disp.axvline(2.0, color="#7f8c8d", linestyle="--", lw=1.2, alpha=0.7)
    
    ax_disp.plot(k_vals, target_f1, "k--", lw=2, label=f"Target Spec (Band 1 @ X={target_f1_k1:.0f} GHz)")
    ax_disp.plot(k_vals, target_f2, "k:", lw=2, label=f"Target Spec (Band 2 @ X={target_f2_k1:.0f} GHz)")
    
    ax_disp.plot(k_vals, best["f1_pred"], color="#2980b9", lw=2.2, label=f"Solution #1: L1={best['L1_um']:.1f}, L2={best['L2_um']:.1f} µm")
    ax_disp.fill_between(k_vals, best["f1_pred"] - 2*best["f1_std"], best["f1_pred"] + 2*best["f1_std"], color="#2980b9", alpha=0.2)
    ax_disp.plot(k_vals, best["f2_pred"], color="#d35400", lw=2.2)
    ax_disp.fill_between(k_vals, best["f2_pred"] - 2*best["f2_std"], best["f2_pred"] + 2*best["f2_std"], color="#d35400", alpha=0.2)
    
    ax_disp.set_xticks([0.1, 1.0, 2.0, 2.9])
    ax_disp.set_xticklabels([r"$\Gamma$", r"$\mathrm{X}$", r"$\mathrm{M}$", r"$\Gamma$"], fontsize=10, fontweight="bold")
    ax_disp.set_title(f"Target vs Synthesized Response (RMSE: {best['rmse_ghz']:.2f} GHz | Solved: {solve_ms:.0f} ms)", fontsize=11, fontweight="bold", pad=10)
    ax_disp.set_xlabel("Bloch Coordinate k", fontsize=10)
    ax_disp.set_ylabel("Frequency (GHz)", fontsize=10)
    ax_disp.grid(True, linestyle=":", alpha=0.4)
    ax_disp.legend(loc="upper left", fontsize=8)
    
    plt.tight_layout()
    
    table_md = f"""
### 🎯 Top 3 Recommended Geometries (Solved in `{solve_ms:.0f} ms` across 40,000 parameter points)
| Rank | L1 (µm) | L2 (µm) | Fit RMSE | Model Uncertainty (2σ) | Synthesized Gap |
|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for c in candidates:
        gap_info = f"`{c['features']['path_gap_ghz']:.1f} GHz`" if c['features']['has_gap'] else "None"
        table_md += f"| **#{c['rank']}** | `{c['L1_um']:.2f} µm` | `{c['L2_um']:.2f} µm` | `{c['rmse_ghz']:.2f} GHz` | `±{c['uncertainty_ghz']:.2f} GHz` | {gap_info} |\n"
        
    return fig, table_md


def create_app():
    custom_css = """
    .gradio-container { max-width: 1400px !important; }
    .header-box { text-align: center; margin-bottom: 20px; }
    """
    with gr.Blocks(title="EM Unit Cell AI Design Studio", theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"), css=custom_css) as demo:
        gr.Markdown(
            r"""
            # 🔬 Electromagnetic Unit Cell AI Design Studio
            ### Ultra-Fast Physics-Informed Forward Surrogates & Inverse Geometric Synthesis
            Explore physical unit cell geometries, dispersion curves across the Brillouin zone ($\Gamma - X - M - \Gamma$), and synthesize optimal unit cells instantaneously without requiring COMSOL finite element licenses.
            """
        )
        
        with gr.Tabs():
            # TAB 1: Live Forward Solver
            with gr.Tab("⚡ Forward Dispersion Solver & Unit Cell Visualizer"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 🎛️ Geometric Parameters")
                        f_struct = gr.Dropdown(choices=["para (Structure 0)", "r1 (Structure 1)"], value="para (Structure 0)", label="Topology Model")
                        f_l1 = gr.Slider(minimum=140.0, maximum=189.0, value=160.0, step=0.5, label="Inclusion Length L1 (µm)")
                        f_l2 = gr.Slider(minimum=91.0, maximum=140.0, value=115.0, step=0.5, label="Inclusion Width L2 (µm)")
                        f_show_gt = gr.Checkbox(value=True, label="Overlay COMSOL Ground Truth (if available)")
                        f_btn = gr.Button("Evaluate Design", variant="primary")
                    
                    with gr.Column(scale=3):
                        f_plot = gr.Plot(label="Live Unit Cell & Band Dispersion Diagram")
                        f_metrics = gr.Markdown()
                
                # Auto-update when sliders change or button clicked
                for comp in [f_l1, f_l2, f_struct, f_show_gt]:
                    comp.change(fn=render_unit_cell_and_bands, inputs=[f_struct, f_l1, f_l2, f_show_gt], outputs=[f_plot, f_metrics])
                f_btn.click(fn=render_unit_cell_and_bands, inputs=[f_struct, f_l1, f_l2, f_show_gt], outputs=[f_plot, f_metrics])
                demo.load(fn=render_unit_cell_and_bands, inputs=[f_struct, f_l1, f_l2, f_show_gt], outputs=[f_plot, f_metrics])

            # TAB 2: Inverse Design Engine
            with gr.Tab("🎯 Inverse Design Geometric Synthesis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 🎯 Target Spectral Specifications")
                        i_struct = gr.Dropdown(choices=["para (Structure 0)", "r1 (Structure 1)"], value="para (Structure 0)", label="Topology Model")
                        i_f1 = gr.Slider(minimum=100.0, maximum=350.0, value=270.0, step=5.0, label="Target Band 1 at X (k=1.0) in GHz")
                        i_f2 = gr.Slider(minimum=200.0, maximum=550.0, value=340.0, step=5.0, label="Target Band 2 at X (k=1.0) in GHz")
                        i_btn = gr.Button("Synthesize Optimal Geometry", variant="primary")
                    
                    with gr.Column(scale=3):
                        i_plot = gr.Plot(label="Synthesized Unit Cell & Spectral Match")
                        i_table = gr.Markdown()
                
                i_btn.click(fn=run_inverse_synthesis, inputs=[i_struct, i_f1, i_f2], outputs=[i_plot, i_table])
                # Auto load initial inverse demo
                demo.load(fn=run_inverse_synthesis, inputs=[i_struct, i_f1, i_f2], outputs=[i_plot, i_table])
                
    return demo


if __name__ == "__main__":
    demo = create_app()
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
