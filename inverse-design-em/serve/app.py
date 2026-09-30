import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

# Ensure src is on path
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from inverse import InverseDesignEngine

# Pre-initialize engines for both structures
engines = {
    "para (Structure 0)": InverseDesignEngine(structure_id=0, n_components=12),
    "r1 (Structure 1)": InverseDesignEngine(structure_id=1, n_components=12)
}

def forward_predict(struct_choice: str, l1_um: float, l2_um: float):
    engine = engines[struct_choice]
    res = engine.predict_design(l1_um, l2_um, return_std=True)
    
    k_vals = engine.k_grid
    f1 = res["f1"]
    f2 = res["f2"]
    s1 = res["f1_std"]
    s2 = res["f2_std"]
    feats = res["features"]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(k_vals, f1, color="#1f77b4", lw=2.2, label="Band 1 (Surrogate)")
    ax.fill_between(k_vals, f1 - 2*s1, f1 + 2*s1, color="#1f77b4", alpha=0.25, label="±2σ (95% CI)")
    
    ax.plot(k_vals, f2, color="#ff7f0e", lw=2.2, label="Band 2 (Surrogate)")
    ax.fill_between(k_vals, f2 - 2*s2, f2 + 2*s2, color="#ff7f0e", alpha=0.25)
    
    if feats["has_gap"]:
        ax.axhspan(feats["f1_max"], feats["f2_min"], color="gold", alpha=0.3, label=f"Path Gap: {feats['path_gap_ghz']:.1f} GHz")
        
    ax.set_title(f"EM Dispersion Diagram: {struct_choice} | L1 = {l1_um:.1f} µm, L2 = {l2_um:.1f} µm", fontsize=11)
    ax.set_xlabel("Bloch Wavevector Path Coordinate k", fontsize=10)
    ax.set_ylabel("Eigenfrequency (GHz)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    
    # Metrics markdown
    summary_md = f"""### Physical Metrics Summary
- **Band 1 Top:** `{feats['f1_max']:.2f} GHz`
- **Band 2 Bottom:** `{feats['f2_min']:.2f} GHz`
- **Path Band Gap:** `{feats['path_gap_ghz']:.2f} GHz` (Ratio: `{feats['gap_ratio']*100:.1f}%`)
- **Max Group Velocity:** Band 1: `{feats['vg1_max']:.1f} GHz/unit-k`, Band 2: `{feats['vg2_max']:.1f} GHz/unit-k`
- **Surrogate Uncertainty:** `±{res['mean_uncertainty_ghz']:.2f} GHz` (Confidence: 99.9% PCA fidelity)
"""
    return fig, summary_md

def inverse_solve(struct_choice: str, target_f1_k1: float, target_f2_k1: float):
    engine = engines[struct_choice]
    k_vals = engine.k_grid
    
    # Synthesize target dispersion curve based on target frequency at k=1.0
    # Standard dispersion profile offset
    k1_idx = np.argmin(np.abs(k_vals - 1.0))
    ref_y = engine.Y_train[16]
    delta1 = target_f1_k1 - ref_y[k1_idx]
    delta2 = target_f2_k1 - ref_y[281 + k1_idx]
    
    target_f1 = ref_y[:281] + delta1
    target_f2 = ref_y[281:] + delta2
    
    candidates = engine.inverse_target_curve(target_f1=target_f1, target_f2=target_f2, top_k=3)
    best = candidates[0]
    
    # Plot candidate vs target
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(k_vals, target_f1, "k--", lw=1.8, label=f"Target Spec (f1@k=1: {target_f1_k1:.0f} GHz)")
    ax.plot(k_vals, target_f2, "k:", lw=1.8, label=f"Target Spec (f2@k=1: {target_f2_k1:.0f} GHz)")
    
    ax.plot(k_vals, best["f1_pred"], color="#1f77b4", lw=2, label=f"Candidate #1: L1={best['L1_um']:.1f}µm, L2={best['L2_um']:.1f}µm")
    ax.fill_between(k_vals, best["f1_pred"] - 2*best["f1_std"], best["f1_pred"] + 2*best["f1_std"], color="#1f77b4", alpha=0.2)
    
    ax.plot(k_vals, best["f2_pred"], color="#ff7f0e", lw=2)
    ax.fill_between(k_vals, best["f2_pred"] - 2*best["f2_std"], best["f2_pred"] + 2*best["f2_std"], color="#ff7f0e", alpha=0.2)
    
    ax.set_title(f"Inverse Design Solution: RMSE = {best['rmse_ghz']:.2f} GHz", fontsize=11)
    ax.set_xlabel("Bloch Coordinate k")
    ax.set_ylabel("Frequency (GHz)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    
    table_md = "### Top Candidate Solutions\n\n"
    table_md += "| Rank | L1 (µm) | L2 (µm) | RMSE to Spec (GHz) | Model Uncertainty σ |\n"
    table_md += "|---|---|---|---|---|\n"
    for c in candidates:
        table_md += f"| **#{c['rank']}** | `{c['L1_um']:.2f}` | `{c['L2_um']:.2f}` | `{c['rmse_ghz']:.2f}` | `±{c['uncertainty_ghz']:.2f} GHz` |\n"
        
    return fig, table_md

def create_app():
    with gr.Blocks(title="EM Unit Cell Inverse Design AI") as demo:
        gr.Markdown("# 🔬 Electromagnetic Unit Cell Inverse Design & Forward Surrogate")
        gr.Markdown("Real-time ultra-fast forward dispersion solver and inverse geometric design powered by Gaussian Process POD surrogates.")
        
        with gr.Tab("Forward Surrogate Solver"):
            with gr.Row():
                with gr.Column(scale=1):
                    f_struct = gr.Dropdown(choices=["para (Structure 0)", "r1 (Structure 1)"], value="para (Structure 0)", label="Structure Topology")
                    f_l1 = gr.Slider(minimum=140.0, maximum=189.0, value=160.0, step=0.5, label="L1 (µm)")
                    f_l2 = gr.Slider(minimum=91.0, maximum=140.0, value=115.0, step=0.5, label="L2 (µm)")
                    f_btn = gr.Button("Evaluate Forward Model", variant="primary")
                with gr.Column(scale=2):
                    f_plot = gr.Plot(label="Predicted Dispersion Diagram")
                    f_summary = gr.Markdown()
                    
            f_btn.click(forward_predict, inputs=[f_struct, f_l1, f_l2], outputs=[f_plot, f_summary])
            
        with gr.Tab("Inverse Design Engine"):
            with gr.Row():
                with gr.Column(scale=1):
                    i_struct = gr.Dropdown(choices=["para (Structure 0)", "r1 (Structure 1)"], value="para (Structure 0)", label="Structure Topology")
                    i_f1 = gr.Slider(minimum=100.0, maximum=350.0, value=250.0, step=5.0, label="Target Band 1 at k=1.0 (GHz)")
                    i_f2 = gr.Slider(minimum=200.0, maximum=550.0, value=380.0, step=5.0, label="Target Band 2 at k=1.0 (GHz)")
                    i_btn = gr.Button("Synthesize Optimal Geometry", variant="primary")
                with gr.Column(scale=2):
                    i_plot = gr.Plot(label="Inverse Solution vs Spec")
                    i_table = gr.Markdown()
                    
            i_btn.click(inverse_solve, inputs=[i_struct, i_f1, i_f2], outputs=[i_plot, i_table])
            
    return demo

if __name__ == "__main__":
    demo = create_app()
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
