"""
Blind Validation Design List Generator (H3 / Phase 6).
Generates data/blind/blind_design_list.csv containing 12 recommended designs (6 per structure):
- 3 Sobol quasi-random off-grid designs inside [140, 189] x [91, 140] um.
- 3 Inverse-designed candidates chosen by the engine from pre-written engineering specs.
Pre-registers and freezes the designs with an immutable SHA-256 header hash.
"""

import os
import sys
import hashlib
import numpy as np
from scipy.stats.qmc import Sobol

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from invdes.inverse.engine import InverseDesignEngine
from invdes.inverse.spec import GapSpec, TargetCurve
from invdes.utils import set_seed

def generate_blind_set():
    print("=" * 60)
    print("Generating Blind Evaluation Set (H3)")
    print("=" * 60)

    set_seed(42)
    os.makedirs("data/blind", exist_ok=True)

    engine = InverseDesignEngine(grid_density=40)

    designs = []
    design_id = 1

    # 1. Sobol Off-Grid Designs (3 per structure)
    # Bounds: L1 in [140.0, 189.0], L2 in [91.0, 140.0]
    sobol_sampler = Sobol(d=2, scramble=True, seed=42)
    sobol_pts = sobol_sampler.random(n=6) # 3 for para, 3 for r1

    l1_sobol = 140.0 + sobol_pts[:, 0] * 49.0
    l2_sobol = 91.0 + sobol_pts[:, 1] * 49.0

    structures = ["para", "r1"]

    for s_idx, s_name in enumerate(structures):
        # 3 Sobol designs
        for i in range(3):
            idx = s_idx * 3 + i
            designs.append({
                "id": f"BLIND_{design_id:02d}",
                "structure": s_name,
                "type": "sobol_off_grid",
                "L1_um": round(float(l1_sobol[idx]), 3),
                "L2_um": round(float(l2_sobol[idx]), 3),
                "spec_description": f"Sobol quasi-random sample {i+1} for {s_name}"
            })
            design_id += 1

        # 3 Inverse-chosen designs from pre-registered specs
        # Spec A: Gap centered at 310 GHz, width >= 25 GHz
        spec_A = GapSpec(center_ghz=310.0, min_width_ghz=25.0)
        cand_A = engine.design(spec_A, structure=s_name, top_k=1)[0]
        designs.append({
            "id": f"BLIND_{design_id:02d}",
            "structure": s_name,
            "type": "inverse_gap_spec_310_25",
            "L1_um": round(float(cand_A["L1_um"]), 3),
            "L2_um": round(float(cand_A["L2_um"]), 3),
            "spec_description": f"Inverse design: gap center 310 GHz, min width 25 GHz ({cand_A['status']})"
        })
        design_id += 1

        # Spec B: Gap centered at 285 GHz, width >= 15 GHz
        spec_B = GapSpec(center_ghz=285.0, min_width_ghz=15.0)
        cand_B = engine.design(spec_B, structure=s_name, top_k=1)[0]
        designs.append({
            "id": f"BLIND_{design_id:02d}",
            "structure": s_name,
            "type": "inverse_gap_spec_285_15",
            "L1_um": round(float(cand_B["L1_um"]), 3),
            "L2_um": round(float(cand_B["L2_um"]), 3),
            "spec_description": f"Inverse design: gap center 285 GHz, min width 15 GHz ({cand_B['status']})"
        })
        design_id += 1

        # Spec C: Gap centered at 340 GHz, width >= 20 GHz
        spec_C = GapSpec(center_ghz=340.0, min_width_ghz=20.0)
        cand_C = engine.design(spec_C, structure=s_name, top_k=1)[0]
        designs.append({
            "id": f"BLIND_{design_id:02d}",
            "structure": s_name,
            "type": "inverse_gap_spec_340_20",
            "L1_um": round(float(cand_C["L1_um"]), 3),
            "L2_um": round(float(cand_C["L2_um"]), 3),
            "spec_description": f"Inverse design: gap center 340 GHz, min width 20 GHz ({cand_C['status']})"
        })
        design_id += 1

    # Format CSV lines
    csv_rows = ["id,structure,type,L1_um,L2_um,spec_description"]
    for d in designs:
        csv_rows.append(f"{d['id']},{d['structure']},{d['type']},{d['L1_um']:.3f},{d['L2_um']:.3f},\"{d['spec_description']}\"")

    body_content = "\n".join(csv_rows)
    content_hash = hashlib.sha256(body_content.encode("utf-8")).hexdigest()

    file_content = f"# SHA256: {content_hash}\n# Pre-registered blind COMSOL validation set generated per PLAN.md H3\n" + body_content

    out_path = "data/blind/blind_design_list.csv"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(file_content)

    print(f"\nSuccessfully generated {len(designs)} blind designs.")
    print(f"File: {out_path}")
    print(f"SHA-256 Digest: {content_hash}\n")
    for d in designs:
        print(f"  {d['id']} | {d['structure']:4s} | {d['type']:25s} | L1={d['L1_um']:6.3f} um | L2={d['L2_um']:6.3f} um")

if __name__ == "__main__":
    generate_blind_set()
