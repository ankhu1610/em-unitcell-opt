"""Read-only diagnostic of near-tied periodic interfaces; no solver changes."""
import json
import numpy as np
from validate_energy_pwe_full import ROOT, atomic_json
from invdes.interface_pwe import normal_projector_grid, nearest_normal


def main():
    config = json.loads((ROOT/"configs/pwe_sensitivity_audit.json").read_text())
    a = config["a_um"]
    sides = np.array(config["side_multipliers"][config["reference_design_index"]])*a/np.sqrt(3)
    grid, h = config["normal_grids"][0], config["steps_um"][-1]
    minus, plus = sides.copy(), sides.copy()
    minus[0] -= h
    plus[0] += h
    pm, pp = [normal_projector_grid(a, s, grid) for s in (minus, plus)]
    difference = np.hypot(pm[0]-pp[0], pm[1]-pp[1])
    i, j = np.unravel_index(np.argmax(difference), difference.shape)
    point = a*np.array([i/grid+j/(2*grid), np.sqrt(3)*j/(2*grid)])
    evidence = {"parameter": "L1", "step_um": h, "base_sides_um": sides.tolist(),
                "normal_grid": grid, "sample_ij": [int(i), int(j)], "point_um": point.tolist(),
                "projector_change_norm": float(difference[i, j]), "perturbations": {}}
    for label, lengths in [("minus", minus), ("plus", plus)]:
        entries = []
        for index, (side, center, up) in enumerate(zip(
                lengths, [(a, a/np.sqrt(3)), (a/2, a/(2*np.sqrt(3)))], [False, True])):
            for du in [-1, 0, 1]:
                for dv in [-1, 0, 1]:
                    shift = np.array(center)+a*np.array([du+dv/2, np.sqrt(3)*dv/2])
                    distance, normal = nearest_normal((point-shift)[None, :], side, up)
                    entries.append({"squared_distance_um2": float(distance[0]),
                                    "hole_index": index, "image": [du, dv],
                                    "normal": normal[0].tolist()})
        entries.sort(key=lambda row: row["squared_distance_um2"])
        evidence["perturbations"][label] = entries[:3]
    evidence["interpretation"] = (
        "Example of competing periodic images with near-equal squared distances but different "
        "normal projectors. Floating-point tie ordering can switch under a tiny length change. "
        "This diagnoses one switching mechanism, not every grid sample.")
    atomic_json(ROOT/"results/pwe_sensitivity_audit/projector_tie_example.json", evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
