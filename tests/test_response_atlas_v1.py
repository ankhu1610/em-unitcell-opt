from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

from invdes.inverse.response_atlas_v1 import (
    choose_next_point, deduplicate_observations, load_comsol_reference,
)


def config():
    return {
        "candidate_grid_size": 5, "minimum_normalized_separation": 0.12,
        "initial_coverage_evaluations": 2, "targets_GHz": [7.5, 15.0],
        "surrogate": {"smoothing": 0.02, "target_proximity_GHz": 4.0, "target_priority": 1.5},
    }


def observations():
    return [
        {"normalized_x": [0.2, 0.2], "gap_GHz": 5.0},
        {"normalized_x": [0.5, 0.5], "gap_GHz": 12.0},
        {"normalized_x": [0.8, 0.8], "gap_GHz": 20.0},
    ]


def test_deduplicate_observations_rejects_conflicting_repeat():
    repeated = observations()+[{"normalized_x": [0.2, 0.2], "gap_GHz": 5.0}]
    assert len(deduplicate_observations(repeated)) == 3
    bad = observations()+[{"normalized_x": [0.2, 0.2], "gap_GHz": 5.2}]
    import pytest
    with pytest.raises(ValueError, match="inconsistent"):
        deduplicate_observations(bad)


def test_choice_is_separated_and_coverage_first():
    point, rationale = choose_next_point(observations(), config(), completed_new_evaluations=0)
    assert rationale == "space_filling_coverage"
    assert min(np.linalg.norm(np.asarray(point)-np.asarray(row["normalized_x"])) for row in observations()) >= 0.12


def test_choice_becomes_target_aware_after_coverage():
    point, rationale = choose_next_point(observations(), config(), completed_new_evaluations=2)
    assert rationale == "target_aware_refinement"
    assert point is not None


def test_choice_can_schedule_overshoot_containment():
    cfg = config()
    cfg["surrogate"].update({"overshoot_priority": 1.0, "overshoot_every": 3})
    point, rationale = choose_next_point(observations(), cfg, completed_new_evaluations=2)
    assert rationale == "overshoot_containment"
    assert point is not None


def test_processed_comsol_reference_uses_signed_gap(tmp_path):
    f = np.zeros((2, 2, 3, 281, 2), dtype=float)
    f[..., 0] = 10.0
    f[..., 1] = 17.0
    path = tmp_path/"bands.npz"
    np.savez(path, F=f, L1_um=np.array([140., 147.]), L2_um=np.array([91., 98., 105.]))
    reference = load_comsol_reference(path, 0)
    assert reference["signed_path_gap_GHz"].shape == (2, 3)
    assert np.allclose(reference["signed_path_gap_GHz"], 7.0)
