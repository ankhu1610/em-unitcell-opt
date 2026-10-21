"""Checkpointed adaptive PWE response atlas with a separate COMSOL reference layer.

The PWE surrogate is fit only to M20 PWE observations.  The processed COMSOL
dataset is rendered in a separate panel so disagreement remains visible.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.interpolate import RBFInterpolator

from validate_energy_pwe_full import atomic_json
from invdes.inverse.quadrature_forward import QuadratureForward
from invdes.projector_experiment import DeadlineReached, digest


VERSION = "pure_pwe_response_atlas_v1"


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def point_key(x):
    x = np.asarray(x, dtype=float)
    if x.shape != (2,) or not np.all(np.isfinite(x)) or np.any(x < 0) or np.any(x > 1):
        raise ValueError("Atlas points must be finite normalized pairs in [0, 1]")
    return tuple(float(v).hex() for v in x)


def deduplicate_observations(observations):
    """Collapse exact repeated PWE points while rejecting contradictory values."""
    grouped = {}
    for row in observations:
        x = np.asarray(row["normalized_x"], dtype=float)
        gap = float(row["gap_GHz"])
        key = point_key(x)
        if not np.isfinite(gap):
            raise ValueError("Atlas observation contains a nonfinite gap")
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = {"normalized_x": x.tolist(), "gap_GHz": gap}
        elif not np.isclose(existing["gap_GHz"], gap, rtol=0.0, atol=1e-10):
            raise ValueError("Repeated PWE point has inconsistent sampled-path gap")
    return [grouped[key] for key in sorted(grouped)]


def fit_pwe_surface(observations, smoothing):
    points = np.asarray([row["normalized_x"] for row in observations], dtype=float)
    values = np.asarray([row["gap_GHz"] for row in observations], dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("At least three distinct PWE observations are required for an atlas surface")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(values)):
        raise ValueError("Atlas surface requires finite observations")
    # Omit the polynomial tail: the seed points may temporarily lie along a
    # search trajectory, where a linear tail would be rank-deficient.
    return RBFInterpolator(points, values, kernel="thin_plate_spline",
                           smoothing=float(smoothing), degree=-1)


def candidate_grid(size):
    if not isinstance(size, int) or size < 3:
        raise ValueError("candidate_grid_size must be an integer of at least three")
    axis = np.linspace(0.0, 1.0, size)
    return np.asarray([(x, y) for x in axis for y in axis], dtype=float)


def choose_next_point(observations, config, completed_new_evaluations):
    """Choose a deterministic coverage point, then target-aware refinement point."""
    known = np.asarray([row["normalized_x"] for row in observations], dtype=float)
    grid = candidate_grid(int(config["candidate_grid_size"]))
    distances = np.min(np.linalg.norm(grid[:, None, :] - known[None, :, :], axis=2), axis=1)
    eligible = distances >= float(config["minimum_normalized_separation"])
    if not np.any(eligible):
        return None, "no_eligible_point"
    if completed_new_evaluations < int(config["initial_coverage_evaluations"]):
        score = distances
        rationale = "space_filling_coverage"
    else:
        model = fit_pwe_surface(observations, config["surrogate"]["smoothing"])
        prediction = np.asarray(model(grid), dtype=float)
        overshoot_priority = float(config["surrogate"].get("overshoot_priority", 0.0))
        overshoot_every = int(config["surrogate"].get("overshoot_every", 0))
        choose_overshoot = (overshoot_priority > 0 and overshoot_every > 0
                            and (completed_new_evaluations+1) % overshoot_every == 0)
        if choose_overshoot:
            values = np.asarray([row["gap_GHz"] for row in observations], dtype=float)
            scale = max(float(np.ptp(values)), np.finfo(float).eps)
            excess = np.maximum(np.maximum(values.min()-prediction, prediction-values.max()), 0.0)/scale
            score = distances*(1.0+overshoot_priority*excess)
            rationale = "overshoot_containment"
        else:
            target_distance = np.min(
                np.abs(prediction[:, None] - np.asarray(config["targets_GHz"], dtype=float)[None, :]), axis=1)
            proximity = max(float(config["surrogate"]["target_proximity_GHz"]), np.finfo(float).eps)
            interest = np.exp(-target_distance/proximity)
            score = distances*(1.0+float(config["surrogate"]["target_priority"])*interest)
            rationale = "target_aware_refinement"
    score = np.where(eligible, score, -np.inf)
    maximum = np.max(score)
    # Lexicographic tie break keeps runs deterministic across platforms.
    indices = np.flatnonzero(np.isclose(score, maximum, rtol=0.0, atol=1e-15))
    index = min(indices, key=lambda i: (grid[i, 0], grid[i, 1]))
    return grid[index].tolist(), rationale


def load_comsol_reference(path, structure_index):
    """Load the signed path-gap field from the processed data contract only."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as dataset:
        required = {"F", "L1_um", "L2_um"}
        if not required.issubset(dataset.files):
            raise ValueError("Processed COMSOL archive does not satisfy the data contract")
        frequencies = np.asarray(dataset["F"], dtype=float)
        l1, l2 = np.asarray(dataset["L1_um"], dtype=float), np.asarray(dataset["L2_um"], dtype=float)
    if (frequencies.ndim != 5 or frequencies.shape[-2:] != (281, 2)
            or not 0 <= structure_index < frequencies.shape[0]
            or frequencies.shape[1:3] != (len(l1), len(l2))):
        raise ValueError("Processed COMSOL archive has an unexpected band-grid shape")
    bands = frequencies[structure_index]
    gap = np.min(bands[..., 1], axis=-1)-np.max(bands[..., 0], axis=-1)
    if not (np.all(np.isfinite(l1)) and np.all(np.isfinite(l2)) and np.all(np.isfinite(gap))):
        raise ValueError("Processed COMSOL reference contains nonfinite values")
    return {"L1_um": l1, "L2_um": l2, "signed_path_gap_GHz": gap}


def render_report(result):
    source = result["seed_observations"]
    new = result["new_evaluations"]
    lines = [
        "# Adaptive pure-PWE response atlas", "",
        f"Status: {result['status']}.",
        f"Validated M20 PWE seed observations: {source}.",
        f"New M20 PWE evaluations: {len(new)}.",
        f"Distinct PWE points in the interpolation: {result['distinct_pwe_points']}.", "",
        "The PWE contour panel is fit only to pure-PWE M20 sampled-path-gap values.",
        "The COMSOL panel is a separately labelled view of data/processed/bands.npz; it is not used",
        "to train, select, calibrate or accept PWE candidates. A visible disagreement is evidence, not an error to hide.",
        "Neither panel certifies a full-Brillouin-zone gap or continuum convergence.",
        "Any contour-derived design must still receive the existing full-path M20 and independent M25 checks.",
    ]
    if new:
        lines += ["", "|New point|L1 um|L2 um|M20 sampled-path gap GHz|Selection|",
                  "|---|---|---|---|---|"]
        for index, row in enumerate(new, start=1):
            sides = row["sides_um"]
            lines.append(f"|{index}|{sides[0]:.8g}|{sides[1]:.8g}|{row['gap_GHz']:.9g}|{row['selection_rationale']}|")
    return "\n".join(lines)+"\n"


class ResponseAtlasExperiment:
    """A resumable, bounded M20 atlas experiment; it never accepts inverse designs."""
    def __init__(self, root, *, deadline, device="cpu", config_path=None):
        self.root = Path(root).resolve()
        self.config_path = self.root/(config_path or "configs/pure_pwe_response_atlas_v1.json")
        self.config = json.loads(self.config_path.read_text())
        self.version = self.config.get("experiment_version", VERSION)
        self.out = self.root/self.config.get("output_directory", f"results/{self.version}")
        if not self.out.is_relative_to(self.root):
            raise ValueError("Atlas output directory must stay inside the repository")
        self.deadline = float(deadline)
        if not np.isfinite(self.deadline) or self.deadline <= time.time():
            raise DeadlineReached("Response-atlas deadline must be in the future")
        maximum = float(self.config["max_runtime_hours"])*3600
        if self.deadline-time.time() > maximum:
            raise ValueError("Requested atlas runtime exceeds its preregistered maximum")
        self.forward_config_path = self.root/self.config["base_config"]
        self.forward_config = json.loads(self.forward_config_path.read_text())
        if self.forward_config["search_cutoff"] != self.config["search_cutoff"]:
            raise ValueError("Atlas must use the base configuration's M20 search cutoff")
        self.seed, self.seed_hashes = self._load_seed()
        protocol = {
            "experiment_version": self.version, "atlas_config": self.config,
            "effective_forward_config": self.forward_config,
            "seed_source_hashes": self.seed_hashes,
            "runtime": {"dtype": "float64/complex128", "allowed_backends": ["cpu", "cuda"]},
        }
        self.fingerprint = json_hash(protocol)
        self.protocol = protocol
        self.evaluator = QuadratureForward(self.forward_config, self.out, self.fingerprint,
                                            device=device, deadline_check=self.check_deadline)

    def check_deadline(self):
        if time.time() >= self.deadline:
            raise DeadlineReached("Response-atlas deadline reached; checkpoints preserved")

    def _load_seed(self):
        seed_dir = (self.root/self.config["seed_experiment"]).resolve()
        results_path, manifest_path, status_path = (seed_dir/name for name in (
            "results.json", "manifest.json", "status.json"))
        if not all(path.is_file() for path in (results_path, manifest_path, status_path)):
            raise ValueError("Completed multistart PWE seed experiment is required")
        status, source, manifest = (json.loads(path.read_text()) for path in (
            status_path, results_path, manifest_path))
        if (status.get("status") != "complete" or source.get("status") != "complete"
                or manifest.get("experiment_version") != "pure_pwe_multistart_v1"
                or manifest.get("effective_forward_config") != self.forward_config):
            raise ValueError("Atlas seeds must come from the completed unchanged pure-PWE experiment")
        observations = []
        for case in source.get("cases", []):
            for call in case.get("evaluation_calls", []):
                if call.get("cached"):
                    continue
                evaluation_path = Path(call["evaluation_path"])
                record = json.loads(evaluation_path.read_text())
                if (record.get("payload_sha256") != QuadratureForward._payload_hash(record)
                        or "result" not in record or record["result"].get("cutoff") != 20):
                    raise ValueError("Seed PWE checkpoint failed integrity validation")
                result = record["result"]
                if (result["normalized_x"] != call["normalized_x"]
                        or not np.isclose(result["gap_GHz"], call["gap_GHz"], rtol=0.0, atol=1e-12)):
                    raise ValueError("Seed result disagrees with its recorded multistart call")
                observations.append({"normalized_x": result["normalized_x"], "gap_GHz": result["gap_GHz"]})
        seed = deduplicate_observations(observations)
        if len(seed) < 3:
            raise ValueError("The source experiment does not provide enough distinct M20 seed points")
        hashes = {str(path.resolve()): digest(path) for path in (results_path, manifest_path, status_path)}
        atlas_seed = self.config.get("seed_atlas")
        if atlas_seed:
            atlas_dir = (self.root/atlas_seed).resolve()
            atlas_results, atlas_manifest = (atlas_dir/name for name in ("results.json", "manifest.json"))
            if not atlas_results.is_file() or not atlas_manifest.is_file():
                raise ValueError("Configured prior atlas seed is incomplete")
            prior, prior_protocol = (json.loads(path.read_text()) for path in (atlas_results, atlas_manifest))
            if prior.get("status") != "complete_for_requested_budget":
                raise ValueError("Prior atlas seed must have completed its requested budget")
            for row in prior.get("new_evaluations", []):
                x = np.asarray(row["normalized_x"], dtype=float)
                path = atlas_dir/"evaluations"/f"{row['evaluation_key']}.json"
                if not path.is_file():
                    raise ValueError("Prior atlas evaluation checkpoint is missing")
                record = json.loads(path.read_text())
                if (record.get("payload_sha256") != QuadratureForward._payload_hash(record)
                        or "result" not in record or record["result"].get("cutoff") != 20):
                    raise ValueError("Prior atlas checkpoint failed integrity validation")
                value = record["result"]
                if (value["normalized_x"] != x.tolist()
                        or not np.isclose(value["gap_GHz"], row["gap_GHz"], rtol=0., atol=1e-12)):
                    raise ValueError("Prior atlas checkpoint disagrees with its result row")
                seed.append({"normalized_x": value["normalized_x"], "gap_GHz": value["gap_GHz"]})
            seed = deduplicate_observations(seed)
            hashes.update({str(path.resolve()): digest(path) for path in (atlas_results, atlas_manifest)})
        return seed, hashes

    def _cached_at(self, x):
        x = np.asarray(x, dtype=float)
        identity = self.evaluator._identity(x, self.config["search_cutoff"])
        path = self.evaluator.output/f"{self.evaluator._identity_hash(identity)}.json"
        if not path.exists():
            raise ValueError("Atlas result references a missing evaluation checkpoint")
        record = json.loads(path.read_text())
        self.evaluator._validate_checkpoint(
            record, identity, self.evaluator._identity_hash(identity), x,
            self.evaluator.sides_from_x(x), self.config["search_cutoff"], path)
        return record["result"]

    def _read_result(self):
        path = self.out/"results.json"
        if not path.exists():
            return {"status": "not_started", "new_evaluations": []}
        value = json.loads(path.read_text())
        if value.get("scientific_fingerprint") != self.fingerprint:
            raise ValueError("Existing atlas output belongs to a different scientific protocol")
        return value

    def _write(self, result):
        observations = deduplicate_observations(self.seed + [
            {"normalized_x": row["normalized_x"], "gap_GHz": row["gap_GHz"]}
            for row in result["new_evaluations"]])
        result["seed_observations"] = len(self.seed)
        result["distinct_pwe_points"] = len(observations)
        result["scientific_fingerprint"] = self.fingerprint
        result["protocol"] = self.protocol
        result["generated_unix"] = time.time()
        atomic_json(self.out/"results.json", result)
        (self.out/"REPORT.md").write_text(render_report(result), encoding="utf-8")
        self._render_html(observations)

    @staticmethod
    def _colour(value, minimum, maximum):
        """Small dependency-free viridis-like colour ramp for the HTML artifact."""
        stops = ((68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37))
        ratio = 0.5 if maximum <= minimum else np.clip((value-minimum)/(maximum-minimum), 0., 1.)
        scaled = ratio*(len(stops)-1)
        index = min(int(scaled), len(stops)-2)
        amount = scaled-index
        rgb = tuple(round((1-amount)*stops[index][i]+amount*stops[index+1][i]) for i in range(3))
        return "#%02x%02x%02x" % rgb

    @staticmethod
    def _marching_segments(values, level):
        """Return normalized line segments for a scalar-grid contour level."""
        values = np.asarray(values, dtype=float)
        ny, nx = values.shape
        segments = []
        for iy in range(ny-1):
            for ix in range(nx-1):
                corners = ((ix, iy, values[iy, ix]), (ix+1, iy, values[iy, ix+1]),
                           (ix+1, iy+1, values[iy+1, ix+1]), (ix, iy+1, values[iy+1, ix]))
                points = []
                for first, second in zip(corners, corners[1:]+corners[:1]):
                    a, b = first[2]-level, second[2]-level
                    if (a < 0 <= b) or (b < 0 <= a):
                        fraction = 0.5 if a == b else -a/(b-a)
                        points.append((first[0]+fraction*(second[0]-first[0]),
                                       first[1]+fraction*(second[1]-first[1])))
                if len(points) == 2:
                    segments.append((points[0], points[1]))
                elif len(points) == 4:
                    segments.extend(((points[0], points[1]), (points[2], points[3])))
        return segments

    def _render_html(self, observations):
        """Write a portable vector contour map without requiring plotting packages."""
        model = fit_pwe_surface(observations, self.config["surrogate"]["smoothing"])
        axis = np.linspace(0., 1., 81)
        xx, yy = np.meshgrid(axis, axis, indexing="xy")
        pwe_values = np.asarray(model(np.column_stack([xx.ravel(), yy.ravel()]))).reshape(xx.shape)
        reference = load_comsol_reference(self.root/self.config["processed_comsol_data"],
                                          int(self.config["structure_index"]))
        comsol_values = reference["signed_path_gap_GHz"].T
        minimum = float(min(np.min(pwe_values), np.min(comsol_values)))
        maximum = float(max(np.max(pwe_values), np.max(comsol_values)))
        panel_width, panel_height, top = 460, 390, 80

        def heatmap(values, left):
            ny, nx = values.shape
            cells = []
            for iy in range(ny-1):
                for ix in range(nx-1):
                    colour = self._colour(values[iy, ix], minimum, maximum)
                    x, y = left+panel_width*ix/(nx-1), top+panel_height*(ny-2-iy)/(ny-1)
                    cells.append(f'<rect x="{x:.3f}" y="{y:.3f}" width="{panel_width/(nx-1)+.4:.3f}" height="{panel_height/(ny-1)+.4:.3f}" fill="{colour}"/>')
            return "".join(cells)

        def contours(values, left):
            output = []
            for target in self.config["targets_GHz"]:
                for first, second in self._marching_segments(values, target):
                    x1, y1 = left+panel_width*first[0]/(values.shape[1]-1), top+panel_height*(1-first[1]/(values.shape[0]-1))
                    x2, y2 = left+panel_width*second[0]/(values.shape[1]-1), top+panel_height*(1-second[1]/(values.shape[0]-1))
                    output.append(f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" stroke="white" stroke-width="1.1"/>')
            return "".join(output)

        pwe_left, comsol_left = 65, 630
        dots = "".join(
            f'<circle cx="{pwe_left+panel_width*row["normalized_x"][0]:.3f}" cy="{top+panel_height*(1-row["normalized_x"][1]):.3f}" r="3" fill="white" stroke="black" stroke-width=".7"/>'
            for row in observations)
        html = f'''<!doctype html><html><head><meta charset="utf-8"><title>Adaptive response atlas</title>
<style>body{{font-family:Arial,sans-serif;background:#f7f7f7;color:#202020;margin:24px}}svg{{background:white;border:1px solid #ccc;max-width:100%;height:auto}}.title{{font-size:18px;font-weight:bold}}.subtitle{{font-size:13px}}.axis{{font-size:12px;fill:#222}}.note{{max-width:1100px;line-height:1.45}}</style></head><body>
<h1>Adaptive inverse-response atlas</h1><p class="note">Left: pure-PWE M20 interpolation, used only to propose locations. Right: processed COMSOL reference. The layers are never fitted together; contour lines mark {', '.join(str(v) for v in self.config['targets_GHz'])} GHz. Any proposed point still needs full-path M20 and independent M25 validation.</p>
<svg viewBox="0 0 1160 530" role="img" aria-label="Separate PWE and COMSOL sampled-path-gap contour maps">
<text x="{pwe_left}" y="34" class="title">Pure-PWE M20 proposal surface</text><text x="{comsol_left}" y="34" class="title">Processed COMSOL reference ({self.config['structure_label']})</text>
<text x="{pwe_left}" y="54" class="subtitle">white dots: evaluated PWE samples</text><text x="{comsol_left}" y="54" class="subtitle">separate empirical layer — not PWE training data</text>
{heatmap(pwe_values, pwe_left)}{contours(pwe_values, pwe_left)}{dots}{heatmap(comsol_values, comsol_left)}{contours(comsol_values, comsol_left)}
<rect x="{pwe_left}" y="{top}" width="{panel_width}" height="{panel_height}" fill="none" stroke="#222"/><rect x="{comsol_left}" y="{top}" width="{panel_width}" height="{panel_height}" fill="none" stroke="#222"/>
<text x="{pwe_left+panel_width/2}" y="500" text-anchor="middle" class="axis">L1 (µm): {self.evaluator.lower[0]:.1f} → {self.evaluator.lower[0]+self.evaluator.width[0]:.1f}</text><text x="{comsol_left+panel_width/2}" y="500" text-anchor="middle" class="axis">L1 (µm): {reference['L1_um'][0]:.1f} → {reference['L1_um'][-1]:.1f}</text>
<text x="10" y="{top+panel_height/2}" transform="rotate(-90 10 {top+panel_height/2})" text-anchor="middle" class="axis">L2 (µm): {self.evaluator.lower[1]:.1f} → {self.evaluator.lower[1]+self.evaluator.width[1]:.1f}</text><text x="575" y="{top+panel_height/2}" transform="rotate(-90 575 {top+panel_height/2})" text-anchor="middle" class="axis">L2 (µm): {reference['L2_um'][0]:.1f} → {reference['L2_um'][-1]:.1f}</text>
</svg></body></html>'''
        (self.out/"response_atlas.html").write_text(html, encoding="utf-8")

    def run(self, max_new_evaluations):
        if (not isinstance(max_new_evaluations, int) or max_new_evaluations < 1
                or max_new_evaluations > int(self.config["maximum_new_evaluations"])):
            raise ValueError("Requested atlas point count is outside the preregistered bound")
        self.out.mkdir(parents=True, exist_ok=True)
        lock = self.out/"run.lock"
        with lock.open("x") as stream:
            stream.write(str(os.getpid()))
        try:
            manifest = self.out/"manifest.json"
            if manifest.exists() and json.loads(manifest.read_text()) != self.protocol:
                raise ValueError("Existing atlas manifest differs from the preregistered protocol")
            atomic_json(manifest, self.protocol)
            result = self._read_result()
            result.setdefault("new_evaluations", [])
            result["status"] = "running"
            self._write(result)
            while len(result["new_evaluations"]) < max_new_evaluations:
                self.check_deadline()
                observations = deduplicate_observations(self.seed + [
                    {"normalized_x": row["normalized_x"], "gap_GHz": row["gap_GHz"]}
                    for row in result["new_evaluations"]])
                x, rationale = choose_next_point(observations, self.config, len(result["new_evaluations"]))
                if x is None:
                    result["status"] = "coverage_saturated"
                    break
                result["current_normalized_x"] = x
                result["current_selection_rationale"] = rationale
                self._write(result)
                value = self.evaluator(x, cutoff=self.config["search_cutoff"])
                # Re-read the exact integrity-checked payload before recording it.
                value = self._cached_at(x)
                result["new_evaluations"].append({
                    "normalized_x": value["normalized_x"], "sides_um": value["sides_um"],
                    "gap_GHz": value["gap_GHz"], "evaluation_key": value["evaluation_key"],
                    "elapsed_seconds": value["elapsed_seconds"], "selection_rationale": rationale,
                })
                result.pop("current_normalized_x", None)
                result.pop("current_selection_rationale", None)
                self._write(result)
            if result["status"] == "running":
                result["status"] = "complete_for_requested_budget"
            self._write(result)
            return result
        except DeadlineReached:
            result = self._read_result()
            result["status"] = "deadline_checkpointed"
            self._write(result)
            return result
        finally:
            lock.unlink(missing_ok=True)
