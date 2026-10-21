"""Fresh, checkpointed physics evaluator for an externally gated inverse driver.

No search, launch gate, saved-design library, COMSOL labels or fitting lives
here. The caller supplies a NEW output directory and a scientific fingerprint
covering its protected inputs. Existing checkpoint identities are immutable.
"""
import copy
import gc
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from audit_pure_pwe import eta_matrix, reciprocal_indices, reciprocal_basis, path_coordinates
from audit_pwe_sensitivity import modes
from validate_energy_pwe_full import atomic_json, lowest_two
from invdes.interface_pwe import energy_factorized_impermittivity
from invdes.radial_projector_quadrature import projector_matrices
from invdes.pwe_sensitivity import (
    eta_derivative, energy_eigenvalue_derivative, frequency_derivative,
)


def assess_gradient_reliability(active_separations, edge_margins, gradient,
                                separation_tolerance, edge_tolerance):
    """Conservative simple-mode and unique-active-edge policy."""
    separations = np.asarray(active_separations, dtype=float)
    margins = np.asarray(edge_margins, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    reasons = []
    if separations.shape != (2,) or not np.all(np.isfinite(separations)) or np.any(
            separations <= separation_tolerance):
        reasons.append("active edge modes are not demonstrably simple")
    if margins.shape != (2,) or not np.all(np.isfinite(margins)) or np.any(margins <= edge_tolerance):
        reasons.append("active sampled edges are not unique within tolerance")
    if gradient.shape != (2,) or not np.all(np.isfinite(gradient)):
        reasons.append("analytic gradient is nonfinite or malformed")
    return {"gradient_reliable": not reasons, "reasons": reasons}


class QuadratureForward:
    """Evaluate one bounded normalized design at a configured Fourier cutoff.

    Projectors are constant across lengths and cached per cutoff in memory.
    Every distinct exact normalized point gets a fresh full-path calculation.
    A deadline callback may raise at safe boundaries; completed k points stay
    in their checkpoint. This class never decides whether a search may launch.
    """
    def __init__(self, config, output_directory, scientific_fingerprint,
                 device="cpu", deadline_check=None):
        self.config = copy.deepcopy(config)
        self.output = Path(output_directory)/"evaluations"
        self.fingerprint = str(scientific_fingerprint)
        if not self.fingerprint:
            raise ValueError("A nonempty scientific fingerprint is required")
        if device not in ("cpu", "cuda", "auto"):
            raise ValueError("device must be cpu, cuda or auto")
        if (config["eps_background"], config["eps_hole"]) != (9, 1):
            raise ValueError("This evaluator supports only the documented eps=9/1 geometry")
        if not np.isfinite(config["a_um"]) or config["a_um"] <= 0:
            raise ValueError("Positive finite lattice constant required")
        self.bounds = np.asarray(config["side_multiplier_bounds"], dtype=float)
        if self.bounds.shape != (2, 2) or not np.array_equal(
                self.bounds, np.array([[1., 1.35], [.65, 1.]])):
            raise ValueError("Only the documented side-length box is supported")
        self.lower = self.bounds[:, 0]*config["a_um"]/np.sqrt(3)
        self.width = np.diff(self.bounds, axis=1).ravel()*config["a_um"]/np.sqrt(3)
        count = config["full_path_count"]
        if not isinstance(count, int) or count < 2:
            raise ValueError("At least two path points required")
        self.parameters = config["full_path_start"]+config["full_path_step"]*np.arange(count)
        if (not np.all(np.isfinite(self.parameters)) or np.any(self.parameters <= 0)
                or np.any(self.parameters >= 3) or config["full_path_step"] <= 0):
            raise ValueError("Path points must increase strictly inside (0,3)")
        self.ks = path_coordinates(self.parameters, config["a_um"]*1e-6)
        self.device = device
        self.deadline_check = deadline_check or (lambda: None)
        self._projectors = {}
        for key in ("near_degenerate_relative_separation", "near_tied_edge_tolerance_GHz"):
            if not np.isfinite(config[key]) or config[key] <= 0:
                raise ValueError(f"{key} must be positive and finite")

    def sides_from_x(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (2,) or not np.all(np.isfinite(x)) or np.any(x < 0) or np.any(x > 1):
            raise ValueError("normalized x must be a finite pair in [0,1]")
        return self.lower+self.width*x

    def _identity(self, x, cutoff):
        return {"scientific_fingerprint": self.fingerprint, "config": self.config,
                "cutoff": int(cutoff), "normalized_x_hex": [float(v).hex() for v in x],
                "evaluator": "fixed-radial-direct-quadrature-v1"}

    @staticmethod
    def _identity_hash(identity):
        return hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()

    @classmethod
    def _payload_hash(cls, record):
        """Integrity checksum, not authentication against deliberate rewriting."""
        return cls._identity_hash({k: v for k, v in record.items() if k != "payload_sha256"})

    @classmethod
    def _save_checkpoint(cls, path, record):
        record["payload_sha256"] = cls._payload_hash(record)
        atomic_json(path, record)

    @staticmethod
    def _finite_bound(value, maximum, label):
        if not isinstance(value, (float, int)) or not np.isfinite(value) or not 0 <= value <= maximum:
            raise ValueError(f"Invalid {label}")

    @classmethod
    def _validate_diagnostic(cls, diagnostic):
        if not isinstance(diagnostic, dict):
            raise ValueError("Malformed point diagnostic")
        cls._finite_bound(diagnostic.get("relative_eigenpair_residual"), 1e-7, "eigenpair residual")
        cls._finite_bound(diagnostic.get("hermitian_defect"), 1e-12, "Hermitian defect")
        cls._finite_bound(diagnostic.get("backward_residual"), float("inf"), "backward residual")

    def _validate_checkpoint(self, record, identity, identity_hash, x, sides, cutoff, path):
        if record.get("identity") != identity or record.get("identity_hash") != identity_hash:
            raise ValueError("Checkpoint immutable identity mismatch")
        try:
            valid_hash = record.get("payload_sha256") == self._payload_hash(record)
        except (TypeError, ValueError) as exc:
            raise ValueError("Checkpoint payload is not finite JSON") from exc
        if not valid_hash:
            raise ValueError("Checkpoint payload checksum mismatch")
        frequencies, diagnostics = record.get("frequencies_GHz"), record.get("point_diagnostics")
        if not isinstance(frequencies, list) or not isinstance(diagnostics, list):
            raise ValueError("Malformed checkpoint arrays")
        f = np.asarray(frequencies, dtype=float)
        if (len(frequencies) != len(diagnostics) or len(frequencies) > len(self.ks)
                or (len(frequencies) and f.shape != (len(frequencies), 2))
                or not np.all(np.isfinite(f)) or np.any(f <= 0)
                or (len(frequencies) and np.any(np.diff(f, axis=1) < 0))):
            raise ValueError("Checkpoint contains invalid band frequencies or counters")
        self._finite_bound(record.get("elapsed_seconds"), float("inf"), "elapsed time")
        for ik, diagnostic in enumerate(diagnostics):
            self._validate_diagnostic(diagnostic)
            if (diagnostic.get("k_index") != ik or diagnostic.get("backend") not in ("cpu", "cuda")
                    or diagnostic.get("requested_backend") not in ("cpu", "cuda", "auto")):
                raise ValueError("Checkpoint point index/backend mismatch")
        if "result" not in record:
            return
        result = record["result"]
        if len(frequencies) != len(self.ks) or not isinstance(result, dict):
            raise ValueError("Completed result has an incomplete path")
        expected = {"identity_hash": identity_hash, "scientific_fingerprint": self.fingerprint,
                    "evaluation_key": identity_hash, "evaluation_path": str(path.resolve()),
                    "cutoff": int(cutoff), "normalized_x": x.tolist(), "sides_um": sides.tolist(),
                    "path_parameters": self.parameters.tolist(), "frequencies_GHz": frequencies,
                    "cached": False, "continuum_accuracy_certified": False,
                    "gap_scope": "sampled path, not full Brillouin zone",
                    "elapsed_seconds": record["elapsed_seconds"]}
        if any(result.get(k) != v for k, v in expected.items()):
            raise ValueError("Completed result does not match checkpoint identity/data")
        resumed = result.get("resumed_point_count")
        if not isinstance(resumed, int) or not 0 <= resumed <= len(self.ks):
            raise ValueError("Invalid resumed point count")
        edges = [int(np.argmax(f[:, 0])), int(np.argmin(f[:, 1]))]
        margins = [float(np.sort(f[:, 0])[-1]-np.sort(f[:, 0])[-2]),
                   float(np.sort(f[:, 1])[1]-np.sort(f[:, 1])[0])]
        if result.get("gap_GHz") != float(f[edges[1], 1]-f[edges[0], 0]):
            raise ValueError("Cached gap disagrees with frequencies")
        checks = result.get("checks")
        if not isinstance(checks, dict):
            raise ValueError("Missing completed numerical checks")
        expected_checks = {"edge_indices": edges, "edge_path_parameters": self.parameters[edges].tolist(),
                           "edge_uniqueness_margins_GHz": margins,
                           "point_backends": [d["backend"] for d in diagnostics]}
        if any(checks.get(k) != v for k, v in expected_checks.items()):
            raise ValueError("Cached active-edge diagnostics disagree with frequencies")
        for key in ("cpu_device_max_absolute_GHz", "edge_three_mode_max_absolute_GHz"):
            self._finite_bound(checks.get(key), 1e-5, key)
        self._finite_bound(checks.get("edge_three_mode_max_relative_eigenpair_residual"),
                           1e-7, "edge eigenpair residual")
        self._finite_bound(checks.get("max_relative_eigenpair_residual"), 1e-7, "summary residual")
        expected_residual = max(checks["edge_three_mode_max_relative_eigenpair_residual"],
                                max(d["relative_eigenpair_residual"] for d in diagnostics))
        if checks["max_relative_eigenpair_residual"] != expected_residual:
            raise ValueError("Cached residual summary disagrees with point diagnostics")
        separation = np.asarray(checks.get("active_mode_relative_separations"), dtype=float)
        if separation.shape != (2,) or not np.all(np.isfinite(separation)) or np.any(separation < 0):
            raise ValueError("Invalid active-mode separations")
        gradient = result.get("gradient_GHz_per_um")
        if gradient is not None:
            gradient = np.asarray(gradient, dtype=float)
            if gradient.shape != (2,) or not np.all(np.isfinite(gradient)):
                raise ValueError("Malformed cached gradient")
            normalized = (gradient*self.width).tolist()
        else:
            gradient = np.full(2, np.nan)
            normalized = None
        if (result.get("gradient_normalized") != normalized
                or result.get("gradient_normalized_x") != normalized):
            raise ValueError("Cached normalized gradient disagrees with units")
        reliability = assess_gradient_reliability(
            separation, margins, gradient, self.config["near_degenerate_relative_separation"],
            self.config["near_tied_edge_tolerance_GHz"])
        if (result.get("gradient_reliable") is not reliability["gradient_reliable"]
                or checks.get("gradient_reliability_reasons") != reliability["reasons"]):
            raise ValueError("Cached gradient reliability disagrees with diagnostics")

    def _backend(self, g, tensor):
        if self.device == "cpu":
            return None, "cpu"
        try:
            import torch
        except ImportError:
            return None, "cpu"
        if not torch.cuda.is_available():
            return None, "cpu"
        from benchmark_pwe_cuda import CudaSolver
        return CudaSolver(g, tensor), "cuda"

    def evaluate(self, x, cutoff=None):
        started = time.perf_counter()
        x = np.asarray(x, dtype=float)
        sides = self.sides_from_x(x)
        if cutoff is None:
            cutoff = self.config["search_cutoff"]
        if (not isinstance(cutoff, (int, np.integer)) or cutoff < 1
                or cutoff not in (self.config["search_cutoff"], self.config["independent_check_cutoff"])):
            raise ValueError("Use one of the configured search/check cutoffs")
        identity = self._identity(x, cutoff)
        identity_hash = self._identity_hash(identity)
        path = self.output/f"{identity_hash}.json"
        record = {"identity": identity, "identity_hash": identity_hash,
                  "frequencies_GHz": [], "point_diagnostics": [], "elapsed_seconds": 0.}
        if path.exists():
            record = json.loads(path.read_text())
            self._validate_checkpoint(record, identity, identity_hash, x, sides, cutoff, path)
            if "result" in record:
                return {**record["result"], "cached": True, "call_elapsed_seconds": time.perf_counter()-started}
        resumed_points = len(record["frequencies_GHz"])
        previous_elapsed = float(record["elapsed_seconds"])
        self.deadline_check()
        self._save_checkpoint(path, record)
        ids = reciprocal_indices(cutoff)
        a = self.config["a_um"]
        eta = eta_matrix(ids, a, sides, self.config["material_arc_order"])
        if cutoff not in self._projectors:
            self._projectors[cutoff] = projector_matrices(
                ids, a, self.config["projector_quadrature_order"])
        p = self._projectors[cutoff]
        identity_matrix = np.eye(len(ids))
        epsilon = 10*identity_matrix-9*eta
        tensor = energy_factorized_impermittivity(eta, epsilon, p)
        g = ids@reciprocal_basis(a*1e-6)
        cuda = None
        try:
            cuda, backend = self._backend(g, tensor)
            for ik in range(resumed_points, len(self.ks)):
                self.deadline_check()
                f, diagnostics = (cuda.solve(self.ks[ik]) if cuda is not None
                                  else lowest_two(g, self.ks[ik], tensor))
                if (np.asarray(f).shape != (2,) or not np.all(np.isfinite(f))
                        or np.any(f <= 0) or f[1] < f[0]):
                    raise ValueError("Invalid freshly computed band frequencies")
                self._validate_diagnostic(diagnostics)
                record["frequencies_GHz"].append(f.tolist())
                record["point_diagnostics"].append({**diagnostics, "backend": backend,
                                                    "requested_backend": self.device, "k_index": ik})
                record["elapsed_seconds"] = previous_elapsed+time.perf_counter()-started
                self._save_checkpoint(path, record)
            f = np.asarray(record["frequencies_GHz"])
            cpu_difference = 0.
            for ik in sorted(set((0, len(self.ks)//2, len(self.ks)-1))):
                self.deadline_check()
                cpu, cpu_diagnostic = lowest_two(g, self.ks[ik], tensor)
                self._validate_diagnostic(cpu_diagnostic)
                if not np.all(np.isfinite(cpu)):
                    raise ValueError("Nonfinite CPU cross-check frequencies")
                cpu_difference = max(cpu_difference, float(np.max(abs(cpu-f[ik]))))
            if not np.isfinite(cpu_difference) or cpu_difference > 1e-5:
                raise ValueError(f"CPU/device frequency disagreement: {cpu_difference} GHz")
            edges = [int(np.argmax(f[:, 0])), int(np.argmin(f[:, 1]))]
            self.deadline_check()
            edge_f, eigenpairs, residuals, separations = modes(g, self.ks[edges], tensor)
            if (not np.all(np.isfinite(edge_f)) or not np.all(np.isfinite(separations))
                    or not np.all(np.isfinite(residuals)) or max(residuals) > 1e-7):
                raise ValueError("Invalid active-edge eigenpairs or residuals")
            edge_difference = float(np.max(abs(edge_f-f[edges])))
            if not np.isfinite(edge_difference) or edge_difference > 1e-5:
                raise ValueError("Active-edge three-mode solve disagrees with path frequencies")
            b = np.linalg.solve(epsilon, identity_matrix)
            gradients = []
            for axis in (0, 1):
                self.deadline_check()
                deta = eta_derivative(ids, a, sides, axis, self.config["material_arc_order"])
                derivative = np.array([
                    frequency_derivative(values, energy_eigenvalue_derivative(
                        g, k, vectors, b, eta-b, p, deta))
                    for k, (values, vectors) in zip(self.ks[edges], eigenpairs)])
                gradients.append(float(derivative[1, 1]-derivative[0, 0]))
            gradients = np.asarray(gradients)
            margins = [float(np.sort(f[:, 0])[-1]-np.sort(f[:, 0])[-2]),
                       float(np.sort(f[:, 1])[1]-np.sort(f[:, 1])[0])]
            active_separations = [float(separations[0, 0]), float(separations[1, 1])]
            reliable = assess_gradient_reliability(
                active_separations, margins, gradients,
                self.config["near_degenerate_relative_separation"],
                self.config["near_tied_edge_tolerance_GHz"])
            gradient_um = gradients.tolist() if np.all(np.isfinite(gradients)) else None
            gradient_x = (gradients*self.width).tolist() if gradient_um is not None else None
            result = {
                "identity_hash": identity_hash, "scientific_fingerprint": self.fingerprint,
                "evaluation_key": identity_hash, "evaluation_path": str(path.resolve()),
                "cutoff": int(cutoff), "normalized_x": x.tolist(), "sides_um": sides.tolist(),
                "path_parameters": self.parameters.tolist(), "frequencies_GHz": f.tolist(),
                "gap_GHz": float(f[edges[1], 1]-f[edges[0], 0]),
                "gradient_GHz_per_um": gradient_um,
                "gradient_normalized_x": gradient_x,
                "gradient_normalized": gradient_x,
                "gradient_reliable": reliable["gradient_reliable"],
                "checks": {"gradient_reliability_reasons": reliable["reasons"],
                           "edge_indices": edges, "edge_path_parameters": self.parameters[edges].tolist(),
                           "edge_uniqueness_margins_GHz": margins,
                           "active_mode_relative_separations": active_separations,
                           "cpu_device_max_absolute_GHz": cpu_difference,
                           "edge_three_mode_max_absolute_GHz": edge_difference,
                           "edge_three_mode_max_relative_eigenpair_residual": max(residuals),
                           "max_relative_eigenpair_residual": max(
                               max(residuals), max(d["relative_eigenpair_residual"]
                                                   for d in record["point_diagnostics"])),
                           "point_backends": [d["backend"] for d in record["point_diagnostics"]]},
                "elapsed_seconds": previous_elapsed+time.perf_counter()-started,
                "cached": False, "resumed_point_count": resumed_points,
                "gap_scope": "sampled path, not full Brillouin zone",
                "continuum_accuracy_certified": False,
            }
            record["result"] = result
            record["elapsed_seconds"] = result["elapsed_seconds"]
            self._save_checkpoint(path, record)
            self._validate_checkpoint(record, identity, identity_hash, x, sides, cutoff, path)
            return {**result, "call_elapsed_seconds": time.perf_counter()-started}
        finally:
            if cuda is not None:
                del cuda
            if self.device in ("cuda", "auto"):
                # Also execute if construction failed after allocating tensors.
                # A live exception traceback can retain tensors until handled;
                # empty_cache only releases allocations no longer referenced.
                gc.collect()
                try:
                    import torch
                except ImportError:
                    pass
                else:
                    torch.cuda.empty_cache()

    __call__ = evaluate
