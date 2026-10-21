"""Read-only verification of the saved full-path angular quadrature supplement.

No PWE evaluation, solver, optimizer, or write is performed. Passing is a
necessary diagnostic safeguard, not authorization to run or a continuum claim.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


class _IncompleteEvidence(ValueError):
    pass


def _read(path):
    if not path.is_file():
        raise _IncompleteEvidence(f"Missing evidence file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _same(left, right):
    """JSON identity also distinguishes a Boolean from the integer 0 or 1."""
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False)


def _within_root(root, name):
    path = Path(name)
    path = (path if path.is_absolute() else root/path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Evidence path escapes the project: {path}")
    return path


def _rehash_groups(root, manifest):
    verified = {}
    for group in ("protected_hashes", "code_hashes"):
        declared = manifest[group]
        if not isinstance(declared, dict) or not declared:
            raise ValueError(f"Missing/non-object manifest {group}.")
        for name, expected in declared.items():
            path = _within_root(root, name)
            if (not isinstance(expected, str) or len(expected) != 64
                    or not path.is_file() or _digest(path) != expected):
                raise ValueError(f"Changed or missing historical protected/code file: {path}")
            key = str(path)
            if key in verified and verified[key] != expected:
                raise ValueError(f"Conflicting historical hashes: {path}")
            verified[key] = expected
    return verified


def validate_angular_evidence(root):
    """Validate stored evidence without executing PWE; return deterministic data.

    Missing/incomplete evidence and invalid or changed provenance fail closed
    with ``passed=False`` and an explanatory ``state``/``reasons``. No exception
    is needed for an ordinary unavailable, malformed, stale, or failed audit.
    ``evidence_hashes`` includes freshly checked protected/code files and the
    report, status, manifests, configs, baseline, and full evaluation checkpoint.
    This function never calls ``QuadratureForward.evaluate`` (or ``__call__``).
    """
    root = Path(root).resolve()
    answer = {
        "passed": False, "state": "incomplete", "reasons": [],
        "run_id": None, "comparison": None, "evidence_hashes": {},
        "launch_authorized": False, "continuum_accuracy_certified": False,
        "scope": "Read-only historical angular-evidence guard; explicit inverse launch review remains required.",
    }
    try:
        _validate(root, answer)
    except _IncompleteEvidence as exc:
        answer["state"] = "incomplete"
        answer["reasons"].append(str(exc))
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
        answer["state"] = "invalid"
        answer["reasons"].append(f"Invalid angular evidence: {exc}")
    return answer


def _validate(root, answer):
    config_path = root/"configs/quadrature_full_order_v1.json"
    config = _read(config_path)
    out = _within_root(root, config["output_directory"])
    manifest_path, report_path, status_path = (
        out/name for name in ("manifest.json", "results.json", "status.json"))
    manifest, report, status = (_read(p) for p in (manifest_path, report_path, status_path))
    hashes = _rehash_groups(root, manifest)
    answer["evidence_hashes"].update(hashes)
    if not _same(config, manifest["config"]) or not _same(config, report["config"]):
        raise ValueError("Current, manifest, and reported angular configurations differ.")
    forward_path = _within_root(root, config["forward_config_source"])
    forward = _read(forward_path)
    if (forward["projector_quadrature_order"] != config["baseline_order"]
            or forward["search_cutoff"] != config["cutoff"]
            or forward["full_path_count"] != 281):
        raise ValueError("Current forward protocol differs from the historical supplement.")
    forward["projector_quadrature_order"] = config["comparison_order"]
    if not _same(forward, manifest["effective_forward_config"]):
        raise ValueError("Current effective forward config differs from the manifest.")

    # Verify the historical sources before importing their pure validation helpers.
    required_protected = (
        config_path, forward_path, root/"scripts/audit_quadrature_full_order.py",
        root/"src/invdes/inverse/quadrature_forward.py",
    )
    if any(str(path.resolve()) not in hashes for path in required_protected):
        raise ValueError("Required angular config/validator source is absent from historical hashes.")
    scripts = Path(__file__).resolve().parents[3]/"scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from audit_quadrature_full_order import content_hash, validate_baseline, compare_results
    from invdes.inverse.quadrature_forward import QuadratureForward

    run_id = content_hash(manifest)
    answer["run_id"] = run_id
    if status["run_id"] != run_id or report["run_id"] != run_id:
        raise ValueError("Angular status/report run_id does not match the manifest fingerprint.")
    if status["status"] != "complete" or report["status"] != "complete":
        raise _IncompleteEvidence("Angular status and report must both be complete.")
    if report["verified_file_count"] != len(hashes):
        raise ValueError("Reported historical verified-file count is inconsistent.")

    upstream_path = _within_root(root, config["baseline_manifest"])
    baseline_path = _within_root(root, config["baseline_case"])
    if any(str(path) not in hashes for path in (upstream_path, baseline_path)):
        raise ValueError("Baseline case or upstream manifest is not historically protected.")
    upstream, saved_baseline = _read(upstream_path), _read(baseline_path)
    upstream_hashes = _rehash_groups(root, upstream)
    if any(hashes.get(path) != value for path, value in upstream_hashes.items()):
        raise ValueError("Angular manifest does not preserve all upstream protected/code hashes.")
    radial_config_path = root/"configs/radial_quadrature_audit_v1.json"
    if (str(radial_config_path.resolve()) not in hashes
            or not _same(_read(radial_config_path), upstream["config"])):
        raise ValueError("Current baseline audit config differs from its protected manifest.")
    if manifest["baseline_run_id"] != content_hash(upstream):
        raise ValueError("Angular manifest baseline run_id is not the upstream fingerprint.")
    baseline = validate_baseline(saved_baseline, upstream, config, forward)

    # Construction computes only coordinates/identity; it does not create files
    # or assemble/eigensolve an operator. Never call evaluate, __call__, or backend.
    validator = QuadratureForward(forward, out, run_id, device="cpu")
    bounds = np.asarray(forward["side_multiplier_bounds"], dtype=float)
    x = (np.asarray(config["side_multipliers"])-bounds[:, 0])/np.diff(bounds, axis=1).ravel()
    sides = validator.sides_from_x(x)
    cutoff = config["cutoff"]
    identity = validator._identity(x, cutoff)
    evaluation_key = validator._identity_hash(identity)
    evaluation_path = (out/"evaluations"/f"{evaluation_key}.json").resolve()
    reported_comparison = report["comparison"]
    if (reported_comparison["comparison_evaluation_key"] != evaluation_key
            or _within_root(root, reported_comparison["comparison_evaluation_path"]) != evaluation_path):
        raise ValueError("Reported angular evaluation identity/path differs from the planned checkpoint.")
    record = _read(evaluation_path)
    validator._validate_checkpoint(record, identity, evaluation_key, x, sides, cutoff, evaluation_path)
    if "result" not in record:
        raise _IncompleteEvidence("Angular checkpoint has no completed full-path result.")
    comparison = compare_results(baseline, record["result"], config)
    if not _same(reported_comparison, comparison):
        raise ValueError("Stored comparison differs from the recomputed full-path comparison.")
    computed_pass = comparison["exploratory_readiness_passed"]
    if report["passes_exploratory_guard"] is not computed_pass:
        raise ValueError("Reported pass flag disagrees with recomputed angular guards.")
    if not isinstance(computed_pass, bool):
        raise ValueError("Recomputed readiness flag is not a Boolean.")
    if (comparison["comparison_gradient_reliable"] is not True
            or comparison["comparison_gradient_reliability_reasons"]):
        answer["reasons"].append("Angular comparison active-edge gradients are not reliable.")
    if computed_pass is not True:
        answer["reasons"].append("Recomputed angular numerical guards did not pass.")
    answer["comparison"] = comparison
    # Hash the actual evidence after all recomputation; no timestamps or live
    # process metadata are synthesized, keeping repeated readiness reads stable.
    for path in (config_path, forward_path, radial_config_path, manifest_path,
                 report_path, status_path, upstream_path, baseline_path, evaluation_path):
        answer["evidence_hashes"][str(path.resolve())] = _digest(path)
    answer["passed"] = not answer["reasons"]
    answer["state"] = "passed" if answer["passed"] else "failed"
