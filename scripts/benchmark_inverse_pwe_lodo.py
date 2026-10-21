"""Checkpointed 61-positive-gap LODO benchmark; no ML or COMSOL labels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import inverse_pwe as inverse
import numpy as np
from invdes.splits import get_lodo_splits

OUT = inverse.ROOT/"results/pure_pwe_inverse_lodo"


def folds(library):
    for available, excluded in get_lodo_splits(len(library)):
        held = library[int(excluded[0])]
        if held["gap_ghz"] > 0:
            entries = [library[int(i)] for i in available]
            assert held["source"] not in {c["source"] for c in entries}
            yield held, entries


def run_fold(held, entries, evaluate, settings):
    if held["source"] in {c["source"] for c in entries}:
        raise ValueError("Held-out design leaked into search library")
    start = time.perf_counter()
    # Only the scalar target, not its geometry or spectrum, enters design_gap.
    result = inverse.design_gap(
        held["gap_ghz"], entries, evaluate,
        tolerance=settings["gap_tolerance_ghz"],
        max_evaluations=settings["max_new_evaluations"],
        fraction_limits=settings["secant_fraction_limits"])
    result.update({"held_out_design": held["source"],
                   "search_library_ids": [c["source"] for c in entries],
                   "elapsed_seconds": time.perf_counter()-start})
    return result


def summarize(records, total):
    results = [r["result"] for r in records if "result" in r]
    errors = np.array([r["absolute_error_ghz"] for r in results])
    matched = sum(r["status"] in ("matched_new_pwe", "matched_saved_pwe") for r in results)
    summary = {"total_targets": total, "completed_cases": len(records),
               "matched_cases": matched, "nonmatching_cases": len(results)-matched,
               "execution_errors": len(records)-len(results),
               "success_rate_completed_pct": 100*matched/len(records) if records else None,
               "metrics_include_nonmatching_returned_candidates": True}
    if len(errors):
        summary.update({"gap_mae_ghz": float(errors.mean()),
                        "gap_rmse_ghz": float(np.sqrt(np.mean(errors**2))),
                        "gap_max_error_ghz": float(errors.max()),
                        "gap_mean_relative_error_pct": float(np.mean([
                            100*r["absolute_error_ghz"]/r["target_gap_ghz"] for r in results]))})
    return summary


def publish(records, total, manifest_id, state, active=None):
    summary = summarize(records, total)
    payload = {"status": state, "active_design": active, "pid": os.getpid(),
               "updated_unix": time.time(), "manifest_id": manifest_id, **summary}
    inverse.atomic_json(OUT/"status.json", payload)
    inverse.atomic_json(OUT/"results.json", {**payload, "cases": records})
    lines = ["# Pure-PWE leave-one-design-out inverse benchmark", "",
             "Each positive-gap target is excluded from its 63-design search library.",
             "Targets and candidate spectra are PWE only. No ML, training or COMSOL spectral fitting.",
             "Same frozen full-path solver and inverse settings; 0.05 GHz matching tolerance.",
             "Each fold has an isolated numerical cache; no cross-fold candidate-library additions.",
             "Unbracketed or budget-exhausted targets count as nonmatches, not proof of physical impossibility.",
             "Nonunique geometry: response recovery, not recovery of original side lengths, is assessed.",
             "Sampled-path gaps only. Not a full-Brillouin-zone or COMSOL validation.", "",
             f"Status: {state}.", "", "```json", json.dumps(summary, indent=2), "```", "",
             "|Held out|Target GHz|Achieved GHz|Abs error GHz|L1 um|L2 um|Status|",
             "|---|---:|---:|---:|---:|---:|---|"]
    for record in records:
        if "result" not in record:
            lines.append(f"|{record['design_id']}||||||execution_error|")
            continue
        r = record["result"]
        c = r["candidate"]
        lines.append(f"|{record['design_id']}|{r['target_gap_ghz']:.9f}|{c['gap_ghz']:.9f}|"
                     f"{r['absolute_error_ghz']:.9f}|{c['L1_um']:.9f}|{c['L2_um']:.9f}|{r['status']}|")
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    cfg, fingerprint, library = inverse.load_library()
    settings = json.loads((inverse.ROOT/"configs/pure_pwe_inverse.json").read_text())
    cases = list(folds(library))
    if len(cases) != 61:
        raise ValueError("Expected 61 positive-gap folds")
    sources = [Path(__file__), inverse.ROOT/"scripts/inverse_pwe.py",
               inverse.ROOT/"src/invdes/inverse/pure_pwe.py",
               inverse.ROOT/"src/invdes/splits.py",
               inverse.ROOT/"scripts/benchmark_pwe_cuda.py"]
    sources += sorted((inverse.OUT/"predictions").glob("D*.json"))
    manifest = {"solver_fingerprint": fingerprint, "settings": settings,
                "device": args.device, "target_ids": [h["source"] for h, _ in cases],
                "source_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                "ml_training": False, "comsol_spectra_used": False}
    manifest_id = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/"manifest.json").exists():
        if json.loads((OUT/"manifest.json").read_text()) != manifest:
            raise ValueError("Benchmark changed; existing results must not be mixed")
    else:
        inverse.atomic_json(OUT/"manifest.json", manifest)
    lock = OUT/"run.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    records = []
    try:
        # All completed folds survive resume without being rewritten.
        for held, _ in cases:
            path = OUT/"folds"/f"{held['source']}.json"
            if path.exists():
                record = json.loads(path.read_text())
                if record["manifest_id"] != manifest_id:
                    raise ValueError("Fold checkpoint fingerprint mismatch")
                records.append(record)
        completed = {r["design_id"] for r in records}
        publish(records, len(cases), manifest_id, "running")
        for held, entries in cases:
            name = held["source"]
            if name in completed:
                continue
            publish(records, len(cases), manifest_id, "running", name)
            print(f"Starting {name}: target={held['gap_ghz']:.9f} GHz", flush=True)
            # Isolate evaluations, including from earlier single-target tests.
            inverse.DEST = OUT/"fold_cache"/name
            evaluate = inverse.FullPathPWE(cfg, fingerprint, args.device)
            record = {"design_id": name, "manifest_id": manifest_id}
            try:
                record["result"] = run_fold(held, entries, evaluate, settings)
                record.update({"fresh_pwe_solves": evaluate.fresh_solves,
                               "cache_hits": evaluate.cache_hits})
            except Exception as exc:
                record["execution_error"] = repr(exc)
            inverse.atomic_json(OUT/"folds"/f"{name}.json", record)
            records.append(record)
            records.sort(key=lambda r: r["design_id"])
            publish(records, len(cases), manifest_id, "running")
            print(f"Finished {name}: {record.get('result', {}).get('status', 'execution_error')}", flush=True)
        for path, expected in manifest["source_hashes"].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Input changed during benchmark: {path}")
        state = "complete_with_execution_errors" if any("execution_error" in r for r in records) else "complete"
        publish(records, len(cases), manifest_id, state)
    except BaseException:
        publish(records, len(cases), manifest_id, "interrupted")
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
