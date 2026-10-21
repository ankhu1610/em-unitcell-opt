"""Usage: python scripts/inverse_pwe.py 20 [--device cuda]

Only required design input is a positive gap width in GHz.
No ML imports, training, COMSOL spectra, or reference-report calls.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from validate_energy_pwe_full import ROOT, OUT, tensor_for, lowest_two, path_coordinates, atomic_json
import numpy as np
from invdes.inverse.pure_pwe import candidate, design_gap

DEST = ROOT/"results/pure_pwe_inverse"


def load_library():
    # Reuse frozen numerical configuration, not the COMSOL comparison report.
    benchmark = json.loads((OUT/"benchmark.json").read_text())
    cfg, fingerprint = benchmark["config"], benchmark["fingerprint"]
    computed = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    if computed != fingerprint:
        raise ValueError("Invalid frozen configuration fingerprint")
    for relative, expected in cfg["input_hashes"].items():
        if hashlib.sha256((ROOT.parent/relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen solver/geometry changed: {relative}")
    library = []
    for path in sorted((OUT/"predictions").glob("D*.json")):
        record = json.loads(path.read_text())
        if record["fingerprint"] != fingerprint:
            raise ValueError(f"Checkpoint fingerprint mismatch: {path.name}")
        if np.shape(record["frequencies_GHz"]) != (len(cfg["k"]), 2):
            raise ValueError(f"Incomplete checkpoint: {path.name}")
        library.append(candidate(record["sides_um"], record["frequencies_GHz"], path.stem))
    if len(library) != 64:
        raise ValueError("Expected all 64 frozen PWE designs")
    return cfg, fingerprint, library


class FullPathPWE:
    def __init__(self, cfg, fingerprint, device):
        self.cfg, self.fingerprint, self.device = cfg, fingerprint, device
        self.fresh_solves = 0
        self.cache_hits = 0

    def __call__(self, sides):
        identity = {"fingerprint": self.fingerprint, "sides_um": list(map(float, sides)),
                    "device": self.device}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = DEST/"evaluations"/f"{key}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["identity"] != identity:
                raise ValueError("Inverse cache identity mismatch")
            if np.shape(saved["frequencies_GHz"]) != (len(self.cfg["k"]), 2):
                raise ValueError("Incomplete inverse cache")
            self.cache_hits += 1
            return candidate(sides, saved["frequencies_GHz"], str(path))
        self.fresh_solves += 1
        start = time.perf_counter()
        print(f"Full PWE: L1={sides[0]:.9f}, L2={sides[1]:.9f} um", flush=True)
        g, tensor = tensor_for({"L1_um": sides[0], "L2_um": sides[1]}, self.cfg)
        solver = None
        if self.device == "cuda":
            from benchmark_pwe_cuda import CudaSolver
            import torch
            torch.set_num_threads(2)
            solver = CudaSolver(g, tensor)
        frequencies, diagnostics = [], []
        for i, k in enumerate(path_coordinates(self.cfg["k"], self.cfg["a_um"]*1e-6)):
            f, diagnostic = solver.solve(k) if solver else lowest_two(g, k, tensor)
            frequencies.append(f.tolist())
            diagnostics.append(diagnostic)
            if (i+1) % 50 == 0:
                print(f"  {i+1}/{len(self.cfg['k'])} k-points", flush=True)
        atomic_json(path, {"identity": identity, "frequencies_GHz": frequencies,
                           "diagnostics": diagnostics, "seconds": time.perf_counter()-start})
        result = candidate(sides, frequencies, str(path))
        print(f"  gap={result['gap_ghz']:.9f} GHz", flush=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gap_ghz", type=float, nargs="?")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--test-existing", action="store_true")
    parser.add_argument("--report-existing", action="store_true",
                        help="Render the saved experiment report without rerunning it")
    args = parser.parse_args()
    if args.report_existing:
        write_test_report(json.loads((DEST/"existing_results_test.json").read_text()))
        return
    if not args.test_existing and args.gap_ghz is None:
        parser.error("Supply target gap width in GHz, or --test-existing")
    settings = json.loads((ROOT/"configs/pure_pwe_inverse.json").read_text())
    cfg, fingerprint, library = load_library()
    evaluate = FullPathPWE(cfg, fingerprint, args.device)

    def solve(target, entries):
        return design_gap(target, entries, evaluate,
                          tolerance=settings["gap_tolerance_ghz"],
                          max_evaluations=settings["max_new_evaluations"],
                          fraction_limits=settings["secant_fraction_limits"])

    start = time.perf_counter()
    if args.test_existing:
        # Retrieval sanity check is explicitly not a generalization experiment.
        retrieval = [solve(c["gap_ghz"], library) for c in library if c["gap_ghz"] > 0]
        held = next(c for c in library if c["source"] == settings["held_out_design"])
        holdout = solve(held["gap_ghz"], [c for c in library if c["source"] != held["source"]])
        off_grid = solve(settings["off_grid_target_ghz"], library)
        output = {"retrieval_checks": retrieval, "excluded_nonpositive_gaps": len(library)-len(retrieval),
                  "held_out_design": held["source"], "held_out_response_test": holdout,
                  "off_grid_target_test": off_grid,
                  "note": "Held-out PWE spectrum supplies only scalar target; its geometry and spectrum are excluded from search library. Not COMSOL validation or a full LODO study."}
        destination = DEST/"existing_results_test.json"
    else:
        output = solve(args.gap_ghz, library)
        destination = DEST/f"target_{args.gap_ghz:g}_GHz.json"
    output.update({"solver_fingerprint": fingerprint, "settings": settings,
                   "device": args.device, "elapsed_seconds": time.perf_counter()-start,
                   "fresh_pwe_solves": evaluate.fresh_solves, "cache_hits": evaluate.cache_hits,
                   "ml_training": False, "comsol_spectra_used": False})
    atomic_json(destination, output)
    print(json.dumps(output if not args.test_existing else {
        "retrieval_checks": len(retrieval), "held_out": holdout,
        "off_grid": off_grid, "elapsed_seconds": output["elapsed_seconds"]}, indent=2), flush=True)
    print(f"Saved {destination}", flush=True)
    if args.test_existing:
        write_test_report(output)


def write_test_report(output):
    retrieval = output["retrieval_checks"]
    holdout = output["held_out_response_test"]
    off_grid = output["off_grid_target_test"]
    lines = ["# Pure-PWE gap-only inverse tests", "",
                 "No ML training or COMSOL spectral fitting. Targets come from saved PWE or an explicit requested width.",
                 "Full frozen M=20, normal grid 256, 281 k-points, two bands; no reduced-fidelity search.", "",
                 f"Positive saved-target retrieval checks: {len(retrieval)}.",
                 f"Nonpositive signed gaps excluded as width targets: {output['excluded_nonpositive_gaps']}.",
                 "Retrieval checks are software sanity checks, not independent inverse validation.", "",
                 "|Experiment|Target GHz|L1 um|L2 um|Achieved GHz|Absolute error GHz|Trial evaluations (may reuse cache)|Status|",
                 "|---|---:|---:|---:|---:|---:|---:|---|"]
    for name, r in [("Held-out "+output["held_out_design"], holdout), ("Requested off-grid gap", off_grid)]:
        c = r["candidate"]
        lines.append(f"|{name}|{r['target_gap_ghz']:.9f}|{c['L1_um']:.9f}|"
                     f"{c['L2_um']:.9f}|{c['gap_ghz']:.9f}|{r['absolute_error_ghz']:.9f}|"
                     f"{r['new_evaluations']}|{r['status']}|")
    lines += ["", f"Elapsed seconds: {output['elapsed_seconds']:.3f}.",
                  "", "One solution per target; geometry recovery is not required for a nonunique scalar inverse.",
                  "Only one held-out design tested, not a full LODO benchmark.",
                  "Matching tolerance is not a bound on physical accuracy. New geometries have PWE-only verification.",
                  "Gaps are sampled-path gaps, not certified full-Brillouin-zone gaps.",
                  "Existing learned-engine parameter-recovery acceptance gates are not certified by this experiment."]
    (DEST/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
