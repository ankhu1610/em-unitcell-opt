"""Bounded CPU geometry parallelism, reusing the unchanged frozen PWE solver.

Completed files are never rewritten. Partial files resume at their saved point.
Original fingerprints remain valid: physics and eigensolver are unchanged.
Execution provenance is recorded separately. No new dependencies.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
from pathlib import Path
import time
import subprocess
from validate_energy_pwe_full import (
    OUT, inputs, tensor_for, lowest_two, path_coordinates, atomic_json, report as reference_report,
)


def checkpoint_io(operation, *, optional=False):
    """Retry transient Windows sharing errors without hiding solver failures.

    Optional live monitoring can wait until the next update. Final reporting
    remains mandatory and raises if access does not recover.
    """
    for attempt in range(5):
        try:
            return operation()
        except PermissionError:
            if attempt == 4:
                if optional:
                    print("Checkpoint access busy; deferring live monitoring update.", flush=True)
                    return None
                raise
            time.sleep(0.1 * (attempt + 1))


def worker(design, cfg, fingerprint, backend="scipy_cpu"):
    name = design["design_id"]
    path = OUT/"predictions"/f"{name}.json"
    if path.exists():
        record = json.loads(path.read_text())
        if record["fingerprint"] != fingerprint:
            raise ValueError(f"Wrong checkpoint fingerprint: {name}")
    else:
        record = {"fingerprint": fingerprint, "design_id": name,
                  "sides_um": [float(design["L1_um"]), float(design["L2_um"])],
                  "frequencies_GHz": [], "diagnostics": [], "elapsed_compute_seconds": 0.}
    offset = len(record["frequencies_GHz"])
    if offset == len(cfg["k"]):
        return name
    if (OUT/"parallel.stop").exists():
        return name
    record.setdefault("execution_segments", []).append({
        "start_k_index": offset, "backend": backend, "cpu_threads": 2,
        "pid": os.getpid(), "started_unix": time.time(),
        "executor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    start = time.perf_counter()
    g, tensor = tensor_for(design, cfg)
    cuda_solver = None
    if backend == "torch_cuda":
        from benchmark_pwe_cuda import CudaSolver
        import torch
        torch.set_num_threads(2)
        torch.set_grad_enabled(False)
        cuda_solver = CudaSolver(g, tensor)
        record["execution_segments"][-1].update({
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "dtype": "complex128/float64", "gpu": torch.cuda.get_device_name(0),
            "cuda_solver_sha256": hashlib.sha256(
                Path(__file__).with_name("benchmark_pwe_cuda.py").read_bytes()).hexdigest()})
    kpoints = path_coordinates(cfg["k"], cfg["a_um"]*1e-6)
    previous = start
    for ik in range(offset, len(kpoints)):
        if cuda_solver is None:
            f, diagnostic = lowest_two(g, kpoints[ik], tensor)
        else:
            f, diagnostic = cuda_solver.solve(kpoints[ik])
        record["frequencies_GHz"].append(f.tolist())
        record["diagnostics"].append(diagnostic)
        stop = (OUT/"parallel.stop").exists()
        if (ik+1) % 10 == 0 or ik+1 == len(kpoints) or stop:
            now = time.perf_counter()
            record["elapsed_compute_seconds"] += now-previous
            previous = now
            atomic_json(path, record)
            print(f'{name}: {ik+1}/{len(kpoints)}; worker {os.getpid()}', flush=True)
        if stop:
            break
    return name


def gpu_assignment(python, name):
    subprocess.run([python, "-u", str(Path(__file__).resolve()), "--gpu-design", name],
                   check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return name


def report(cfg, fingerprint, designs):
    result = reference_report(cfg, fingerprint, designs)
    backends = set()
    for path in (OUT/"predictions").glob("D*.json"):
        record = json.loads(path.read_text())
        segments = record.get("execution_segments", [])
        if not segments or segments[0]["start_k_index"] > 0:
            backends.add("scipy_cpu")
        backends.update(s["backend"] for s in segments)
    result["execution_backends"] = sorted(backends)
    result["execution_note"] = (
        "Config/fingerprint identifies the frozen CPU-reference calculation. "
        "Actual execution backends and their provenance are recorded in checkpoint execution_segments.")
    atomic_json(OUT/"results.json", result)
    with (OUT/"REPORT.md").open("a", encoding="utf-8") as stream:
        stream.write("\n## Execution backends\n\n"+", ".join(sorted(backends))+"\n\n"+
                     result["execution_note"]+"\n")
    return result


def progress(active, gpu=False):
    complete, points = 0, 0
    for path in (OUT/"predictions").glob("D*.json"):
        r = json.loads(path.read_text())
        count = len(r["frequencies_GHz"])
        points += count
        complete += count == 281
    atomic_json(OUT/"status.json", {
        "status": "running", "pid": os.getpid(),
        "executor": "parallel CPU + CUDA" if gpu else "parallel CPU",
        "active_geometries": sorted(active), "completed_geometries": complete,
        "computed_k_points": points, "total_k_points": 64*281,
        "updated_unix": time.time()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--gpu-python")
    parser.add_argument("--gpu-design", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stop:
        atomic_json(OUT/"parallel.stop", {"requested_unix": time.time()})
        return
    if not 1 <= args.workers <= 4:
        raise ValueError("Worker limit is 1..4")
    if (OUT/"parallel.stop").exists():
        raise ValueError("A stop request exists; inspect before restarting")
    cfg, fingerprint, designs = inputs()
    if args.gpu_design:
        worker(next(d for d in designs if d["design_id"] == args.gpu_design), cfg, fingerprint, "torch_cuda")
        return
    bench = json.loads((OUT/"benchmark.json").read_text())
    if bench["fingerprint"] != fingerprint:
        raise ValueError("Frozen solver/config no longer matches original benchmark")
    if args.gpu_python:
        gpu_bench = json.loads((OUT/"cuda_benchmark.json").read_text())
        if gpu_bench["fingerprint"] != fingerprint or gpu_bench["cpu_to_gpu_speed_ratio"] <= 1:
            raise ValueError("GPU benchmark must match frozen operator and show a speed benefit")
    pending = []
    preserved = {}
    for d in designs:
        path = OUT/"predictions"/f'{d["design_id"]}.json'
        if path.exists():
            r = json.loads(path.read_text())
            if r["fingerprint"] != fingerprint:
                raise ValueError("Checkpoint mismatch")
            if len(r["frequencies_GHz"]) == len(cfg["k"]):
                preserved[d["design_id"]] = hashlib.sha256(path.read_bytes()).hexdigest()
                continue
        pending.append(d)
    lock = OUT/"run.lock"
    with lock.open("x") as f:
        f.write(str(os.getpid()))
    atomic_json(OUT/"parallel_launch.json", {
        "pid": os.getpid(), "workers": args.workers, "threads_per_worker": 2,
        "gpu_python": args.gpu_python, "gpu_workers": int(bool(args.gpu_python)),
        "fingerprint": fingerprint, "completed_files_preserved_sha256": preserved,
        "started_unix": time.time()})
    failure = None
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool, ThreadPoolExecutor(max_workers=1) as gpu_pool:
            futures = {}
            backends = {}
            iterator = iter(pending)
            if args.gpu_python and pending:
                d = next(iterator)
                future = gpu_pool.submit(gpu_assignment, args.gpu_python, d["design_id"])
                futures[future], backends[future] = d["design_id"], "cuda"
            for _ in range(args.workers):
                d = next(iterator, None)
                if d is not None:
                    future = pool.submit(worker, d, cfg, fingerprint)
                    futures[future], backends[future] = d["design_id"], "cpu"
            while futures:
                done, _ = wait(futures, timeout=5, return_when=FIRST_COMPLETED)
                checkpoint_io(lambda: progress(futures.values(), bool(args.gpu_python)),
                              optional=True)
                for future in done:
                    name = futures.pop(future)
                    backend = backends.pop(future)
                    try:
                        future.result()
                        checkpoint_io(lambda: report(cfg, fingerprint, designs), optional=True)
                        print(f"Finished worker assignment: {name}", flush=True)
                    except Exception as exc:
                        failure = repr(exc)
                        atomic_json(OUT/"parallel.stop", {"error": failure})
                    if not (OUT/"parallel.stop").exists():
                        d = next(iterator, None)
                        if d is not None:
                            new = (gpu_pool.submit(gpu_assignment, args.gpu_python, d["design_id"])
                                   if backend == "cuda" else pool.submit(worker, d, cfg, fingerprint))
                            futures[new], backends[new] = d["design_id"], backend
        for name, checksum in preserved.items():
            if hashlib.sha256((OUT/"predictions"/f"{name}.json").read_bytes()).hexdigest() != checksum:
                raise ValueError(f"Completed file changed: {name}")
        result = checkpoint_io(lambda: report(cfg, fingerprint, designs))
        atomic_json(OUT/"status.json", {
            "status": "failed" if failure else result["status"], "error": failure,
            "completed_geometries": result["completed_geometries"], "updated_unix": time.time()})
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
