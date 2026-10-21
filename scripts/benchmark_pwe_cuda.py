"""Benchmark CUDA complex128 against the frozen CPU PWE, without COMSOL data."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import json
import time
import importlib.metadata
import numpy as np
import torch
from validate_energy_pwe_full import (
    ROOT, OUT, inputs, tensor_for, lowest_two, atomic_json, path_coordinates, C,
)


class CudaSolver:
    """Same Hz operator and precision; no autograd, training or fitting."""
    def __init__(self, g, tensor):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        self.g = torch.as_tensor(g, dtype=torch.float64, device="cuda")
        self.tensor = tuple(torch.as_tensor(t, dtype=torch.complex128, device="cuda") for t in tensor)
        torch.cuda.synchronize()

    @torch.no_grad()
    def solve(self, k):
        xx, xy, yx, yy = self.tensor
        x, y = (self.g + torch.as_tensor(k, dtype=torch.float64, device="cuda")).T
        operator = (y[:, None]*xx*y[None, :] - y[:, None]*xy*x[None, :]
                    - x[:, None]*yx*y[None, :] + x[:, None]*yy*x[None, :])
        defect = (torch.linalg.norm(operator-operator.mH)/torch.linalg.norm(operator)).item()
        if not np.isfinite(defect) or defect > 1e-12:
            raise ValueError(f"Non-Hermitian CUDA operator: {defect}")
        values, vectors = torch.linalg.eigh(operator)
        low, v = values[:2], vectors[:, :2]
        if low[0].item() <= 0:
            raise ValueError("Nonpositive eigenvalue away from Gamma")
        residual = torch.linalg.vector_norm(operator@v-v*low, dim=0)
        relative = torch.max(residual/torch.clamp(torch.abs(low), min=1.)).item()
        if relative > 1e-7:
            raise ValueError(f"CUDA residual too large: {relative}")
        scale = torch.max(torch.sum(torch.abs(operator), dim=1))
        f = (C*torch.sqrt(low)/(2*np.pi*1e9)).cpu().numpy()
        return f, {"hermitian_defect": defect, "relative_eigenpair_residual": relative,
                   "backward_residual": (torch.max(residual)/scale).item()}


def main():
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    if not torch.cuda.is_available():
        raise RuntimeError("Installed torch cannot access CUDA")
    cfg, fingerprint, designs = inputs()
    result = {"fingerprint": fingerprint, "torch": torch.__version__,
              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
              "dtype": "complex128/float64", "samples": [],
              "environment": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}
    for name in ["D001", "D028", "D064"]:
        design = next(d for d in designs if d["design_id"] == name)
        g, tensor = tensor_for(design, cfg)
        solver = CudaSolver(g, tensor)
        kpoints = path_coordinates([.1, .37, 1., 2., 2.9], cfg["a_um"]*1e-6)
        # Warm up the CUDA library once per tensor outside timed solves.
        solver.solve(kpoints[0])
        for k, point in zip([.1, .37, 1., 2., 2.9], kpoints):
            start = time.perf_counter()
            cpu, cpu_check = lowest_two(g, point, tensor)
            cpu_seconds = time.perf_counter()-start
            torch.cuda.synchronize()
            start = time.perf_counter()
            gpu, gpu_check = solver.solve(point)
            torch.cuda.synchronize()
            gpu_seconds = time.perf_counter()-start
            error = float(np.max(abs(cpu-gpu)))
            if error > 1e-5:
                raise ValueError(f"CPU/GPU disagreement {error} GHz")
            row = {"design_id": name, "k": k, "max_abs_difference_GHz": error,
                   "cpu_seconds": cpu_seconds, "gpu_seconds": gpu_seconds,
                   "cpu": cpu_check, "gpu": gpu_check}
            result["samples"].append(row)
            print(json.dumps(row), flush=True)
        del solver
        torch.cuda.empty_cache()
    result["median_cpu_seconds"] = float(np.median([r["cpu_seconds"] for r in result["samples"]]))
    result["median_gpu_seconds"] = float(np.median([r["gpu_seconds"] for r in result["samples"]]))
    result["cpu_to_gpu_speed_ratio"] = result["median_cpu_seconds"]/result["median_gpu_seconds"]
    atomic_json(OUT/"cuda_benchmark.json", result)
    lines = ["# CUDA PWE benchmark", "",
             "Frozen physical operator; no COMSOL labels, training, fitting or reduced precision.", "",
             f'GPU: {result["gpu"]}. PyTorch: {result["torch"]}. CUDA: {result["cuda"]}.',
             f'Median CPU solve: {result["median_cpu_seconds"]:.6f} seconds.',
             f'Median GPU solve: {result["median_gpu_seconds"]:.6f} seconds.',
             f'CPU time / GPU time: {result["cpu_to_gpu_speed_ratio"]:.3f}.',
             "CPU uses a lowest-two subset eigensolve; CUDA uses a full Hermitian eigensolve.",
             "Timing includes operator assembly and residual checks, after warmup.",
             "No production checkpoints were modified by this benchmark."]
    (OUT/"CUDA_BENCHMARK.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ["samples", "environment"]}), flush=True)


if __name__ == "__main__":
    main()
