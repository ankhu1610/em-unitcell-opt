# CUDA PWE benchmark

Frozen physical operator; no COMSOL labels, training, fitting or reduced precision.

GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU. PyTorch: 2.11.0+cu128. CUDA: 12.8.
Median CPU solve: 1.544938 seconds.
Median GPU solve: 0.315642 seconds.
CPU time / GPU time: 4.895.
CPU uses a lowest-two subset eigensolve; CUDA uses a full Hermitian eigensolve.
Timing includes operator assembly and residual checks, after warmup.
No production checkpoints were modified by this benchmark.
