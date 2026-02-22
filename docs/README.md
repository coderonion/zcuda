# zCUDA Documentation

## Documentation Index

### API Reference

- [**API Overview & Cross-Reference**](API.md) — Complete module listing with function signatures and CUDA mapping

### Modules

| Module   | Doc                          | Description                                                      |
| -------- | ---------------------------- | ---------------------------------------------------------------- |
| driver   | [README](driver/README.md)   | Device management, memory, kernel launch, streams, events        |
| nvrtc    | [README](nvrtc/README.md)    | Runtime compilation of CUDA C++ to PTX / CUBIN                   |
| cublas   | [README](cublas/README.md)   | BLAS Level 1/2/3 (SAXPY, SGEMM, DGEMM, batched, mixed-precision) |
| cublaslt | [README](cublaslt/README.md) | Lightweight GEMM with algorithm heuristics                       |
| curand   | [README](curand/README.md)   | GPU random number generation                                     |
| cudnn    | [README](cudnn/README.md)    | Convolution, activation, pooling, softmax, batch norm            |
| cusolver | [README](cusolver/README.md) | LU, QR, SVD, Cholesky, eigenvalue decomposition                  |
| cusparse | [README](cusparse/README.md) | SpMV, SpMM, SpGEMM with CSR/COO formats                          |
| cufft    | [README](cufft/README.md)    | 1D/2D/3D Fast Fourier Transform                                  |
| nvtx     | [README](nvtx/README.md)     | Profiling annotations for NVIDIA Nsight                          |
| **kernel**   | [**API**](kernel/API.md)     | **Kernel DSL — write CUDA kernels in pure Zig, compiled to PTX** |

### Guides

- [Kernel DSL API Reference](kernel/API.md) — intrinsics, shared memory, WMMA/MMA, TMA, cluster, tcgen05
- [CUDA C++ → Zig Migration](kernel/MIGRATION.md) — port existing CUDA C++ kernels to pure Zig
- [Examples Guide](../examples/README.md) — 162 examples: 58 host (10 categories with per-category READMEs) + 80 kernel (11 categories) + 24 integration
- [Project README](../README.md) — Quick start, build options, and project overview
