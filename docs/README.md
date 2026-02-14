# zCUDA Documentation

## Overview

Comprehensive, safe, and idiomatic Zig bindings for the entire CUDA ecosystem.

| Stat         | Value                     |
| ------------ | ------------------------- |
| Zig Version  | 0.16.0-dev.2535+b5bd49460 |
| CUDA Toolkit | 12.8                      |
| Modules      | 10                        |
| Examples     | 50                        |

## Documentation Index

### API Reference

- [**API Overview & Cross-Reference**](api.md) — Complete module listing with function signatures and CUDA mapping

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

### Guides

- [Examples Guide](../examples/README.md) — 50 runnable examples with descriptions
- [Project README](../README.md) — Quick start, build options, and project overview
