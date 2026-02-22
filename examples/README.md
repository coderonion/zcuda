# zCUDA Examples

A comprehensive collection of **159 examples** demonstrating every module and capability of the zCUDA library.
All host examples use the **safe API layer** exclusively.

## Overview

| Category | Examples | Notes |
|----------|----------|-------|
| [Basics (Driver API)](basics/README.md) | 16 | Contexts, streams, memory, events |
| [cuBLAS](cublas/README.md) | 19 | BLAS L1/L2/L3 + mixed-precision |
| [cuBLAS LT](cublaslt/README.md) | 1 | Lightweight BLAS with heuristics |
| [cuDNN](cudnn/README.md) | 3 | Conv, activation, pooling |
| [cuFFT](cufft/README.md) | 4 | 1D/2D/3D complex and real FFTs |
| [cuRAND](curand/README.md) | 3 | GPU random number generation |
| [cuSOLVER](cusolver/README.md) | 5 | LU, QR, Cholesky, SVD, eigenvalue |
| [cuSPARSE](cusparse/README.md) | 4 | SpMV (CSR/COO), SpMM, SpGEMM |
| [NVRTC](nvrtc/README.md) | 2 | JIT kernel compilation |
| [NVTX](nvtx/README.md) | 1 | Profiling annotations |
| **Kernel DSL — [all categories](kernel/)** | **80** | Pure-Zig GPU kernels (11 categories) |
| Integration | 24 | End-to-end pipelines and benchmarks |
| **Total** | **162** | — |

---

## Building & Running

```bash
# Run a host example (run-<category>-<name>)
zig build run-basics-vector_add
zig build run-cublas-gemm -Dcublas=true
zig build run-cudnn-conv2d -Dcudnn=true

# Build a specific kernel example (example-kernel-<cat>-<name>)
zig build example-kernel-0-basic-kernel_vector_add -Dgpu-arch=sm_86

# Build all integration examples at once
zig build example-integration -Dcublas=true -Dcufft=true ...

# Run all integration binaries
zig-out/bin/integration-<name>
```

Library flags: `-Dcublas=true`, `-Dcublaslt=true`, `-Dcudnn=true`, `-Dcufft=true`,
`-Dcurand=true`, `-Dcusolver=true`, `-Dcusparse=true`, `-Dnvtx=true`.

---

## [Basics — CUDA Driver API](basics/README.md) (16 examples)

Core GPU programming: contexts, streams, memory management, events, kernels.

| Example | Description | Run Command |
|---------|-------------|-------------|
| [vector_add](basics/vector_add.zig) | Vector addition via JIT kernel | `run-basics-vector_add` |
| [device_info](basics/device_info.zig) | GPU specs: memory, compute, features | `run-basics-device_info` |
| [event_timing](basics/event_timing.zig) | Event-based timing & bandwidth measurement | `run-basics-event_timing` |
| [streams](basics/streams.zig) | Multi-stream concurrent execution | `run-basics-streams` |
| [peer_to_peer](basics/peer_to_peer.zig) | Multi-GPU peer access and cross-device copy | `run-basics-peer_to_peer` |
| [constant_memory](basics/constant_memory.zig) | GPU constant memory for polynomial eval | `run-basics-constant_memory` |
| [struct_kernel](basics/struct_kernel.zig) | Pass Zig `extern struct` to GPU kernel | `run-basics-struct_kernel` |
| [kernel_attributes](basics/kernel_attributes.zig) | Query kernel registers, shared mem, occupancy | `run-basics-kernel_attributes` |
| [alloc_patterns](basics/alloc_patterns.zig) | Device, host, pinned, and unified allocation | `run-basics-alloc_patterns` |
| [async_memcpy](basics/async_memcpy.zig) | Async H2D/D2H transfers with streams | `run-basics-async_memcpy` |
| [pinned_memory](basics/pinned_memory.zig) | Pinned (page-locked) memory for faster transfers | `run-basics-pinned_memory` |
| [unified_memory](basics/unified_memory.zig) | Unified memory (UM) migration and access | `run-basics-unified_memory` |
| [context_lifecycle](basics/context_lifecycle.zig) | Context creation, binding, and destruction | `run-basics-context_lifecycle` |
| [dtod_copy_chain](basics/dtod_copy_chain.zig) | Device-to-device chained copy pipeline | `run-basics-dtod_copy_chain` |
| [memset_patterns](basics/memset_patterns.zig) | Device memset patterns and initialization | `run-basics-memset_patterns` |
| [multi_device_query](basics/multi_device_query.zig) | Enumerate and query all CUDA devices | `run-basics-multi_device_query` |

---

## [cuBLAS — Dense Linear Algebra](cublas/README.md) (19 examples)

BLAS Level 1, 2, and 3 operations. Enable with `-Dcublas=true`.

### Level 1 — Vector–Vector Operations

| Example | Description | Run Command |
|---------|-------------|-------------|
| [axpy](cublas/axpy.zig) | SAXPY: y = α·x + y | `run-cublas-axpy -Dcublas=true` |
| [dot](cublas/dot.zig) | Dot product | `run-cublas-dot -Dcublas=true` |
| [nrm2_asum](cublas/nrm2_asum.zig) | L1 and L2 vector norms | `run-cublas-nrm2_asum -Dcublas=true` |
| [scal](cublas/scal.zig) | Vector scaling: x = α·x | `run-cublas-scal -Dcublas=true` |
| [amax_amin](cublas/amax_amin.zig) | Index of max/min absolute value | `run-cublas-amax_amin -Dcublas=true` |
| [swap_copy](cublas/swap_copy.zig) | Vector swap and copy | `run-cublas-swap_copy -Dcublas=true` |
| [rot](cublas/rot.zig) | Givens rotation | `run-cublas-rot -Dcublas=true` |
| [cosine_similarity](cublas/cosine_similarity.zig) | Cosine similarity via L1 ops | `run-cublas-cosine_similarity -Dcublas=true` |

### Level 2 — Matrix–Vector Operations

| Example | Description | Run Command |
|---------|-------------|-------------|
| [gemv](cublas/gemv.zig) | Matrix-vector multiply (SGEMV) | `run-cublas-gemv -Dcublas=true` |
| [symv_syr](cublas/symv_syr.zig) | Symmetric matrix-vector ops | `run-cublas-symv_syr -Dcublas=true` |
| [trmv_trsv](cublas/trmv_trsv.zig) | Triangular multiply and solve | `run-cublas-trmv_trsv -Dcublas=true` |

### Level 3 — Matrix–Matrix Operations

| Example | Description | Run Command |
|---------|-------------|-------------|
| [gemm](cublas/gemm.zig) | Matrix-matrix multiply (SGEMM) | `run-cublas-gemm -Dcublas=true` |
| [gemm_batched](cublas/gemm_batched.zig) | Strided batched GEMM | `run-cublas-gemm_batched -Dcublas=true` |
| [gemm_ex](cublas/gemm_ex.zig) | Mixed-precision GemmEx | `run-cublas-gemm_ex -Dcublas=true` |
| [symm](cublas/symm.zig) | Symmetric matrix multiply | `run-cublas-symm -Dcublas=true` |
| [trsm](cublas/trsm.zig) | Triangular solve (STRSM) | `run-cublas-trsm -Dcublas=true` |
| [syrk](cublas/syrk.zig) | Symmetric rank-k update | `run-cublas-syrk -Dcublas=true` |
| [geam](cublas/geam.zig) | Matrix add / transpose | `run-cublas-geam -Dcublas=true` |
| [dgmm](cublas/dgmm.zig) | Diagonal matrix multiply | `run-cublas-dgmm -Dcublas=true` |

---

## [cuBLAS LT — Lightweight BLAS](cublaslt/README.md) (1 example)

Advanced GEMM with algorithm heuristics and mixed-precision support. Enable with `-Dcublaslt=true`.

| Example | Description | Run Command |
|---------|-------------|-------------|
| [lt_sgemm](cublaslt/lt_sgemm.zig) | SGEMM with heuristic algorithm selection | `run-cublaslt-lt_sgemm -Dcublaslt=true` |

---

## [cuDNN — Deep Neural Networks](cudnn/README.md) (3 examples)

Neural network primitives: convolution, activation, pooling, softmax. Enable with `-Dcudnn=true`.

| Example | Description | Run Command |
|---------|-------------|-------------|
| [activation](cudnn/activation.zig) | ReLU, sigmoid, tanh activation functions | `run-cudnn-activation -Dcudnn=true` |
| [pooling_softmax](cudnn/pooling_softmax.zig) | Max pooling + softmax pipeline | `run-cudnn-pooling_softmax -Dcudnn=true` |
| [conv2d](cudnn/conv2d.zig) | 2D convolution forward pass | `run-cudnn-conv2d -Dcudnn=true` |

---

## [cuFFT — Fast Fourier Transform](cufft/README.md) (4 examples)

1D, 2D, and 3D FFTs with complex and real data. Enable with `-Dcufft=true`.

| Example | Description | Run Command |
|---------|-------------|-------------|
| [fft_1d_c2c](cufft/fft_1d_c2c.zig) | 1D complex-to-complex FFT | `run-cufft-fft_1d_c2c -Dcufft=true` |
| [fft_1d_r2c](cufft/fft_1d_r2c.zig) | 1D real-to-complex with frequency filtering | `run-cufft-fft_1d_r2c -Dcufft=true` |
| [fft_2d](cufft/fft_2d.zig) | 2D complex FFT | `run-cufft-fft_2d -Dcufft=true` |
| [fft_3d](cufft/fft_3d.zig) | 3D complex FFT | `run-cufft-fft_3d -Dcufft=true` |

---

## [cuRAND — Random Number Generation](curand/README.md) (3 examples)

GPU-accelerated random number generation. Enable with `-Dcurand=true`.

| Example | Description | Run Command |
|---------|-------------|-------------|
| [distributions](curand/distributions.zig) | Uniform, normal, Poisson distributions | `run-curand-distributions -Dcurand=true` |
| [generators](curand/generators.zig) | Generator comparison (XORWOW, MRG32k3a, …) | `run-curand-generators -Dcurand=true` |
| [monte_carlo_pi](curand/monte_carlo_pi.zig) | Monte Carlo π estimation | `run-curand-monte_carlo_pi -Dcurand=true` |

---

## [cuSOLVER — Dense Solvers](cusolver/README.md) (5 examples)

LU, QR, Cholesky, SVD, and eigenvalue decomposition. Enable with `-Dcusolver=true`.

> **Note:** `devInfo` is a GPU-side pointer (`CudaSlice(i32)`) per cuSOLVER API contract.
> Use `stream.memcpyDtoH` after `ctx.synchronize()` to read it on the host.

| Example | Description | Run Command |
|---------|-------------|-------------|
| [getrf](cusolver/getrf.zig) | LU factorization (PA = LU) + linear solve | `run-cusolver-getrf -Dcusolver=true` |
| [gesvd](cusolver/gesvd.zig) | Singular value decomposition (A = UΣVᵀ) | `run-cusolver-gesvd -Dcusolver=true` |
| [potrf](cusolver/potrf.zig) | Cholesky factorization (A = LLᵀ) + solve | `run-cusolver-potrf -Dcusolver=true` |
| [syevd](cusolver/syevd.zig) | Symmetric eigenvalue decomposition | `run-cusolver-syevd -Dcusolver=true` |
| [geqrf](cusolver/geqrf.zig) | QR factorization + Q extraction | `run-cusolver-geqrf -Dcusolver=true` |

---

## [cuSPARSE — Sparse Linear Algebra](cusparse/README.md) (4 examples)

Sparse matrix operations with CSR, COO, and SpGEMM formats. Enable with `-Dcusparse=true`.

| Example | Description | Run Command |
|---------|-------------|-------------|
| [spmv_csr](cusparse/spmv_csr.zig) | Sparse matrix-vector multiply (CSR) | `run-cusparse-spmv_csr -Dcusparse=true` |
| [spmv_coo](cusparse/spmv_coo.zig) | Sparse matrix-vector multiply (COO) | `run-cusparse-spmv_coo -Dcusparse=true` |
| [spmm_csr](cusparse/spmm_csr.zig) | Sparse × dense matrix multiply | `run-cusparse-spmm_csr -Dcusparse=true` |
| [spgemm](cusparse/spgemm.zig) | Sparse × sparse matrix multiply | `run-cusparse-spgemm -Dcusparse=true` |

---

## [NVRTC — Runtime Compilation](nvrtc/README.md) (2 examples)

Just-in-time CUDA kernel compilation.

| Example | Description | Run Command |
|---------|-------------|-------------|
| [jit_compile](nvrtc/jit_compile.zig) | Runtime CUDA C++ → PTX compilation | `run-nvrtc-jit_compile` |
| [template_kernel](nvrtc/template_kernel.zig) | Multi-kernel pipeline with templated types | `run-nvrtc-template_kernel` |

---

## [NVTX — Profiling Annotations](nvtx/README.md) (1 example)

Nsight-compatible range markers.

| Example | Description | Run Command |
|---------|-------------|-------------|
| [profiling](nvtx/profiling.zig) | Range push/pop and point mark annotations | `run-nvtx-profiling -Dnvtx=true` |

---

## Kernel DSL — Pure-Zig GPU Kernels (80 examples)

All kernels are written in pure Zig and compiled to PTX via Zig's built-in LLVM NVPTX backend.
See [kernel/README.md](kernel/README.md) for the full index.

| Category | Examples | Topics |
|----------|----------|--------|
| [0_Basic](kernel/0_Basic/) | 8 | SAXPY, RELU, dot, grid-stride, normalize |
| [1_Reduction](kernel/1_Reduction/) | 5 | Warp reduce, multi-block, prefix sum, scalar product |
| [2_Matrix](kernel/2_Matrix/) | 6 | Naive & tiled matmul, matvec, transpose, pad, diag |
| [3_Atomics](kernel/3_Atomics/) | 5 | Atomic ops, histograms, warp-aggregated atomics |
| [4_SharedMemory](kernel/4_SharedMemory/) | 3 | Dynamic SMEM, 1D stencil, shared mem demo |
| [5_Warp](kernel/5_Warp/) | 5 | Ballot, broadcast, match, reduce, scan |
| [6_MathAndTypes](kernel/6_MathAndTypes/) | 9 | FP16, complex, FFT filter, fast math, type conversion |
| [7_Debug](kernel/7_Debug/) | 2 | Error checking, `printf` debug from GPU |
| [8_TensorCore](kernel/8_TensorCore/) | 11 | WMMA (f16/bf16/int8/tf32), MMA (f16/fp8) |
| [9_Advanced](kernel/9_Advanced/) | 8 | Async copy pipeline, cooperative groups, softmax |
| [10_Integration](kernel/10_Integration/) | 24 | End-to-end pipelines and benchmarks |

---

## Integration Examples (24 examples)

End-to-end integration examples using Zig kernels with CUDA libraries.

```bash
# Build all integration examples
zig build example-integration -Dgpu-arch=sm_86 -Dcublas=true -Dcufft=true ...

# Run a specific binary
./zig-out/bin/integration-<name>
```

| Binary | Description |
|--------|-------------|
| integration-module-load-launch | Driver lifecycle: PTX load + kernel launch |
| integration-ptx-compile-execute | NVRTC compile + execute pipeline |
| integration-stream-callback | Stream callback pattern (event-driven) |
| integration-stream-concurrency | Multi-stream concurrent execution |
| integration-basic-graph | CUDA Graph basics: capture and replay |
| integration-graph-replay-update | Graph replay with node update |
| integration-graph-with-deps | Graph with explicit dependencies |
| integration-scale-bias-gemm | cuBLAS Scale+Bias→GEMM→ReLU pipeline |
| integration-residual-gemm | Residual connection with GEMM |
| integration-error-recovery | CUDA error recovery patterns |
| integration-oob-launch | Out-of-bounds launch detection |
| integration-fft-filter | FFT-based filter pipeline |
| integration-conv2d-fft | 2D convolution via FFT |
| integration-occupancy-calc | Occupancy calculator utilities |
| integration-monte-carlo-option | Monte Carlo option pricing (GPU) |
| integration-particle-system | Particle system simulation |
| integration-matmul-e2e | Matrix multiply end-to-end |
| integration-reduction-e2e | Reduction end-to-end |
| integration-saxpy-e2e | SAXPY end-to-end |
| integration-multi-library | Multi-library pipeline (cuBLAS + cuDNN + cuFFT) |
| integration-wmma-gemm-verify | WMMA GEMM correctness verification |
| integration-attention-pipeline | Attention pipeline (QK^T, softmax, V) |
| integration-mixed-precision-train | Mixed-precision training pipeline (FP16+TF32) |
| integration-perf-benchmark | Zig kernel vs cuBLAS (event-timed benchmark) |
