# zCUDA Examples

A collection of 50 examples demonstrating every module of the zCUDA library.
All examples use the **safe API layer** exclusively.

## Building & Running

```bash
# Build a single example
zig build example-<category>-<name> -D<lib>=true

# Run a single example
zig build run-<category>-<name> -D<lib>=true

# Examples:
zig build run-basics-vector_add
zig build run-cublas-gemm -Dcublas=true
zig build run-cudnn-conv2d -Dcudnn=true
```

Library flags: `-Dcublas=true`, `-Dcublaslt=true`, `-Dcudnn=true`, `-Dcufft=true`,
`-Dcurand=true`, `-Dcusolver=true`, `-Dcusparse=true`, `-Dnvtx=true`.

---

## Basics — CUDA Driver API (8 examples)

Core GPU programming: contexts, streams, events, kernels.

| Example | Description | Build Command |
|---------|------------|---------------|
| [vector_add](basics/vector_add.zig) | Vector addition via JIT-compiled kernel | `run-basics-vector_add` |
| [device_info](basics/device_info.zig) | GPU specs: memory, compute, features, limits | `run-basics-device_info` |
| [event_timing](basics/event_timing.zig) | Event-based GPU timing & bandwidth measurement | `run-basics-event_timing` |
| [streams](basics/streams.zig) | Multi-stream concurrent kernel execution | `run-basics-streams` |
| [peer_to_peer](basics/peer_to_peer.zig) | Multi-GPU peer access and cross-device copy | `run-basics-peer_to_peer` |
| [constant_memory](basics/constant_memory.zig) | GPU constant memory for polynomial evaluation | `run-basics-constant_memory` |
| [struct_kernel](basics/struct_kernel.zig) | Pass Zig `extern struct` to GPU kernel | `run-basics-struct_kernel` |
| [kernel_attributes](basics/kernel_attributes.zig) | Query kernel registers, shared memory, occupancy | `run-basics-kernel_attributes` |

---

## cuBLAS — Dense Linear Algebra (18 examples)

BLAS Level 1, 2, and 3 operations on GPU.

### Level 1 (Vector-Vector)

| Example | Description | Build Command |
|---------|------------|---------------|
| [axpy](cublas/axpy.zig) | SAXPY: y = α·x + y | `run-cublas-axpy -Dcublas=true` |
| [dot](cublas/dot.zig) | Dot product | `run-cublas-dot -Dcublas=true` |
| [nrm2_asum](cublas/nrm2_asum.zig) | L1 and L2 vector norms | `run-cublas-nrm2_asum -Dcublas=true` |
| [scal](cublas/scal.zig) | Vector scaling: x = α·x | `run-cublas-scal -Dcublas=true` |
| [amax_amin](cublas/amax_amin.zig) | Index of max/min absolute value | `run-cublas-amax_amin -Dcublas=true` |
| [swap_copy](cublas/swap_copy.zig) | Vector swap and copy | `run-cublas-swap_copy -Dcublas=true` |
| [rot](cublas/rot.zig) | Givens rotation | `run-cublas-rot -Dcublas=true` |
| [cosine_similarity](cublas/cosine_similarity.zig) | Cosine similarity via L1 ops | `run-cublas-cosine_similarity -Dcublas=true` |

### Level 2 (Matrix-Vector)

| Example | Description | Build Command |
|---------|------------|---------------|
| [gemv](cublas/gemv.zig) | Matrix-vector multiply (SGEMV) | `run-cublas-gemv -Dcublas=true` |
| [symv_syr](cublas/symv_syr.zig) | Symmetric matrix-vector ops | `run-cublas-symv_syr -Dcublas=true` |
| [trmv_trsv](cublas/trmv_trsv.zig) | Triangular multiply and solve | `run-cublas-trmv_trsv -Dcublas=true` |

### Level 3 (Matrix-Matrix)

| Example | Description | Build Command |
|---------|------------|---------------|
| [gemm](cublas/gemm.zig) | Matrix-matrix multiply (SGEMM) | `run-cublas-gemm -Dcublas=true` |
| [gemm_batched](cublas/gemm_batched.zig) | Strided batched GEMM | `run-cublas-gemm_batched -Dcublas=true` |
| [gemm_ex](cublas/gemm_ex.zig) | Mixed-precision GemmEx | `run-cublas-gemm_ex -Dcublas=true` |
| [symm](cublas/symm.zig) | Symmetric matrix multiply | `run-cublas-symm -Dcublas=true` |
| [trsm](cublas/trsm.zig) | Triangular solve (STRSM) | `run-cublas-trsm -Dcublas=true` |
| [syrk](cublas/syrk.zig) | Symmetric rank-k update | `run-cublas-syrk -Dcublas=true` |
| [geam](cublas/geam.zig) | Matrix add / transpose | `run-cublas-geam -Dcublas=true` |
| [dgmm](cublas/dgmm.zig) | Diagonal matrix multiply | `run-cublas-dgmm -Dcublas=true` |

---

## cuBLAS LT — Lightweight BLAS (1 example)

Advanced GEMM with algorithm heuristics and mixed-precision support.

| Example | Description | Build Command |
|---------|------------|---------------|
| [lt_sgemm](cublaslt/lt_sgemm.zig) | SGEMM with heuristic algorithm selection | `run-cublaslt-lt_sgemm -Dcublaslt=true` |

---

## cuDNN — Deep Neural Networks (3 examples)

Neural network primitives: convolution, activation, pooling, softmax.

| Example | Description | Build Command |
|---------|------------|---------------|
| [activation](cudnn/activation.zig) | ReLU, sigmoid, tanh activation functions | `run-cudnn-activation -Dcudnn=true` |
| [pooling_softmax](cudnn/pooling_softmax.zig) | Max pooling + softmax pipeline | `run-cudnn-pooling_softmax -Dcudnn=true` |
| [conv2d](cudnn/conv2d.zig) | 2D convolution forward pass | `run-cudnn-conv2d -Dcudnn=true` |

---

## cuFFT — Fast Fourier Transform (4 examples)

1D, 2D, and 3D FFTs with complex and real data.

| Example | Description | Build Command |
|---------|------------|---------------|
| [fft_1d_c2c](cufft/fft_1d_c2c.zig) | 1D complex-to-complex FFT | `run-cufft-fft_1d_c2c -Dcufft=true` |
| [fft_1d_r2c](cufft/fft_1d_r2c.zig) | 1D real-to-complex with frequency filtering | `run-cufft-fft_1d_r2c -Dcufft=true` |
| [fft_2d](cufft/fft_2d.zig) | 2D complex FFT | `run-cufft-fft_2d -Dcufft=true` |
| [fft_3d](cufft/fft_3d.zig) | 3D complex FFT | `run-cufft-fft_3d -Dcufft=true` |

---

## cuRAND — Random Number Generation (3 examples)

GPU-accelerated random number generation.

| Example | Description | Build Command |
|---------|------------|---------------|
| [distributions](curand/distributions.zig) | Uniform, normal, Poisson distributions | `run-curand-distributions -Dcurand=true` |
| [generators](curand/generators.zig) | Generator type comparison (XORWOW, MRG32k3a, etc.) | `run-curand-generators -Dcurand=true` |
| [monte_carlo_pi](curand/monte_carlo_pi.zig) | Monte Carlo π estimation | `run-curand-monte_carlo_pi -Dcurand=true` |

---

## cuSOLVER — Dense Solvers (5 examples)

LU, QR, Cholesky, SVD, and eigenvalue decomposition.

| Example | Description | Build Command |
|---------|------------|---------------|
| [getrf](cusolver/getrf.zig) | LU factorization + linear solve | `run-cusolver-getrf -Dcusolver=true` |
| [gesvd](cusolver/gesvd.zig) | Singular value decomposition | `run-cusolver-gesvd -Dcusolver=true` |
| [potrf](cusolver/potrf.zig) | Cholesky factorization + solve | `run-cusolver-potrf -Dcusolver=true` |
| [syevd](cusolver/syevd.zig) | Eigenvalue decomposition | `run-cusolver-syevd -Dcusolver=true` |
| [geqrf](cusolver/geqrf.zig) | QR factorization | `run-cusolver-geqrf -Dcusolver=true` |

---

## cuSPARSE — Sparse Linear Algebra (4 examples)

Sparse matrix operations with CSR, COO, and SpGEMM.

| Example | Description | Build Command |
|---------|------------|---------------|
| [spmv_csr](cusparse/spmv_csr.zig) | Sparse matrix-vector multiply (CSR) | `run-cusparse-spmv_csr -Dcusparse=true` |
| [spmv_coo](cusparse/spmv_coo.zig) | Sparse matrix-vector multiply (COO) | `run-cusparse-spmv_coo -Dcusparse=true` |
| [spmm_csr](cusparse/spmm_csr.zig) | Sparse × dense matrix multiply | `run-cusparse-spmm_csr -Dcusparse=true` |
| [spgemm](cusparse/spgemm.zig) | Sparse × sparse matrix multiply | `run-cusparse-spgemm -Dcusparse=true` |

---

## NVRTC — Runtime Compilation (2 examples)

Just-in-time CUDA kernel compilation.

| Example | Description | Build Command |
|---------|------------|---------------|
| [jit_compile](nvrtc/jit_compile.zig) | Runtime kernel compilation and execution | `run-nvrtc-jit_compile` |
| [template_kernel](nvrtc/template_kernel.zig) | Multi-kernel pipeline with templated types | `run-nvrtc-template_kernel` |

---

## NVTX — Profiling Annotations (1 example)

Nsight-compatible profiling markers.

| Example | Description | Build Command |
|---------|------------|---------------|
| [profiling](nvtx/profiling.zig) | Range push/pop and mark annotations | `run-nvtx-profiling -Dnvtx=true` |
