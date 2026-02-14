# zCUDA Project Structure

## Directory Layout

```
zcuda/
├── build.zig                  # Build configuration (library, tests, examples)
├── build.zig.zon              # Package manifest
├── src/                       # Source code
│   ├── cuda.zig               # Root module — re-exports all public types
│   ├── types.zig              # Shared types (Dim3, LaunchConfig, DevicePtr)
│   ├── driver/                # CUDA Driver API (always enabled)
│   │   ├── sys.zig            # Raw FFI (@cImport cuda.h)
│   │   ├── result.zig         # Error wrapping (CUresult → DriverError)
│   │   ├── safe.zig           # CudaContext, CudaStream, CudaSlice, CudaEvent, CudaGraph
│   │   └── driver.zig         # Module entry point
│   ├── nvrtc/                 # NVRTC — runtime compilation (always enabled)
│   │   ├── sys.zig            # Raw FFI (@cImport nvrtc.h)
│   │   ├── result.zig         # Error wrapping
│   │   ├── safe.zig           # compilePtx, compileCubin, CompileOptions
│   │   └── nvrtc.zig          # Module entry point
│   ├── cublas/                # cuBLAS — BLAS L1/L2/L3 (-Dcublas=true)
│   │   ├── sys.zig            # Raw FFI
│   │   ├── result.zig         # Error wrapping
│   │   ├── safe.zig           # CublasContext, GEMM/AXPY/TRSM etc.
│   │   └── cublas.zig         # Module entry point
│   ├── cublaslt/              # cuBLAS LT — lightweight GEMM (-Dcublaslt=true)
│   │   ├── sys.zig, result.zig, safe.zig, cublaslt.zig
│   │   └── ...
│   ├── curand/                # cuRAND — GPU random numbers (-Dcurand=true)
│   │   ├── sys.zig, result.zig, safe.zig, curand.zig
│   │   └── ...
│   ├── cudnn/                 # cuDNN — deep learning (-Dcudnn=true)
│   │   ├── sys.zig, result.zig, safe.zig, cudnn.zig
│   │   └── ...
│   ├── cusolver/              # cuSOLVER — direct solvers (-Dcusolver=true)
│   │   ├── sys.zig, result.zig, safe.zig, cusolver.zig
│   │   └── ...
│   ├── cusparse/              # cuSPARSE — sparse matrices (-Dcusparse=true)
│   │   ├── sys.zig, result.zig, safe.zig, cusparse.zig
│   │   └── ...
│   ├── cufft/                 # cuFFT — FFT (-Dcufft=true)
│   │   ├── sys.zig, result.zig, safe.zig, cufft.zig
│   │   └── ...
│   ├── nvtx/                  # NVTX — profiling annotations (-Dnvtx=true)
│   │   ├── sys.zig, safe.zig, nvtx.zig
│   │   └── ...
│   └── runtime/               # CUDA Runtime API (internal)
│       ├── sys.zig, result.zig, safe.zig, runtime.zig
│       └── ...
├── test/                      # Tests
│   ├── unit/                  # Unit tests (12 files)
│   │   ├── driver_test.zig    # Context, stream, memory, events, graphs
│   │   ├── nvrtc_test.zig     # PTX/CUBIN compilation
│   │   ├── cublas_test.zig    # BLAS L1/L2/L3 operations
│   │   ├── cublaslt_test.zig  # Lightweight GEMM
│   │   ├── curand_test.zig    # Random number generation
│   │   ├── cudnn_test.zig     # Conv, activation, pooling, softmax
│   │   ├── cusolver_test.zig  # LU, SVD, Cholesky, eigensolve
│   │   ├── cusparse_test.zig  # SpMV, SpMM, SpGEMM
│   │   ├── cufft_test.zig     # FFT plans and execution
│   │   ├── nvtx_test.zig      # Profiling annotations
│   │   ├── runtime_test.zig   # CUDA runtime API
│   │   └── types_test.zig     # Shared type tests
│   └── integration/           # Integration tests (10 files)
│       ├── gemm_roundtrip_test.zig    # cuBLAS GEMM round-trip
│       ├── jit_kernel_test.zig        # NVRTC compile + launch
│       ├── lu_solve_test.zig          # cuSOLVER LU solve pipeline
│       ├── svd_reconstruct_test.zig   # SVD reconstruction
│       ├── fft_roundtrip_test.zig     # FFT forward + inverse
│       ├── curand_fft_test.zig        # cuRAND → cuFFT pipeline
│       ├── conv_pipeline_test.zig     # cuDNN conv pipeline
│       ├── conv_relu_test.zig         # cuDNN conv + activation
│       ├── sparse_pipeline_test.zig   # cuSPARSE pipeline
│       └── syrk_geam_test.zig         # cuBLAS SYRK + GEAM
├── examples/                  # 50 runnable examples
│   ├── README.md              # Categorized example index
│   ├── basics/                # 8 examples — contexts, streams, events, kernels
│   │   ├── vector_add.zig, streams.zig, device_info.zig, event_timing.zig
│   │   ├── struct_kernel.zig, kernel_attributes.zig, constant_memory.zig
│   │   └── peer_to_peer.zig
│   ├── cublas/                # 19 examples — BLAS L1/L2/L3, batched, mixed-precision
│   │   ├── gemm.zig, axpy.zig, dot.zig, scal.zig, nrm2_asum.zig
│   │   ├── gemv.zig, symv_syr.zig, trmv_trsv.zig, trsm.zig
│   │   ├── gemm_batched.zig, gemm_ex.zig, geam.zig, dgmm.zig
│   │   ├── swap_copy.zig, rot.zig, amax_amin.zig, symm.zig, syrk.zig
│   │   └── cosine_similarity.zig
│   ├── cublaslt/              # 1 example
│   │   └── lt_sgemm.zig
│   ├── cudnn/                 # 3 examples — convolution, activation, pooling
│   │   ├── conv2d.zig, activation.zig, pooling_softmax.zig
│   │   └── ...
│   ├── cufft/                 # 4 examples — 1D/2D/3D FFT
│   │   ├── fft_1d_c2c.zig, fft_1d_r2c.zig, fft_2d.zig, fft_3d.zig
│   │   └── ...
│   ├── curand/                # 3 examples — RNG, distributions, Monte Carlo
│   │   ├── generators.zig, distributions.zig, monte_carlo_pi.zig
│   │   └── ...
│   ├── cusolver/              # 5 examples — LU, SVD, Cholesky, QR, eigensolve
│   │   ├── getrf.zig, gesvd.zig, potrf.zig, geqrf.zig, syevd.zig
│   │   └── ...
│   ├── cusparse/              # 4 examples — CSR/COO SpMV, SpMM, SpGEMM
│   │   ├── spmv_csr.zig, spmv_coo.zig, spmm_csr.zig, spgemm.zig
│   │   └── ...
│   ├── nvrtc/                 # 2 examples — JIT compilation
│   │   ├── jit_compile.zig, template_kernel.zig
│   │   └── ...
│   └── nvtx/                  # 1 example — Nsight profiling
│       └── profiling.zig
├── docs/                      # Documentation
│   ├── README.md              # Documentation index
│   ├── api.md                 # Complete API reference
│   ├── driver/README.md       # Driver module docs
│   ├── nvrtc/README.md        # NVRTC module docs
│   ├── cublas/README.md       # cuBLAS module docs
│   ├── cublaslt/README.md     # cuBLAS LT module docs
│   ├── curand/README.md       # cuRAND module docs
│   ├── cudnn/README.md        # cuDNN module docs
│   ├── cusolver/README.md     # cuSOLVER module docs
│   ├── cusparse/README.md     # cuSPARSE module docs
│   ├── cufft/README.md        # cuFFT module docs
│   └── nvtx/README.md         # NVTX module docs
└── CHANGELOG.md
```

## Module Overview

### Driver (`src/driver/` — 4 files)

Core CUDA types: `CudaContext`, `CudaStream`, `CudaSlice(T)`, `CudaView(T)`, `CudaViewMut(T)`, `CudaModule`, `CudaFunction`, `CudaEvent`, `CudaGraph`. Device management, memory allocation, host ↔ device transfers, kernel launch, stream synchronization, event timing, graph capture, and unified memory.

### NVRTC (`src/nvrtc/` — 4 files)

Runtime compilation: `compilePtx`, `compileCubin`, `compilePtxWithOptions`, `compileCubinWithOptions`. `CompileOptions` for target architecture, optimization, register limits, and arbitrary flags.

### cuBLAS (`src/cublas/` — 4 files)

`CublasContext` wrapping cuBLAS handle. Level 1 (AXPY, SCAL, DOT, NRM2, AMAX, AMIN, SWAP, COPY, ROT, ROTG), Level 2 (GEMV, SYMV, TRMV, TRSV, SYR), Level 3 (SGEMM, DGEMM, strided batched, pointer-array batched, GemmEx, SYMM, TRSM, TRMM, SYRK, GEAM, DGMM, grouped batched GEMM). Single and double precision throughout.

### cuBLAS LT (`src/cublaslt/` — 4 files)

`CublasLtContext` for lightweight GEMM with fine-grained algorithm selection via `getHeuristics`, layout descriptors, and `matmul`/`matmulWithAlgo`. Supports mixed-precision with f16/bf16/f32/f64 data types and TF32 compute.

### cuRAND (`src/curand/` — 4 files)

`CurandContext` with 8 generator types (XORWOW, MRG32k3a, MTGP32, MT19937, Philox, Sobol, etc.). Distributions: uniform, normal, log-normal, Poisson. Single and double precision.

### cuDNN (`src/cudnn/` — 4 files)

`CudnnContext` for deep learning primitives. 2D and N-dimensional convolution (forward, backward data, backward filter), activation, pooling, softmax (with backward), batch normalization, dropout, element-wise tensor operations (`opTensor`, `addTensor`, `scaleTensor`, `reduceTensor`). Multiple algorithms (implicit GEMM, Winograd, FFT, etc.).

### cuSOLVER (`src/cusolver/` — 4 files)

`CusolverDnContext` for LU factorization and SVD. `CusolverDnExt` extends with Cholesky (potrf/potrs), QR (geqrf/orgqr), eigenvalue decomposition (syevd), and Jacobi SVD (gesvdj) with configurable tolerance and max sweeps. Single and double precision.

### cuSPARSE (`src/cusparse/` — 4 files)

`CusparseContext` for CSR and COO sparse matrix creation. SpMV (sparse × dense vector), SpMM (sparse × dense matrix), SpGEMM (sparse × sparse) with work estimation / compute / copy phases. Algorithm selection for deterministic vs non-deterministic compute.

### cuFFT (`src/cufft/` — 4 files)

`CufftPlan` for 1D/2D/3D and batched FFT plans. Six execution modes: C2C, R2C, C2R for float and double (execC2C, execZ2Z, execR2C, execC2R, execD2Z, execZ2D).

### NVTX (`src/nvtx/` — 3 files)

`rangePush`/`rangePop` for named range markers, `mark` for point markers, `ScopedRange` for RAII-style ranges, `Domain` for per-module profiling isolation.

### Shared Types (`src/types.zig`)

`Dim3`, `LaunchConfig` (with `forNumElems` auto-configuration), `DevicePtr(T)`, and cuBLAS types (`Operation`, `FillMode`, `DiagType`, `SideMode`).

## Build Targets

```bash
zig build                  # Build library (driver + nvrtc)
zig build test             # All tests (unit + integration)
zig build test-unit        # Unit tests only
zig build test-integration # Integration tests only
zig build run-<cat>-<name> # Run an example (e.g. run-basics-vector_add)
```
