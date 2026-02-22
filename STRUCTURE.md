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
│   ├── runtime/               # CUDA Runtime API (internal)
│       ├── sys.zig, result.zig, safe.zig, runtime.zig
│       └── ...
│   └── kernel/                # GPU Kernel DSL — device-side intrinsics & types
│       ├── device.zig         # Module entry point (re-exports all sub-modules)
│       ├── intrinsics.zig     # 98 inline fns: threadIdx, atomics, warp, math, cache hints
│       ├── tensor_core.zig    # 56 inline fns: WMMA/MMA/wgmma/tcgen05/TMA/cluster
│       ├── shared_mem.zig     # SharedArray (addrspace(3)), dynamicShared, cooperative utils
│       ├── arch.zig           # SM version guards (requireSM, SmVersion enum)
│       ├── types.zig          # DeviceSlice(T), DevicePtr(T), GridStrideIterator
│       ├── shared_types.zig   # Host-device shared: Vec2/3/4, Int2/3, Matrix3x3/4x4, LaunchConfig
│       ├── bridge_gen.zig     # Type-safe kernel bridge generator (Fn enum, load, getFunction)
│       └── debug.zig          # assertf, ErrorFlag, printf, checkNaN, CycleTimer, __trap
├── test/                      # Tests
│   ├── helpers.zig            # Shared test helpers (initCuda, readPtxFile)
│   ├── unit/                  # Unit tests (12 files + 10 kernel unit tests)
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
│   │   ├── types_test.zig     # Shared type tests
│   │   └── kernel/            # Kernel DSL unit tests (host-side, no GPU required)
│   │       ├── kernel_arch_test.zig           # SM version guards
│   │       ├── kernel_debug_test.zig          # ErrorFlag, CycleTimer declarations
│   │       ├── kernel_device_test.zig         # Device kernel compilation & launch
│   │       ├── kernel_device_types_test.zig   # DeviceSlice, DevicePtr, GridStrideIterator
│   │       ├── kernel_grid_stride_test.zig    # GridStrideIterator logic
│   │       ├── kernel_intrinsics_host_test.zig # Intrinsic type/signature validation
│   │       ├── kernel_shared_mem_test.zig     # SharedArray comptime API
│   │       ├── kernel_shared_types_test.zig   # Vec2/3/4, Matrix, LaunchConfig
│   │       ├── kernel_tensor_core_host_test.zig # Fragment types, SM guards
│   │       └── kernel_types_test.zig          # Device type layout tests
│   └── integration/           # Integration tests (10 library + 7 kernel = 17 files)
│       ├── gemm_roundtrip_test.zig    # cuBLAS GEMM round-trip
│       ├── jit_kernel_test.zig        # NVRTC compile + launch
│       ├── lu_solve_test.zig          # cuSOLVER LU solve pipeline
│       ├── svd_reconstruct_test.zig   # SVD reconstruction
│       ├── fft_roundtrip_test.zig     # FFT forward + inverse
│       ├── curand_fft_test.zig        # cuRAND → cuFFT pipeline
│       ├── conv_pipeline_test.zig     # cuDNN conv pipeline
│       ├── conv_relu_test.zig         # cuDNN conv + activation
│       ├── sparse_pipeline_test.zig   # cuSPARSE pipeline
│       ├── syrk_geam_test.zig         # cuBLAS SYRK + GEAM
│       └── kernel/                    # Kernel DSL integration tests (GPU required)
│           ├── kernel_device_test.zig         # Basic kernel launch correctness
│           ├── kernel_event_timing_test.zig   # Event timing + multi-stream
│           ├── kernel_intrinsics_gpu_test.zig # Math/atomic intrinsics on real GPU
│           ├── kernel_memory_lifecycle_test.zig # Alloc/free/copy lifecycle
│           ├── kernel_pipeline_test.zig       # Tiled matmul, softmax, dot product
│           ├── kernel_reduction_test.zig      # Warp reduce, histogram, matmul
│           ├── kernel_shared_mem_gpu_test.zig # Shared mem reduce/transpose
│           └── kernel_softmax_test.zig        # Online softmax correctness
├── examples/                  # Runnable examples
│   ├── README.md              # Categorized example index (with links to per-category READMEs)
│   ├── basics/                # 16 examples — contexts, streams, events, memory, kernels
│   │   ├── README.md          # Category index with API key snippets
│   │   ├── vector_add.zig, streams.zig, device_info.zig, event_timing.zig
│   │   ├── struct_kernel.zig, kernel_attributes.zig, constant_memory.zig
│   │   ├── peer_to_peer.zig, alloc_patterns.zig, async_memcpy.zig
│   │   ├── pinned_memory.zig, unified_memory.zig, context_lifecycle.zig
│   │   └── dtod_copy_chain.zig, memset_patterns.zig, multi_device_query.zig
│   ├── kernel/                # 80 GPU kernel examples (11 categories, compiled to PTX)
│   │   ├── README.md          # Per-category kernel example index
│   │   ├── 0_Basic/           # 8 kernels — SAXPY, ReLU, dot product, grid stride
│   │   ├── 1_Reduction/       # 5 kernels — warp shuffle, prefix scan, multi-block
│   │   ├── 2_Matrix/          # 6 kernels — naive matmul, tiled matmul, transpose
│   │   ├── 3_Atomics/         # 5 kernels — atomic ops, histogram, warp-aggregated
│   │   ├── 4_SharedMemory/    # 3 kernels — static/dynamic smem, 1D stencil
│   │   ├── 5_Warp/            # 5 kernels — ballot, broadcast, match, scan
│   │   ├── 6_MathAndTypes/    # 9 kernels — FP16, complex, fast math, type conversion
│   │   ├── 7_Debug/           # 2 kernels — error checking, GPU printf
│   │   ├── 8_TensorCore/      # 11 kernels — WMMA (f16/bf16/int8/tf32), MMA PTX, FP8
│   │   ├── 9_Advanced/        # 8 kernels — async copy, cooperative groups, softmax
│   │   └── 10_Integration/    # 24 kernels — end-to-end pipelines and benchmarks
│   ├── cublas/                # 19 examples — BLAS L1/L2/L3, batched, mixed-precision
│   │   ├── README.md          # Category index with row-major note and API key snippets
│   │   ├── gemm.zig, axpy.zig, dot.zig, scal.zig, nrm2_asum.zig
│   │   ├── gemv.zig, symv_syr.zig, trmv_trsv.zig, trsm.zig
│   │   ├── gemm_batched.zig, gemm_ex.zig, geam.zig, dgmm.zig
│   │   ├── swap_copy.zig, rot.zig, amax_amin.zig, symm.zig, syrk.zig
│   │   └── cosine_similarity.zig
│   ├── cublaslt/              # 1 example — lightweight GEMM with heuristics
│   │   ├── README.md          # Category index
│   │   └── lt_sgemm.zig
│   ├── cudnn/                 # 3 examples — convolution, activation, pooling
│   │   ├── README.md          # Category index
│   │   ├── conv2d.zig, activation.zig, pooling_softmax.zig
│   │   └── ...\n│   ├── cufft/                 # 4 examples — 1D/2D/3D FFT
│   │   ├── README.md          # Category index with transform type table
│   │   ├── fft_1d_c2c.zig, fft_1d_r2c.zig, fft_2d.zig, fft_3d.zig
│   │   └── ...\n│   ├── curand/                # 3 examples — RNG, distributions, Monte Carlo
│   │   ├── README.md          # Category index with generator type table
│   │   ├── generators.zig, distributions.zig, monte_carlo_pi.zig
│   │   └── ...\n│   ├── cusolver/              # 5 examples — LU, SVD, Cholesky, QR, eigensolve
│   │   ├── README.md          # Category index with devInfo note
│   │   ├── getrf.zig, gesvd.zig, potrf.zig, geqrf.zig, syevd.zig
│   │   └── ...\n│   ├── cusparse/              # 4 examples — CSR/COO SpMV, SpMM, SpGEMM
│   │   ├── README.md          # Category index with sparse format table
│   │   ├── spmv_csr.zig, spmv_coo.zig, spmm_csr.zig, spgemm.zig
│   │   └── ...\n│   ├── nvrtc/                 # 2 examples — JIT compilation
│   │   ├── README.md          # Category index with CompileOptions table
│   │   ├── jit_compile.zig, template_kernel.zig
│   │   └── ...\n│   └── nvtx/                  # 1 example — Nsight profiling
│       ├── README.md          # Category index with Nsight usage
│       └── profiling.zig
├── docs/                      # Documentation
│   ├── README.md              # Documentation index
│   ├── API.md                 # Complete API reference (binding layer + Kernel DSL overview)
│   ├── kernel/
│   │   ├── API.md             # Kernel DSL full API reference (intrinsics, smem, tensor cores)
│   │   └── MIGRATION.md       # CUDA C++ → Zig migration guide
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

### Device / Kernel DSL (`src/kernel/` — 9 files)

Device-side module for writing GPU kernels in pure Zig, compiled to PTX via the NVPTX backend. Contains 175 inline functions across:
- **intrinsics.zig** (98 fns): `threadIdx`, `blockIdx`, `__syncthreads`, atomics (`atomicAdd`–`atomicDec`), warp shuffle/vote/match/reduce, fast math, bit ops, cache hints, type conversions, `__nanosleep`, `__byte_perm`
- **tensor_core.zig** (56 fns): WMMA (sm_70+), MMA PTX (sm_80+), FP8 MMA (sm_89+), wgmma/TMA/cluster (sm_90+), tcgen05 (sm_100+)
- **shared_mem.zig**: `SharedArray(T, N)` via `addrspace(.shared)`, `dynamicShared(T)`, `clearShared`, `loadToShared`, `storeFromShared`, `reduceSum`
- **arch.zig**: `SmVersion` enum (sm_52–sm_100+), `requireSM` comptime guard, `atLeast`, `codename`
- **types.zig**: `DeviceSlice(T)` (get/set/len), `DevicePtr(T)` (load/store/atomicAdd), `GridStrideIterator`, `globalThreadIdx`, `gridStride`
- **shared_types.zig**: `Vec2/3/4`, `Int2/3`, `Matrix3x3/4x4`, `LaunchConfig` (init1D/2D, forElementCount)
- **debug.zig**: `assertf`, `assertInBounds`, `safeGet`, `ErrorFlag` (5 error codes + `setError`/`checkNaN`), `printf`, `CycleTimer`, `__trap`, `__brkpt`
- **bridge_gen.zig**: `init(Config)` — comptime `Fn` enum, `load`, `loadFromPtx`, `getFunction`, `getFunctionByName`

→ Full API reference: [`docs/kernel/API.md`](docs/kernel/API.md)

## Build Targets

```bash
zig build                  # Build library (driver + nvrtc + cublas + curand)
zig build test             # All tests (unit + integration, 235 total)
zig build test-unit        # Unit tests only
zig build test-integration # Integration tests only
zig build run-<cat>-<name> # Run a host example (e.g. run-basics-vector_add)
zig build example-integration -Dgpu-arch=sm_86 -Dcublas=true -Dcufft=true  # Build all integration examples
zig build compile-kernels  # Compile all GPU kernels to PTX (default sm_80)
zig build compile-kernels -Dgpu-arch=sm_80  # Target Ampere
zig build compile-kernels -Dgpu-arch=sm_90  # Target Hopper
zig build example-kernel-<cat>-<name> -Dgpu-arch=sm_86  # Build one kernel example
```
