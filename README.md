# zCUDA

zCUDA: Comprehensive, safe, and idiomatic Zig bindings for the entire [CUDA](https://developer.nvidia.com/cuda-zone) ecosystem — from driver API to cuBLAS, cuDNN, cuFFT, cuSOLVER, cuSPARSE, cuRAND, and beyond.

## Overview

| Metric           | Value                         |
| ---------------- | ----------------------------- |
| **Version**      | 0.1.0                         |
| **Zig**          | 0.16.0-dev.2535+b5bd49460     |
| **CUDA Toolkit** | 12.8                          |
| **Modules**      | 10                            |
| **Tests**        | 22 (12 unit + 10 integration) |
| **Examples**     | 50                            |

## Features

- ✅ **Type-safe** — Idiomatic Zig API with compile-time type checking
- ✅ **Memory-safe** — RAII-style resource management with `defer`
- ✅ **Zero-cost** — Direct C API calls via `@cImport` with minimal overhead
- ✅ **Comprehensive** — 10 CUDA library bindings with full API coverage
- ✅ **Three-layer architecture** — sys (raw FFI) → result (error wrapping) → safe (user API)
- ✅ **Modular** — Enable only the libraries you need via build flags

## Quick Start

### Prerequisites

- **Zig** 0.16.0-dev.2535+b5bd49460
- **CUDA Toolkit 12.x** (with `nvcc`, `libcuda`, `libcudart`, `libnvrtc`)
- **cuDNN 9.x** (optional, for `cudnn` module)
- **NVIDIA GPU** with Compute Capability 8.0+ (RTX series)

### Build & Test

```bash
git clone https://github.com/coderonion/zcuda
cd zcuda

zig build                                    # Build library (driver + nvrtc)
zig build test                               # Run all tests
zig build test-unit                          # Unit tests only
zig build test-integration                   # Integration tests only

# Enable optional modules
zig build -Dcublas=true -Dcurand=true -Dcudnn=true

# All modules
zig build -Dcublas=true -Dcublaslt=true -Dcurand=true -Dcudnn=true \
          -Dcusolver=true -Dcusparse=true -Dcufft=true -Dnvtx=true
```

### Basic Usage

```zig
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Create a CUDA context on device 0
    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();

    const stream = ctx.defaultStream();

    // Allocate and transfer data
    const host_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const dev_data = try stream.cloneHtod(f32, &host_data);
    defer dev_data.deinit();

    // Compile and launch a kernel
    const ptx = try cuda.nvrtc.compilePtx(allocator,
        \\extern "C" __global__ void add1(float *data, int n) {
        \\    int i = blockIdx.x * blockDim.x + threadIdx.x;
        \\    if (i < n) data[i] += 1.0f;
        \\}
    );
    defer allocator.free(ptx);

    const module = try ctx.loadModule(ptx);
    defer module.deinit();
    const kernel = try module.getFunction("add1");

    try stream.launch(kernel, cuda.LaunchConfig.forNumElems(4),
        .{ &dev_data, @as(i32, 4) });
    try stream.synchronize();

    // Read back results
    var result: [4]f32 = undefined;
    try stream.memcpyDtoh(f32, &result, dev_data);
    // result = { 2.0, 3.0, 4.0, 5.0 }
}
```

## 📦 Use as Zig Package

Add zCUDA as a dependency in your project — **CUDA library linking is handled automatically**.

### Step 1: Add dependency to `build.zig.zon`

**Local path (for development):**

```zig
.dependencies = .{
    .zcuda = .{
        .path = "../zcuda",
    },
},
```

**Git URL (for release):**

```zig
.dependencies = .{
    .zcuda = .{
        .url = "https://github.com/coderonion/zcuda/archive/v0.1.0.tar.gz",
        .hash = "HASH_VALUE",
    },
},
```

> [!TIP]
> **How to get the hash:** First, add the `.url` field **without** `.hash`, then run `zig build`. Zig will download the package, compute the hash, and display the correct `.hash = "..."` value in the error output. Copy that value into your `build.zig.zon`.

### Step 2: Import in `build.zig`

**Option A — Simple (hardcoded flags):**

```zig
const zcuda = b.dependency("zcuda", .{
    .target    = target,
    .optimize  = optimize,
    .cublas    = true,   // cuBLAS     (default: true)
    .cublaslt  = true,   // cuBLAS LT  (default: true)
    .curand    = true,   // cuRAND     (default: true)
    .nvrtc     = true,   // NVRTC      (default: true)
    .cudnn     = false,  // cuDNN      (default: false)
    .cusolver  = false,  // cuSOLVER   (default: false)
    .cusparse  = false,  // cuSPARSE   (default: false)
    .cufft     = false,  // cuFFT      (default: false)
    .cupti     = false,  // CUPTI      (default: false)
    .cufile    = false,  // cuFile     (default: false)
    .nvtx      = false,  // NVTX       (default: false)
    // .@"cuda-path" = "/usr/local/cuda",  // optional: override auto-detect
});
exe.root_module.addImport("zcuda", zcuda.module("zcuda"));
```

**Option B — Dynamic (forward to CLI):**

Expose zcuda flags as your project's build options, so users can toggle modules at build time:

```zig
const enable_cublas   = b.option(bool, "cublas",   "Enable cuBLAS")   orelse true;
const enable_cublaslt = b.option(bool, "cublaslt", "Enable cuBLAS LT") orelse true;
const enable_curand   = b.option(bool, "curand",   "Enable cuRAND")   orelse true;
const enable_nvrtc    = b.option(bool, "nvrtc",    "Enable NVRTC")    orelse true;
const enable_cudnn    = b.option(bool, "cudnn",    "Enable cuDNN")    orelse false;
const enable_cusolver = b.option(bool, "cusolver", "Enable cuSOLVER") orelse false;
const enable_cusparse = b.option(bool, "cusparse", "Enable cuSPARSE") orelse false;
const enable_cufft    = b.option(bool, "cufft",    "Enable cuFFT")    orelse false;
const enable_nvtx     = b.option(bool, "nvtx",     "Enable NVTX")    orelse false;
const cuda_path       = b.option([]const u8, "cuda-path", "Path to CUDA installation (default: auto-detect)");

const zcuda = b.dependency("zcuda", .{
    .target    = target,
    .optimize  = optimize,
    .cublas    = enable_cublas,
    .cublaslt  = enable_cublaslt,
    .curand    = enable_curand,
    .nvrtc     = enable_nvrtc,
    .cudnn     = enable_cudnn,
    .cusolver  = enable_cusolver,
    .cusparse  = enable_cusparse,
    .cufft     = enable_cufft,
    .nvtx      = enable_nvtx,
    .@"cuda-path" = cuda_path,
});
exe.root_module.addImport("zcuda", zcuda.module("zcuda"));
```

```bash
zig build                                  # defaults (cublas, curand, nvrtc enabled)
zig build -Dcudnn=true -Dcusolver=true     # add cuDNN + cuSOLVER
zig build -Dcublas=false                   # disable cuBLAS
```

### Step 3: Use in your code

```zig
const cuda = @import("zcuda");

pub fn main() !void {
    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    // ...
}
```

## Modules

| Module         | Description                                               | Flag               |
| -------------- | --------------------------------------------------------- | ------------------ |
| **Driver API** | Device management, memory, kernel launch, streams, events | *(always enabled)* |
| **NVRTC**      | Runtime compilation of CUDA C++ to PTX/CUBIN              | *(always enabled)* |
| **cuBLAS**     | BLAS Level 1/2/3 (SAXPY, SGEMM, DGEMM, etc.)              | `-Dcublas=true`    |
| **cuBLAS LT**  | Lightweight GEMM with algorithm heuristics                | `-Dcublaslt=true`  |
| **cuRAND**     | GPU random number generation                              | `-Dcurand=true`    |
| **cuDNN**      | Convolution, activation, pooling, softmax, batch norm     | `-Dcudnn=true`     |
| **cuSOLVER**   | LU, QR, SVD, Cholesky, eigenvalue decomposition           | `-Dcusolver=true`  |
| **cuSPARSE**   | SpMV, SpMM, SpGEMM with CSR/COO formats                   | `-Dcusparse=true`  |
| **cuFFT**      | 1D/2D/3D Fast Fourier Transform                           | `-Dcufft=true`     |
| **NVTX**       | Profiling annotations for Nsight                          | `-Dnvtx=true`      |

### Build Options

| Option             | Default | Description                         |
| ------------------ | ------- | ----------------------------------- |
| `-Dcublas=true`    | `true`  | Enable cuBLAS (BLAS operations)     |
| `-Dcublaslt=true`  | `true`  | Enable cuBLAS LT (lightweight GEMM) |
| `-Dcurand=true`    | `true`  | Enable cuRAND (random numbers)      |
| `-Dcudnn=true`     | `false` | Enable cuDNN (deep learning)        |
| `-Dcusolver=true`  | `false` | Enable cuSOLVER (direct solvers)    |
| `-Dcusparse=true`  | `false` | Enable cuSPARSE (sparse matrices)   |
| `-Dcufft=true`     | `false` | Enable cuFFT (FFT)                  |
| `-Dnvtx=true`      | `false` | Enable NVTX (annotations)           |
| `-Dcuda-path=...`  | auto    | CUDA toolkit path                   |
| `-Dcudnn-path=...` | auto    | cuDNN path                          |

## Examples

50 working examples in the [`examples/`](examples/) directory. See [examples/README.md](examples/README.md) for the full categorized index.

```bash
# Build and run
zig build run-basics-vector_add
zig build run-cublas-gemm -Dcublas=true
zig build run-cusolver-gesvd -Dcusolver=true
zig build run-cudnn-conv2d -Dcudnn=true
zig build run-cufft-fft_2d -Dcufft=true
```

### Example Categories

| Category      | Count | Examples                                  | What You'll Learn                             |
| ------------- | ----- | ----------------------------------------- | --------------------------------------------- |
| **Basics**    | 8     | vector_add, streams, device_info, …       | Contexts, streams, events, kernels, multi-GPU |
| **cuBLAS**    | 19    | gemm, axpy, trsm, cosine_similarity, …    | L1/L2/L3 BLAS, batched GEMM, mixed-precision  |
| **cuDNN**     | 3     | conv2d, activation, pooling_softmax       | Neural network primitives                     |
| **cuFFT**     | 4     | fft_1d_c2c, fft_2d, fft_3d, fft_1d_r2c    | 1D/2D/3D FFT, filtering                       |
| **cuRAND**    | 3     | distributions, generators, monte_carlo_pi | RNG types, Monte Carlo                        |
| **cuSOLVER**  | 5     | getrf, gesvd, potrf, syevd, geqrf         | LU, SVD, Cholesky, QR, eigensolve             |
| **cuSPARSE**  | 4     | spmv_csr, spmm_csr, spmv_coo, spgemm      | CSR/COO SpMV, SpMM, SpGEMM                    |
| **cuBLAS LT** | 1     | lt_sgemm                                  | GEMM with algorithm heuristics                |
| **NVRTC**     | 2     | jit_compile, template_kernel              | Runtime compilation                           |
| **NVTX**      | 1     | profiling                                 | Nsight annotations                            |

## Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory:

- **[Documentation Index](docs/README.md)** — Full navigation guide
- **[API Reference](docs/api.md)** — Complete safe-layer API for all modules
- **[Examples](examples/README.md)** — 50 runnable examples with build commands
- **[Project Structure](STRUCTURE.md)** — Source code organization and module overview

Each module has its own detailed README in `docs/<module>/README.md`.

## Testing

```bash
zig build test                               # All tests (unit + integration)
zig build test-unit                          # Unit tests only
zig build test-integration                   # Integration tests only
```

Test coverage includes:
- **Unit tests** (12) — Each module's core functionality and error handling
- **Integration tests** (10) — Cross-module workflows (GEMM round-trip, JIT kernel, FFT, conv pipeline, etc.)

## Architecture

Each module follows a consistent three-layer design:

```
┌──────────────────────────────────────────────┐
│  Safe Layer (safe.zig)                       │  ← Recommended API
│  Type-safe abstractions, RAII, Zig idioms    │
├──────────────────────────────────────────────┤
│  Result Layer (result.zig)                   │  ← Error wrapping
│  C error codes → Zig error unions            │
├──────────────────────────────────────────────┤
│  Sys Layer (sys.zig)                         │  ← Raw FFI
│  Direct @cImport of C headers               │
└──────────────────────────────────────────────┘
```

```
zcuda/
├── src/                       # Zig API layer (10 modules)
│   ├── cuda.zig               # Root module — re-exports all modules
│   ├── types.zig              # Shared types (Dim3, LaunchConfig, DevicePtr)
│   ├── driver/                # CUDA Driver API (sys, result, safe)
│   ├── nvrtc/                 # NVRTC (runtime compilation)
│   └── ...                    # 8 more module directories
├── examples/                  # 50 working examples
├── test/                      # 22 tests
│   ├── unit/                  # Per-module unit tests (12)
│   └── integration/           # Cross-module integration tests (10)
├── docs/                      # Comprehensive API documentation
├── build.zig                  # Build configuration
└── build.zig.zon              # Package manifest
```

> **Users should only use the Safe Layer.** The `result` and `sys` layers are implementation details — all public types and functions are re-exported from each module's top-level file.

## Contributing

1. ⭐ Star and Fork this repository
2. Create a feature branch (`git checkout -b feature/new-module`)
3. Implement sys/result/safe layers in `src/<module>/`
4. Add unit tests in `test/unit/` and integration tests in `test/integration/`
5. Create an example in `examples/<module>/`
6. Update documentation in `docs/<module>/`
7. Submit a Pull Request

## License

MIT License

## Acknowledgments

Built with gratitude on the shoulders of giants:

- **[CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)** — NVIDIA's parallel computing platform and API, providing the underlying runtime, compiler, and libraries.
- **[Zig](https://ziglang.org/)** — A modern systems programming language focused on safety, performance, and simplicity, created by Andrew Kelley and the Zig Software Foundation.
- **[cudarc](https://github.com/coreylowman/cudarc)** — A safe Rust wrapper for CUDA whose three-layer architecture (sys → result → safe) served as the foundational reference for this project.
