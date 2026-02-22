# zCUDA

**Write CUDA kernels in pure Zig. Or keep your C++ kernels. Ship them all to GPU.**

zCUDA is a dual-ecosystem GPU programming framework for [Zig](https://ziglang.org/):

- 🔗 **CUDA Binding Layer** — Complete, type-safe Zig bindings for CUDA C++ libraries (driver API, cuBLAS, cuDNN, cuFFT, cuSOLVER, cuSPARSE, cuRAND, NVRTC, NVTX, CUPTI, cuFile, and beyond). Seamlessly call existing, battle-tested CUDA C++ kernels (`.cu` files) from Zig — JIT compile via NVRTC or load pre-compiled PTX. Your mature C++ kernel ecosystem stays intact.
- ⚡ **CUDA Kernel DSL** — A zero-overhead device-side API (`zcuda_kernel`) that compiles pure Zig directly to PTX at build time. No CUDA C++. No `nvcc`. Just Zig, all the way down to the CUDA registers — with full compile-time type safety, auto-discovery, and type-safe bridge generation.

**The best of both worlds**: Reuse the industry's vast collection of mature CUDA C++ kernels via NVRTC JIT or pre-compiled PTX — zero rewrite, zero migration friction. Write new custom kernels in pure Zig with auto-discovery, type-safe bridges, and `zig build` integration. Both paths produce PTX, both launch through the same `stream.launch()` API, and both ship in your final binary. Start with C++, migrate to Zig at your own pace, or keep both forever.

**Why pure Zig?** Zig brings a rare combination to GPU programming: C-level real-time performance (zero-cost abstractions, no GC, no hidden allocations, deterministic memory layout), Python-like readability (clean syntax, no header files, no forward declarations, no preprocessor macros. One unified toolchain: `zig build` handles build, test, package, and deploy — with built-in package manager (`build.zig.zon`, reproducible hashes, semantic versioning), built-in unit test runner, and first-class cross-compilation to any CUDA-capable platform (x86_64, aarch64, riscv64, and beyond) from any host. Compile-time type safety catches errors at build time and `defer`-based resource management.


## Overview

| Metric           | Value                         |
| ---------------- | ----------------------------- |
| **Zig**          | 0.16.0-dev.2535+b5bd49460     |
| **CUDA Toolkit** | 12.8                          |
| **Modules**      | 11 (driver, nvrtc, cublas, cublaslt, curand, cudnn, cusolver, cusparse, cufft, nvtx, cupti) |
| **Tests**        | 39 files (22 unit + 17 integration) |
| **Host Examples**| 58 across 10 categories (50 `run-xxx` targets) + 24 integration |
| **Kernel Examples** | 80 pure-Zig GPU kernels across 11 categories |
| **GPU Validated** | ✅ sm_86 (RTX 3080 Ti, CUDA 12.8) — all unit + integration tests PASS, **50/50 run-xxx examples PASS**, **zcuda-demo** All correctness PASSED (`zig build test` + `zig build run` EXIT=0) |

## Features

**CUDA Binding Layer:**
- ✅ **Type-safe** — Idiomatic Zig API with compile-time type checking, Zig error unions instead of C error codes
- ✅ **Memory-safe** — RAII-style resource management with `defer`, no leaked GPU memory, streams, or contexts
- ✅ **Zero-cost** — Direct C API calls via `@cImport`, zero-overhead wrappers, no runtime reflection
- ✅ **Comprehensive** — 11 CUDA library bindings (driver, nvrtc, cublas, cublaslt, curand, cudnn, cusolver, cusparse, cufft, nvtx, cupti) with full API coverage
- ✅ **Three-layer architecture** — `sys` (raw FFI) → `result` (error wrapping) → `safe` (ergonomic user API)
- ✅ **Modular** — Enable only the libraries you need via `-D` build flags, unused modules are zero-cost
- ✅ **C++ Compatible** — Call existing CUDA C++ kernels via NVRTC JIT or pre-compiled PTX, zero migration friction

**CUDA Kernel DSL:**
- ✅ **Pure Zig Kernels** — Write CUDA kernels in pure Zig with `zcuda_kernel`, compile to PTX at `zig build` time
- ✅ **Auto-discovery** — Kernels detected by content (`export fn`), no manual registration or config files
- ✅ **Full device intrinsics** — shared memory, atomics (`atomicAdd`, `atomicCAS`), warp shuffles (`__shfl_sync`, `__shfl_xor_sync`), `__syncthreads`, `printf`, thread/block/grid indexing
- ✅ **Tensor Core support** — WMMA and MMA intrinsics for f16, bf16, tf32, int8, fp8 matrix operations
- ✅ **Type-safe bridge generation** — Auto-generated `Fn` enum per kernel, function name typos caught at compile time

**Ecosystem & Tooling:**
- ✅ **5 kernel loading methods** — filesystem PTX, NVRTC JIT (inline), `@embedFile` JIT, `@embedFile` PTX, build.zig auto-generated bridge module
- ✅ **Hybrid Ready** — Mix CUDA C++ and Zig kernels in the same project, both produce PTX, both use the same `stream.launch()` API
- ✅ **Cross-compilation** — Target x86_64, aarch64, riscv64, and beyond from any host
- ✅ **Downstream export** — Use as a Zig package via `build.zig.zon`, exposes `pub const build_helpers` for downstream kernel compilation
- ✅ **Built-in testing** — comptime-verifiable unit tests, `zig build test` with no external test framework

## Quick Start

### Prerequisites

- **Zig** 0.16.0-dev.2535+b5bd49460
- **CUDA Toolkit 12.x** (with `nvcc`, `libcuda`, `libcudart`, `libnvrtc`)
- **cuDNN 9.x** (optional, for `cudnn` module)
- **NVIDIA GPU** with Compute Capability 7.0+ (Volta and later)

### Build & Test

```bash
git clone https://github.com/coderonion/zcuda
cd zcuda

zig build                                    # Build library (driver + nvrtc + cublas + curand)
zig build test                               # Run all tests (unit + integration)
zig build test-unit                          # Unit tests only
zig build test-integration                   # Integration tests only

# Enable optional modules
zig build -Dcudnn=true -Dcusolver=true

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
    const dev_data = try stream.cloneHtoD(f32, &host_data);
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
    try stream.memcpyDtoH(f32, &result, dev_data);
    // result = { 2.0, 3.0, 4.0, 5.0 }
}
```

## 📦 Use as Zig Package

Add zCUDA as a dependency — CUDA library linking, kernel compilation, and bridge generation are all handled automatically.

> [!TIP]
> **[`zcuda-demo`](https://github.com/coderonion/zcuda-demo)** is a fully worked reference project that imports zcuda as a local package.
> It covers NVRTC JIT, pure Zig GPU kernels (bridge module), cuBLAS SGEMM, cross-validation, and performance benchmarking.
> Clone it alongside `zcuda` to see a complete, runnable example of every step below:
> ```bash
> git clone https://github.com/coderonion/zcuda
> git clone https://github.com/coderonion/zcuda-demo
> cd zcuda-demo && zig build run -Dgpu-arch=sm_86
> # → ✓ All correctness checks PASSED (Sections 1–4)
> ```

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
> **How to get the hash:** Add `.url` without `.hash`, then run `zig build`. Zig will display the correct hash in the error output — copy it into your `build.zig.zon`.

### Step 2: Configure `build.zig`

A typical `build.zig` has three parts: **1. import zcuda**, **2. discover & compile kernels**, **3. wire everything to your executable**.

#### 1. Import zcuda dependency

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
```

#### 2. Set up GPU architecture target

`build_helpers` provides ready-made helpers so you don't need to copy GPU arch
boilerplate into every downstream project:

```zig
const bridge    = @import("zcuda").build_helpers;
const gpu_arch  = b.option([]const u8, "gpu-arch", "Target GPU SM arch (default: sm_80)") orelse "sm_80";
const embed_ptx = b.option(bool, "embed-ptx", "Embed PTX in binary") orelse false;

// Resolve nvptx64 target from "sm_XX" string (one call, no boilerplate)
const nvptx_target = bridge.resolveNvptxTarget(b, gpu_arch);

// Create the GPU-side device intrinsics module (compiled for nvptx64, not the host)
// sm_version build_options and internal paths are handled automatically
const device_mod = bridge.makeDeviceModule(b, zcuda_dep, nvptx_target, gpu_arch);
```

Supported SM versions: `sm_52`, `sm_60`, `sm_70`, `sm_75`, `sm_80`, `sm_86`, `sm_89`, `sm_90`, `sm_100`.

#### 3. Discover & compile GPU kernels

```zig
const kernel_step = b.step("compile-kernels", "Compile Zig GPU kernels to PTX");
const kernel_dir  = b.option([]const u8, "kernel-dir",
    "Root dir for kernel discovery (default: src/kernel/)") orelse "src/kernel/";

// Recursively scan kernel directory for .zig files containing `export fn`
const kernels = bridge.discoverKernels(b, kernel_dir);

// Compile Zig → PTX + generate type-safe bridge modules
const result = bridge.addBridgeModules(b, kernels, .{
    .embed_ptx        = embed_ptx,
    .zcuda_bridge_mod = zcuda_dep.module("zcuda_bridge"),
    .zcuda_mod        = zcuda_dep.module("zcuda"),
    .device_mod       = device_mod,
    .nvptx_target     = nvptx_target,
    .kernel_step      = kernel_step,
    .target           = target,
    .optimize         = optimize,
});
```

#### 4. Create executable + wire imports

```zig
const exe = b.addExecutable(.{
    .name = "my_app",
    .root_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target   = target,
        .optimize = optimize,
    }),
});

exe.root_module.addImport("zcuda", zcuda_dep.module("zcuda"));

// Link libc + CUDA libraries.
// Zig 0.16.0-dev does not propagate mod.linkSystemLibrary to downstream exe,
// so these must match the flags you passed to b.dependency("zcuda", ...).
exe.root_module.link_libc = true;
exe.root_module.linkSystemLibrary("cuda",   .{});
exe.root_module.linkSystemLibrary("cudart", .{});
// Add libraries matching your enabled dependency flags, e.g.:
// if (enable_nvrtc)    exe.root_module.linkSystemLibrary("nvrtc",    .{});
// if (enable_cublas)   exe.root_module.linkSystemLibrary("cublas",   .{});
// if (enable_cublaslt) exe.root_module.linkSystemLibrary("cublasLt", .{});
// if (enable_curand)   exe.root_module.linkSystemLibrary("curand",   .{});

// Mount kernel bridge modules (choose one):
//
// Option A — Mount all (single-exe projects):
for (result.modules) |entry| {
    exe.root_module.addImport(entry.name, entry.module);
    // For disk-mode (non-embedded) PTX, ensure PTX is installed before exe:
    if (entry.install_step) |s| b.getInstallStep().dependOn(s);
}
//
// Option B — Selective (multi-exe projects):
// if (bridge.findBridge(result.modules, "my_kernel")) |mod| {
//     exe.root_module.addImport("my_kernel", mod);
// }

b.installArtifact(exe);

const run_cmd = b.addRunArtifact(exe);
run_cmd.step.dependOn(b.getInstallStep());
b.step("run", "Build and run").dependOn(&run_cmd.step);
```

> [!NOTE]
> **`linkSystemLibrary` is not auto-propagated** in Zig 0.16.0-dev.
> You must call `linkSystemLibrary` on your `exe` for each library you enabled in
> `b.dependency("zcuda", ...)`. Using **Option B** (flag-forwarding) lets you keep
> these in sync automatically:
> ```zig
> if (enable_cublas) exe.root_module.linkSystemLibrary("cublas", .{});
> ```

> [!TIP]
> **Mount all** adds every discovered kernel bridge — ideal for single-app projects.
> **Selective** uses `findBridge()` to pick specific kernels — useful when multiple executables each need different kernels.

### Step 3: Write your code

#### GPU kernel (`zcuda_kernel`)

Create `.zig` files anywhere under your kernel directory. Import `zcuda_kernel` for the full device-side API:

```zig
// kernels/my_kernel.zig
const cuda = @import("zcuda_kernel");

export fn myAdd(
    A: [*]const f32, B: [*]const f32, C: [*]f32, n: u32,
) callconv(.kernel) void {
    const i = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (i < n) C[i] = A[i] + B[i];
}
```

> [!NOTE]
> Detection is **content-based**: any `.zig` file containing `export fn` is auto-recognized as a kernel. No naming conventions or manual registration required.

**`zcuda_kernel` API quick reference** — naming matches CUDA C++ for seamless migration:

| Category | Zig (`zcuda_kernel`) | CUDA C++ Equivalent |
|---|---|---|
| **Thread Indexing** | `cuda.threadIdx()`, `cuda.blockIdx()`, `cuda.blockDim()`, `cuda.gridDim()` | `threadIdx.x`, `blockIdx.x`, `blockDim.x`, `gridDim.x` |
| **Synchronization** | `cuda.__syncthreads()`, `cuda.__threadfence()`, `cuda.__syncwarp(mask)` | `__syncthreads()`, `__threadfence()`, `__syncwarp()` |
| **Atomics** | `cuda.atomicAdd(ptr, val)`, `atomicCAS`, `atomicExch`, `atomicMin/Max`, `atomicAnd/Or/Xor`, `atomicInc/Dec` | `atomicAdd()`, `atomicCAS()`, etc. |
| **Warp Shuffle** | `cuda.__shfl_sync(mask, val, src, w)`, `__shfl_down_sync`, `__shfl_up_sync`, `__shfl_xor_sync` | `__shfl_sync()`, `__shfl_down_sync()`, etc. |
| **Warp Vote** | `cuda.__ballot_sync(mask, pred)`, `__all_sync`, `__any_sync`, `__activemask()` | `__ballot_sync()`, `__all_sync()`, etc. |
| **Warp Reduce** *(sm_80+)* | `cuda.__reduce_add_sync(mask, val)`, `__reduce_min/max/and/or/xor_sync` | `__reduce_add_sync()`, etc. |
| **Fast Math** | `cuda.__sinf(x)`, `__cosf`, `__expf`, `__logf`, `__log2f`, `rsqrtf`, `sqrtf`, `__fmaf_rn`, `__powf` | `__sinf()`, `__cosf()`, etc. |
| **Integer** | `cuda.__clz(x)`, `__popc`, `__brev`, `__ffs`, `__byte_perm`, `__dp4a` | `__clz()`, `__popc()`, etc. |
| **Cache Hints** | `cuda.__ldg(ptr)`, `__ldca`, `__ldcs`, `__ldcg`, `__stcg`, `__stcs`, `__stwb` | `__ldg()`, etc. |
| **Shared Memory** | `cuda.shared_mem.SharedArray(f32, 256)`, `.dynamicShared(f32)`, `.reduceSum(...)` | `__shared__ float tile[256]`, `extern __shared__` |
| **Tensor Cores** *(sm_70+)* | `cuda.tensor_core.wmma_mma_f16_f32(a, b, c)`, `mma_f16_f32`, `mma_bf16_f32`, `mma_tf32_f32` | `wmma::mma_sync()`, `mma` PTX |
| **Debug** | `cuda.debug.assertf(cond)`, `.assertInBounds(i, n)`, `.CycleTimer`, `ErrorFlag` | `assert()`, `__trap()` |
| **Shared Types** | `cuda.shared.Vec2/Vec3/Vec4`, `.Matrix3x3`, `.Matrix4x4` | `float2/float3/float4` |
| **Clock** | `cuda.clock()`, `cuda.clock64()`, `cuda.globaltimer()` | `clock()`, `clock64()` |
| **Misc** | `cuda.warpSize` (32), `cuda.FULL_MASK`, `cuda.SM`, `cuda.__nanosleep(ns)` | `warpSize`, `__nanosleep()` |

**Kernel examples:**

```zig
const cuda = @import("zcuda_kernel");

// ── Vector addition ──
export fn vectorAdd(A: [*]const f32, B: [*]const f32, C: [*]f32, n: u32) callconv(.kernel) void {
    const i = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (i < n) C[i] = A[i] + B[i];
}
```

#### Host application

Use zcuda bindings + kernel bridge together:

```zig
// src/main.zig
const std = @import("std");
const cuda = @import("zcuda");          // zcuda binding API
const my_kernel = @import("my_kernel"); // type-safe kernel bridge

pub fn main() !void {
    // ── Driver API (from zcuda bindings) ──
    var ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    var stream = try ctx.newStream();
    defer stream.deinit();

    // ── Allocate GPU memory ──
    const n: u32 = 1024;
    var d_a = try stream.alloc(f32, n);
    defer d_a.deinit();
    var d_b = try stream.alloc(f32, n);
    defer d_b.deinit();
    var d_c = try stream.alloc(f32, n);
    defer d_c.deinit();

    // ── Load kernel (auto-detects embedded PTX vs disk file) ──
    const module = try my_kernel.load(ctx);
    defer module.deinit();

    // Function names are compile-time enums — typos cause build errors!
    const func = try my_kernel.getFunction(module, .myAdd);

    // ── Launch kernel ──
    try stream.launch(func,
        cuda.LaunchConfig.forNumElems(n),
        .{ d_a.ptr, d_b.ptr, d_c.ptr, n },
    );
    try stream.synchronize();

    // ── Read back results ──
    var result: [1024]f32 = undefined;
    try stream.memcpyDtoH(f32, &result, d_c);
}
```

#### Using CUDA C++ kernels (`.cu` files)

zCUDA fully supports existing CUDA C++ kernels — you don't need to rewrite anything in Zig. There are three ways to integrate C++ kernels:

**Way 1 — NVRTC JIT compilation (inline C++ source):**

Embed CUDA C++ source as a string in Zig and compile at runtime via NVRTC. Best for small kernels or rapid prototyping:

```zig
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    var ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    const stream = ctx.defaultStream();

    // ── Existing CUDA C++ kernel, used as-is ──
    const cuda_cpp_source =
        \\extern "C" __global__ void saxpy(float a, float *x, float *y, int n) {
        \\    int i = blockIdx.x * blockDim.x + threadIdx.x;
        \\    if (i < n) y[i] = a * x[i] + y[i];
        \\}
    ;

    // JIT compile C++ → PTX at runtime
    const ptx = try cuda.nvrtc.compilePtx(allocator, cuda_cpp_source);
    defer allocator.free(ptx);

    const module = try ctx.loadModule(ptx);
    defer module.deinit();
    const kernel = try module.getFunction("saxpy");

    // Allocate + launch, same API as Zig kernels
    const n: u32 = 1024;
    var d_x = try stream.alloc(f32, n);
    defer d_x.deinit();
    var d_y = try stream.alloc(f32, n);
    defer d_y.deinit();

    try stream.launch(kernel, cuda.LaunchConfig.forNumElems(n),
        .{ @as(f32, 2.0), d_x.ptr, d_y.ptr, @as(i32, @intCast(n)) });
    try stream.synchronize();
}
```

**Way 2 — NVRTC JIT compilation (read `.cu` file from disk):**

Load an existing `.cu` file and JIT compile it. Perfect for reusing large, mature C++ kernel libraries:

```zig
// Read existing .cu file — no modification needed
const cu_source = @embedFile("kernels/matmul_optimized.cu");

// Or load at runtime from any path:
// const cu_source = try std.fs.cwd().readFileAlloc(allocator, "vendor/kernels/matmul.cu", 1024 * 1024);
// defer allocator.free(cu_source);

const ptx = try cuda.nvrtc.compilePtx(allocator, cu_source);
defer allocator.free(ptx);

const module = try ctx.loadModule(ptx);
defer module.deinit();
const kernel = try module.getFunction("matmul_optimized");

// Launch with the same zcuda API
try stream.launch(kernel, .{ .grid = .{ .x = grid_x, .y = grid_y }, .block = .{ .x = 16, .y = 16 } },
    .{ d_A.ptr, d_B.ptr, d_C.ptr, @as(i32, @intCast(N)) });
```

**Way 3 — Pre-compiled PTX (offline `nvcc` compilation):**

Use `nvcc` to compile `.cu` → `.ptx` offline, then load the PTX in Zig. Best for production or when you need `nvcc`-specific flags:

```bash
# Compile with nvcc (your existing build pipeline)
nvcc -ptx -arch=sm_80 -o matmul.ptx matmul.cu
```

```zig
// Load pre-compiled PTX at build time (embed in binary)
const ptx = @embedFile("matmul.ptx");
const module = try ctx.loadModule(ptx);
defer module.deinit();
const kernel = try module.getFunction("matmul");

// Same launch API — it's all PTX under the hood
try stream.launch(kernel, config, .{ d_A.ptr, d_B.ptr, d_C.ptr, @as(i32, @intCast(N)) });
```

> [!TIP]
> **Migration strategy**: Start by wrapping your existing C++ kernels with Way 1 or 2 — zero rewrite needed. Then gradually port performance-critical kernels to pure Zig (Step 3 above) and enjoy compile-time type safety and `zig build` integration. Both C++ and Zig kernels produce PTX, both use the same `stream.launch()` API, and both can coexist in the same project.

### Step 4: Build & Run

#### Common commands

```bash
zig build                    # Build with defaults (driver + nvrtc + cublas + curand)
zig build run                # Build & run your application
zig build test               # Run all tests
```

#### CUDA module flags

Enable/disable CUDA library bindings at build time. These flags control which libraries get linked:

| Flag | Default | Module | Description |
|------|---------|--------|-------------|
| `-Dcublas=BOOL` | `true` | **cuBLAS** | BLAS Level 1/2/3 (SAXPY, SGEMM, DGEMM, etc.) |
| `-Dcublaslt=BOOL` | `true` | **cuBLAS LT** | Lightweight GEMM with algorithm heuristics |
| `-Dcurand=BOOL` | `true` | **cuRAND** | GPU random number generation |
| `-Dnvrtc=BOOL` | `true` | **NVRTC** | Runtime kernel compilation |
| `-Dcudnn=BOOL` | `false` | **cuDNN** | Convolution, activation, pooling, softmax, batch norm |
| `-Dcusolver=BOOL` | `false` | **cuSOLVER** | LU, QR, SVD, Cholesky, eigenvalue decomposition |
| `-Dcusparse=BOOL` | `false` | **cuSPARSE** | SpMV, SpMM, SpGEMM with CSR/COO formats |
| `-Dcufft=BOOL` | `false` | **cuFFT** | 1D/2D/3D Fast Fourier Transform |
| `-Dcupti=BOOL` | `false` | **CUPTI** | Profiling and tracing via CUDA Profiling Tools Interface |
| `-Dcufile=BOOL` | `false` | **cuFile** | GPUDirect Storage for direct GPU↔storage I/O |
| `-Dnvtx=BOOL` | `false` | **NVTX** | Profiling annotations for Nsight |

> **Driver API** is always enabled (no flag needed).

```bash
zig build                                  # defaults (cublas + cublaslt + curand + nvrtc)
zig build -Dcudnn=true -Dcusolver=true     # add cuDNN + cuSOLVER
zig build -Dcublas=false                   # disable cuBLAS
zig build -Dcublas=true -Dcudnn=true -Dcufft=true  # multi-module combo
```

#### GPU kernel flags

Control kernel compilation and PTX handling:

| Flag | Default | Description |
|------|---------|-------------|
| `-Dgpu-arch=ARCH` | `sm_80` | Target GPU architecture (e.g. `sm_75`, `sm_89`, `sm_90`) |
| `-Dembed-ptx=BOOL` | `false` | Embed PTX in binary — no `.ptx` files needed at runtime |
| `-Dkernel-dir=PATH` | `src/kernel/` | Root directory for kernel auto-discovery |

```bash
zig build compile-kernels                     # Compile all Zig kernels to PTX
zig build kernel-my_kernel                    # Compile single kernel only
zig build compile-kernels -Dgpu-arch=sm_80    # Target Ampere GPUs
zig build -Dembed-ptx=true                    # Production: PTX baked into binary
zig build compile-kernels -Dkernel-dir=my/kernels/  # Custom Zig kernels location
```

PTX output: `zig-out/bin/kernel/*.ptx`

#### Path overrides

| Flag | Default | Description |
|------|---------|-------------|
| `-Dcuda-path=PATH` | auto-detect | Path to CUDA toolkit installation |
| `-Dkernel-dir=PATH` | `src/kernel/` | Root directory for kernel auto-discovery |

> **💡 Combining flags**: Module flags and kernel flags are orthogonal — use them together freely:
> ```bash
> zig build compile-kernels -Dgpu-arch=sm_80 -Dcublas=true -Dcufft=true -Dembed-ptx=true
> ```

## Examples

### Host Examples (58 files, 74 build targets)

Working host-side examples in the [`examples/`](examples/) directory. See [examples/README.md](examples/README.md) for the full categorized index.

```bash
# Build and run
zig build run-basics-vector_add
zig build run-cublas-gemm
zig build run-cusolver-gesvd -Dcusolver=true
zig build run-cudnn-conv2d -Dcudnn=true
zig build run-cufft-fft_2d -Dcufft=true
```

| Category      | Count | Examples                                  | What You'll Learn                             |
| ------------- | ----- | ----------------------------------------- | --------------------------------------------- |
| **Basics**    | 16    | vector_add, streams, device_info, alloc_patterns, async_memcpy, pinned_memory, unified_memory, … | Contexts, streams, events, kernels, multi-GPU, memory patterns |
| **cuBLAS**    | 19    | gemm, axpy, trsm, cosine_similarity, gemm_batched, gemm_ex, dgmm, … | L1/L2/L3 BLAS, batched GEMM, mixed-precision  |
| **cuDNN**     | 3     | conv2d, activation, pooling_softmax       | Neural network primitives                     |
| **cuFFT**     | 4     | fft_1d_c2c, fft_2d, fft_3d, fft_1d_r2c    | 1D/2D/3D FFT, filtering                       |
| **cuRAND**    | 3     | distributions, generators, monte_carlo_pi | RNG types, Monte Carlo                        |
| **cuSOLVER**  | 5     | getrf, gesvd, potrf, syevd, geqrf         | LU, SVD, Cholesky, QR, eigensolve             |
| **cuSPARSE**  | 4     | spmv_csr, spmm_csr, spmv_coo, spgemm      | CSR/COO SpMV, SpMM, SpGEMM                    |
| **cuBLAS LT** | 1     | lt_sgemm                                  | GEMM with algorithm heuristics                |
| **NVRTC**     | 2     | jit_compile, template_kernel              | Runtime compilation                           |
| **NVTX**      | 1     | profiling                                 | Nsight annotations                            |

### Kernel Examples (80 files, 11 categories)

GPU kernel source files in [`examples/kernel/`](examples/kernel/), organized by difficulty and feature:

| Category | Count | Kernels | Features Demonstrated |
| -------- | ----- | ------- | --------------------- |
| **0_Basic** | 8 | vector_add, saxpy, grid_stride, dot_product, relu, scale_bias, residual_norm, vec3_normalize | Thread indexing, grid-stride loops, elementwise ops |
| **1_Reduction** | 5 | reduce_sum, reduce_warp, reduce_multiblock, prefix_sum, scalar_product | Parallel reduction patterns |
| **2_Matrix** | 6 | matmul_naive, matmul_tiled, matvec, transpose, extract_diag, pad_2d | Matrix operations, tiling, 2D indexing |
| **3_Atomics** | 5 | histogram, histogram_256bin, atomic_ops, system_atomics, warp_aggregated_atomics | Atomic operations, histogramming |
| **4_SharedMemory** | 3 | shared_mem_demo, dynamic_smem, stencil_1d | Static/dynamic shared memory, stencils |
| **5_Warp** | 5 | warp shuffle, vote, reduce, ballot, cooperative | Warp-level primitives |
| **6_MathAndTypes** | 8 | fast_math, half_precision, integer_intrinsics, type_conversion, sigmoid, signal_gen, complex_mul, freq_filter | Math intrinsics, type conversion, signal processing |
| **7_Debug** | 2 | debug kernels | Debug assertions, error flags |
| **8_TensorCore** | 6 | WMMA GEMM, fragment ops | Tensor core operations (sm_70+) |
| **9_Advanced** | 8 | softmax, async_copy, cooperative_groups, thread_fence, gbm_paths, particle_init/step, intrinsics_coverage | Advanced patterns, simulation |
| **10_Integration** | 24 | Driver lifecycle, streams, CUDA graphs, cuBLAS pipelines, cuFFT pipelines, cuRAND apps, end-to-end, tensor core pipelines, **perf benchmark (Zig vs cuBLAS)** | Multi-library integration, end-to-end workflows, performance comparison |

## Testing

```bash
zig build test                               # All tests (unit + integration)
zig build test-unit                          # Unit tests only
zig build test-integration                   # Integration tests only

# Enable all optional module tests
zig build test -Dcudnn=true -Dcusolver=true -Dcusparse=true -Dcufft=true -Dnvtx=true
```

### Test Summary

| Category | Files | Tests | Requires GPU |
|----------|-------|-------|:---:|
| **Unit — kernel types** (pure Zig) | 10 | 222 | ❌ |
| **Unit — core** (driver, runtime, nvrtc) | 4 | — | ✅ |
| **Unit — conditional** (cublas, cudnn, etc.) | 8 | — | ✅ |
| **Integration — core** | 10 | — | ✅ |
| **Integration — kernel GPU** | 7 | — | ✅ |
| **Total** | **39** | **222+** | |

> [!NOTE]
> **222 pure-Zig tests** verify kernel DSL types, compile-time logic, and memory layouts — they run on any machine without CUDA.
> GPU-dependent tests require CUDA libraries and will fail to link on macOS or systems without CUDA installed.

### Test Architecture

The 10 kernel type test files (`test/unit/kernel/`) test **pure Zig data structures** that don't call any CUDA APIs:

- `kernel_shared_types_test` — Vec2/3/4, Matrix, LaunchConfig struct layout
- `kernel_types_test` — SharedMemory type tag validation
- `kernel_arch_test` — SM architecture enum & feature tables
- `kernel_device_types_test` — DevicePtr, GridStrideIterator struct verification
- `kernel_debug_test` — Printf format string compile-time logic
- `kernel_shared_mem_test` — SharedMemory compile-time meta-info
- `kernel_intrinsics_host_test` — Intrinsic function signatures & type inference
- `kernel_tensor_core_host_test` — WMMA fragment type definitions
- `kernel_grid_stride_test` — GridStrideIterator field validation
- `kernel_device_test` — Device pointer load/store patterns

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

### Project Structure

```
zcuda/
├── src/                           # Zig API layer (11 modules)
│   ├── cuda.zig                   # Root module — re-exports all modules
│   ├── types.zig                  # Shared types (Dim3, LaunchConfig, DevicePtr)
│   ├── driver/                    # CUDA Driver API (sys, result, safe)
│   ├── nvrtc/                     # NVRTC (runtime compilation)
│   ├── kernel/                    # GPU Kernel DSL (pure Zig → PTX, no CUDA C++ needed)
│   │   ├── device.zig             #   Module root (re-exports all sub-modules)
│   │   ├── types.zig              #   DeviceSlice(T), DevicePtr(T), GridStrideIterator
│   │   ├── shared_types.zig       #   Vec2/3/4, Int2/3, Matrix3x3/4x4, LaunchConfig
│   │   ├── arch.zig               #   SmVersion enum, requireSM comptime guard
│   │   ├── intrinsics.zig         #   98 inline fns: threadIdx, atomics, warp, math, cache
│   │   ├── shared_mem.zig         #   SharedArray(T,N), dynamicShared, cooperative utils
│   │   ├── tensor_core.zig        #   56 inline fns: WMMA/MMA/wgmma/TMA/cluster/tcgen05
│   │   ├── bridge_gen.zig         #   Type-safe kernel bridge (Fn enum, load, getFunction)
│   │   └── debug.zig              #   assertf, ErrorFlag, printf, CycleTimer, __trap
│   └── ...                        # 8 more binding modules (cublas, cudnn, ...)
├── examples/                      # 58 host-side examples
│   ├── basics/                    #   16 fundamentals (vector_add, streams, ...)
│   ├── cublas/                    #   19 BLAS examples
│   ├── cudnn/                     #   3 neural network examples
│   ├── cufft/                     #   4 FFT examples
│   ├── curand/                    #   3 RNG examples
│   ├── cusolver/                  #   5 linear algebra examples
│   ├── cusparse/                  #   4 sparse matrix examples
│   ├── cublaslt/                  #   1 LT GEMM example
│   ├── nvrtc/                     #   2 JIT compilation examples
│   ├── nvtx/                      #   1 profiling example
│   └── kernel/                    #   79 GPU kernel source files
│       ├── 0_Basic/               #     8 basic kernels
│       ├── 1_Reduction/           #     5 reduction kernels
│       ├── 2_Matrix/              #     6 matrix kernels
│       ├── 3_Atomics/             #     5 atomic kernels
│       ├── 4_SharedMemory/        #     3 shared memory kernels
│       ├── 5_Warp/                #     5 warp-level kernels
│       ├── 6_MathAndTypes/        #     8 math/type kernels
│       ├── 7_Debug/               #     2 debug kernels
│       ├── 8_TensorCore/          #     6 tensor core kernels
│       ├── 9_Advanced/            #     8 advanced kernels
│       └── 10_Integration/        #     24 integration host examples
├── test/                          # 39 test files
│   ├── unit/                      #   12 core unit tests
│   │   └── kernel/                #   10 kernel type tests (pure Zig, no GPU)
│   └── integration/               #   10 core integration tests
│       └── kernel/                #   7 kernel GPU integration tests
├── docs/                          # Comprehensive API documentation
├── build.zig                      # Build configuration + pub const build_helpers
├── build_helpers.zig              # Kernel discovery, PTX compilation, bridge generation
├── build.zig.zon                  # Package manifest
└── BUG_TRACKER.md                 # Known issues and fixes
```

> **Users should only use the Safe Layer.** The `result` and `sys` layers are implementation details — all public types and functions are re-exported from each module's top-level file.

## Build System

The build system (`build.zig` + `build_helpers.zig`) provides:

- **Auto-detection**: CUDA installation path discovery across common locations
- **Modular linking**: Only link libraries you enable via `-D` flags
- **Kernel pipeline**: `discoverKernels()` → content-based scan → `addBridgeModules()` → Zig→PTX compilation + type-safe bridge generation
- **Downstream export**: `pub const build_helpers = @import("build_helpers.zig")` — downstream packages access via `@import("zcuda").build_helpers`
- **Configurable paths**: `-Dkernel-dir`, `-Dcuda-path`, `-Dgpu-arch`, `-Dembed-ptx`

## Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory:

- **[Documentation Index](docs/README.md)** — Full navigation guide
- **[API Reference](docs/API.md)** — Complete safe-layer API for all binding modules + Kernel DSL overview
- **[Kernel DSL API](docs/kernel/API.md)** — Full device-side intrinsics, smem, Tensor Cores, bridge_gen
- **[CUDA C++ → Zig Migration](docs/kernel/MIGRATION.md)** — Side-by-side migration guide
- **[Examples](examples/README.md)** — 58 host + 80 kernel examples with build commands
- **[Project Structure](STRUCTURE.md)** — Source code organization and module overview

Each module has its own detailed README in `docs/<module>/README.md`.

## Contributing

1. ⭐ Star and Fork this repository
2. Create a feature branch (`git checkout -b feature/new-module`)
3. Implement sys/result/safe layers in `src/<module>/`
4. Add unit tests in `test/unit/` and integration tests in `test/integration/`
5. Create a host example in `examples/<module>/`
6. Add kernel examples in `examples/kernel/<category>/`
7. Update documentation in `docs/<module>/`
8. Submit a Pull Request

## License

MIT License

## Acknowledgments

Built with gratitude on the shoulders of giants:

- **[Zig](https://ziglang.org/)** — A modern systems programming language focused on safety, performance, and simplicity, created by Andrew Kelley and the Zig Software Foundation.

- **[CUDA](https://developer.nvidia.com/cuda-toolkit)** — NVIDIA's parallel computing platform and API, providing the underlying runtime, compiler, and libraries.

- **[cudarc](https://github.com/coreylowman/cudarc)** — A safe Rust wrapper for CUDA whose three-layer architecture (sys → result → safe) served as the foundational reference for this project.
