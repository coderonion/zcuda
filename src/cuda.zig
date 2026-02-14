/// zCUDA: Safe and minimal CUDA bindings for Zig.
///
/// This library provides safe, idiomatic Zig bindings for the NVIDIA CUDA
/// ecosystem, closely following the architecture of [cudarc](https://github.com/chelsea0x3b/cudarc).
///
/// # Architecture
///
/// Each module is organized into three layers:
/// 1. `sys` — Raw FFI bindings (direct @cImport of C headers)
/// 2. `result` — Thin wrapper converting C error codes to Zig errors
/// 3. `safe` — High-level, type-safe abstractions (recommended for general use)
///
/// # Quick Start
///
/// ```zig
/// const std = @import("std");
/// const cuda = @import("zcuda");
///
/// pub fn main() !void {
///     // Create a CUDA context on device 0
///     const ctx = try cuda.driver.CudaContext.new(0);
///     defer ctx.deinit();
///
///     // Get the default stream
///     const stream = ctx.defaultStream();
///
///     // Allocate device memory and copy data
///     const host_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
///     const dev_data = try stream.htod(f32, allocator, &host_data);
///     defer dev_data.deinit();
///
///     // Compile and launch a kernel
///     const ptx = try cuda.nvrtc.compilePtx(allocator,
///         \\extern "C" __global__ void saxpy(float a, float *x, float *y, int n) {
///         \\    int i = blockIdx.x * blockDim.x + threadIdx.x;
///         \\    if (i < n) y[i] = a * x[i] + y[i];
///         \\}
///     );
///     defer allocator.free(ptx);
///
///     const module = try ctx.loadModule(ptx);
///     defer module.deinit();
///
///     const kernel = try module.getFunction("saxpy");
///     try stream.launch(kernel, .{
///         .grid_dim = .{ .x = 1, .y = 1, .z = 1 },
///         .block_dim = .{ .x = 4, .y = 1, .z = 1 },
///         .shared_mem_bytes = 0,
///     }, .{ 2.0, &x, &y, 4 });
/// }
/// ```
///
/// # Supported Modules
///
/// | Library | Status |
/// | --- | --- |
/// | CUDA Driver API | ✅ |
/// | NVRTC (runtime compilation) | ✅ |
/// | cuBLAS (linear algebra) | ✅ |
/// | cuBLAS LT (lightweight GEMM) | ✅ |
/// | cuRAND (random numbers) | ✅ |
/// | CUDA Runtime API | ✅ |
/// | cuDNN (deep neural networks) | ✅ (optional) |
/// | cuSOLVER (linear solvers) | ✅ (optional) |
/// | cuSPARSE (sparse matrices) | ✅ (optional) |
/// | cuFFT (FFT) | ✅ (optional) |
/// | CUPTI (profiling) | ✅ (optional) |
/// | cuFile (GPUDirect Storage) | ✅ (optional) |
/// | NVTX (annotations) | ✅ (optional) |
///
/// # CUDA Versions Supported
/// - CUDA 12.8 (primary)
/// - CUDA 13.1+ (compatible)
const std = @import("std");
const build_options = @import("build_options");

// ============================================================================
// Core Modules (always available)
// ============================================================================

/// Shared type definitions (Dim3, LaunchConfig, DevicePtr, etc.)
pub const types = @import("types.zig");

/// CUDA Driver API — device management, memory allocation, kernel launch, streams, events.
pub const driver = @import("driver/driver.zig");

/// NVRTC — runtime compilation of CUDA C++ source to PTX.
pub const nvrtc = @import("nvrtc/nvrtc.zig");

// ============================================================================
// Optional Modules (controlled by build options)
// ============================================================================

/// cuBLAS — Basic Linear Algebra Subroutines on GPU.
pub const cublas = if (build_options.enable_cublas) @import("cublas/cublas.zig") else @compileError("cuBLAS is not enabled. Pass -Dcublas=true to enable.");

/// cuBLAS LT — Lightweight matrix multiply with algorithm selection.
pub const cublaslt = if (build_options.enable_cublaslt) @import("cublaslt/cublaslt.zig") else @compileError("cuBLAS LT is not enabled. Pass -Dcublaslt=true to enable.");

/// cuRAND — GPU random number generation.
pub const curand = if (build_options.enable_curand) @import("curand/curand.zig") else @compileError("cuRAND is not enabled. Pass -Dcurand=true to enable.");

/// CUDA Runtime API — simplified device/memory/stream management.
pub const runtime = @import("runtime/runtime.zig");

/// cuDNN — Deep Neural Network library (convolutions, activations, pooling, etc.).
pub const cudnn = if (build_options.enable_cudnn) @import("cudnn/cudnn.zig") else @compileError("cuDNN is not enabled. Pass -Dcudnn=true to enable.");

/// cuSOLVER — Dense and sparse direct solvers (LU, QR, SVD, eigenvalue).
pub const cusolver = if (build_options.enable_cusolver) @import("cusolver/cusolver.zig") else @compileError("cuSOLVER is not enabled. Pass -Dcusolver=true to enable.");

/// cuSPARSE — Sparse matrix operations.
pub const cusparse = if (build_options.enable_cusparse) @import("cusparse/cusparse.zig") else @compileError("cuSPARSE is not enabled. Pass -Dcusparse=true to enable.");

/// cuFFT — Fast Fourier Transform.
pub const cufft = if (build_options.enable_cufft) @import("cufft/cufft.zig") else @compileError("cuFFT is not enabled. Pass -Dcufft=true to enable.");

/// NVTX — NVIDIA Tools Extension for annotations and markers.
pub const nvtx = if (build_options.enable_nvtx) @import("nvtx/nvtx.zig") else @compileError("NVTX is not enabled. Pass -Dnvtx=true to enable.");

// Re-export commonly used types for convenience
pub const CudaContext = driver.CudaContext;
pub const CudaStream = driver.CudaStream;
pub const CudaModule = driver.CudaModule;
pub const CudaFunction = driver.CudaFunction;
pub const CudaSlice = driver.CudaSlice;
pub const LaunchConfig = types.LaunchConfig;
pub const Dim3 = types.Dim3;

// ============================================================================
// Tests
// ============================================================================
