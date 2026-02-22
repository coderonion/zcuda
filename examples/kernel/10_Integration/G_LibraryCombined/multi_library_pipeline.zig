/// Multi-Library Pipeline: cuRAND + Custom Kernel + cuBLAS + cuFFT
///
/// Full pipeline combining multiple CUDA libraries with custom Zig kernels:
///   1. cuRAND → generate random input matrix
///   2. Custom kernel → apply non-linear transform (sigmoid)
///   3. cuBLAS GEMM → matrix multiply
///   4. Custom kernel → extract diagonal + prepare for FFT
///   5. cuFFT → spectral analysis of diagonal
///
/// This demonstrates the real power of zcuda: seamlessly mixing hand-written
/// Zig kernels with multiple CUDA library calls in a single pipeline.
///
/// Reference: cuda-samples (various multi-library examples)
//
// ── Kernel Loading: Way 5 (enhanced) build.zig auto-generated bridge module ──
// Bridge provides: .load(), .getFunction(mod, .enum), .source_path, .source
const std = @import("std");
const cuda = @import("zcuda");

// kernel: sigmoid_transform(data, n) — in-place sigmoid: x = 1 / (1 + exp(-x))
const kernel_sigmoid = @import("kernel_sigmoid");

// kernel: extract_diagonal(matrix, diagonal, n, stride)
const kernel_extract_diag = @import("kernel_extract_diag");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    std.debug.print("=== Multi-Library Pipeline: cuRAND → Kernel → cuBLAS → Kernel → cuFFT ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    const stream = ctx.defaultStream();

    // ── Initialize libraries ──
    const blas = try cuda.cublas.CublasContext.init(ctx);
    defer blas.deinit();
    const rng = try cuda.curand.CurandContext.init(ctx, .philox4_32_10);
    defer rng.deinit();
    try rng.setStream(stream);
    try rng.setSeed(42);

    // ── Load custom kernels ──
    const mod_sig = try kernel_sigmoid.load(ctx, std.heap.page_allocator);
    defer mod_sig.deinit();
    const sigmoid_fn = try kernel_sigmoid.getFunction(mod_sig, .sigmoidTransform);

    const mod_diag = try kernel_extract_diag.load(ctx, std.heap.page_allocator);
    defer mod_diag.deinit();
    const diag_fn = try kernel_extract_diag.getFunction(mod_diag, .extractDiagonal);

    const n: i32 = 128;
    const nn: usize = @intCast(n * n);

    // ── Stage 1: cuRAND — fill random matrices A and B ──
    const d_A = try stream.alloc(f32, allocator, nn);
    defer d_A.deinit();
    const d_B = try stream.alloc(f32, allocator, nn);
    defer d_B.deinit();
    try rng.fillNormal(d_A, 0.0, 1.0);
    try rng.fillNormal(d_B, 0.0, 1.0);
    std.debug.print("Stage 1: cuRAND generated {0}×{0} random matrices A, B\n", .{n});

    // ── Stage 2: Custom kernel — sigmoid(A) ──
    const config = cuda.LaunchConfig.forNumElems(@intCast(nn));
    const nn_i32: i32 = @intCast(nn);
    try stream.launch(sigmoid_fn, config, .{ &d_A, nn_i32 });
    std.debug.print("Stage 2: Applied sigmoid transform to A\n", .{});

    // ── Stage 3: cuBLAS GEMM — C = sigmoid(A) × B ──
    const d_C = try stream.allocZeros(f32, allocator, nn);
    defer d_C.deinit();
    try blas.sgemm(.no_transpose, .no_transpose, n, n, n, 1.0, d_A, n, d_B, n, 0.0, d_C, n);
    std.debug.print("Stage 3: cuBLAS GEMM C = sigmoid(A) × B\n", .{});

    // ── Stage 4: Custom kernel — extract diagonal of C ──
    const diag_size: usize = @intCast(n);
    const complex_diag = diag_size * 2; // for FFT (real + imag)
    const d_diag = try stream.allocZeros(f32, allocator, complex_diag);
    defer d_diag.deinit();
    const diag_config = cuda.LaunchConfig.forNumElems(diag_size);
    try stream.launch(diag_fn, diag_config, .{ &d_C, &d_diag, n, n });
    std.debug.print("Stage 4: Extracted diagonal ({} elements) for spectral analysis\n", .{diag_size});

    // ── Stage 5: cuFFT — spectral analysis of diagonal ──
    const plan = try cuda.cufft.CufftPlan.plan1d(n, .c2c_f32, 1);
    defer plan.deinit();
    try plan.setStream(stream);
    try plan.execC2C(d_diag, d_diag, .forward);
    std.debug.print("Stage 5: cuFFT spectral analysis of diagonal\n", .{});

    // ── Read back ──
    var diag_fft: [256]f32 = undefined; // first 128 complex pairs
    try stream.memcpyDtoH(f32, &diag_fft, d_diag);

    std.debug.print("\nFirst 5 frequency components (magnitude):\n", .{});
    for (0..5) |i| {
        const re = diag_fft[2 * i];
        const im = diag_fft[2 * i + 1];
        const mag = @sqrt(re * re + im * im);
        std.debug.print("  |F[{}]| = {d:.4}\n", .{ i, mag });
    }

    std.debug.print("\n✓ Multi-library pipeline complete:\n", .{});
    std.debug.print("  cuRAND → sigmoid kernel → cuBLAS GEMM → extract_diag kernel → cuFFT\n", .{});
}
