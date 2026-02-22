/// cuBLAS Pipeline: Custom Kernel + GEMM
///
/// Pipeline: custom Zig kernel (scale+bias data) → cuBLAS SGEMM → custom kernel (ReLU activation)
/// Demonstrates combining hand-written device kernels with cuBLAS library calls.
///
/// Reference: cuda-samples/batchCUBLAS, cuda-samples/matmulCUBLAS
//
// ── Kernel Loading: Way 5 (enhanced) build.zig auto-generated bridge module ──
// Bridge provides: .load(), .getFunction(mod, .enum), .source_path, .source
const std = @import("std");
const cuda = @import("zcuda");

// ── Device Kernels ──────────────────────────────────────────────
// kernel: scale each element by `alpha` and add `bias`
const kernel_scale_bias = @import("kernel_scale_bias");

// kernel: apply ReLU activation in-place: x = max(x, 0)
const kernel_relu = @import("kernel_relu");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    std.debug.print("=== cuBLAS Pipeline: Scale+Bias → GEMM → ReLU ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();

    const stream = ctx.defaultStream();
    const blas = try cuda.cublas.CublasContext.init(ctx);
    defer blas.deinit();

    // ── Load custom kernels ──
    const mod_sb = try kernel_scale_bias.load(ctx, std.heap.page_allocator);
    defer mod_sb.deinit();
    const scale_bias_fn = try kernel_scale_bias.getFunction(mod_sb, .scaleBias);

    const mod_relu = try kernel_relu.load(ctx, std.heap.page_allocator);
    defer mod_relu.deinit();
    const relu_fn = try kernel_relu.getFunction(mod_relu, .relu);

    // ── Prepare data ──
    const m: i32 = 4;
    const n: i32 = 3;
    const k: i32 = 5;
    const mn: usize = @intCast(m * n);
    const mk: usize = @intCast(m * k);
    const kn: usize = @intCast(k * n);

    var A: [mk]f32 = undefined;
    var B: [kn]f32 = undefined;

    var rng = std.Random.DefaultPrng.init(42);
    const random = rng.random();
    for (&A) |*v| v.* = @as(f32, @floatFromInt(random.intRangeAtMost(i32, -5, 5)));
    for (&B) |*v| v.* = @as(f32, @floatFromInt(random.intRangeAtMost(i32, -5, 5)));

    const d_A = try stream.cloneHtoD(f32, &A);
    defer d_A.deinit();
    const d_B = try stream.cloneHtoD(f32, &B);
    defer d_B.deinit();
    const d_C = try stream.allocZeros(f32, allocator, mn);
    defer d_C.deinit();

    // ── Stage 1: Custom kernel — scale A by 0.5 and add bias 1.0 ──
    const config_a = cuda.LaunchConfig.forNumElems(@intCast(mk));
    const alpha: f32 = 0.5;
    const bias: f32 = 1.0;
    const mk_i32: i32 = @intCast(mk);
    try stream.launch(scale_bias_fn, config_a, .{ &d_A, alpha, bias, mk_i32 });

    // ── Stage 2: cuBLAS SGEMM — C = A' × B ──
    try blas.sgemm(.no_transpose, .no_transpose, m, n, k, 1.0, d_A, m, d_B, k, 0.0, d_C, m);

    // ── Stage 3: Custom kernel — ReLU on C ──
    const config_c = cuda.LaunchConfig.forNumElems(@intCast(mn));
    const mn_i32: i32 = @intCast(mn);
    try stream.launch(relu_fn, config_c, .{ &d_C, mn_i32 });

    // ── Read back ──
    var C: [mn]f32 = undefined;
    try stream.memcpyDtoH(f32, &C, d_C);

    std.debug.print("C = ReLU(ScaleBias(A) × B):\n", .{});
    for (0..@intCast(m)) |r| {
        std.debug.print("  [", .{});
        for (0..@intCast(n)) |c| {
            std.debug.print(" {d:8.2}", .{C[c * @as(usize, @intCast(m)) + r]});
        }
        std.debug.print(" ]\n", .{});
    }

    // Verify: all values should be >= 0 (ReLU)
    for (&C) |v| {
        if (v < 0.0) {
            std.debug.print("✗ FAILED: ReLU output contains negative value\n", .{});
            return error.ValidationFailed;
        }
    }
    std.debug.print("\n✓ Pipeline complete: ScaleBias → GEMM → ReLU\n", .{});
}
