/// cuBLAS Pipeline: Residual Connection with GEMM
///
/// Pipeline: cuBLAS GEMM → custom Zig kernel (add residual + layer norm)
/// Models a transformer‑style residual block: y = LayerNorm(x + W·x)
///
/// Reference: cuda-samples/batchCUBLAS
//
// ── Kernel Loading: Way 5 (enhanced) build.zig auto-generated bridge module ──
// Bridge provides: .load(), .getFunction(mod, .enum), .source_path, .source
const std = @import("std");
const cuda = @import("zcuda");

// kernel: residual_add_norm(output, residual, n, eps)
// output[i] = (output[i] + residual[i] - mean) / sqrt(var + eps)
const kernel_residual_norm = @import("kernel_residual_norm");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    std.debug.print("=== cuBLAS Pipeline: GEMM → Residual + LayerNorm ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();

    const stream = ctx.defaultStream();
    const blas = try cuda.cublas.CublasContext.init(ctx);
    defer blas.deinit();

    // ── Load custom kernel ──
    const mod = try kernel_residual_norm.load(ctx, std.heap.page_allocator);
    defer mod.deinit();
    const residual_fn = try kernel_residual_norm.getFunction(mod, .residualNorm);

    // square matrix for residual: y = W·x + x
    const n: i32 = 64;
    const nn: usize = @intCast(n * n);

    var W: [nn]f32 = undefined;
    var X: [nn]f32 = undefined;

    var rng = std.Random.DefaultPrng.init(123);
    const random = rng.random();
    for (&W) |*v| v.* = (@as(f32, @floatFromInt(random.intRangeAtMost(i32, -10, 10)))) / 100.0;
    for (&X) |*v| v.* = (@as(f32, @floatFromInt(random.intRangeAtMost(i32, -10, 10)))) / 10.0;

    const d_W = try stream.cloneHtoD(f32, &W);
    defer d_W.deinit();
    const d_X = try stream.cloneHtoD(f32, &X);
    defer d_X.deinit();
    const d_Y = try stream.allocZeros(f32, allocator, nn);
    defer d_Y.deinit();

    // ── Stage 1: cuBLAS GEMM — Y = W · X ──
    try blas.sgemm(.no_transpose, .no_transpose, n, n, n, 1.0, d_W, n, d_X, n, 0.0, d_Y, n);
    std.debug.print("Stage 1: GEMM Y = W·X done\n", .{});

    // ── Stage 2: Custom kernel — Y = LayerNorm(Y + X) ──
    const config = cuda.LaunchConfig.forNumElems(@intCast(nn));
    const nn_i32: i32 = @intCast(nn);
    const eps: f32 = 1e-5;
    try stream.launch(residual_fn, config, .{ &d_Y, &d_X, nn_i32, eps });
    std.debug.print("Stage 2: Residual + LayerNorm done\n", .{});

    // ── Read back ──
    var Y: [nn]f32 = undefined;
    try stream.memcpyDtoH(f32, &Y, d_Y);

    // Print first 8 values
    std.debug.print("\nFirst 8 output values: [", .{});
    for (0..8) |i| std.debug.print(" {d:.4}", .{Y[i]});
    std.debug.print(" ]\n", .{});

    std.debug.print("\n✓ Pipeline complete: GEMM → ResidualAdd → LayerNorm\n", .{});
}
