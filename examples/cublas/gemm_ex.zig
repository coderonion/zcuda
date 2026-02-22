/// cuBLAS GemmEx Example: Mixed-Precision GEMM
///
/// Performs GEMM with explicit data types, enabling mixed-precision workloads
/// (e.g., FP16 data with FP32 accumulation for deep learning inference).
///
/// Reference: CUDALibrarySamples/cuBLAS/Extensions/gemmEx
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("=== cuBLAS GemmEx (Mixed Precision) Example ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();

    const stream = ctx.defaultStream();
    const blas = try cuda.cublas.CublasContext.init(ctx);
    defer blas.deinit();

    // Simple FP32 GemmEx to verify the API works
    // C (4×3) = A (4×5) × B (5×3) in FP32
    const m: i32 = 4;
    const n: i32 = 3;
    const k: i32 = 5;

    var A: [20]f32 = undefined;
    var B: [15]f32 = undefined;

    var rng = std.Random.DefaultPrng.init(42);
    const random = rng.random();
    for (&A) |*v| v.* = @as(f32, @floatFromInt(random.intRangeAtMost(i32, 0, 5)));
    for (&B) |*v| v.* = @as(f32, @floatFromInt(random.intRangeAtMost(i32, 0, 5)));

    const d_A = try stream.cloneHtoD(f32, &A);
    defer d_A.deinit();
    const d_B = try stream.cloneHtoD(f32, &B);
    defer d_B.deinit();
    const d_C = try stream.allocZeros(f32, allocator, @intCast(m * n));
    defer d_C.deinit();

    // GemmEx with FP32 data and FP32 compute
    try blas.gemmEx(
        .no_transpose,
        .no_transpose,
        m,
        n,
        k,
        1.0,
        d_A,
        .f32,
        m,
        d_B,
        .f32,
        k,
        0.0,
        d_C,
        .f32,
        m,
    );

    var C: [12]f32 = undefined;
    try stream.memcpyDtoH(f32, &C, d_C);

    std.debug.print("C = A × B ({}×{}, computed via GemmEx FP32):\n", .{ m, n });
    for (0..@intCast(m)) |r| {
        std.debug.print("  [", .{});
        for (0..@intCast(n)) |c| {
            std.debug.print(" {d:6.0}", .{C[c * @as(usize, @intCast(m)) + r]});
        }
        std.debug.print(" ]\n", .{});
    }

    // Verify
    var max_error: f32 = 0.0;
    for (0..@intCast(m)) |r| {
        for (0..@intCast(n)) |c| {
            var expected: f32 = 0.0;
            for (0..@intCast(k)) |p| {
                expected += A[p * @as(usize, @intCast(m)) + r] * B[c * @as(usize, @intCast(k)) + p];
            }
            max_error = @max(max_error, @abs(expected - C[c * @as(usize, @intCast(m)) + r]));
        }
    }

    std.debug.print("\nMax error: {e}\n", .{max_error});
    if (max_error > 1e-4) return error.ValidationFailed;
    std.debug.print("✓ cuBLAS GemmEx verified\n", .{});
}
