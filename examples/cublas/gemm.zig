/// cuBLAS GEMM Example: C = α·A·B + β·C
///
/// General matrix-matrix multiply, the most important BLAS operation.
/// Demonstrates SGEMM with matrix setup, computation, and verification.
///
/// Reference: CUDALibrarySamples/cuBLAS/Level-3/gemm
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("=== cuBLAS GEMM Example ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();

    const stream = ctx.defaultStream();
    const blas = try cuda.cublas.CublasContext.init(ctx);
    defer blas.deinit();

    // C (m×n) = alpha * A (m×k) * B (k×n) + beta * C (m×n)
    const m: i32 = 4;
    const n: i32 = 3;
    const k: i32 = 5;

    // Column-major storage
    // A: 4×5 matrix
    var A: [20]f32 = undefined;
    var B: [15]f32 = undefined;
    var C: [12]f32 = undefined;

    var rng = std.Random.DefaultPrng.init(42);
    const random = rng.random();
    for (&A) |*v| v.* = @as(f32, @floatFromInt(random.intRangeAtMost(i32, 0, 9)));
    for (&B) |*v| v.* = @as(f32, @floatFromInt(random.intRangeAtMost(i32, 0, 9)));
    @memset(&C, 0.0);

    std.debug.print("A ({}×{}):\n", .{ m, k });
    for (0..@intCast(m)) |r| {
        std.debug.print("  [", .{});
        for (0..@intCast(k)) |c| {
            std.debug.print(" {d:3.0}", .{A[c * @as(usize, @intCast(m)) + r]});
        }
        std.debug.print(" ]\n", .{});
    }

    std.debug.print("B ({}×{}):\n", .{ k, n });
    for (0..@intCast(k)) |r| {
        std.debug.print("  [", .{});
        for (0..@intCast(n)) |c| {
            std.debug.print(" {d:3.0}", .{B[c * @as(usize, @intCast(k)) + r]});
        }
        std.debug.print(" ]\n", .{});
    }

    // Copy to device
    const d_A = try stream.cloneHtoD(f32, &A);
    defer d_A.deinit();
    const d_B = try stream.cloneHtoD(f32, &B);
    defer d_B.deinit();
    const d_C = try stream.allocZeros(f32, allocator, @intCast(m * n));
    defer d_C.deinit();

    // SGEMM: C = 1.0 * A * B + 0.0 * C
    try blas.sgemm(.no_transpose, .no_transpose, m, n, k, 1.0, d_A, m, d_B, k, 0.0, d_C, m);

    // Copy back
    try stream.memcpyDtoH(f32, &C, d_C);

    std.debug.print("\nC = A·B ({}×{}):\n", .{ m, n });
    for (0..@intCast(m)) |r| {
        std.debug.print("  [", .{});
        for (0..@intCast(n)) |c| {
            std.debug.print(" {d:6.0}", .{C[c * @as(usize, @intCast(m)) + r]});
        }
        std.debug.print(" ]\n", .{});
    }

    // Verify against CPU computation
    var max_error: f32 = 0.0;
    for (0..@intCast(m)) |r| {
        for (0..@intCast(n)) |c| {
            var expected: f32 = 0.0;
            for (0..@intCast(k)) |p| {
                const a_val = A[p * @as(usize, @intCast(m)) + r];
                const b_val = B[c * @as(usize, @intCast(k)) + p];
                expected += a_val * b_val;
            }
            const actual = C[c * @as(usize, @intCast(m)) + r];
            max_error = @max(max_error, @abs(expected - actual));
        }
    }

    std.debug.print("\nMax error: {e}\n", .{max_error});
    if (max_error > 1e-4) {
        std.debug.print("✗ FAILED\n", .{});
        return error.ValidationFailed;
    }
    std.debug.print("✓ cuBLAS GEMM verified\n", .{});
}
