/// cuBLAS TRSM Example: Triangular Solve
///
/// Solves op(A) * X = α·B where A is triangular.
/// Real use case: solving linear systems after LU factorization.
///
/// Reference: CUDALibrarySamples/cuBLAS/Level-3/trsm
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    std.debug.print("=== cuBLAS TRSM Example ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();

    const stream = ctx.defaultStream();
    const blas = try cuda.cublas.CublasContext.init(ctx);
    defer blas.deinit();

    // Solve A * X = B where A is lower triangular 3×3, B is 3×1
    const m: i32 = 3;
    const n: i32 = 1;

    // A (lower triangular):
    // | 2  0  0 |
    // | 3  4  0 |
    // | 1  5  6 |
    // Column-major
    const A_data = [_]f32{ 2, 3, 1, 0, 4, 5, 0, 0, 6 };
    // B = | 4 |
    //     | 23 |
    //     | 58 |
    // Solution should be X = | 2 |
    //                        | 4.25 |
    //                        | 5.625 |
    var B_data = [_]f32{ 4, 23, 58 };

    std.debug.print("A (lower triangular):\n", .{});
    for (0..@intCast(m)) |r| {
        std.debug.print("  [", .{});
        for (0..@intCast(m)) |c| {
            std.debug.print(" {d:3.0}", .{A_data[c * @as(usize, @intCast(m)) + r]});
        }
        std.debug.print(" ]\n", .{});
    }
    std.debug.print("B = [ ", .{});
    for (&B_data) |v| std.debug.print("{d:.0} ", .{v});
    std.debug.print("]\n\n", .{});

    const d_A = try stream.cloneHtod(f32, &A_data);
    defer d_A.deinit();
    const d_B = try stream.cloneHtod(f32, &B_data);
    defer d_B.deinit();

    // Solve: A * X = 1.0 * B  (result stored in B)
    try blas.strsm(.left, .lower, .no_transpose, .non_unit, m, n, 1.0, d_A, m, d_B, m);

    var X: [3]f32 = undefined;
    try stream.memcpyDtoh(f32, &X, d_B);

    std.debug.print("X (solution of A·X = B):\n  [ ", .{});
    for (&X) |v| std.debug.print("{d:.4} ", .{v});
    std.debug.print("]\n", .{});

    // Verify: A * X should equal original B
    const orig_B = [_]f32{ 4, 23, 58 };
    std.debug.print("\nVerification A·X:\n", .{});
    for (0..@intCast(m)) |r| {
        var sum: f32 = 0.0;
        for (0..r + 1) |c| {
            sum += A_data[c * @as(usize, @intCast(m)) + r] * X[c];
        }
        std.debug.print("  Row {}: {d:.4} (expected {d:.0})\n", .{ r, sum, orig_B[r] });
        if (@abs(sum - orig_B[r]) > 1e-3) return error.ValidationFailed;
    }

    std.debug.print("\n✓ cuBLAS TRSM verified\n", .{});
}
