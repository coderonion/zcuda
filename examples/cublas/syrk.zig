/// cuBLAS SYRK Example: Symmetric Rank-k Update
///
/// C = alpha * A * A^T + beta * C  (result is symmetric)
///
/// Reference: CUDALibrarySamples/cuBLAS/Level-3/syrk
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    std.debug.print("=== cuBLAS SYRK Example ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();

    const stream = ctx.defaultStream();
    const blas = try cuda.cublas.CublasContext.init(ctx);
    defer blas.deinit();

    // A is 3×2 (col-major), C is 3×3 symmetric
    const n: i32 = 3;
    const k: i32 = 2;

    // A = | 1  4 |
    //     | 2  5 |
    //     | 3  6 |
    // Col-major: column 0 = {1,2,3}, column 1 = {4,5,6}
    const A_data = [_]f32{ 1, 2, 3, 4, 5, 6 };

    // C initialized to zeros
    var C_data = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    const d_A = try stream.cloneHtod(f32, &A_data);
    defer d_A.deinit();
    const d_C = try stream.cloneHtod(f32, &C_data);
    defer d_C.deinit();

    // C = 1.0 * A * A^T + 0.0 * C
    // A*A^T = | 1*1+4*4  1*2+4*5  1*3+4*6 |   | 17  22  27 |
    //         | 2*1+5*4  2*2+5*5  2*3+5*6 | = | 22  29  36 |
    //         | 3*1+6*4  3*2+6*5  3*3+6*6 |   | 27  36  45 |
    try blas.ssyrk(.lower, .no_transpose, n, k, 1.0, d_A, n, 0.0, d_C, n);

    try stream.memcpyDtoh(f32, &C_data, d_C);

    std.debug.print("A (3×2):\n", .{});
    for (0..3) |r| {
        std.debug.print("  [", .{});
        for (0..2) |c| std.debug.print(" {d:3.0}", .{A_data[c * 3 + r]});
        std.debug.print(" ]\n", .{});
    }

    std.debug.print("\nC = A·Aᵀ (lower triangular stored):\n", .{});
    for (0..3) |r| {
        std.debug.print("  [", .{});
        for (0..3) |c| std.debug.print(" {d:3.0}", .{C_data[c * 3 + r]});
        std.debug.print(" ]\n", .{});
    }

    // Verify diagonal and lower triangle
    const expected = [_]f32{ 17, 22, 27, 22, 29, 36, 27, 36, 45 };
    // Only lower triangle is written, so check lower part
    for (0..3) |r| {
        for (0..r + 1) |c| {
            const idx = c * 3 + r;
            if (@abs(C_data[idx] - expected[idx]) > 1e-3) return error.ValidationFailed;
        }
    }
    std.debug.print("\n✓ cuBLAS SYRK verified\n", .{});
}
