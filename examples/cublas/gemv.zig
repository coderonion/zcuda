/// cuBLAS GEMV Example: y = α·A·x + β·y
///
/// General matrix-vector multiply, the core Level-2 BLAS operation.
/// Demonstrates both non-transposed and transposed forms.
///
/// Reference: CUDALibrarySamples/cuBLAS/Level-2/gemv
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    std.debug.print("=== cuBLAS GEMV Example ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();

    const stream = ctx.defaultStream();
    const blas = try cuda.cublas.CublasContext.init(ctx);
    defer blas.deinit();

    // Matrix A (3×4, column-major) and vector x (4×1)
    // A = | 1  2  3  4 |     x = | 1 |
    //     | 5  6  7  8 |         | 2 |
    //     | 9 10 11 12 |         | 3 |
    //                            | 4 |
    const m: i32 = 3; // rows of A
    const n: i32 = 4; // cols of A

    // Column-major: column 0 = {1,5,9}, column 1 = {2,6,10}, ...
    const A_data = [_]f32{ 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 };
    const x_data = [_]f32{ 1, 2, 3, 4 };
    const y_data = [_]f32{ 0.0, 0.0, 0.0 };

    std.debug.print("A ({}×{}, column-major):\n", .{ m, n });
    for (0..@intCast(m)) |row| {
        std.debug.print("  [", .{});
        for (0..@intCast(n)) |col| {
            const idx = col * @as(usize, @intCast(m)) + row;
            std.debug.print(" {d:4.0}", .{A_data[idx]});
        }
        std.debug.print(" ]\n", .{});
    }

    std.debug.print("x = [ ", .{});
    for (&x_data) |v| std.debug.print("{d:.0} ", .{v});
    std.debug.print("]\n\n", .{});

    // Copy to device
    const d_A = try stream.cloneHtod(f32, &A_data);
    defer d_A.deinit();
    const d_x = try stream.cloneHtod(f32, &x_data);
    defer d_x.deinit();
    const d_y = try stream.cloneHtod(f32, &y_data);
    defer d_y.deinit();

    // y = 1.0 * A * x + 0.0 * y
    try blas.sgemv(.no_transpose, m, n, 1.0, d_A, m, d_x, 0.0, d_y);

    var h_result: [3]f32 = undefined;
    try stream.memcpyDtoh(f32, &h_result, d_y);

    // Expected: y = A*x = [1*1+2*2+3*3+4*4, 5*1+6*2+7*3+8*4, 9*1+10*2+11*3+12*4] = [30, 70, 110]
    std.debug.print("y = A·x = [ ", .{});
    for (&h_result) |v| std.debug.print("{d:.0} ", .{v});
    std.debug.print("]\n", .{});

    // Verify
    const expected = [_]f32{ 30.0, 70.0, 110.0 };
    for (&expected, &h_result) |exp, actual| {
        if (@abs(exp - actual) > 1e-4) {
            std.debug.print("✗ FAILED\n", .{});
            return error.ValidationFailed;
        }
    }
    std.debug.print("Expected: [ 30 70 110 ]\n", .{});
    std.debug.print("✓ Verified\n", .{});

    std.debug.print("\n✓ cuBLAS GEMV complete\n", .{});
}
