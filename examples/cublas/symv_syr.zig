/// cuBLAS SYMV/SYR Example: Symmetric Matrix-Vector Operations
///
/// SYMV: y = alpha * A * x + beta * y  (A is symmetric)
/// SYR:  A = A + alpha * x * x^T       (rank-1 update)
///
/// Uses the safe layer.
///
/// Reference: CUDALibrarySamples/cuBLAS/Level-2/symv, syr
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    std.debug.print("=== cuBLAS SYMV/SYR Example ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();

    const stream = ctx.defaultStream();
    const blas = try cuda.cublas.CublasContext.init(ctx);
    defer blas.deinit();
    const allocator = std.heap.page_allocator;

    const n: i32 = 3;

    // Symmetric matrix A (col-major):
    // A = | 4  2  1 |
    //     | 2  5  3 |
    //     | 1  3  6 |
    var A_data = [_]f32{ 4, 2, 1, 2, 5, 3, 1, 3, 6 };
    const x_data = [_]f32{ 1, 2, 3 };
    var y_data = [_]f32{ 0, 0, 0 };

    const d_A = try stream.cloneHtod(f32, &A_data);
    defer d_A.deinit();
    const d_x = try stream.cloneHtod(f32, &x_data);
    defer d_x.deinit();
    var d_y = try stream.allocZeros(f32, allocator, 3);
    defer d_y.deinit();

    // --- SYMV: y = 1.0 * A * x + 0.0 * y ---
    std.debug.print("─── SYMV: y = A·x ───\n", .{});
    try blas.ssymv(.lower, n, 1.0, d_A, n, d_x, 1, 0.0, d_y, 1);

    try stream.memcpyDtoh(f32, &y_data, d_y);

    std.debug.print("A:\n", .{});
    for (0..3) |r| {
        std.debug.print("  [", .{});
        for (0..3) |c| std.debug.print(" {d:2.0}", .{A_data[c * 3 + r]});
        std.debug.print(" ]\n", .{});
    }
    std.debug.print("x = [{d:.0}, {d:.0}, {d:.0}]\n", .{ x_data[0], x_data[1], x_data[2] });
    std.debug.print("y = A·x = [{d:.0}, {d:.0}, {d:.0}]\n", .{ y_data[0], y_data[1], y_data[2] });

    // Expected: A*x = [4+4+3, 2+10+9, 1+6+18] = [11, 21, 25]
    const exp_y = [_]f32{ 11, 21, 25 };
    for (&y_data, &exp_y) |got, exp| {
        if (@abs(got - exp) > 1e-3) return error.ValidationFailed;
    }
    std.debug.print("  ✓ Verified\n\n", .{});

    // --- SYR: A = A + 1.0 * x * x^T ---
    std.debug.print("─── SYR: A = A + x·xᵀ ───\n", .{});
    try blas.ssyr(.lower, n, 1.0, d_x, 1, d_A, n);

    try stream.memcpyDtoh(f32, &A_data, d_A);

    std.debug.print("A + x·xᵀ:\n", .{});
    for (0..3) |r| {
        std.debug.print("  [", .{});
        for (0..3) |c| std.debug.print(" {d:2.0}", .{A_data[c * 3 + r]});
        std.debug.print(" ]\n", .{});
    }

    // Expected lower triangle: (0,0):4+1=5, (1,0):2+2=4, (2,0):1+3=4, (1,1):5+4=9, (2,1):3+6=9, (2,2):6+9=15
    if (@abs(A_data[0] - 5) > 1e-3 or @abs(A_data[1] - 4) > 1e-3 or @abs(A_data[4] - 9) > 1e-3) {
        return error.ValidationFailed;
    }
    std.debug.print("  ✓ Verified\n", .{});

    std.debug.print("\n✓ cuBLAS SYMV/SYR complete\n", .{});
}
