/// cuSOLVER Cholesky Factorization + Solve Example
///
/// Computes L*L^T = A (Cholesky) for positive-definite A, then solves A*x = b.
///
/// Reference: CUDALibrarySamples/cuSOLVER/potrf
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    std.debug.print("=== cuSOLVER Cholesky Factorization + Solve ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    const stream = ctx.defaultStream();
    const allocator = std.heap.page_allocator;

    const sol = try cuda.cusolver.CusolverDnContext.init(ctx);
    defer sol.deinit();
    const ext = cuda.cusolver.CusolverDnExt.init(&sol);

    // Positive-definite symmetric 3x3 matrix (col-major)
    // A = | 4  2  1 |
    //     | 2  5  3 |
    //     | 1  3  6 |
    const n: i32 = 3;
    var A_data = [_]f32{ 4, 2, 1, 2, 5, 3, 1, 3, 6 };
    var b_data = [_]f32{ 7, 10, 10 }; // b = A * [1, 1, 1]

    std.debug.print("A (positive-definite):\n", .{});
    for (0..3) |r| {
        std.debug.print("  [", .{});
        for (0..3) |c| std.debug.print(" {d:.0}", .{A_data[c * 3 + r]});
        std.debug.print(" ]\n", .{});
    }
    std.debug.print("b = [7, 10, 10]\n\n", .{});

    var d_A = try stream.cloneHtod(f32, &A_data);
    defer d_A.deinit();
    var d_b = try stream.cloneHtod(f32, &b_data);
    defer d_b.deinit();

    // Cholesky factorization: A = L * L^T
    const buf_size = try ext.spotrf_bufferSize(.lower, n, d_A, n);
    const d_ws = try stream.alloc(f32, allocator, @intCast(buf_size));
    defer d_ws.deinit();

    var info: i32 = -1;
    try ext.spotrf(.lower, n, d_A, n, d_ws, buf_size, &info);
    try ctx.synchronize();

    if (info != 0) {
        std.debug.print("Cholesky factorization failed: info = {}\n", .{info});
        return error.FactorizationFailed;
    }
    std.debug.print("Cholesky factorization: info = {} (success)\n", .{info});

    try stream.memcpyDtoh(f32, &A_data, d_A);
    std.debug.print("L (lower Cholesky factor):\n", .{});
    for (0..3) |r| {
        std.debug.print("  [", .{});
        for (0..3) |c| {
            if (c <= r) {
                std.debug.print(" {d:.3}", .{A_data[c * 3 + r]});
            } else {
                std.debug.print("     0", .{});
            }
        }
        std.debug.print(" ]\n", .{});
    }
    std.debug.print("\n", .{});

    // Solve A*x = b using Cholesky factors
    try ext.spotrs(.lower, n, 1, d_A, n, d_b, n, &info);
    try ctx.synchronize();

    try stream.memcpyDtoh(f32, &b_data, d_b);
    std.debug.print("Solution x = [{d:.4}, {d:.4}, {d:.4}]\n", .{ b_data[0], b_data[1], b_data[2] });
    std.debug.print("Expected    [1, 1, 1]\n\n", .{});

    for (&b_data) |v| {
        if (@abs(v - 1.0) > 0.01) return error.ValidationFailed;
    }
    std.debug.print("✓ Cholesky solve verified\n", .{});
}
