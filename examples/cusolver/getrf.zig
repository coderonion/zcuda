/// cuSOLVER LU Factorization + Solve Example
///
/// Demonstrates PA = LU factorization followed by solving Ax = b.
/// devInfo is allocated on GPU (cuSOLVER requirement) and copied back
/// to host after synchronization.
///
/// Reference: CUDALibrarySamples/cuSOLVER/getrf
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    std.debug.print("=== cuSOLVER LU Factorization + Solve ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    const stream = ctx.defaultStream();
    const allocator = std.heap.page_allocator;

    const sol = try cuda.cusolver.CusolverDnContext.init(ctx);
    defer sol.deinit();

    // 3x3 system: A*x = b
    // A = | 1  2  3 |   b = | 14 |   Expected x = | 1 |
    //     | 4  5  6 |       | 32 |                 | 2 |
    //     | 7  8  0 |       | 23 |                 | 3 |
    const n: i32 = 3;
    var A_data = [_]f32{ 1, 4, 7, 2, 5, 8, 3, 6, 0 }; // col-major
    var b_data = [_]f32{ 14, 32, 23 };

    std.debug.print("A = | 1  2  3 |\n    | 4  5  6 |\n    | 7  8  0 |\n\n", .{});
    std.debug.print("b = [14, 32, 23]\n\n", .{});

    var d_A = try stream.cloneHtoD(f32, &A_data);
    defer d_A.deinit();
    var d_b = try stream.cloneHtoD(f32, &b_data);
    defer d_b.deinit();

    // cuSOLVER requires devInfo to be a GPU-side pointer
    var d_info = try stream.allocZeros(i32, allocator, 1);
    defer d_info.deinit();
    var h_info: i32 = 0;

    // LU factorization
    const buf_size = try sol.sgetrf_bufferSize(n, n, d_A, n);
    const d_ws = try stream.alloc(f32, allocator, @intCast(buf_size));
    defer d_ws.deinit();
    var d_ipiv = try stream.allocZeros(i32, allocator, @intCast(n));
    defer d_ipiv.deinit();

    try sol.sgetrf(n, n, d_A, n, d_ws, d_ipiv, d_info);
    try ctx.synchronize();
    try stream.memcpyDtoH(i32, @as(*[1]i32, &h_info), d_info);

    if (h_info != 0) {
        std.debug.print("LU factorization failed: info = {}\n", .{h_info});
        return error.FactorizationFailed;
    }
    std.debug.print("LU factorization: info = {} (success)\n", .{h_info});

    // Solve Ax = b
    try sol.sgetrs(n, 1, d_A, n, d_ipiv, d_b, n, d_info);
    try ctx.synchronize();
    try stream.memcpyDtoH(i32, @as(*[1]i32, &h_info), d_info);

    try stream.memcpyDtoH(f32, &b_data, d_b);
    std.debug.print("Solution x = [{d:.4}, {d:.4}, {d:.4}]\n", .{ b_data[0], b_data[1], b_data[2] });
    std.debug.print("Expected    [1, 2, 3]\n\n", .{});

    for (&b_data, &[_]f32{ 1, 2, 3 }) |got, exp| {
        if (@abs(got - exp) > 0.01) return error.ValidationFailed;
    }
    std.debug.print("✓ LU solve verified\n", .{});
}
