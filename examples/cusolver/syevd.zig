/// cuSOLVER Eigenvalue Decomposition Example
///
/// Computes eigenvalues and eigenvectors of a symmetric matrix via SYEVD.
///
/// Reference: CUDALibrarySamples/cuSOLVER/syevd
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    std.debug.print("=== cuSOLVER Eigenvalue Decomposition (SYEVD) ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    const stream = ctx.defaultStream();
    const allocator = std.heap.page_allocator;

    const sol = try cuda.cusolver.CusolverDnContext.init(ctx);
    defer sol.deinit();
    const ext = cuda.cusolver.CusolverDnExt.init(&sol);

    // Symmetric 3x3 matrix (col-major)
    // A = | 2  1  0 |
    //     | 1  3  1 |
    //     | 0  1  2 |
    const n: i32 = 3;
    var A_data = [_]f32{ 2, 1, 0, 1, 3, 1, 0, 1, 2 };

    std.debug.print("A (symmetric):\n", .{});
    for (0..3) |r| {
        std.debug.print("  [", .{});
        for (0..3) |c| std.debug.print(" {d:.0}", .{A_data[c * 3 + r]});
        std.debug.print(" ]\n", .{});
    }
    std.debug.print("\n", .{});

    var d_A = try stream.cloneHtoD(f32, &A_data);
    defer d_A.deinit();
    var d_W = try stream.allocZeros(f32, allocator, @intCast(n));
    defer d_W.deinit();

    const buf_size = try ext.ssyevd_bufferSize(.vector, .lower, n, d_A, n, d_W);
    const d_ws = try stream.alloc(f32, allocator, @intCast(buf_size));
    defer d_ws.deinit();

    // cuSOLVER requires devInfo to be a GPU-side pointer
    var d_info = try stream.allocZeros(i32, allocator, 1);
    defer d_info.deinit();
    var h_info: i32 = 0;

    try ext.ssyevd(.vector, .lower, n, d_A, n, d_W, d_ws, buf_size, d_info);
    try ctx.synchronize();
    try stream.memcpyDtoH(i32, @as(*[1]i32, &h_info), d_info);

    if (h_info != 0) {
        std.debug.print("Eigenvalue decomposition failed: info = {}\n", .{h_info});
        return error.EigenFailed;
    }

    var h_W: [3]f32 = undefined;
    try stream.memcpyDtoH(f32, &h_W, d_W);

    std.debug.print("Eigenvalues (ascending):\n", .{});
    for (0..3) |i| {
        std.debug.print("  lambda[{}] = {d:.6}\n", .{ i, h_W[i] });
    }
    std.debug.print("\n", .{});

    // Eigenvalues of [[2,1,0],[1,3,1],[0,1,2]] are 1, 2, 4
    const expected = [_]f32{ 1.0, 2.0, 4.0 };
    for (&h_W, &expected) |got, exp| {
        if (@abs(got - exp) > 0.05) return error.ValidationFailed;
    }

    // Read eigenvectors (columns of A after decomposition)
    try stream.memcpyDtoH(f32, &A_data, d_A);
    std.debug.print("Eigenvectors (columns):\n", .{});
    for (0..3) |c| {
        std.debug.print("  v[{}] = [{d:.4}, {d:.4}, {d:.4}]\n", .{
            c,
            A_data[c * 3],
            A_data[c * 3 + 1],
            A_data[c * 3 + 2],
        });
    }

    std.debug.print("\n✓ Eigenvalue decomposition verified (eigenvalues: 1, 2, 4)\n", .{});
}
