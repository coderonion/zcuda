/// cuSOLVER SVD Example
///
/// Computes A = U * S * V^T via singular value decomposition.
/// devInfo is allocated on GPU (cuSOLVER requirement).
///
/// Reference: CUDALibrarySamples/cuSOLVER/gesvd
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    std.debug.print("=== cuSOLVER SVD (Singular Value Decomposition) ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    const stream = ctx.defaultStream();
    const allocator = std.heap.page_allocator;

    const sol = try cuda.cusolver.CusolverDnContext.init(ctx);
    defer sol.deinit();

    // 3x2 matrix A (col-major)
    // A = | 1  4 |
    //     | 2  5 |
    //     | 3  6 |
    const m: i32 = 3;
    const n: i32 = 2;
    var A_data = [_]f32{ 1, 2, 3, 4, 5, 6 }; // col-major

    std.debug.print("A (3x2):\n", .{});
    for (0..3) |r| {
        std.debug.print("  [{d:.0}, {d:.0}]\n", .{ A_data[r], A_data[3 + r] });
    }
    std.debug.print("\n", .{});

    var d_A = try stream.cloneHtoD(f32, &A_data);
    defer d_A.deinit();

    // Allocate output buffers
    var d_S = try stream.allocZeros(f32, allocator, @intCast(n));
    defer d_S.deinit();
    var d_U = try stream.allocZeros(f32, allocator, @intCast(m * m));
    defer d_U.deinit();
    var d_VT = try stream.allocZeros(f32, allocator, @intCast(n * n));
    defer d_VT.deinit();

    const lwork = try sol.sgesvd_bufferSize(m, n);
    const d_work = try stream.alloc(f32, allocator, @intCast(lwork));
    defer d_work.deinit();

    // cuSOLVER requires devInfo to be a GPU-side pointer
    var d_info = try stream.allocZeros(i32, allocator, 1);
    defer d_info.deinit();
    var h_info: i32 = 0;

    try sol.sgesvd('A', 'A', m, n, d_A, m, d_S, d_U, m, d_VT, n, d_work, lwork, d_info);
    try ctx.synchronize();
    try stream.memcpyDtoH(i32, @as(*[1]i32, &h_info), d_info);

    if (h_info != 0) {
        std.debug.print("SVD failed: info = {}\n", .{h_info});
        return error.SvdFailed;
    }

    var h_S: [2]f32 = undefined;
    try stream.memcpyDtoH(f32, &h_S, d_S);

    std.debug.print("Singular values:\n", .{});
    std.debug.print("  S[0] = {d:.6}\n  S[1] = {d:.6}\n\n", .{ h_S[0], h_S[1] });

    // For [[1,4],[2,5],[3,6]], singular values should be ~9.508 and ~0.773
    if (h_S[0] < 9.0 or h_S[0] > 10.0) return error.ValidationFailed;
    if (h_S[1] < 0.5 or h_S[1] > 1.0) return error.ValidationFailed;

    std.debug.print("✓ SVD verified (S[0] ~ 9.5, S[1] ~ 0.77)\n", .{});
}
