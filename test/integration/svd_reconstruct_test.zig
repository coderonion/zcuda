/// Integration Test: cuSOLVER SVD → reconstruct pipeline
///
/// Computes SVD of a matrix, then reconstructs it from U, S, V^T
/// to verify the factorization is correct.
/// devInfo must be a GPU-side pointer per cuSOLVER API contract.
const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;
const cusolver = cuda.cusolver;

test "SVD reconstruction pipeline" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const stream = ctx.defaultStream();

    const sol = cusolver.CusolverDnContext.init(ctx) catch return error.SkipZigTest;
    defer sol.deinit();

    // 3x2 matrix A (col-major)
    const m: i32 = 3;
    const n: i32 = 2;
    const original = [_]f32{ 1, 2, 3, 4, 5, 6 };

    var d_A = try stream.cloneHtoD(f32, &original);
    defer d_A.deinit();
    var d_S = try stream.allocZeros(f32, allocator, @intCast(n));
    defer d_S.deinit();
    var d_U = try stream.allocZeros(f32, allocator, @intCast(m * m));
    defer d_U.deinit();
    var d_VT = try stream.allocZeros(f32, allocator, @intCast(n * n));
    defer d_VT.deinit();

    const lwork = try sol.sgesvd_bufferSize(m, n);
    const d_work = try stream.alloc(f32, allocator, @intCast(lwork));
    defer d_work.deinit();

    // cuSOLVER devInfo must be a GPU-side pointer
    var d_info = try stream.allocZeros(i32, allocator, 1);
    defer d_info.deinit();
    var h_info: i32 = -1;

    try sol.sgesvd('A', 'A', m, n, d_A, m, d_S, d_U, m, d_VT, n, d_work, lwork, d_info);
    try ctx.synchronize();
    try stream.memcpyDtoH(i32, @as(*[1]i32, &h_info), d_info);

    try std.testing.expectEqual(@as(i32, 0), h_info);

    var h_S: [2]f32 = undefined;
    var h_U: [9]f32 = undefined;
    var h_VT: [4]f32 = undefined;
    try stream.memcpyDtoH(f32, &h_S, d_S);
    try stream.memcpyDtoH(f32, &h_U, d_U);
    try stream.memcpyDtoH(f32, &h_VT, d_VT);

    // Reconstruct: A_approx[r][c] = sum_k U[r][k] * S[k] * VT[k][c]  for k=0..n-1
    // col-major: U[r][k] = h_U[k*m+r], VT[k][c] = h_VT[c*n+k]
    for (0..3) |r| {
        for (0..2) |c| {
            var val: f64 = 0;
            for (0..2) |k| {
                val += @as(f64, h_U[k * 3 + r]) * @as(f64, h_S[k]) * @as(f64, h_VT[c * 2 + k]);
            }
            const expected: f64 = @floatCast(original[c * 3 + r]);
            try std.testing.expect(@abs(val - expected) < 0.1);
        }
    }
}
