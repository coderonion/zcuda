/// zCUDA Integration Test: LU Solve Pipeline
/// Tests the cuSOLVER sgetrf → sgetrs pipeline to solve Ax=b.
const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;
const cusolver = cuda.cusolver;
const CusolverDnContext = cusolver.CusolverDnContext;

test "LU Solve: sgetrf → sgetrs → verify Ax=b" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const sol = CusolverDnContext.init(ctx) catch return error.SkipZigTest;
    defer sol.deinit();
    const stream = ctx.defaultStream();

    // System: A x = b
    // A = [[2, 1], [5, 3]] (col-major: [2, 5, 1, 3])
    // b = [4, 7]
    // Solution: x = [5, -6]   (2*5 + 1*(-6) = 4, 5*5 + 3*(-6) = 7)
    const n: i32 = 2;
    const a_data = [_]f32{ 2, 5, 1, 3 }; // col-major
    const b_data = [_]f32{ 4, 7 };

    // Upload A and b to device
    var d_a = try stream.cloneHtod(f32, &a_data);
    defer d_a.deinit();
    var d_b = try stream.cloneHtod(f32, &b_data);
    defer d_b.deinit();

    // Step 1: LU factorize A
    const buf_size = try sol.sgetrf_bufferSize(n, n, d_a, n);
    const d_workspace = try stream.alloc(f32, allocator, @intCast(buf_size));
    defer d_workspace.deinit();
    var d_ipiv = try stream.allocZeros(i32, allocator, @intCast(n));
    defer d_ipiv.deinit();

    var info: i32 = -1;
    try sol.sgetrf(n, n, d_a, n, d_workspace, d_ipiv, &info);
    try ctx.synchronize();
    try std.testing.expectEqual(@as(i32, 0), info);

    // Step 2: Solve using LU factors
    var info2: i32 = -1;
    try sol.sgetrs(n, 1, d_a, n, d_ipiv, d_b, n, &info2);
    try ctx.synchronize();
    try std.testing.expectEqual(@as(i32, 0), info2);

    // Step 3: Verify solution
    var result: [2]f32 = undefined;
    try stream.memcpyDtoh(f32, &result, d_b);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), result[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -6.0), result[1], 1e-4);
}
