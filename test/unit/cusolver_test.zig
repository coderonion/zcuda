/// zCUDA Unit Tests: cuSOLVER
const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;
const cusolver = cuda.cusolver;
const CusolverDnContext = cusolver.CusolverDnContext;

test "cuSOLVER context creation" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const sol = CusolverDnContext.init(ctx) catch |err| {
        std.debug.print("Cannot create cuSOLVER context: {}\n", .{err});
        return error.SkipZigTest;
    };
    defer sol.deinit();
}

test "cuSOLVER sgetrf buffer size query" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const sol = CusolverDnContext.init(ctx) catch return error.SkipZigTest;
    defer sol.deinit();
    const stream = ctx.defaultStream();

    // Need a dummy device allocation for buffer size query
    const n: i32 = 4;
    const a_data = [_]f32{0} ** 16;
    const d_a = try stream.cloneHtoD(f32, &a_data);
    defer d_a.deinit();

    const buf_size = try sol.sgetrf_bufferSize(n, n, d_a, n);
    _ = allocator;
    try std.testing.expect(buf_size >= 0);
}

test "cuSOLVER sgetrf — LU factorization" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const sol = CusolverDnContext.init(ctx) catch return error.SkipZigTest;
    defer sol.deinit();
    const stream = ctx.defaultStream();

    // 2x2 matrix in col-major: [[4, 2], [1, 3]] => [4, 1, 2, 3]
    const n: i32 = 2;
    const a_data = [_]f32{ 4, 1, 2, 3 };

    var d_a = try stream.cloneHtoD(f32, &a_data);
    defer d_a.deinit();

    const buf_size = try sol.sgetrf_bufferSize(n, n, d_a, n);
    const d_workspace = try stream.alloc(f32, allocator, @intCast(buf_size));
    defer d_workspace.deinit();

    var d_ipiv = try stream.allocZeros(i32, allocator, @intCast(n));
    defer d_ipiv.deinit();

    // cuSOLVER devInfo must be a GPU-side pointer
    var d_info = try stream.allocZeros(i32, allocator, 1);
    defer d_info.deinit();
    var h_info: i32 = -1;

    try sol.sgetrf(n, n, d_a, n, d_workspace, d_ipiv, d_info);
    try ctx.synchronize();
    try stream.memcpyDtoH(i32, @as(*[1]i32, &h_info), d_info);

    try std.testing.expectEqual(@as(i32, 0), h_info);
}

test "cuSOLVER gesvdj — Jacobi SVD" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const sol = CusolverDnContext.init(ctx) catch return error.SkipZigTest;
    defer sol.deinit();
    const ext = cusolver.CusolverDnExt.init(&sol);
    const stream = ctx.defaultStream();

    // Create gesvdj params using safe wrapper
    const params = cusolver.GesvdjInfo.init() catch return error.SkipZigTest;
    defer params.deinit();

    try params.setTolerance(1e-7);
    try params.setMaxSweeps(100);

    // 2x2 matrix in col-major: [[3, 1], [1, 3]] => singular values 4, 2
    const n: i32 = 2;
    const a_data = [_]f32{ 3, 1, 1, 3 };

    var d_a = try stream.cloneHtoD(f32, &a_data);
    defer d_a.deinit();

    var d_s = try stream.allocZeros(f32, allocator, 2);
    defer d_s.deinit();
    var d_u = try stream.allocZeros(f32, allocator, 4);
    defer d_u.deinit();
    var d_v = try stream.allocZeros(f32, allocator, 4);
    defer d_v.deinit();

    const econ: c_int = 0; // full SVD
    const buf_size = ext.sgesvdj_bufferSize(.vector, econ, n, n, d_a, n, d_s, d_u, n, d_v, n, params) catch return error.SkipZigTest;

    const d_work = try stream.alloc(f32, allocator, @intCast(buf_size));
    defer d_work.deinit();

    // cuSOLVER devInfo must be a GPU-side pointer
    var d_info = try stream.allocZeros(i32, allocator, 1);
    defer d_info.deinit();
    var h_info: i32 = -1;

    try ext.sgesvdj(.vector, econ, n, n, d_a, n, d_s, d_u, n, d_v, n, d_work, buf_size, d_info, params);
    try ctx.synchronize();
    try stream.memcpyDtoH(i32, @as(*[1]i32, &h_info), d_info);

    try std.testing.expectEqual(@as(i32, 0), h_info);

    var s_result: [2]f32 = undefined;
    try stream.memcpyDtoH(f32, &s_result, d_s);
    // Singular values of [[3,1],[1,3]] are 4 and 2
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), s_result[0], 0.05);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), s_result[1], 0.05);
}

test "cuSOLVER sgetrf + sgetrs: LU solve pipeline" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const sol = CusolverDnContext.init(ctx) catch return error.SkipZigTest;
    defer sol.deinit();
    const stream = ctx.defaultStream();

    // Solve A*x = b where A = [[4, 1], [1, 3]], b = [5, 4]
    // Expected: x = [1, 1]
    const n: i32 = 2;
    var a_data = [_]f32{ 4, 1, 1, 3 }; // col-major
    var b_data = [_]f32{ 5, 4 };

    var d_a = try stream.cloneHtoD(f32, &a_data);
    defer d_a.deinit();
    var d_b = try stream.cloneHtoD(f32, &b_data);
    defer d_b.deinit();

    const buf_size = try sol.sgetrf_bufferSize(n, n, d_a, n);
    const d_ws = try stream.alloc(f32, allocator, @intCast(buf_size));
    defer d_ws.deinit();

    var d_ipiv = try stream.allocZeros(i32, allocator, @intCast(n));
    defer d_ipiv.deinit();

    // cuSOLVER devInfo must be a GPU-side pointer
    var d_info = try stream.allocZeros(i32, allocator, 1);
    defer d_info.deinit();
    var h_info: i32 = -1;

    try sol.sgetrf(n, n, d_a, n, d_ws, d_ipiv, d_info);
    try ctx.synchronize();
    try stream.memcpyDtoH(i32, @as(*[1]i32, &h_info), d_info);
    try std.testing.expectEqual(@as(i32, 0), h_info);

    // Solve: getrs(A_factored, ipiv, b) using safe layer
    try sol.sgetrs(n, 1, d_a, n, d_ipiv, d_b, n, d_info);
    try ctx.synchronize();

    try stream.memcpyDtoH(f32, &b_data, d_b);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), b_data[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), b_data[1], 0.01);
}

test "cuSOLVER Cholesky: spotrf factorization" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const sol = CusolverDnContext.init(ctx) catch return error.SkipZigTest;
    defer sol.deinit();
    const ext = cusolver.CusolverDnExt.init(&sol);
    const stream = ctx.defaultStream();

    // Positive definite matrix: A = [[4, 2], [2, 5]]
    const n: i32 = 2;
    var a_data = [_]f32{ 4, 2, 2, 5 };

    var d_a = try stream.cloneHtoD(f32, &a_data);
    defer d_a.deinit();

    const buf_size = try ext.spotrf_bufferSize(.lower, n, d_a, n);
    const d_ws = try stream.alloc(f32, allocator, @intCast(buf_size));
    defer d_ws.deinit();

    // cuSOLVER devInfo must be a GPU-side pointer
    var d_info = try stream.allocZeros(i32, allocator, 1);
    defer d_info.deinit();
    var h_info: i32 = -1;

    try ext.spotrf(.lower, n, d_a, n, d_ws, buf_size, d_info);
    try ctx.synchronize();
    try stream.memcpyDtoH(i32, @as(*[1]i32, &h_info), d_info);

    try std.testing.expectEqual(@as(i32, 0), h_info);

    try stream.memcpyDtoH(f32, &a_data, d_a);
    // L[0,0] = sqrt(4) = 2
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), a_data[0], 0.01);
}
