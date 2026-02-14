/// Integration Test: cuSPARSE SpMV pipeline
///
/// Creates sparse matrix from dense data, performs SpMV, and verifies
/// against a known dense result.
const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;
const cusparse = cuda.cusparse;

test "sparse build → SpMV → verify pipeline" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const stream = ctx.defaultStream();
    const sp = cusparse.CusparseContext.init(ctx) catch return error.SkipZigTest;
    defer sp.deinit();

    // Build CSR for 3x3 tridiagonal matrix:
    // A = | 2 -1  0 |
    //     |-1  2 -1 |
    //     | 0 -1  2 |
    const row_ptr = [_]i32{ 0, 2, 5, 7 };
    const col_idx = [_]i32{ 0, 1, 0, 1, 2, 1, 2 };
    const vals = [_]f32{ 2, -1, -1, 2, -1, -1, 2 };

    const d_rp = try stream.cloneHtod(i32, &row_ptr);
    defer d_rp.deinit();
    const d_ci = try stream.cloneHtod(i32, &col_idx);
    defer d_ci.deinit();
    const d_vs = try stream.cloneHtod(f32, &vals);
    defer d_vs.deinit();

    const mat = try sp.createCsr(3, 3, 7, d_rp, d_ci, d_vs);
    defer sp.destroySpMat(mat);

    // x = [1, 2, 3]
    const x_h = [_]f32{ 1.0, 2.0, 3.0 };
    const d_x = try stream.cloneHtod(f32, &x_h);
    defer d_x.deinit();
    const vec_x = try sp.createDnVec(d_x);
    defer sp.destroyDnVec(vec_x);

    var d_y = try stream.allocZeros(f32, allocator, 3);
    defer d_y.deinit();
    const vec_y = try sp.createDnVec(d_y);
    defer sp.destroyDnVec(vec_y);

    const buf_size = try sp.spMV_bufferSize(.non_transpose, 1.0, mat, vec_x, 0.0, vec_y);
    var workspace: ?driver.CudaSlice(u8) = null;
    if (buf_size > 0) workspace = try stream.alloc(u8, allocator, buf_size);
    defer if (workspace) |ws| ws.deinit();

    try sp.spMV(.non_transpose, 1.0, mat, vec_x, 0.0, vec_y, workspace);
    try ctx.synchronize();

    var res: [3]f32 = undefined;
    try stream.memcpyDtoh(f32, &res, d_y);

    // A*x: [2*1-1*2, -1*1+2*2-1*3, -1*2+2*3] = [0, 0, 4]
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), res[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), res[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), res[2], 1e-5);
}
