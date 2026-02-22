/// zCUDA Unit Tests: cuSPARSE
const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;
const cusparse = cuda.cusparse;
const CusparseContext = cusparse.CusparseContext;

test "cuSPARSE context creation" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const sp = CusparseContext.init(ctx) catch |err| {
        std.debug.print("Cannot create cuSPARSE context: {}\n", .{err});
        return error.SkipZigTest;
    };
    defer sp.deinit();
}

test "cuSPARSE CSR creation" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const sp = CusparseContext.init(ctx) catch return error.SkipZigTest;
    defer sp.deinit();
    const stream = ctx.defaultStream();

    // 3x3 identity in CSR
    const row_ptrs_h = [_]i32{ 0, 1, 2, 3 };
    const col_idxs_h = [_]i32{ 0, 1, 2 };
    const vals_h = [_]f32{ 1.0, 1.0, 1.0 };

    const d_row_ptrs = try stream.cloneHtoD(i32, &row_ptrs_h);
    defer d_row_ptrs.deinit();
    const d_col_idxs = try stream.cloneHtoD(i32, &col_idxs_h);
    defer d_col_idxs.deinit();
    const d_vals = try stream.cloneHtoD(f32, &vals_h);
    defer d_vals.deinit();

    const mat = try sp.createCsr(3, 3, 3, d_row_ptrs, d_col_idxs, d_vals);
    sp.destroySpMat(mat);
}

test "cuSPARSE SpMV — y = A*x (identity)" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const sp = CusparseContext.init(ctx) catch return error.SkipZigTest;
    defer sp.deinit();
    const stream = ctx.defaultStream();

    // 3x3 identity in CSR
    const row_ptrs_h = [_]i32{ 0, 1, 2, 3 };
    const col_idxs_h = [_]i32{ 0, 1, 2 };
    const vals_h = [_]f32{ 1.0, 1.0, 1.0 };

    const d_row_ptrs = try stream.cloneHtoD(i32, &row_ptrs_h);
    defer d_row_ptrs.deinit();
    const d_col_idxs = try stream.cloneHtoD(i32, &col_idxs_h);
    defer d_col_idxs.deinit();
    const d_vals = try stream.cloneHtoD(f32, &vals_h);
    defer d_vals.deinit();

    const mat = try sp.createCsr(3, 3, 3, d_row_ptrs, d_col_idxs, d_vals);
    defer sp.destroySpMat(mat);

    // x = [1, 2, 3]
    const x_h = [_]f32{ 1.0, 2.0, 3.0 };
    const d_x = try stream.cloneHtoD(f32, &x_h);
    defer d_x.deinit();
    const vec_x = try sp.createDnVec(d_x);
    defer sp.destroyDnVec(vec_x);

    // y = [0, 0, 0]
    var d_y = try stream.allocZeros(f32, allocator, 3);
    defer d_y.deinit();
    const vec_y = try sp.createDnVec(d_y);
    defer sp.destroyDnVec(vec_y);

    // Get buffer size
    const buf_size = try sp.spMV_bufferSize(.non_transpose, 1.0, mat, vec_x, 0.0, vec_y);

    var workspace: ?driver.CudaSlice(u8) = null;
    if (buf_size > 0) {
        workspace = try stream.alloc(u8, allocator, buf_size);
    }
    defer if (workspace) |ws| ws.deinit();

    // y = 1.0 * I * x + 0.0 * y  => y = x
    try sp.spMV(.non_transpose, 1.0, mat, vec_x, 0.0, vec_y, workspace);
    try ctx.synchronize();

    var result: [3]f32 = undefined;
    try stream.memcpyDtoH(f32, &result, d_y);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), result[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), result[2], 1e-5);
}

test "cuSPARSE SpGEMM descriptor creation" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const sp = CusparseContext.init(ctx) catch return error.SkipZigTest;
    defer sp.deinit();

    const spgemm = try sp.createSpGEMMDescr();
    defer spgemm.deinit();
}

test "cuSPARSE COO creation" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const sp = CusparseContext.init(ctx) catch return error.SkipZigTest;
    defer sp.deinit();
    const stream = ctx.defaultStream();

    // 3x3 identity in COO
    const row_idxs_h = [_]i32{ 0, 1, 2 };
    const col_idxs_h = [_]i32{ 0, 1, 2 };
    const vals_h = [_]f32{ 1.0, 1.0, 1.0 };

    const d_row = try stream.cloneHtoD(i32, &row_idxs_h);
    defer d_row.deinit();
    const d_col = try stream.cloneHtoD(i32, &col_idxs_h);
    defer d_col.deinit();
    const d_vals = try stream.cloneHtoD(f32, &vals_h);
    defer d_vals.deinit();

    const mat = try sp.createCoo(3, 3, 3, d_row, d_col, d_vals);
    defer sp.destroySpMat(mat);
}

test "cuSPARSE SpMM — C = A*B (CSR × dense)" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const sp = CusparseContext.init(ctx) catch return error.SkipZigTest;
    defer sp.deinit();
    const stream = ctx.defaultStream();

    // 2x2 identity in CSR
    const row_h = [_]i32{ 0, 1, 2 };
    const col_h = [_]i32{ 0, 1 };
    const val_h = [_]f32{ 2.0, 3.0 };
    const d_row = try stream.cloneHtoD(i32, &row_h);
    defer d_row.deinit();
    const d_col = try stream.cloneHtoD(i32, &col_h);
    defer d_col.deinit();
    const d_val = try stream.cloneHtoD(f32, &val_h);
    defer d_val.deinit();
    const mat_a = try sp.createCsr(2, 2, 2, d_row, d_col, d_val);
    defer sp.destroySpMat(mat_a);

    // Dense B (2x2 col-major): [[1,3],[2,4]]
    const b_h = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const d_b = try stream.cloneHtoD(f32, &b_h);
    defer d_b.deinit();
    const mat_b = try sp.createDnMat(2, 2, 2, d_b);
    defer sp.destroyDnMat(mat_b);

    // Output C (2x2)
    var d_c = try stream.allocZeros(f32, allocator, 4);
    defer d_c.deinit();
    const mat_c = try sp.createDnMat(2, 2, 2, d_c);
    defer sp.destroyDnMat(mat_c);

    const buf_size = try sp.spMMBufferSize(.non_transpose, .non_transpose, 1.0, mat_a, mat_b, 0.0, mat_c);
    var workspace: ?driver.CudaSlice(u8) = null;
    if (buf_size > 0) workspace = try stream.alloc(u8, allocator, buf_size);
    defer if (workspace) |ws| ws.deinit();

    try sp.spMM(.non_transpose, .non_transpose, 1.0, mat_a, mat_b, 0.0, mat_c, workspace);
    try ctx.synchronize();

    var res: [4]f32 = undefined;
    try stream.memcpyDtoH(f32, &res, d_c);
    // diag(2,3) * [[1,3],[2,4]] = [[2,6],[6,12]]
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), res[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), res[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), res[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), res[3], 1e-5);
}

test "cuSPARSE COO SpMV" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const sp = CusparseContext.init(ctx) catch return error.SkipZigTest;
    defer sp.deinit();
    const stream = ctx.defaultStream();

    // 2x2 matrix [[1,0],[0,2]] in COO
    const ri = [_]i32{ 0, 1 };
    const ci = [_]i32{ 0, 1 };
    const vs = [_]f32{ 1.0, 2.0 };
    const d_ri = try stream.cloneHtoD(i32, &ri);
    defer d_ri.deinit();
    const d_ci = try stream.cloneHtoD(i32, &ci);
    defer d_ci.deinit();
    const d_vs = try stream.cloneHtoD(f32, &vs);
    defer d_vs.deinit();
    const mat = try sp.createCoo(2, 2, 2, d_ri, d_ci, d_vs);
    defer sp.destroySpMat(mat);

    const x_h = [_]f32{ 3.0, 5.0 };
    const d_x = try stream.cloneHtoD(f32, &x_h);
    defer d_x.deinit();
    const vec_x = try sp.createDnVec(d_x);
    defer sp.destroyDnVec(vec_x);

    var d_y = try stream.allocZeros(f32, allocator, 2);
    defer d_y.deinit();
    const vec_y = try sp.createDnVec(d_y);
    defer sp.destroyDnVec(vec_y);

    const buf_size = try sp.spMV_bufferSize(.non_transpose, 1.0, mat, vec_x, 0.0, vec_y);
    var workspace: ?driver.CudaSlice(u8) = null;
    if (buf_size > 0) workspace = try stream.alloc(u8, allocator, buf_size);
    defer if (workspace) |ws| ws.deinit();

    try sp.spMV(.non_transpose, 1.0, mat, vec_x, 0.0, vec_y, workspace);
    try ctx.synchronize();

    var res: [2]f32 = undefined;
    try stream.memcpyDtoH(f32, &res, d_y);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), res[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), res[1], 1e-5);
}
