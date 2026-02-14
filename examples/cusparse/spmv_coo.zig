/// cuSPARSE COO SpMV Example
///
/// Sparse matrix-vector multiply using COO (Coordinate) format.
///
/// Reference: CUDALibrarySamples/cuSPARSE/spMV
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    std.debug.print("=== cuSPARSE SpMV (COO Format) ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    const stream = ctx.defaultStream();
    const allocator = std.heap.page_allocator;

    const sp = try cuda.cusparse.CusparseContext.init(ctx);
    defer sp.deinit();

    // 3x3 sparse identity-like matrix in COO format:
    // A = | 2  0  0 |
    //     | 0  3  0 |
    //     | 1  0  4 |
    const rows: i64 = 3;
    const cols: i64 = 3;
    const nnz: i64 = 4;

    const h_row_ind = [_]i32{ 0, 1, 2, 2 };
    const h_col_ind = [_]i32{ 0, 1, 0, 2 };
    const h_values = [_]f32{ 2, 3, 1, 4 };

    const d_row_ind = try stream.cloneHtod(i32, &h_row_ind);
    defer d_row_ind.deinit();
    const d_col_ind = try stream.cloneHtod(i32, &h_col_ind);
    defer d_col_ind.deinit();
    const d_values = try stream.cloneHtod(f32, &h_values);
    defer d_values.deinit();

    const h_x = [_]f32{ 1, 2, 3 };
    var h_y = [_]f32{ 0, 0, 0 };

    const d_x = try stream.cloneHtod(f32, &h_x);
    defer d_x.deinit();
    var d_y = try stream.cloneHtod(f32, &h_y);
    defer d_y.deinit();

    std.debug.print("A (COO, {} nnz):\n  | 2  0  0 |\n  | 0  3  0 |\n  | 1  0  4 |\n\n", .{nnz});
    std.debug.print("x = [1, 2, 3]\n\n", .{});

    const mat_a = try sp.createCoo(rows, cols, nnz, d_row_ind, d_col_ind, d_values);
    const vec_x = try sp.createDnVec(d_x);
    const vec_y = try sp.createDnVec(d_y);

    const buf_size = try sp.spMV_bufferSize(.non_transpose, 1.0, mat_a, vec_x, 0.0, vec_y);
    const workspace = if (buf_size > 0)
        try stream.alloc(u8, allocator, buf_size)
    else
        null;
    defer if (workspace) |ws| ws.deinit();

    try sp.spMV(.non_transpose, 1.0, mat_a, vec_x, 0.0, vec_y, workspace);
    try ctx.synchronize();

    try stream.memcpyDtoh(f32, &h_y, d_y);
    std.debug.print("y = A * x = [{d:.1}, {d:.1}, {d:.1}]\n", .{ h_y[0], h_y[1], h_y[2] });

    // Expected: [2*1, 3*2, 1*1+4*3] = [2, 6, 13]
    const expected = [_]f32{ 2, 6, 13 };
    std.debug.print("Expected  = [2.0, 6.0, 13.0]\n\n", .{});
    for (&h_y, &expected) |got, exp| {
        if (@abs(got - exp) > 0.01) return error.ValidationFailed;
    }
    std.debug.print("✓ SpMV COO verified\n", .{});
}
