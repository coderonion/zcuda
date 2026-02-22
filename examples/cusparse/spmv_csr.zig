/// cuSPARSE SpMV (CSR) Example
///
/// Sparse matrix-vector multiply: y = alpha * A * x + beta * y
/// using CSR (Compressed Sparse Row) format.
///
/// Reference: CUDALibrarySamples/cuSPARSE/spMV_csr
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    std.debug.print("=== cuSPARSE SpMV (CSR Format) ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    const stream = ctx.defaultStream();
    const allocator = std.heap.page_allocator;

    const sp = try cuda.cusparse.CusparseContext.init(ctx);
    defer sp.deinit();

    // 4x4 sparse matrix in CSR format:
    // A = | 1  0  2  0 |
    //     | 0  3  0  0 |
    //     | 4  0  5  6 |
    //     | 0  7  0  8 |
    // 8 non-zero elements
    const rows: i64 = 4;
    const cols: i64 = 4;
    const nnz: i64 = 8;

    // CSR format: row_offsets (size rows+1), col_indices (size nnz), values (size nnz)
    const h_row_offsets = [_]i32{ 0, 2, 3, 6, 8 };
    const h_col_indices = [_]i32{ 0, 2, 1, 0, 2, 3, 1, 3 };
    const h_values = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };

    const d_row_offsets = try stream.cloneHtoD(i32, &h_row_offsets);
    defer d_row_offsets.deinit();
    const d_col_indices = try stream.cloneHtoD(i32, &h_col_indices);
    defer d_col_indices.deinit();
    const d_values = try stream.cloneHtoD(f32, &h_values);
    defer d_values.deinit();

    // Dense vectors x and y
    const h_x = [_]f32{ 1, 2, 3, 4 };
    var h_y = [_]f32{ 0, 0, 0, 0 };

    const d_x = try stream.cloneHtoD(f32, &h_x);
    defer d_x.deinit();
    var d_y = try stream.cloneHtoD(f32, &h_y);
    defer d_y.deinit();

    std.debug.print("Sparse A (CSR, {} nnz):\n", .{nnz});
    std.debug.print("  | 1  0  2  0 |\n  | 0  3  0  0 |\n  | 4  0  5  6 |\n  | 0  7  0  8 |\n\n", .{});
    std.debug.print("x = [1, 2, 3, 4]\n\n", .{});

    // Create sparse matrix and dense vector descriptors
    const mat_a = try sp.createCsr(rows, cols, nnz, d_row_offsets, d_col_indices, d_values);
    const vec_x = try sp.createDnVec(d_x);
    const vec_y = try sp.createDnVec(d_y);

    // Get workspace size
    const buf_size = try sp.spMV_bufferSize(.non_transpose, 1.0, mat_a, vec_x, 0.0, vec_y);
    const workspace = if (buf_size > 0)
        try stream.alloc(u8, allocator, buf_size)
    else
        null;
    defer if (workspace) |ws| ws.deinit();

    // SpMV: y = 1.0 * A * x + 0.0 * y
    try sp.spMV(.non_transpose, 1.0, mat_a, vec_x, 0.0, vec_y, workspace);
    try ctx.synchronize();

    try stream.memcpyDtoH(f32, &h_y, d_y);

    std.debug.print("y = A * x = [{d:.1}, {d:.1}, {d:.1}, {d:.1}]\n", .{ h_y[0], h_y[1], h_y[2], h_y[3] });

    // Expected: [1*1+2*3, 3*2, 4*1+5*3+6*4, 7*2+8*4] = [7, 6, 43, 46]
    const expected = [_]f32{ 7, 6, 43, 46 };
    std.debug.print("Expected  = [7.0, 6.0, 43.0, 46.0]\n\n", .{});

    for (&h_y, &expected) |got, exp| {
        if (@abs(got - exp) > 0.01) return error.ValidationFailed;
    }
    std.debug.print("✓ SpMV CSR verified\n", .{});
}
