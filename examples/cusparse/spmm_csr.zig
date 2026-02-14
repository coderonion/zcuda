/// cuSPARSE SpMM (CSR) Example
///
/// Sparse matrix × dense matrix: C = alpha * A * B + beta * C
///
/// Reference: CUDALibrarySamples/cuSPARSE/spMM_csr
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    std.debug.print("=== cuSPARSE SpMM (CSR × Dense) ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    const stream = ctx.defaultStream();
    const allocator = std.heap.page_allocator;

    const sp = try cuda.cusparse.CusparseContext.init(ctx);
    defer sp.deinit();

    // 3x3 sparse matrix (CSR):
    // A = | 1  0  2 |
    //     | 0  3  0 |
    //     | 4  0  5 |
    const rows_a: i64 = 3;
    const cols_a: i64 = 3;
    const nnz: i64 = 5;

    const h_row_off = [_]i32{ 0, 2, 3, 5 };
    const h_col_idx = [_]i32{ 0, 2, 1, 0, 2 };
    const h_vals = [_]f32{ 1, 2, 3, 4, 5 };

    // Dense 3x2 matrix B (col-major):
    // B = | 1  4 |
    //     | 2  5 |
    //     | 3  6 |
    const cols_b: i64 = 2;
    const h_B = [_]f32{ 1, 2, 3, 4, 5, 6 }; // col-major

    const d_row_off = try stream.cloneHtod(i32, &h_row_off);
    defer d_row_off.deinit();
    const d_col_idx = try stream.cloneHtod(i32, &h_col_idx);
    defer d_col_idx.deinit();
    const d_vals = try stream.cloneHtod(f32, &h_vals);
    defer d_vals.deinit();
    const d_B = try stream.cloneHtod(f32, &h_B);
    defer d_B.deinit();

    // Output C (3x2, col-major)
    var d_C = try stream.allocZeros(f32, allocator, 6);
    defer d_C.deinit();

    std.debug.print("A (3x3 sparse, {} nnz):\n", .{nnz});
    std.debug.print("  | 1  0  2 |\n  | 0  3  0 |\n  | 4  0  5 |\n\n", .{});
    std.debug.print("B (3x2 dense):\n  | 1  4 |\n  | 2  5 |\n  | 3  6 |\n\n", .{});

    const mat_a = try sp.createCsr(rows_a, cols_a, nnz, d_row_off, d_col_idx, d_vals);
    const mat_b = try sp.createDnMat(rows_a, cols_b, rows_a, d_B);
    const mat_c = try sp.createDnMat(rows_a, cols_b, rows_a, d_C);

    // Get workspace
    const buf_size = try sp.spMMBufferSize(.non_transpose, .non_transpose, 1.0, mat_a, mat_b, 0.0, mat_c);
    const workspace = if (buf_size > 0)
        try stream.alloc(u8, allocator, buf_size)
    else
        null;
    defer if (workspace) |ws| ws.deinit();

    // SpMM: C = A * B
    try sp.spMM(.non_transpose, .non_transpose, 1.0, mat_a, mat_b, 0.0, mat_c, workspace);
    try ctx.synchronize();

    var h_C: [6]f32 = undefined;
    try stream.memcpyDtoh(f32, &h_C, d_C);

    // Expected C = A * B:
    // Row 0: [1*1+2*3, 1*4+2*6] = [7, 16]
    // Row 1: [3*2, 3*5]          = [6, 15]
    // Row 2: [4*1+5*3, 4*4+5*6] = [19, 46]
    std.debug.print("C = A * B (3x2):\n", .{});
    for (0..3) |r| {
        std.debug.print("  [{d:6.1}, {d:6.1}]\n", .{ h_C[r], h_C[3 + r] });
    }

    const expected = [_]f32{ 7, 6, 19, 16, 15, 46 }; // col-major
    for (&h_C, &expected) |got, exp| {
        if (@abs(got - exp) > 0.01) return error.ValidationFailed;
    }

    std.debug.print("\n✓ SpMM CSR verified\n", .{});
}
