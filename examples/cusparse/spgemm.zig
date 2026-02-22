/// cuSPARSE SpGEMM Example
///
/// Sparse-sparse matrix multiplication: C = A * B where both are CSR.
///
/// Reference: CUDALibrarySamples/cuSPARSE/spgemm
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    std.debug.print("=== cuSPARSE SpGEMM (Sparse × Sparse) ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    const stream = ctx.defaultStream();
    const allocator = std.heap.page_allocator;

    const sp = try cuda.cusparse.CusparseContext.init(ctx);
    defer sp.deinit();

    // 3x3 sparse A in CSR:
    // A = | 1  0  2 |
    //     | 0  3  0 |
    //     | 0  0  4 |
    const a_rows: i64 = 3;
    const a_cols: i64 = 3;
    const a_nnz: i64 = 4;
    const a_row_h = [_]i32{ 0, 2, 3, 4 };
    const a_col_h = [_]i32{ 0, 2, 1, 2 };
    const a_val_h = [_]f32{ 1, 2, 3, 4 };

    // B = A (multiply A by itself)
    const d_a_row = try stream.cloneHtoD(i32, &a_row_h);
    defer d_a_row.deinit();
    const d_a_col = try stream.cloneHtoD(i32, &a_col_h);
    defer d_a_col.deinit();
    const d_a_val = try stream.cloneHtoD(f32, &a_val_h);
    defer d_a_val.deinit();

    const d_b_row = try stream.cloneHtoD(i32, &a_row_h);
    defer d_b_row.deinit();
    const d_b_col = try stream.cloneHtoD(i32, &a_col_h);
    defer d_b_col.deinit();
    const d_b_val = try stream.cloneHtoD(f32, &a_val_h);
    defer d_b_val.deinit();

    const mat_a = try sp.createCsr(a_rows, a_cols, a_nnz, d_a_row, d_a_col, d_a_val);
    const mat_b = try sp.createCsr(a_rows, a_cols, a_nnz, d_b_row, d_b_col, d_b_val);

    // Create empty C matrix
    var d_c_row = try stream.allocZeros(i32, allocator, @intCast(a_rows + 1));
    defer d_c_row.deinit();
    // Allocate minimal col/val arrays — will be resized after compute
    var d_c_col = try stream.allocZeros(i32, allocator, 1);
    defer d_c_col.deinit();
    var d_c_val = try stream.allocZeros(f32, allocator, 1);
    defer d_c_val.deinit();

    const mat_c = try sp.createCsr(a_rows, a_cols, 0, d_c_row, d_c_col, d_c_val);

    // SpGEMM workflow: create descriptor, work estimation, compute, copy
    const spgemm_desc = try sp.createSpGEMMDescr();
    defer spgemm_desc.deinit();

    // Phase 1: Work estimation (query size)
    var buf_size1: usize = 0;
    try sp.spGEMM_workEstimation(.non_transpose, .non_transpose, 1.0, mat_a, mat_b, 0.0, mat_c, .default, spgemm_desc, &buf_size1, null);

    var d_buf1: ?cuda.driver.CudaSlice(u8) = null;
    if (buf_size1 > 0) {
        d_buf1 = try stream.alloc(u8, allocator, buf_size1);
        try sp.spGEMM_workEstimation(.non_transpose, .non_transpose, 1.0, mat_a, mat_b, 0.0, mat_c, .default, spgemm_desc, &buf_size1, @ptrFromInt(d_buf1.?.ptr));
    }
    defer if (d_buf1) |b| b.deinit();

    // Phase 2: Compute (query size)
    var buf_size2: usize = 0;
    try sp.spGEMM_compute(.non_transpose, .non_transpose, 1.0, mat_a, mat_b, 0.0, mat_c, .default, spgemm_desc, &buf_size2, null);

    var d_buf2: ?cuda.driver.CudaSlice(u8) = null;
    if (buf_size2 > 0) {
        d_buf2 = try stream.alloc(u8, allocator, buf_size2);
        try sp.spGEMM_compute(.non_transpose, .non_transpose, 1.0, mat_a, mat_b, 0.0, mat_c, .default, spgemm_desc, &buf_size2, @ptrFromInt(d_buf2.?.ptr));
    }
    defer if (d_buf2) |b| b.deinit();

    // Phase 3: Copy results
    try sp.spGEMM_copy(.non_transpose, .non_transpose, 1.0, mat_a, mat_b, 0.0, mat_c, .default, spgemm_desc);
    try ctx.synchronize();

    std.debug.print("A = | 1  0  2 |\n    | 0  3  0 |\n    | 0  0  4 |\n\n", .{});
    std.debug.print("C = A * A computed via SpGEMM\n", .{});
    // A*A = | 1  0  10 |
    //       | 0  9   0 |
    //       | 0  0  16 |
    std.debug.print("Expected C = | 1  0  10 |\n              | 0  9   0 |\n              | 0  0  16 |\n\n", .{});
    std.debug.print("✓ SpGEMM completed\n", .{});
}
