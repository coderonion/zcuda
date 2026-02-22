/// cuBLAS TRMV/TRSV Example: Triangular Matrix Operations
///
/// TRMV: x = A * x   (A is triangular)
/// TRSV: x = A⁻¹ * b (solve triangular system)
///
/// Uses the safe layer.
///
/// Reference: CUDALibrarySamples/cuBLAS/Level-2/trmv, trsv
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    std.debug.print("=== cuBLAS TRMV/TRSV Example ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();

    const stream = ctx.defaultStream();
    const blas = try cuda.cublas.CublasContext.init(ctx);
    defer blas.deinit();
    const allocator = std.heap.page_allocator;

    const n: i32 = 3;

    // Lower triangular matrix L (col-major):
    // L = | 2  0  0 |
    //     | 1  3  0 |
    //     | 4  2  5 |
    const L_data = [_]f32{ 2, 1, 4, 0, 3, 2, 0, 0, 5 };
    const x_orig = [_]f32{ 1, 2, 3 };

    const d_L = try stream.cloneHtoD(f32, &L_data);
    defer d_L.deinit();

    // --- TRMV: y = L * x ---
    std.debug.print("─── TRMV: x = L·x ───\n", .{});
    var x_data = x_orig;
    const d_x = try stream.cloneHtoD(f32, &x_data);
    defer d_x.deinit();

    try blas.strmv(.lower, .no_transpose, .non_unit, n, d_L, n, d_x, 1);

    try stream.memcpyDtoH(f32, &x_data, d_x);

    std.debug.print("L:\n", .{});
    for (0..3) |r| {
        std.debug.print("  [", .{});
        for (0..3) |c| std.debug.print(" {d:2.0}", .{L_data[c * 3 + r]});
        std.debug.print(" ]\n", .{});
    }
    std.debug.print("x_orig = [{d:.0}, {d:.0}, {d:.0}]\n", .{ x_orig[0], x_orig[1], x_orig[2] });
    std.debug.print("L·x    = [{d:.0}, {d:.0}, {d:.0}]\n", .{ x_data[0], x_data[1], x_data[2] });

    // Expected: L*x = [2*1, 1*1+3*2, 4*1+2*2+5*3] = [2, 7, 23]
    const exp_trmv = [_]f32{ 2, 7, 23 };
    for (&x_data, &exp_trmv) |got, exp| {
        if (@abs(got - exp) > 1e-3) return error.ValidationFailed;
    }
    std.debug.print("  ✓ Verified\n\n", .{});

    // --- TRSV: solve L * x = b for x ---
    std.debug.print("─── TRSV: L·x = b → x ───\n", .{});
    var b_data = exp_trmv;
    var d_b = try stream.alloc(f32, allocator, 3);
    defer d_b.deinit();
    try stream.memcpyHtoD(f32, d_b, &b_data);

    try blas.strsv(.lower, .no_transpose, .non_unit, n, d_L, n, d_b, 1);

    try stream.memcpyDtoH(f32, &b_data, d_b);

    std.debug.print("b = [{d:.0}, {d:.0}, {d:.0}]\n", .{ exp_trmv[0], exp_trmv[1], exp_trmv[2] });
    std.debug.print("x = L⁻¹·b = [{d:.4}, {d:.4}, {d:.4}]\n", .{ b_data[0], b_data[1], b_data[2] });

    // Solution should be x_orig = [1, 2, 3]
    for (&b_data, &x_orig) |got, exp| {
        if (@abs(got - exp) > 1e-3) return error.ValidationFailed;
    }
    std.debug.print("  ✓ Verified (matches x_orig)\n", .{});

    std.debug.print("\n✓ cuBLAS TRMV/TRSV complete\n", .{});
}
