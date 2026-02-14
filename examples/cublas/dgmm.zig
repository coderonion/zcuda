/// cuBLAS DGMM Example: Diagonal Matrix Multiply
///
/// C = A * diag(x)  (right) or C = diag(x) * A (left)
///
/// Reference: CUDALibrarySamples/cuBLAS/Extensions/dgmm
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    std.debug.print("=== cuBLAS DGMM Example ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();

    const stream = ctx.defaultStream();
    const blas = try cuda.cublas.CublasContext.init(ctx);
    defer blas.deinit();
    const allocator = std.heap.page_allocator;

    const m: i32 = 3;
    const n: i32 = 3;

    // A = | 1  4  7 |    x = | 10 |
    //     | 2  5  8 |        | 20 |
    //     | 3  6  9 |        | 30 |
    const A_data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const x_data = [_]f32{ 10, 20, 30 };

    const d_A = try stream.cloneHtod(f32, &A_data);
    defer d_A.deinit();
    const d_x = try stream.cloneHtod(f32, &x_data);
    defer d_x.deinit();
    const d_C = try stream.alloc(f32, allocator, 9);
    defer d_C.deinit();

    // --- Right multiply: C = A * diag(x) ---
    // Each column j of A is scaled by x[j]
    // Col 0: {1,2,3}*10 = {10,20,30}
    // Col 1: {4,5,6}*20 = {80,100,120}
    // Col 2: {7,8,9}*30 = {210,240,270}
    std.debug.print("─── C = A · diag(x)  (right) ───\n", .{});
    try blas.sdgmm(.right, m, n, d_A, m, d_x, d_C, m);

    var C: [9]f32 = undefined;
    try stream.memcpyDtoh(f32, &C, d_C);

    std.debug.print("A:\n", .{});
    for (0..3) |r| {
        std.debug.print("  [", .{});
        for (0..3) |c| std.debug.print(" {d:2.0}", .{A_data[c * 3 + r]});
        std.debug.print(" ]\n", .{});
    }
    std.debug.print("x = [{d:.0}, {d:.0}, {d:.0}]\n\n", .{ x_data[0], x_data[1], x_data[2] });

    std.debug.print("C = A · diag(x):\n", .{});
    for (0..3) |r| {
        std.debug.print("  [", .{});
        for (0..3) |c| std.debug.print(" {d:4.0}", .{C[c * 3 + r]});
        std.debug.print(" ]\n", .{});
    }

    // Verify: C[r][c] = A[r][c] * x[c]
    for (0..3) |r| {
        for (0..3) |c| {
            const exp = A_data[c * 3 + r] * x_data[c];
            if (@abs(C[c * 3 + r] - exp) > 1e-3) return error.ValidationFailed;
        }
    }
    std.debug.print("  ✓ Verified\n", .{});

    // --- Left multiply: C = diag(x) * A ---
    // Each row i of A is scaled by x[i]
    std.debug.print("\n─── C = diag(x) · A  (left) ───\n", .{});
    try blas.sdgmm(.left, m, n, d_A, m, d_x, d_C, m);
    try stream.memcpyDtoh(f32, &C, d_C);

    std.debug.print("C = diag(x) · A:\n", .{});
    for (0..3) |r| {
        std.debug.print("  [", .{});
        for (0..3) |c| std.debug.print(" {d:4.0}", .{C[c * 3 + r]});
        std.debug.print(" ]\n", .{});
    }

    // Verify: C[r][c] = x[r] * A[r][c]
    for (0..3) |r| {
        for (0..3) |c| {
            const exp = x_data[r] * A_data[c * 3 + r];
            if (@abs(C[c * 3 + r] - exp) > 1e-3) return error.ValidationFailed;
        }
    }
    std.debug.print("  ✓ Verified\n", .{});

    std.debug.print("\n✓ cuBLAS DGMM complete\n", .{});
}
