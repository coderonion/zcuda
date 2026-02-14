/// cuBLAS GEAM Example: Matrix Add/Transpose
///
/// C = alpha * op(A) + beta * op(B)
/// Demonstrates matrix addition and in-place transpose.
///
/// Reference: CUDALibrarySamples/cuBLAS/Extensions/geam
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    std.debug.print("=== cuBLAS GEAM Example ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();

    const stream = ctx.defaultStream();
    const blas = try cuda.cublas.CublasContext.init(ctx);
    defer blas.deinit();

    const m: i32 = 3;
    const n: i32 = 3;

    // A = | 1  4  7 |    B = | 10  40  70 |
    //     | 2  5  8 |        | 20  50  80 |
    //     | 3  6  9 |        | 30  60  90 |
    const A_data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const B_data = [_]f32{ 10, 20, 30, 40, 50, 60, 70, 80, 90 };

    const d_A = try stream.cloneHtod(f32, &A_data);
    defer d_A.deinit();
    const d_B = try stream.cloneHtod(f32, &B_data);
    defer d_B.deinit();
    const allocator = std.heap.page_allocator;
    const d_C = try stream.alloc(f32, allocator, 9);
    defer d_C.deinit();

    // --- Test 1: C = 2*A + 1*B ---
    std.debug.print("─── C = 2·A + B ───\n", .{});
    try blas.sgeam(.no_transpose, .no_transpose, m, n, 2.0, d_A, m, 1.0, d_B, m, d_C, m);

    var C: [9]f32 = undefined;
    try stream.memcpyDtoh(f32, &C, d_C);

    for (0..3) |r| {
        std.debug.print("  [", .{});
        for (0..3) |c| std.debug.print(" {d:4.0}", .{C[c * 3 + r]});
        std.debug.print(" ]\n", .{});
    }

    // Verify: C[i] = 2*A[i] + B[i]
    for (&A_data, &B_data, &C) |a, b, c| {
        if (@abs(c - (2 * a + b)) > 1e-5) return error.ValidationFailed;
    }
    std.debug.print("  ✓ Verified\n\n", .{});

    // --- Test 2: C = A^T (matrix transpose) ---
    std.debug.print("─── C = Aᵀ (transpose) ───\n", .{});
    try blas.sgeam(.transpose, .no_transpose, n, m, 1.0, d_A, m, 0.0, d_A, n, d_C, n);

    try stream.memcpyDtoh(f32, &C, d_C);

    for (0..3) |r| {
        std.debug.print("  [", .{});
        for (0..3) |c| std.debug.print(" {d:2.0}", .{C[c * 3 + r]});
        std.debug.print(" ]\n", .{});
    }

    // A^T: row r, col c of transpose = col r, row c of original = A[r*3 + c]
    for (0..3) |r| {
        for (0..3) |c| {
            const got = C[c * 3 + r];
            const exp = A_data[r * 3 + c];
            if (@abs(got - exp) > 1e-5) return error.ValidationFailed;
        }
    }
    std.debug.print("  ✓ Verified\n", .{});

    std.debug.print("\n✓ cuBLAS GEAM complete\n", .{});
}
