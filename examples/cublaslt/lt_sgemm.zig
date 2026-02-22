/// cuBLAS LT SGEMM Example
///
/// Advanced matrix multiply using cuBLAS LT with algorithm heuristics.
/// D = alpha * A * B + beta * C
///
/// Reference: CUDALibrarySamples/cuBLASLt/LtSgemm
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    std.debug.print("=== cuBLAS LT SGEMM ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    const stream = ctx.defaultStream();
    const allocator = std.heap.page_allocator;

    const lt = try cuda.cublaslt.CublasLtContext.init(ctx);
    defer lt.deinit();

    // C = alpha * A * B + beta * C
    // A: 4x3, B: 3x2, C: 4x2 (all col-major)
    const m: u64 = 4;
    const n: u64 = 2;
    const k: u64 = 3;

    // A (4x3 col-major)
    const h_A = [_]f32{
        1, 2, 3, 4, // col 0
        5, 6, 7, 8, // col 1
        9, 10, 11, 12, // col 2
    };
    // B (3x2 col-major)
    const h_B = [_]f32{
        1, 2, 3, // col 0
        4, 5, 6, // col 1
    };
    // C initialized to zeros
    var h_C = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };

    const d_A = try stream.cloneHtoD(f32, &h_A);
    defer d_A.deinit();
    const d_B = try stream.cloneHtoD(f32, &h_B);
    defer d_B.deinit();
    var d_C = try stream.cloneHtoD(f32, &h_C);
    defer d_C.deinit();
    var d_D = try stream.allocZeros(f32, allocator, @intCast(m * n));
    defer d_D.deinit();

    std.debug.print("A (4x3):\n", .{});
    for (0..4) |r| {
        std.debug.print("  [", .{});
        for (0..3) |c| std.debug.print(" {d:3.0}", .{h_A[c * 4 + r]});
        std.debug.print(" ]\n", .{});
    }
    std.debug.print("B (3x2):\n", .{});
    for (0..3) |r| {
        std.debug.print("  [", .{});
        for (0..2) |c| std.debug.print(" {d:3.0}", .{h_B[c * 3 + r]});
        std.debug.print(" ]\n", .{});
    }
    std.debug.print("\n", .{});

    // Create matmul descriptor (f32 compute)
    const matmul_desc = try lt.createMatmulDesc(.f32, .f32);
    const layout_a = try lt.createMatrixLayout(.f32, m, k, @intCast(m));
    const layout_b = try lt.createMatrixLayout(.f32, k, n, @intCast(k));
    const layout_c = try lt.createMatrixLayout(.f32, m, n, @intCast(m));
    const layout_d = try lt.createMatrixLayout(.f32, m, n, @intCast(m));

    // Get heuristics
    const pref = try lt.createPreference();
    var heuristics: [3]cuda.cublaslt.MatmulHeuristicResult = undefined;
    const algo_count = try lt.getHeuristics(matmul_desc, layout_a, layout_b, layout_c, layout_d, pref, &heuristics);
    std.debug.print("Found {} algorithm(s) via heuristics\n\n", .{algo_count});

    // Execute: D = 1.0 * A * B + 0.0 * C
    try lt.matmul(f32, matmul_desc, 1.0, d_A, layout_a, d_B, layout_b, 0.0, d_C, layout_c, d_D, layout_d, stream);
    try ctx.synchronize();

    var h_D: [8]f32 = undefined;
    try stream.memcpyDtoH(f32, &h_D, d_D);

    // Expected: A*B
    // Row 0: 1*1+5*2+9*3  = 38,   1*4+5*5+9*6  = 83
    // Row 1: 2*1+6*2+10*3 = 44,   2*4+6*5+10*6 = 98
    // Row 2: 3*1+7*2+11*3 = 50,   3*4+7*5+11*6 = 113
    // Row 3: 4*1+8*2+12*3 = 56,   4*4+8*5+12*6 = 128
    std.debug.print("D = A * B (4x2):\n", .{});
    for (0..4) |r| {
        std.debug.print("  [{d:6.1}, {d:6.1}]\n", .{ h_D[r], h_D[4 + r] });
    }
    std.debug.print("\n", .{});

    const expected = [_]f32{ 38, 44, 50, 56, 83, 98, 113, 128 };
    for (&h_D, &expected) |got, exp| {
        if (@abs(got - exp) > 0.1) return error.ValidationFailed;
    }
    std.debug.print("✓ cuBLAS LT SGEMM verified\n", .{});
}
