/// cuBLAS Symmetric Matrix Multiply (SYMM) Example
///
/// C = α·A·B + β·C where A is symmetric (only upper or lower triangle stored).
/// Real use case: covariance matrices, kernel matrices in ML.
///
/// Reference: CUDALibrarySamples/cuBLAS/Level-3/symm
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("=== cuBLAS SYMM Example ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();

    const stream = ctx.defaultStream();
    const blas = try cuda.cublas.CublasContext.init(ctx);
    defer blas.deinit();

    // Symmetric matrix A (3×3) and matrix B (3×2)
    const m: i32 = 3;
    const n: i32 = 2;

    // A is symmetric: A[i][j] = A[j][i]
    // A = | 1  2  3 |
    //     | 2  5  4 |
    //     | 3  4  6 |
    // Column-major, storing full matrix (cuBLAS reads from specified triangle)
    const A_data = [_]f32{ 1, 2, 3, 2, 5, 4, 3, 4, 6 };

    // B = | 1  4 |
    //     | 2  5 |
    //     | 3  6 |
    const B_data = [_]f32{ 1, 2, 3, 4, 5, 6 };

    std.debug.print("A (symmetric {}×{}):\n", .{ m, m });
    for (0..@intCast(m)) |r| {
        std.debug.print("  [", .{});
        for (0..@intCast(m)) |c| {
            std.debug.print(" {d:3.0}", .{A_data[c * @as(usize, @intCast(m)) + r]});
        }
        std.debug.print(" ]\n", .{});
    }

    std.debug.print("B ({}×{}):\n", .{ m, n });
    for (0..@intCast(m)) |r| {
        std.debug.print("  [", .{});
        for (0..@intCast(n)) |c| {
            std.debug.print(" {d:3.0}", .{B_data[c * @as(usize, @intCast(m)) + r]});
        }
        std.debug.print(" ]\n", .{});
    }

    const d_A = try stream.cloneHtod(f32, &A_data);
    defer d_A.deinit();
    const d_B = try stream.cloneHtod(f32, &B_data);
    defer d_B.deinit();
    const d_C = try stream.allocZeros(f32, allocator, @intCast(m * n));
    defer d_C.deinit();

    // C = 1.0 * A * B + 0.0 * C, A on the left, lower triangle
    try blas.ssymm(.left, .lower, m, n, 1.0, d_A, m, d_B, m, 0.0, d_C, m);

    var C: [6]f32 = undefined;
    try stream.memcpyDtoh(f32, &C, d_C);

    std.debug.print("\nC = A·B ({}×{}):\n", .{ m, n });
    for (0..@intCast(m)) |r| {
        std.debug.print("  [", .{});
        for (0..@intCast(n)) |c| {
            std.debug.print(" {d:6.0}", .{C[c * @as(usize, @intCast(m)) + r]});
        }
        std.debug.print(" ]\n", .{});
    }

    // Expected: C = A*B
    // Row 0: 1*1+2*2+3*3=14,  1*4+2*5+3*6=32
    // Row 1: 2*1+5*2+4*3=24,  2*4+5*5+4*6=57
    // Row 2: 3*1+4*2+6*3=29,  3*4+4*5+6*6=68
    const expected = [_]f32{ 14, 24, 29, 32, 57, 68 };
    std.debug.print("Expected:\n", .{});
    for (0..@intCast(m)) |r| {
        std.debug.print("  [", .{});
        for (0..@intCast(n)) |c| {
            std.debug.print(" {d:6.0}", .{expected[c * @as(usize, @intCast(m)) + r]});
        }
        std.debug.print(" ]\n", .{});
    }

    for (&expected, &C) |exp, actual| {
        if (@abs(exp - actual) > 1e-4) return error.ValidationFailed;
    }
    std.debug.print("\n✓ cuBLAS SYMM verified\n", .{});
}
