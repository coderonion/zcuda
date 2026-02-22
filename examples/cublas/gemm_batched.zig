/// cuBLAS Strided Batched GEMM Example
///
/// Performs multiple independent matrix multiplications in a single call.
/// All matrices are contiguously stored with stride offsets.
///
/// Reference: CUDALibrarySamples/cuBLAS/Level-3/gemmStridedBatched
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("=== cuBLAS Strided Batched GEMM Example ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();

    const stream = ctx.defaultStream();
    const blas = try cuda.cublas.CublasContext.init(ctx);
    defer blas.deinit();

    // 3 batched matrix multiplies: each Cᵢ (2×2) = Aᵢ (2×3) × Bᵢ (3×2)
    const m: i32 = 2;
    const n: i32 = 2;
    const k: i32 = 3;
    const batch: i32 = 3;

    const stride_a: i64 = m * k; // 6 elements per batch
    const stride_b: i64 = k * n; // 6 elements per batch
    const stride_c: i64 = m * n; // 4 elements per batch

    // Column-major storage for 3 batches
    // Batch 0: A = [[1,2,3],[4,5,6]], B = [[1,2],[3,4],[5,6]]
    // Batch 1: A = identity-like, B = scaled
    // Batch 2: A = random, B = random
    var A: [18]f32 = undefined;
    var B: [18]f32 = undefined;
    var C: [12]f32 = undefined;

    var rng = std.Random.DefaultPrng.init(42);
    const random = rng.random();
    for (&A) |*v| v.* = @as(f32, @floatFromInt(random.intRangeAtMost(i32, 1, 5)));
    for (&B) |*v| v.* = @as(f32, @floatFromInt(random.intRangeAtMost(i32, 1, 5)));
    @memset(&C, 0.0);

    // Copy to device
    const d_A = try stream.cloneHtoD(f32, &A);
    defer d_A.deinit();
    const d_B = try stream.cloneHtoD(f32, &B);
    defer d_B.deinit();
    const d_C = try stream.allocZeros(f32, allocator, @intCast(m * n * batch));
    defer d_C.deinit();

    // Execute batched GEMM
    try blas.sgemmStridedBatched(
        .no_transpose,
        .no_transpose,
        m,
        n,
        k,
        1.0,
        d_A,
        m,
        stride_a,
        d_B,
        k,
        stride_b,
        0.0,
        d_C,
        m,
        stride_c,
        batch,
    );

    try stream.memcpyDtoH(f32, &C, d_C);

    // Print and verify results
    for (0..@intCast(batch)) |b| {
        std.debug.print("─── Batch {} ───\n", .{b});
        std.debug.print("  C = A × B:\n", .{});
        for (0..@intCast(m)) |r| {
            std.debug.print("  [", .{});
            for (0..@intCast(n)) |c| {
                const idx = b * @as(usize, @intCast(stride_c)) + c * @as(usize, @intCast(m)) + r;
                std.debug.print(" {d:6.0}", .{C[idx]});
            }
            std.debug.print(" ]\n", .{});
        }
    }

    // Verify batch 0
    var max_error: f32 = 0.0;
    for (0..@intCast(batch)) |b| {
        for (0..@intCast(m)) |r| {
            for (0..@intCast(n)) |c| {
                var expected: f32 = 0.0;
                for (0..@intCast(k)) |p| {
                    const a_idx = b * @as(usize, @intCast(stride_a)) + p * @as(usize, @intCast(m)) + r;
                    const b_idx = b * @as(usize, @intCast(stride_b)) + c * @as(usize, @intCast(k)) + p;
                    expected += A[a_idx] * B[b_idx];
                }
                const c_idx = b * @as(usize, @intCast(stride_c)) + c * @as(usize, @intCast(m)) + r;
                max_error = @max(max_error, @abs(expected - C[c_idx]));
            }
        }
    }

    std.debug.print("\nMax error across {} batches: {e}\n", .{ batch, max_error });
    if (max_error > 1e-4) return error.ValidationFailed;
    std.debug.print("✓ cuBLAS Strided Batched GEMM verified\n", .{});
}
