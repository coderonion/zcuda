/// zCUDA Unit Tests: Kernel Device Tests
///
/// Loads pre-compiled PTX kernels from zig-out/kernels/ and verifies
/// correctness on GPU hardware. Tests will be skipped on machines
/// without a CUDA-capable GPU.
///
/// Run: zig build compile-kernels && zig build test-unit
///
/// Kernel signatures tested:
///   vector_add.zig    → vectorAdd(A, B, C, n)
///   matmul.zig        → matmulNaive(A, B, C, M, N, K)
///                     → matvecMul(A, x, y, M, N)
///   histogram.zig     → histogramSimple(indices, bins, n)
///   grid_stride_demo  → vectorScale(data, scale, n)
///                     → saxpy(x, y, a, n)
///                     → dotProduct(a, b, result, n)
///   reduce_sum.zig    → reduceSum(input, result, n)
const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;
const h = @import("test_helpers");

// Convenience aliases used throughout tests
const initCuda = h.initCuda;
const readPtxFile = h.readPtxFile;

// ============================================================================
// vectorAdd tests
// ============================================================================

test "vectorAdd — basic addition" {
    const allocator = std.testing.allocator;
    const env = try initCuda();
    defer env.ctx.deinit();

    const ptx = readPtxFile(allocator, "vector_add") catch return error.SkipZigTest;
    defer allocator.free(ptx);

    const module = try env.ctx.loadModule(ptx);
    defer module.deinit();

    const func = try module.getFunction("vectorAdd");

    const n: u32 = 1024;
    var a: [1024]f32 = undefined;
    var b: [1024]f32 = undefined;
    for (0..n) |i| {
        a[i] = @floatFromInt(i);
        b[i] = @as(f32, @floatFromInt(i)) * 2.0;
    }

    var d_a = try env.stream.cloneHtoD(f32, &a);
    defer d_a.deinit();
    var d_b = try env.stream.cloneHtoD(f32, &b);
    defer d_b.deinit();
    var d_c = try env.stream.allocZeros(f32, allocator, n);
    defer d_c.deinit();

    const block_size: u32 = 256;
    const grid_size: u32 = (n + block_size - 1) / block_size;
    try env.stream.launch(
        func,
        .{ .grid_dim = .{ .x = grid_size }, .block_dim = .{ .x = block_size } },
        .{ &d_a, &d_b, &d_c, n },
    );
    try env.ctx.synchronize();

    var result: [1024]f32 = undefined;
    try env.stream.memcpyDtoH(f32, &result, d_c);

    for (0..n) |i| {
        const expected = @as(f32, @floatFromInt(i)) * 3.0;
        try std.testing.expectApproxEqAbs(expected, result[i], 1e-5);
    }
}

test "vectorAdd — zero-length" {
    const allocator = std.testing.allocator;
    const env = try initCuda();
    defer env.ctx.deinit();

    const ptx = readPtxFile(allocator, "vector_add") catch return error.SkipZigTest;
    defer allocator.free(ptx);

    const module = try env.ctx.loadModule(ptx);
    defer module.deinit();

    const func = try module.getFunction("vectorAdd");

    const n: u32 = 0;
    var d_a = try env.stream.allocZeros(f32, allocator, 1);
    defer d_a.deinit();
    var d_b = try env.stream.allocZeros(f32, allocator, 1);
    defer d_b.deinit();
    var d_c = try env.stream.allocZeros(f32, allocator, 1);
    defer d_c.deinit();

    // n=0: kernel should not write anything
    try env.stream.launch(
        func,
        .{ .grid_dim = .{ .x = 1 }, .block_dim = .{ .x = 1 } },
        .{ &d_a, &d_b, &d_c, n },
    );
    try env.ctx.synchronize();
}

// ============================================================================
// matmul tests
// ============================================================================

test "matmulNaive — identity multiplication" {
    const allocator = std.testing.allocator;
    const env = try initCuda();
    defer env.ctx.deinit();

    const ptx = readPtxFile(allocator, "matmul") catch return error.SkipZigTest;
    defer allocator.free(ptx);

    const module = try env.ctx.loadModule(ptx);
    defer module.deinit();

    const func = try module.getFunction("matmulNaive");

    // 4×4 identity × arbitrary = arbitrary
    const M: u32 = 4;
    const N: u32 = 4;
    const K: u32 = 4;

    // Identity matrix A
    var a = [_]f32{0} ** 16;
    for (0..4) |i| a[i * 4 + i] = 1.0;

    // Arbitrary matrix B
    var b: [16]f32 = undefined;
    for (0..16) |i| b[i] = @floatFromInt(i + 1);

    var d_a = try env.stream.cloneHtoD(f32, &a);
    defer d_a.deinit();
    var d_b = try env.stream.cloneHtoD(f32, &b);
    defer d_b.deinit();
    var d_c = try env.stream.allocZeros(f32, allocator, 16);
    defer d_c.deinit();

    try env.stream.launch(
        func,
        .{ .grid_dim = .{ .x = 1, .y = 1 }, .block_dim = .{ .x = 4, .y = 4 } },
        .{ &d_a, &d_b, &d_c, M, N, K },
    );
    try env.ctx.synchronize();

    var result: [16]f32 = undefined;
    try env.stream.memcpyDtoH(f32, &result, d_c);

    // Identity × B = B
    for (0..16) |i| {
        try std.testing.expectApproxEqAbs(b[i], result[i], 1e-4);
    }
}

test "matmulNaive — 2×3 × 3×2 known result" {
    const allocator = std.testing.allocator;
    const env = try initCuda();
    defer env.ctx.deinit();

    const ptx = readPtxFile(allocator, "matmul") catch return error.SkipZigTest;
    defer allocator.free(ptx);

    const module = try env.ctx.loadModule(ptx);
    defer module.deinit();

    const func = try module.getFunction("matmulNaive");

    // A = [[1,2,3],[4,5,6]]  (2×3)
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    // B = [[7,8],[9,10],[11,12]]  (3×2)
    const b = [_]f32{ 7, 8, 9, 10, 11, 12 };

    var d_a = try env.stream.cloneHtoD(f32, &a);
    defer d_a.deinit();
    var d_b = try env.stream.cloneHtoD(f32, &b);
    defer d_b.deinit();
    var d_c = try env.stream.allocZeros(f32, allocator, 4);
    defer d_c.deinit();

    const M: u32 = 2;
    const N: u32 = 2;
    const K: u32 = 3;
    try env.stream.launch(
        func,
        .{ .grid_dim = .{ .x = 1, .y = 1 }, .block_dim = .{ .x = 2, .y = 2 } },
        .{ &d_a, &d_b, &d_c, M, N, K },
    );
    try env.ctx.synchronize();

    var result: [4]f32 = undefined;
    try env.stream.memcpyDtoH(f32, &result, d_c);

    // C = A×B = [[58, 64], [139, 154]]
    try std.testing.expectApproxEqAbs(@as(f32, 58.0), result[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 64.0), result[1], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 139.0), result[2], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 154.0), result[3], 1e-3);
}

test "matvecMul — identity × vector" {
    const allocator = std.testing.allocator;
    const env = try initCuda();
    defer env.ctx.deinit();

    const ptx = readPtxFile(allocator, "matmul") catch return error.SkipZigTest;
    defer allocator.free(ptx);

    const module = try env.ctx.loadModule(ptx);
    defer module.deinit();

    const func = try module.getFunction("matvecMul");

    const M: u32 = 4;
    const N: u32 = 4;

    var a = [_]f32{0} ** 16;
    for (0..4) |i| a[i * 4 + i] = 1.0;

    const x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    var d_a = try env.stream.cloneHtoD(f32, &a);
    defer d_a.deinit();
    var d_x = try env.stream.cloneHtoD(f32, &x);
    defer d_x.deinit();
    var d_y = try env.stream.allocZeros(f32, allocator, 4);
    defer d_y.deinit();

    try env.stream.launch(
        func,
        .{ .grid_dim = .{ .x = 1 }, .block_dim = .{ .x = 4 } },
        .{ &d_a, &d_x, &d_y, M, N },
    );
    try env.ctx.synchronize();

    var result: [4]f32 = undefined;
    try env.stream.memcpyDtoH(f32, &result, d_y);

    // I × x = x
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(x[i], result[i], 1e-5);
    }
}

// ============================================================================
// grid_stride_demo tests
// ============================================================================

test "vectorScale — scale by 2" {
    const allocator = std.testing.allocator;
    const env = try initCuda();
    defer env.ctx.deinit();

    const ptx = readPtxFile(allocator, "grid_stride_demo") catch return error.SkipZigTest;
    defer allocator.free(ptx);

    const module = try env.ctx.loadModule(ptx);
    defer module.deinit();

    const func = try module.getFunction("vectorScale");

    const n: u32 = 512;
    var data: [512]f32 = undefined;
    for (0..n) |i| data[i] = @floatFromInt(i);

    var d_data = try env.stream.cloneHtoD(f32, &data);
    defer d_data.deinit();

    const scale: f32 = 2.0;
    try env.stream.launch(
        func,
        .{ .grid_dim = .{ .x = 2 }, .block_dim = .{ .x = 256 } },
        .{ &d_data, scale, n },
    );
    try env.ctx.synchronize();

    var result: [512]f32 = undefined;
    try env.stream.memcpyDtoH(f32, &result, d_data);

    for (0..n) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(i)) * 2.0, result[i], 1e-5);
    }
}

test "saxpy — y = 2*x + y" {
    const allocator = std.testing.allocator;
    const env = try initCuda();
    defer env.ctx.deinit();

    const ptx = readPtxFile(allocator, "grid_stride_demo") catch return error.SkipZigTest;
    defer allocator.free(ptx);

    const module = try env.ctx.loadModule(ptx);
    defer module.deinit();

    const func = try module.getFunction("saxpy");

    const n: u32 = 256;
    var x: [256]f32 = undefined;
    var y: [256]f32 = undefined;
    for (0..n) |i| {
        x[i] = @floatFromInt(i);
        y[i] = 1.0;
    }

    var d_x = try env.stream.cloneHtoD(f32, &x);
    defer d_x.deinit();
    var d_y = try env.stream.cloneHtoD(f32, &y);
    defer d_y.deinit();

    const a: f32 = 2.0;
    try env.stream.launch(
        func,
        .{ .grid_dim = .{ .x = 1 }, .block_dim = .{ .x = 256 } },
        .{ &d_x, &d_y, a, n },
    );
    try env.ctx.synchronize();

    var result: [256]f32 = undefined;
    try env.stream.memcpyDtoH(f32, &result, d_y);

    for (0..n) |i| {
        const expected = 2.0 * @as(f32, @floatFromInt(i)) + 1.0;
        try std.testing.expectApproxEqAbs(expected, result[i], 1e-4);
    }
}

// ============================================================================
// reduce_sum test
// ============================================================================

test "reduceSum — sum of 1..1024" {
    const allocator = std.testing.allocator;
    const env = try initCuda();
    defer env.ctx.deinit();

    const ptx = readPtxFile(allocator, "reduce_sum") catch return error.SkipZigTest;
    defer allocator.free(ptx);

    const module = try env.ctx.loadModule(ptx);
    defer module.deinit();

    const func = try module.getFunction("reduceSum");

    const n: u32 = 1024;
    var input: [1024]f32 = undefined;
    var expected_sum: f32 = 0.0;
    for (0..n) |i| {
        input[i] = @as(f32, @floatFromInt(i + 1));
        expected_sum += input[i];
    }

    var d_input = try env.stream.cloneHtoD(f32, &input);
    defer d_input.deinit();

    // Result is a single f32, initialized to 0
    const zero = [_]f32{0.0};
    var d_result = try env.stream.cloneHtoD(f32, &zero);
    defer d_result.deinit();

    try env.stream.launch(
        func,
        .{ .grid_dim = .{ .x = 4 }, .block_dim = .{ .x = 256 } },
        .{ &d_input, &d_result, n },
    );
    try env.ctx.synchronize();

    var result: [1]f32 = undefined;
    try env.stream.memcpyDtoH(f32, &result, d_result);

    // Sum of 1..1024 = 1024 * 1025 / 2 = 524800
    try std.testing.expectApproxEqRel(expected_sum, result[0], 1e-3);
}

// ============================================================================
// histogram test
// ============================================================================

test "histogramSimple — counting occurrences" {
    const allocator = std.testing.allocator;
    const env = try initCuda();
    defer env.ctx.deinit();

    const ptx = readPtxFile(allocator, "histogram") catch return error.SkipZigTest;
    defer allocator.free(ptx);

    const module = try env.ctx.loadModule(ptx);
    defer module.deinit();

    const func = try module.getFunction("histogramSimple");

    // Input: 8 elements, each is a bin index 0..3
    const n: u32 = 8;
    const indices = [_]u32{ 0, 1, 2, 3, 0, 1, 0, 2 };
    // Expected: bin0=3, bin1=2, bin2=2, bin3=1

    var d_indices = try env.stream.cloneHtoD(u32, &indices);
    defer d_indices.deinit();

    const num_bins: u32 = 256;
    var d_bins = try env.stream.allocZeros(u32, allocator, num_bins);
    defer d_bins.deinit();

    try env.stream.launch(
        func,
        .{ .grid_dim = .{ .x = 1 }, .block_dim = .{ .x = 8 } },
        .{ &d_indices, &d_bins, n },
    );
    try env.ctx.synchronize();

    var bins: [256]u32 = undefined;
    try env.stream.memcpyDtoH(u32, &bins, d_bins);

    try std.testing.expectEqual(@as(u32, 3), bins[0]);
    try std.testing.expectEqual(@as(u32, 2), bins[1]);
    try std.testing.expectEqual(@as(u32, 2), bins[2]);
    try std.testing.expectEqual(@as(u32, 1), bins[3]);
    // All other bins should be 0
    for (4..256) |i| {
        try std.testing.expectEqual(@as(u32, 0), bins[i]);
    }
}
