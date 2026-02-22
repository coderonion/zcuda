/// zCUDA Integration Test: Multi-Kernel Pipeline
///
/// Tests end-to-end GPU compute pipelines that chain multiple Zig kernels.
/// Each test loads PTX, prepares data, runs kernel(s), and validates results.
///
/// Run: zig build compile-kernels && zig build test-integration
const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;
const h = @import("test_helpers");

// ============================================================================
// Cross-validation: tiled_matmul vs matmulNaive
// ============================================================================

test "tiled_matmul matches matmulNaive on 32×32 matrix" {
    const allocator = std.testing.allocator;
    const env = try h.initCuda();
    defer env.ctx.deinit();

    const ptx_naive = h.readPtxFile(allocator, "matmul") catch return error.SkipZigTest;
    defer allocator.free(ptx_naive);
    const ptx_tiled = h.readPtxFile(allocator, "tiled_matmul") catch return error.SkipZigTest;
    defer allocator.free(ptx_tiled);

    const mod_naive = try env.ctx.loadModule(ptx_naive);
    defer mod_naive.deinit();
    const mod_tiled = try env.ctx.loadModule(ptx_tiled);
    defer mod_tiled.deinit();

    const func_naive = try mod_naive.getFunction("matmulNaive");
    const func_tiled = try mod_tiled.getFunction("tiled_matmul");

    const M: u32 = 32;
    const N: u32 = 32;
    const K: u32 = 32;
    const size = M * K;

    var a: [32 * 32]f32 = undefined;
    var b: [32 * 32]f32 = undefined;
    for (0..size) |i| {
        a[i] = @sin(@as(f32, @floatFromInt(i)) * 0.1);
        b[i] = @cos(@as(f32, @floatFromInt(i)) * 0.07);
    }

    var d_a = try env.stream.cloneHtoD(f32, &a);
    defer d_a.deinit();
    var d_b = try env.stream.cloneHtoD(f32, &b);
    defer d_b.deinit();
    var d_c_naive = try env.stream.allocZeros(f32, allocator, size);
    defer d_c_naive.deinit();
    var d_c_tiled = try env.stream.allocZeros(f32, allocator, size);
    defer d_c_tiled.deinit();

    // Run naive matmul
    try env.stream.launch(
        func_naive,
        .{ .grid_dim = .{ .x = 2, .y = 2 }, .block_dim = .{ .x = 16, .y = 16 } },
        .{ &d_a, &d_b, &d_c_naive, M, N, K },
    );

    // Run tiled matmul
    try env.stream.launch(
        func_tiled,
        .{ .grid_dim = .{ .x = 2, .y = 2 }, .block_dim = .{ .x = 16, .y = 16 } },
        .{ &d_a, &d_b, &d_c_tiled, M, N, K },
    );
    try env.ctx.synchronize();

    var result_naive: [32 * 32]f32 = undefined;
    var result_tiled: [32 * 32]f32 = undefined;
    try env.stream.memcpyDtoH(f32, &result_naive, d_c_naive);
    try env.stream.memcpyDtoH(f32, &result_tiled, d_c_tiled);

    for (0..size) |i| {
        try std.testing.expectApproxEqRel(result_naive[i], result_tiled[i], 1e-3);
    }
}

// ============================================================================
// Softmax numerical properties
// ============================================================================

test "softmax — output sums to 1.0 per row" {
    const allocator = std.testing.allocator;
    const env = try h.initCuda();
    defer env.ctx.deinit();

    const ptx = h.readPtxFile(allocator, "softmax") catch return error.SkipZigTest;
    defer allocator.free(ptx);

    const module = try env.ctx.loadModule(ptx);
    defer module.deinit();

    const func = try module.getFunction("softmax");

    const rows: u32 = 4;
    const cols: u32 = 64;
    const total = rows * cols;

    var input: [4 * 64]f32 = undefined;
    for (0..total) |i| {
        input[i] = @as(f32, @floatFromInt(i % 7)) - 3.0;
    }

    var d_in = try env.stream.cloneHtoD(f32, &input);
    defer d_in.deinit();
    var d_out = try env.stream.allocZeros(f32, allocator, total);
    defer d_out.deinit();

    try env.stream.launch(
        func,
        .{ .grid_dim = .{ .x = rows }, .block_dim = .{ .x = 256 } },
        .{ &d_in, &d_out, rows, cols },
    );
    try env.ctx.synchronize();

    var output: [4 * 64]f32 = undefined;
    try env.stream.memcpyDtoH(f32, &output, d_out);

    for (0..rows) |row| {
        var sum: f64 = 0.0;
        for (0..cols) |col| {
            const val = output[row * cols + col];
            try std.testing.expect(val >= 0.0);
            try std.testing.expect(val <= 1.0 + 1e-5);
            sum += val;
        }
        try std.testing.expectApproxEqAbs(@as(f64, 1.0), sum, 1e-3);
    }
}

test "softmax — uniform input produces uniform output" {
    const allocator = std.testing.allocator;
    const env = try h.initCuda();
    defer env.ctx.deinit();

    const ptx = h.readPtxFile(allocator, "softmax") catch return error.SkipZigTest;
    defer allocator.free(ptx);

    const module = try env.ctx.loadModule(ptx);
    defer module.deinit();

    const func = try module.getFunction("softmax");

    const rows: u32 = 1;
    const cols: u32 = 32;

    var input: [32]f32 = undefined;
    for (&input) |*v| v.* = 5.0;

    var d_in = try env.stream.cloneHtoD(f32, &input);
    defer d_in.deinit();
    var d_out = try env.stream.allocZeros(f32, allocator, cols);
    defer d_out.deinit();

    try env.stream.launch(
        func,
        .{ .grid_dim = .{ .x = 1 }, .block_dim = .{ .x = 256 } },
        .{ &d_in, &d_out, rows, cols },
    );
    try env.ctx.synchronize();

    var output: [32]f32 = undefined;
    try env.stream.memcpyDtoH(f32, &output, d_out);

    const expected: f32 = 1.0 / @as(f32, @floatFromInt(cols));
    for (output) |val| {
        try std.testing.expectApproxEqAbs(expected, val, 1e-4);
    }
}

// ============================================================================
// Multi-kernel pipeline: scale → add → reduce
// ============================================================================

test "pipeline: vectorScale → vectorAdd → reduceSum" {
    const allocator = std.testing.allocator;
    const env = try h.initCuda();
    defer env.ctx.deinit();

    const ptx_demo = h.readPtxFile(allocator, "grid_stride_demo") catch return error.SkipZigTest;
    defer allocator.free(ptx_demo);
    const ptx_vadd = h.readPtxFile(allocator, "vector_add") catch return error.SkipZigTest;
    defer allocator.free(ptx_vadd);
    const ptx_reduce = h.readPtxFile(allocator, "reduce_sum") catch return error.SkipZigTest;
    defer allocator.free(ptx_reduce);

    const mod_demo = try env.ctx.loadModule(ptx_demo);
    defer mod_demo.deinit();
    const mod_vadd = try env.ctx.loadModule(ptx_vadd);
    defer mod_vadd.deinit();
    const mod_reduce = try env.ctx.loadModule(ptx_reduce);
    defer mod_reduce.deinit();

    const fn_scale = try mod_demo.getFunction("vectorScale");
    const fn_add = try mod_vadd.getFunction("vectorAdd");
    const fn_sum = try mod_reduce.getFunction("reduceSum");

    const n: u32 = 256;

    var a: [256]f32 = undefined;
    var b: [256]f32 = undefined;
    for (0..n) |i| {
        a[i] = @as(f32, @floatFromInt(i + 1));
        b[i] = 1.0;
    }

    var d_a = try env.stream.cloneHtoD(f32, &a);
    defer d_a.deinit();
    var d_b = try env.stream.cloneHtoD(f32, &b);
    defer d_b.deinit();
    var d_c = try env.stream.allocZeros(f32, allocator, n);
    defer d_c.deinit();

    const grid_cfg = cuda.LaunchConfig{
        .grid_dim = .{ .x = 1 },
        .block_dim = .{ .x = 256 },
    };

    // Step 1: a *= 2
    const scale: f32 = 2.0;
    try env.stream.launch(fn_scale, grid_cfg, .{ &d_a, scale, n });

    // Step 2: c = a + b
    try env.stream.launch(fn_add, grid_cfg, .{ &d_a, &d_b, &d_c, n });

    // Step 3: sum(c)
    const zero = [_]f32{0.0};
    var d_result = try env.stream.cloneHtoD(f32, &zero);
    defer d_result.deinit();

    try env.stream.launch(fn_sum, grid_cfg, .{ &d_c, &d_result, n });
    try env.ctx.synchronize();

    var result: [1]f32 = undefined;
    try env.stream.memcpyDtoH(f32, &result, d_result);

    // c[i] = 2*(i+1) + 1 = 2i + 3 for i=0..255
    var expected: f64 = 0.0;
    for (0..n) |i| {
        expected += 2.0 * @as(f64, @floatFromInt(i + 1)) + 1.0;
    }
    try std.testing.expectApproxEqRel(@as(f32, @floatCast(expected)), result[0], 1e-3);
}

// ============================================================================
// dotProduct test
// ============================================================================

test "dotProduct — orthonormal vectors" {
    const allocator = std.testing.allocator;
    const env = try h.initCuda();
    defer env.ctx.deinit();

    const ptx = h.readPtxFile(allocator, "grid_stride_demo") catch return error.SkipZigTest;
    defer allocator.free(ptx);

    const module = try env.ctx.loadModule(ptx);
    defer module.deinit();

    const func = try module.getFunction("dotProduct");

    const n: u32 = 128;
    var a: [128]f32 = undefined;
    var b: [128]f32 = undefined;

    for (0..n) |i| {
        a[i] = if (i % 2 == 0) @as(f32, 1.0) else @as(f32, 0.0);
        b[i] = if (i % 2 == 0) @as(f32, 0.0) else @as(f32, 1.0);
    }

    var d_a = try env.stream.cloneHtoD(f32, &a);
    defer d_a.deinit();
    var d_b = try env.stream.cloneHtoD(f32, &b);
    defer d_b.deinit();

    const zero = [_]f32{0.0};
    var d_result = try env.stream.cloneHtoD(f32, &zero);
    defer d_result.deinit();

    try env.stream.launch(
        func,
        .{ .grid_dim = .{ .x = 1 }, .block_dim = .{ .x = 128 } },
        .{ &d_a, &d_b, &d_result, n },
    );
    try env.ctx.synchronize();

    var result: [1]f32 = undefined;
    try env.stream.memcpyDtoH(f32, &result, d_result);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[0], 1e-4);
}

test "dotProduct — self dot product = sum of squares" {
    const allocator = std.testing.allocator;
    const env = try h.initCuda();
    defer env.ctx.deinit();

    const ptx = h.readPtxFile(allocator, "grid_stride_demo") catch return error.SkipZigTest;
    defer allocator.free(ptx);

    const module = try env.ctx.loadModule(ptx);
    defer module.deinit();

    const func = try module.getFunction("dotProduct");

    const n: u32 = 64;
    var a: [64]f32 = undefined;
    var expected: f64 = 0.0;
    for (0..n) |i| {
        a[i] = @floatFromInt(i + 1);
        expected += @as(f64, a[i]) * @as(f64, a[i]);
    }

    var d_a = try env.stream.cloneHtoD(f32, &a);
    defer d_a.deinit();

    const zero = [_]f32{0.0};
    var d_result = try env.stream.cloneHtoD(f32, &zero);
    defer d_result.deinit();

    try env.stream.launch(
        func,
        .{ .grid_dim = .{ .x = 1 }, .block_dim = .{ .x = 64 } },
        .{ &d_a, &d_a, &d_result, n },
    );
    try env.ctx.synchronize();

    var result: [1]f32 = undefined;
    try env.stream.memcpyDtoH(f32, &result, d_result);

    try std.testing.expectApproxEqRel(@as(f32, @floatCast(expected)), result[0], 1e-3);
}
