/// zCUDA Integration Test: Reduction Kernels
///
/// Tests reduction algorithms (sum, min, max) on GPU with correctness
/// verified against CPU reference computations.
/// Requires: zig build compile-kernels && zig build test-integration
const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;
const h = @import("test_helpers");

fn cpuSum(data: []const f32) f64 {
    var total: f64 = 0;
    for (data) |v| total += @as(f64, v);
    return total;
}

// ============================================================================
// reduceSum — warp shuffle based
// ============================================================================

test "reduce — warp shuffle sum of 1024 sequential values" {
    const allocator = std.testing.allocator;
    const env = try h.initCuda();
    defer env.ctx.deinit();

    const ptx = h.readPtxFile(allocator, "reduce_sum") catch return error.SkipZigTest;
    defer allocator.free(ptx);

    const module = try env.ctx.loadModule(ptx);
    defer module.deinit();

    const func = try module.getFunction("reduceSum");
    const n: u32 = 1024;

    var input: [1024]f32 = undefined;
    for (0..n) |i| {
        input[i] = @as(f32, @floatFromInt(i + 1));
    }
    const expected_sum = cpuSum(&input);

    var d_input = try env.stream.cloneHtoD(f32, &input);
    defer d_input.deinit();
    var d_output = try env.stream.allocZeros(f32, allocator, 1);
    defer d_output.deinit();

    try env.stream.launch(func, .{
        .grid_dim = .{ .x = 1 },
        .block_dim = .{ .x = 256 },
    }, .{ &d_input, &d_output, n });

    var result: [1]f32 = undefined;
    try env.stream.memcpyDtoH(f32, &result, d_output);
    try env.stream.synchronize();

    try std.testing.expectApproxEqRel(
        @as(f32, @floatCast(expected_sum)),
        result[0],
        1e-4,
    );
}

test "reduce — sum of all-ones array equals N" {
    const allocator = std.testing.allocator;
    const env = try h.initCuda();
    defer env.ctx.deinit();

    const ptx = h.readPtxFile(allocator, "reduce_sum") catch return error.SkipZigTest;
    defer allocator.free(ptx);

    const module = try env.ctx.loadModule(ptx);
    defer module.deinit();

    const func = try module.getFunction("reduceSum");
    const n: u32 = 512;

    var input: [512]f32 = undefined;
    @memset(&input, 1.0);

    var d_input = try env.stream.cloneHtoD(f32, &input);
    defer d_input.deinit();
    var d_output = try env.stream.allocZeros(f32, allocator, 1);
    defer d_output.deinit();

    try env.stream.launch(func, .{
        .grid_dim = .{ .x = 1 },
        .block_dim = .{ .x = 256 },
    }, .{ &d_input, &d_output, n });

    var result: [1]f32 = undefined;
    try env.stream.memcpyDtoH(f32, &result, d_output);
    try env.stream.synchronize();

    try std.testing.expectApproxEqAbs(@as(f32, 512.0), result[0], 1e-2);
}

test "reduce — sum of all-zeros is zero" {
    const allocator = std.testing.allocator;
    const env = try h.initCuda();
    defer env.ctx.deinit();

    const ptx = h.readPtxFile(allocator, "reduce_sum") catch return error.SkipZigTest;
    defer allocator.free(ptx);

    const module = try env.ctx.loadModule(ptx);
    defer module.deinit();

    const func = try module.getFunction("reduceSum");
    const n: u32 = 256;

    var input: [256]f32 = undefined;
    @memset(&input, 0.0);

    var d_input = try env.stream.cloneHtoD(f32, &input);
    defer d_input.deinit();
    var d_output = try env.stream.allocZeros(f32, allocator, 1);
    defer d_output.deinit();

    try env.stream.launch(func, .{
        .grid_dim = .{ .x = 1 },
        .block_dim = .{ .x = 256 },
    }, .{ &d_input, &d_output, n });

    var result: [1]f32 = undefined;
    try env.stream.memcpyDtoH(f32, &result, d_output);
    try env.stream.synchronize();

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[0], 1e-6);
}

test "reduce — sum with negative values cancellation" {
    const allocator = std.testing.allocator;
    const env = try h.initCuda();
    defer env.ctx.deinit();

    const ptx = h.readPtxFile(allocator, "reduce_sum") catch return error.SkipZigTest;
    defer allocator.free(ptx);

    const module = try env.ctx.loadModule(ptx);
    defer module.deinit();

    const func = try module.getFunction("reduceSum");
    const n: u32 = 256;

    var input: [256]f32 = undefined;
    for (0..n) |i| {
        input[i] = if (i % 2 == 0) @as(f32, 1.0) else @as(f32, -1.0);
    }

    var d_input = try env.stream.cloneHtoD(f32, &input);
    defer d_input.deinit();
    var d_output = try env.stream.allocZeros(f32, allocator, 1);
    defer d_output.deinit();

    try env.stream.launch(func, .{
        .grid_dim = .{ .x = 1 },
        .block_dim = .{ .x = 256 },
    }, .{ &d_input, &d_output, n });

    var result: [1]f32 = undefined;
    try env.stream.memcpyDtoH(f32, &result, d_output);
    try env.stream.synchronize();

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[0], 1e-4);
}

// ============================================================================
// vectorAdd — basic element-wise correctness
// ============================================================================

test "vectorAdd — C[i] = A[i] + B[i] for 1024 elements" {
    const allocator = std.testing.allocator;
    const env = try h.initCuda();
    defer env.ctx.deinit();

    const ptx = h.readPtxFile(allocator, "vector_add") catch return error.SkipZigTest;
    defer allocator.free(ptx);

    const module = try env.ctx.loadModule(ptx);
    defer module.deinit();

    const func = try module.getFunction("vectorAdd");
    const n: u32 = 1024;

    var a: [1024]f32 = undefined;
    var b: [1024]f32 = undefined;
    for (0..n) |i| {
        a[i] = @as(f32, @floatFromInt(i)) * 0.1;
        b[i] = @as(f32, @floatFromInt(n - i)) * 0.2;
    }

    var d_a = try env.stream.cloneHtoD(f32, &a);
    defer d_a.deinit();
    var d_b = try env.stream.cloneHtoD(f32, &b);
    defer d_b.deinit();
    var d_c = try env.stream.allocZeros(f32, allocator, n);
    defer d_c.deinit();

    try env.stream.launch(func, .{
        .grid_dim = .{ .x = (n + 255) / 256 },
        .block_dim = .{ .x = 256 },
    }, .{ &d_a, &d_b, &d_c, n });

    var c: [1024]f32 = undefined;
    try env.stream.memcpyDtoH(f32, &c, d_c);
    try env.stream.synchronize();

    for (0..n) |i| {
        const expected = a[i] + b[i];
        try std.testing.expectApproxEqAbs(expected, c[i], 1e-5);
    }
}

// ============================================================================
// matmulNaive — matrix multiply correctness
// ============================================================================

test "matmul — 16×16 naive matmul matches CPU reference" {
    const allocator = std.testing.allocator;
    const env = try h.initCuda();
    defer env.ctx.deinit();

    const ptx = h.readPtxFile(allocator, "matmul") catch return error.SkipZigTest;
    defer allocator.free(ptx);

    const module = try env.ctx.loadModule(ptx);
    defer module.deinit();

    const func = try module.getFunction("matmulNaive");
    const M: u32 = 16;

    var a: [16 * 16]f32 = undefined;
    var b: [16 * 16]f32 = undefined;
    for (0..M * M) |i| {
        a[i] = @sin(@as(f32, @floatFromInt(i)) * 0.1);
        b[i] = @cos(@as(f32, @floatFromInt(i)) * 0.07);
    }

    // CPU reference
    var cpu_c: [16 * 16]f32 = [_]f32{0.0} ** (16 * 16);
    for (0..M) |row| {
        for (0..M) |col| {
            var sum: f64 = 0;
            for (0..M) |k| {
                sum += @as(f64, a[row * M + k]) * @as(f64, b[k * M + col]);
            }
            cpu_c[row * M + col] = @floatCast(sum);
        }
    }

    var d_a = try env.stream.cloneHtoD(f32, &a);
    defer d_a.deinit();
    var d_b = try env.stream.cloneHtoD(f32, &b);
    defer d_b.deinit();
    var d_c = try env.stream.allocZeros(f32, allocator, M * M);
    defer d_c.deinit();

    try env.stream.launch(func, .{
        .grid_dim = .{ .x = 1, .y = 1 },
        .block_dim = .{ .x = M, .y = M },
    }, .{ &d_a, &d_b, &d_c, M, M, M });

    var gpu_c: [16 * 16]f32 = undefined;
    try env.stream.memcpyDtoH(f32, &gpu_c, d_c);
    try env.stream.synchronize();

    for (0..M * M) |i| {
        try std.testing.expectApproxEqAbs(cpu_c[i], gpu_c[i], 1e-3);
    }
}

// ============================================================================
// histogram — atomic histogram correctness
// ============================================================================

test "histogram — 256-bin histogram: sum of all bins equals N" {
    const allocator = std.testing.allocator;
    const env = try h.initCuda();
    defer env.ctx.deinit();

    const ptx = h.readPtxFile(allocator, "histogram") catch return error.SkipZigTest;
    defer allocator.free(ptx);

    const module = try env.ctx.loadModule(ptx);
    defer module.deinit();

    const func = try module.getFunction("histogramSimple");
    const n: u32 = 4096;
    const n_bins: u32 = 256;

    var input: [4096]u32 = undefined;
    for (0..n) |i| {
        input[i] = @intCast(i % n_bins);
    }

    var d_input = try env.stream.cloneHtoD(u32, &input);
    defer d_input.deinit();
    var d_bins = try env.stream.allocZeros(u32, allocator, n_bins);
    defer d_bins.deinit();

    try env.stream.launch(func, .{
        .grid_dim = .{ .x = (n + 255) / 256 },
        .block_dim = .{ .x = 256 },
    }, .{ &d_input, &d_bins, n, n_bins });

    var bins: [256]u32 = undefined;
    try env.stream.memcpyDtoH(u32, &bins, d_bins);
    try env.stream.synchronize();

    var total: u64 = 0;
    for (bins) |count| total += count;
    try std.testing.expectEqual(@as(u64, n), total);

    for (bins) |count| {
        try std.testing.expectEqual(@as(u32, n / n_bins), count);
    }
}
