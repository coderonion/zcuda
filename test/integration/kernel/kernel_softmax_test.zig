/// zCUDA Integration Test: Softmax Kernel Correctness
///
/// Tests softmax kernel properties on GPU:
/// - Output sums to 1.0 (probability distribution)
/// - All values in [0, 1]
/// - Numerically stable (handles large values)
/// Requires: zig build compile-kernels && zig build test-integration
const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;
const h = @import("test_helpers");

// ============================================================================
// Softmax properties
// ============================================================================

test "softmax — output sums to 1.0 for single row" {
    const allocator = std.testing.allocator;
    const env = try h.initCuda();
    defer env.ctx.deinit();

    const ptx = h.readPtxFile(allocator, "softmax") catch return error.SkipZigTest;
    defer allocator.free(ptx);

    const module = try env.ctx.loadModule(ptx);
    defer module.deinit();

    const func = try module.getFunction("softmax");
    const n: u32 = 256;

    var input: [256]f32 = undefined;
    for (0..n) |i| {
        input[i] = @sin(@as(f32, @floatFromInt(i)) * 0.7) * 3.0;
    }

    var d_input = try env.stream.cloneHtoD(f32, &input);
    defer d_input.deinit();
    var d_output = try env.stream.allocZeros(f32, allocator, n);
    defer d_output.deinit();

    const rows: u32 = 1;
    try env.stream.launch(func, .{
        .grid_dim = .{ .x = rows },
        .block_dim = .{ .x = 256 },
    }, .{ &d_input, &d_output, rows, n });

    var output: [256]f32 = undefined;
    try env.stream.memcpyDtoH(f32, &output, d_output);
    try env.stream.synchronize();

    var sum: f64 = 0;
    for (output) |v| sum += @as(f64, v);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), sum, 1e-3);

    for (output) |v| {
        try std.testing.expect(v >= 0.0);
        try std.testing.expect(v <= 1.0);
    }
}

test "softmax — numerically stable with large input values" {
    const allocator = std.testing.allocator;
    const env = try h.initCuda();
    defer env.ctx.deinit();

    const ptx = h.readPtxFile(allocator, "softmax") catch return error.SkipZigTest;
    defer allocator.free(ptx);

    const module = try env.ctx.loadModule(ptx);
    defer module.deinit();

    const func = try module.getFunction("softmax");
    const n: u32 = 256;

    var input: [256]f32 = undefined;
    for (0..n) |i| {
        input[i] = @as(f32, @floatFromInt(i)) + 500.0;
    }

    var d_input = try env.stream.cloneHtoD(f32, &input);
    defer d_input.deinit();
    var d_output = try env.stream.allocZeros(f32, allocator, n);
    defer d_output.deinit();

    const rows: u32 = 1;
    try env.stream.launch(func, .{
        .grid_dim = .{ .x = rows },
        .block_dim = .{ .x = 256 },
    }, .{ &d_input, &d_output, rows, n });

    var output: [256]f32 = undefined;
    try env.stream.memcpyDtoH(f32, &output, d_output);
    try env.stream.synchronize();

    for (output) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }

    var sum: f64 = 0;
    for (output) |v| sum += @as(f64, v);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), sum, 1e-3);
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
    const n: u32 = 256;

    var input: [256]f32 = undefined;
    @memset(&input, 5.0);

    var d_input = try env.stream.cloneHtoD(f32, &input);
    defer d_input.deinit();
    var d_output = try env.stream.allocZeros(f32, allocator, n);
    defer d_output.deinit();

    const rows: u32 = 1;
    try env.stream.launch(func, .{
        .grid_dim = .{ .x = rows },
        .block_dim = .{ .x = 256 },
    }, .{ &d_input, &d_output, rows, n });

    var output: [256]f32 = undefined;
    try env.stream.memcpyDtoH(f32, &output, d_output);
    try env.stream.synchronize();

    const expected: f32 = 1.0 / @as(f32, @floatFromInt(n));
    for (output) |v| {
        try std.testing.expectApproxEqAbs(expected, v, 1e-4);
    }
}

test "softmax — argmax is at the maximum input" {
    const allocator = std.testing.allocator;
    const env = try h.initCuda();
    defer env.ctx.deinit();

    const ptx = h.readPtxFile(allocator, "softmax") catch return error.SkipZigTest;
    defer allocator.free(ptx);

    const module = try env.ctx.loadModule(ptx);
    defer module.deinit();

    const func = try module.getFunction("softmax");
    const n: u32 = 128;

    var input: [128]f32 = undefined;
    @memset(&input, 0.0);
    input[42] = 100.0;

    var d_input = try env.stream.cloneHtoD(f32, &input);
    defer d_input.deinit();
    var d_output = try env.stream.allocZeros(f32, allocator, n);
    defer d_output.deinit();

    const rows: u32 = 1;
    try env.stream.launch(func, .{
        .grid_dim = .{ .x = rows },
        .block_dim = .{ .x = 256 },
    }, .{ &d_input, &d_output, rows, n });

    var output: [128]f32 = undefined;
    try env.stream.memcpyDtoH(f32, &output, d_output);
    try env.stream.synchronize();

    var max_idx: usize = 0;
    var max_val: f32 = -1.0;
    for (output, 0..) |v, i| {
        if (v > max_val) {
            max_val = v;
            max_idx = i;
        }
    }
    try std.testing.expectEqual(@as(usize, 42), max_idx);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), max_val, 1e-3);
}
