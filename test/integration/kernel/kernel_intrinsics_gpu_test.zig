/// zCUDA Integration Test: GPU Intrinsics Correctness
///
/// Tests that device intrinsics produce mathematically correct results on GPU.
/// Requires: zig build compile-kernels && zig build test-integration
const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;
const h = @import("test_helpers");

// ============================================================================
// Intrinsics Test Kernel: math_test
// ============================================================================

test "fast math — sinf/cosf identity (sin²x + cos²x = 1)" {
    const allocator = std.testing.allocator;
    const env = try h.initCuda();
    defer env.ctx.deinit();

    const ptx = h.readPtxFile(allocator, "math_test") catch return error.SkipZigTest;
    defer allocator.free(ptx);

    const module = try env.ctx.loadModule(ptx);
    defer module.deinit();

    const func = try module.getFunction("sincos_identity");
    const n: u32 = 1024;

    var input: [1024]f32 = undefined;
    for (0..n) |i| {
        input[i] = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n)) * 2.0 * std.math.pi;
    }

    var d_input = try env.stream.cloneHtoD(f32, &input);
    defer d_input.deinit();
    var d_output = try env.stream.allocZeros(f32, allocator, n);
    defer d_output.deinit();

    try env.stream.launch(func, .{
        .grid_dim = .{ .x = (n + 255) / 256 },
        .block_dim = .{ .x = 256 },
    }, .{ &d_input, &d_output, n });

    var output: [1024]f32 = undefined;
    try env.stream.memcpyDtoH(f32, &output, d_output);
    try env.stream.synchronize();

    for (0..n) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[i], 1e-4);
    }
}

test "fast math — expf/logf roundtrip (log(exp(x)) ≈ x)" {
    const allocator = std.testing.allocator;
    const env = try h.initCuda();
    defer env.ctx.deinit();

    const ptx = h.readPtxFile(allocator, "math_test") catch return error.SkipZigTest;
    defer allocator.free(ptx);

    const module = try env.ctx.loadModule(ptx);
    defer module.deinit();

    const func = try module.getFunction("explog_roundtrip");
    const n: u32 = 256;

    var input: [256]f32 = undefined;
    for (0..n) |i| {
        input[i] = @as(f32, @floatFromInt(i)) / 25.0 - 5.0;
    }

    var d_input = try env.stream.cloneHtoD(f32, &input);
    defer d_input.deinit();
    var d_output = try env.stream.allocZeros(f32, allocator, n);
    defer d_output.deinit();

    try env.stream.launch(func, .{
        .grid_dim = .{ .x = 1 },
        .block_dim = .{ .x = 256 },
    }, .{ &d_input, &d_output, n });

    var output: [256]f32 = undefined;
    try env.stream.memcpyDtoH(f32, &output, d_output);
    try env.stream.synchronize();

    for (0..n) |i| {
        if (input[i] > -80 and input[i] < 80) {
            try std.testing.expectApproxEqAbs(input[i], output[i], 1e-3);
        }
    }
}

// ============================================================================
// Atomics: parallel sum
// ============================================================================

test "atomicAdd — 1024 threads each add 1.0, result = 1024.0" {
    const allocator = std.testing.allocator;
    const env = try h.initCuda();
    defer env.ctx.deinit();

    const ptx = h.readPtxFile(allocator, "math_test") catch return error.SkipZigTest;
    defer allocator.free(ptx);

    const module = try env.ctx.loadModule(ptx);
    defer module.deinit();

    const func = try module.getFunction("atomic_sum");
    const n: u32 = 1024;

    var d_output = try env.stream.allocZeros(f32, allocator, 1);
    defer d_output.deinit();

    try env.stream.launch(func, .{
        .grid_dim = .{ .x = 1 },
        .block_dim = .{ .x = n },
    }, .{ &d_output, n });

    var result: [1]f32 = undefined;
    try env.stream.memcpyDtoH(f32, &result, d_output);
    try env.stream.synchronize();

    try std.testing.expectApproxEqAbs(@as(f32, 1024.0), result[0], 1e-2);
}

// ============================================================================
// Integer intrinsics: clz, popc, brev
// ============================================================================

test "integer intrinsics — clz correctness" {
    const allocator = std.testing.allocator;
    const env = try h.initCuda();
    defer env.ctx.deinit();

    const ptx = h.readPtxFile(allocator, "math_test") catch return error.SkipZigTest;
    defer allocator.free(ptx);

    const module = try env.ctx.loadModule(ptx);
    defer module.deinit();

    const func = try module.getFunction("test_clz");

    const inputs = [_]u32{ 1, 2, 0x80000000, 0xFF, 0x10000 };
    const expected = [_]u32{ 31, 30, 0, 24, 15 };
    const n: u32 = inputs.len;

    var d_input = try env.stream.cloneHtoD(u32, &inputs);
    defer d_input.deinit();
    var d_output = try env.stream.allocZeros(u32, allocator, n);
    defer d_output.deinit();

    try env.stream.launch(func, .{
        .grid_dim = .{ .x = 1 },
        .block_dim = .{ .x = n },
    }, .{ &d_input, &d_output, n });

    var output: [5]u32 = undefined;
    try env.stream.memcpyDtoH(u32, &output, d_output);
    try env.stream.synchronize();

    for (0..n) |i| {
        try std.testing.expectEqual(expected[i], output[i]);
    }
}

test "integer intrinsics — popc (popcount) correctness" {
    const allocator = std.testing.allocator;
    const env = try h.initCuda();
    defer env.ctx.deinit();

    const ptx = h.readPtxFile(allocator, "math_test") catch return error.SkipZigTest;
    defer allocator.free(ptx);

    const module = try env.ctx.loadModule(ptx);
    defer module.deinit();

    const func = try module.getFunction("test_popc");

    const inputs = [_]u32{ 0, 1, 0xFF, 0xFFFF, 0xFFFFFFFF };
    const expected = [_]u32{ 0, 1, 8, 16, 32 };
    const n: u32 = inputs.len;

    var d_input = try env.stream.cloneHtoD(u32, &inputs);
    defer d_input.deinit();
    var d_output = try env.stream.allocZeros(u32, allocator, n);
    defer d_output.deinit();

    try env.stream.launch(func, .{
        .grid_dim = .{ .x = 1 },
        .block_dim = .{ .x = n },
    }, .{ &d_input, &d_output, n });

    var output: [5]u32 = undefined;
    try env.stream.memcpyDtoH(u32, &output, d_output);
    try env.stream.synchronize();

    for (0..n) |i| {
        try std.testing.expectEqual(expected[i], output[i]);
    }
}

test "integer intrinsics — brev (bit reverse) involution: brev(brev(x)) == x" {
    const allocator = std.testing.allocator;
    const env = try h.initCuda();
    defer env.ctx.deinit();

    const ptx = h.readPtxFile(allocator, "math_test") catch return error.SkipZigTest;
    defer allocator.free(ptx);

    const module = try env.ctx.loadModule(ptx);
    defer module.deinit();

    const func = try module.getFunction("test_brev_roundtrip");

    const inputs = [_]u32{ 0, 1, 0xDEADBEEF, 0x12345678, 0xFFFFFFFF };
    const n: u32 = inputs.len;

    var d_input = try env.stream.cloneHtoD(u32, &inputs);
    defer d_input.deinit();
    var d_output = try env.stream.allocZeros(u32, allocator, n);
    defer d_output.deinit();

    try env.stream.launch(func, .{
        .grid_dim = .{ .x = 1 },
        .block_dim = .{ .x = n },
    }, .{ &d_input, &d_output, n });

    var output: [5]u32 = undefined;
    try env.stream.memcpyDtoH(u32, &output, d_output);
    try env.stream.synchronize();

    for (0..n) |i| {
        try std.testing.expectEqual(inputs[i], output[i]);
    }
}
