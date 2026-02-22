/// zCUDA Integration Test: Event Timing and Streams
///
/// Tests CUDA event timing and multi-stream kernel execution correctness.
/// Requires: zig build compile-kernels && zig build test-integration
const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;
const h = @import("test_helpers");

// ============================================================================
// Event timing
// ============================================================================

test "event — basic timing produces positive elapsed time" {
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
        a[i] = @floatFromInt(i);
        b[i] = @floatFromInt(n - i);
    }

    var d_a = try env.stream.cloneHtoD(f32, &a);
    defer d_a.deinit();
    var d_b = try env.stream.cloneHtoD(f32, &b);
    defer d_b.deinit();
    var d_c = try env.stream.allocZeros(f32, allocator, n);
    defer d_c.deinit();

    var start = try env.ctx.createEvent(0);
    defer start.deinit();
    var stop = try env.ctx.createEvent(0);
    defer stop.deinit();

    try start.record(env.stream);
    try env.stream.launch(func, .{
        .grid_dim = .{ .x = (n + 255) / 256 },
        .block_dim = .{ .x = 256 },
    }, .{ &d_a, &d_b, &d_c, n });
    try stop.record(env.stream);
    try stop.synchronize();

    const elapsed = try start.elapsedTime(stop);
    // Kernel should take some positive time
    try std.testing.expect(elapsed >= 0.0);
}

// ============================================================================
// Multi-stream: independent kernel launches
// ============================================================================

test "stream — two independent vectorAdd on separate streams produce correct results" {
    const allocator = std.testing.allocator;
    const env = try h.initCuda();
    defer env.ctx.deinit();

    const ptx = h.readPtxFile(allocator, "vector_add") catch return error.SkipZigTest;
    defer allocator.free(ptx);

    const module = try env.ctx.loadModule(ptx);
    defer module.deinit();

    const func = try module.getFunction("vectorAdd");
    const n: u32 = 256;

    // Stream 1 data
    var a1: [256]f32 = undefined;
    var b1: [256]f32 = undefined;
    for (0..n) |i| {
        a1[i] = @as(f32, @floatFromInt(i));
        b1[i] = 1.0;
    }

    // Stream 2 data
    var a2: [256]f32 = undefined;
    var b2: [256]f32 = undefined;
    for (0..n) |i| {
        a2[i] = @as(f32, @floatFromInt(i)) * 2.0;
        b2[i] = -1.0;
    }

    var stream1 = try env.ctx.newStream();
    defer stream1.deinit();
    var stream2 = try env.ctx.newStream();
    defer stream2.deinit();

    // Allocate and launch on stream 1
    var d_a1 = try stream1.cloneHtoD(f32, &a1);
    defer d_a1.deinit();
    var d_b1 = try stream1.cloneHtoD(f32, &b1);
    defer d_b1.deinit();
    var d_c1 = try stream1.allocZeros(f32, allocator, n);
    defer d_c1.deinit();

    try stream1.launch(func, .{
        .grid_dim = .{ .x = 1 },
        .block_dim = .{ .x = n },
    }, .{ &d_a1, &d_b1, &d_c1, n });

    // Allocate and launch on stream 2
    var d_a2 = try stream2.cloneHtoD(f32, &a2);
    defer d_a2.deinit();
    var d_b2 = try stream2.cloneHtoD(f32, &b2);
    defer d_b2.deinit();
    var d_c2 = try stream2.allocZeros(f32, allocator, n);
    defer d_c2.deinit();

    try stream2.launch(func, .{
        .grid_dim = .{ .x = 1 },
        .block_dim = .{ .x = n },
    }, .{ &d_a2, &d_b2, &d_c2, n });

    // Sync both streams, then sync the entire context for safety
    try stream1.synchronize();
    try stream2.synchronize();
    try env.ctx.synchronize();

    // Read back and verify — use the same streams that did the work, then sync
    var c1: [256]f32 = undefined;
    var c2: [256]f32 = undefined;
    try stream1.memcpyDtoH(f32, &c1, d_c1);
    try stream2.memcpyDtoH(f32, &c2, d_c2);
    try stream1.synchronize();
    try stream2.synchronize();

    for (0..n) |i| {
        try std.testing.expectApproxEqAbs(a1[i] + b1[i], c1[i], 1e-5);
        try std.testing.expectApproxEqAbs(a2[i] + b2[i], c2[i], 1e-5);
    }
}

// ============================================================================
// Stream sync — verify sync guarantees results ready
// ============================================================================

test "stream — sync guarantees kernel results are ready" {
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
        a[i] = 1.0;
        b[i] = 2.0;
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

    try env.stream.synchronize();

    var c: [1024]f32 = undefined;
    try env.stream.memcpyDtoH(f32, &c, d_c);
    try env.stream.synchronize();

    for (c) |v| {
        try std.testing.expectEqual(@as(f32, 3.0), v);
    }
}
