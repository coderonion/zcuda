/// zCUDA Integration Test: Memory Lifecycle and Device Info
///
/// Tests GPU memory allocation, initialization, transfer, and cleanup.
/// Verifies memory lifecycle correctness: alloc→init→kernel→readback→verify→deinit.
/// Requires: zig build compile-kernels && zig build test-integration
const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;
const h = @import("test_helpers");

// ============================================================================
// allocZeros — initial values
// ============================================================================

test "mem — allocZeros produces all zeros" {
    const allocator = std.testing.allocator;
    const env = try h.initCuda();
    defer env.ctx.deinit();

    const n: u32 = 1024;
    var d_data = try env.stream.allocZeros(f32, allocator, n);
    defer d_data.deinit();

    var host: [1024]f32 = undefined;
    try env.stream.memcpyDtoH(f32, &host, d_data);
    try env.stream.synchronize();

    for (host) |v| {
        try std.testing.expectEqual(@as(f32, 0.0), v);
    }
}

// ============================================================================
// cloneHtoD — roundtrip correctness
// ============================================================================

test "mem — cloneHtoD then readback matches original" {
    const env = try h.initCuda();
    defer env.ctx.deinit();

    const n: u32 = 512;
    var original: [512]f32 = undefined;
    for (0..n) |i| {
        original[i] = @as(f32, @floatFromInt(i)) * 1.23 - 100.0;
    }

    var d_data = try env.stream.cloneHtoD(f32, &original);
    defer d_data.deinit();

    var readback: [512]f32 = undefined;
    try env.stream.memcpyDtoH(f32, &readback, d_data);
    try env.stream.synchronize();

    for (0..n) |i| {
        try std.testing.expectEqual(original[i], readback[i]);
    }
}

// ============================================================================
// Kernel writes then readback — full lifecycle
// ============================================================================

test "mem — kernel write → DtoH readback correctness" {
    const allocator = std.testing.allocator;
    const env = try h.initCuda();
    defer env.ctx.deinit();

    const ptx = h.readPtxFile(allocator, "vector_add") catch return error.SkipZigTest;
    defer allocator.free(ptx);

    const module = try env.ctx.loadModule(ptx);
    defer module.deinit();

    const func = try module.getFunction("vectorAdd");
    const n: u32 = 256;

    var a: [256]f32 = undefined;
    var b: [256]f32 = undefined;
    for (0..n) |i| {
        a[i] = @as(f32, @floatFromInt(i));
        b[i] = @as(f32, @floatFromInt(n - i));
    }

    var d_a = try env.stream.cloneHtoD(f32, &a);
    defer d_a.deinit();
    var d_b = try env.stream.cloneHtoD(f32, &b);
    defer d_b.deinit();
    var d_c = try env.stream.allocZeros(f32, allocator, n);
    defer d_c.deinit();

    try env.stream.launch(func, .{
        .grid_dim = .{ .x = 1 },
        .block_dim = .{ .x = n },
    }, .{ &d_a, &d_b, &d_c, n });

    var c: [256]f32 = undefined;
    try env.stream.memcpyDtoH(f32, &c, d_c);
    try env.stream.synchronize();

    for (0..n) |i| {
        try std.testing.expectEqual(a[i] + b[i], c[i]);
    }
}

// ============================================================================
// alloc/deinit — repeated lifecycle (leak check)
// ============================================================================

test "mem — alloc/deinit 100 times without leak" {
    const allocator = std.testing.allocator;
    const env = try h.initCuda();
    defer env.ctx.deinit();

    for (0..100) |_| {
        var d_data = try env.stream.allocZeros(f32, allocator, 256);
        d_data.deinit();
    }
}

// ============================================================================
// u32 data type — non-f32 memory operations
// ============================================================================

test "mem — u32 cloneHtoD/memcpyDtoH roundtrip" {
    const env = try h.initCuda();
    defer env.ctx.deinit();

    const n: u32 = 128;
    var original: [128]u32 = undefined;
    for (0..n) |i| {
        original[i] = @intCast(i * 0x01010101);
    }

    var d_data = try env.stream.cloneHtoD(u32, &original);
    defer d_data.deinit();

    var readback: [128]u32 = undefined;
    try env.stream.memcpyDtoH(u32, &readback, d_data);
    try env.stream.synchronize();

    for (0..n) |i| {
        try std.testing.expectEqual(original[i], readback[i]);
    }
}

// ============================================================================
// Device info
// ============================================================================

test "mem — computeCapability returns valid values" {
    const env = try h.initCuda();
    defer env.ctx.deinit();

    const cc = try env.ctx.computeCapability();
    try std.testing.expect(cc.major >= 3);
    try std.testing.expect(cc.minor >= 0);
}

test "mem — device name is non-empty" {
    const env = try h.initCuda();
    defer env.ctx.deinit();

    const name = env.ctx.name();
    try std.testing.expect(name.len > 0);
}

// ============================================================================
// Large allocation
// ============================================================================

test "mem — large allocation 1M floats roundtrip" {
    const allocator = std.testing.allocator;
    const env = try h.initCuda();
    defer env.ctx.deinit();

    const n: u32 = 1024 * 1024;
    const host_data = try allocator.alloc(f32, n);
    defer allocator.free(host_data);

    for (0..n) |i| {
        host_data[i] = @as(f32, @floatFromInt(i % 1000)) / 1000.0;
    }

    var d_data = try env.stream.cloneHtoD(f32, host_data);
    defer d_data.deinit();

    const readback = try allocator.alloc(f32, n);
    defer allocator.free(readback);
    try env.stream.memcpyDtoH(f32, readback, d_data);
    try env.stream.synchronize();

    for (0..n) |i| {
        try std.testing.expectEqual(host_data[i], readback[i]);
    }
}
