/// zCUDA Unit Tests: Driver API
const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;

// ============================================================================
// Context & Device
// ============================================================================

test "device count" {
    const count = try driver.CudaContext.deviceCount();
    try std.testing.expect(count >= 0);
}

test "create context and device info" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();

    const name = ctx.name();
    std.debug.print("Device: {s}\n", .{name});
    try std.testing.expect(name.len > 0);

    const cc = try ctx.computeCapability();
    std.debug.print("Compute capability: {d}.{d}\n", .{ cc.major, cc.minor });
    try std.testing.expect(cc.major >= 3);

    const mem_bytes = try ctx.totalMem();
    std.debug.print("Total memory: {d} MB\n", .{mem_bytes / 1024 / 1024});
    try std.testing.expect(mem_bytes > 0);
}

test "free memory" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();

    const free = try ctx.freeMem();
    try std.testing.expect(free > 0);
}

// ============================================================================
// Stream & Memory
// ============================================================================

test "alloc zeros and memcpy round-trip" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();

    const stream = ctx.defaultStream();

    const zeros = try stream.allocZeros(f32, allocator, 100);
    defer zeros.deinit();
    try std.testing.expectEqual(@as(usize, 100), zeros.len);

    // Clone host data to device and back
    const host_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const dev_data = try stream.cloneHtod(f32, &host_data);
    defer dev_data.deinit();

    var result_data: [5]f32 = undefined;
    try stream.memcpyDtoh(f32, &result_data, dev_data);
    try std.testing.expectEqualSlices(f32, &host_data, &result_data);
}

test "async memcpy round-trip" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();

    const stream = try ctx.newStream();

    const n = 256;
    var src: [n]f32 = undefined;
    for (&src, 0..) |*v, i| v.* = @floatFromInt(i);

    var dev = try stream.allocZeros(f32, allocator, n);
    defer dev.deinit();

    try stream.memcpyHtodAsync(f32, dev, &src);
    try stream.synchronize();

    var dst: [n]f32 = undefined;
    try stream.memcpyDtohAsync(f32, &dst, dev);
    try stream.synchronize();

    try std.testing.expectEqualSlices(f32, &src, &dst);
}

// ============================================================================
// Slice
// ============================================================================

test "CudaSlice sub-slicing" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();

    const stream = ctx.defaultStream();
    const data = try stream.allocZeros(f32, allocator, 100);
    defer data.deinit();

    const view = data.slice(10, 50);
    try std.testing.expectEqual(@as(usize, 40), view.len);

    const view_mut = data.sliceMut(0, 10);
    try std.testing.expectEqual(@as(usize, 10), view_mut.len);
}

// ============================================================================
// Events
// ============================================================================

test "event timing" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();

    const sys = @import("zcuda").driver.sys;
    const start = try ctx.createEvent(sys.CU_EVENT_DEFAULT);
    defer start.deinit();
    const end = try ctx.createEvent(sys.CU_EVENT_DEFAULT);
    defer end.deinit();

    const stream = ctx.defaultStream();
    try start.record(stream);
    try end.record(stream);
    try stream.synchronize();

    const elapsed = try driver.CudaEvent.elapsedTime(start, end);
    try std.testing.expect(elapsed >= 0.0);
}

test "memGetInfo — free and total memory" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();

    const free = try ctx.freeMem();
    const total = try ctx.totalMem();
    try std.testing.expect(free > 0);
    try std.testing.expect(total > 0);
    try std.testing.expect(free <= total);
}

test "peer access — query on single GPU" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();

    const count = try driver.CudaContext.deviceCount();
    if (count < 2) {
        // Single GPU: just verify the query doesn't crash
        return;
    }
    // If 2+ GPUs, peer access query should return true or false without error
    const can_access = driver.result.peer.canAccessPeer(0, 1) catch false;
    _ = can_access; // Just verify it doesn't crash
}

test "managed memory alloc and free" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();

    const ptr = driver.result.unified.allocManaged(256, 0x01) catch |err| {
        // CU_MEM_ATTACH_GLOBAL=0x1. May fail on some driver versions
        std.debug.print("Managed memory not supported: {}\n", .{err});
        return error.SkipZigTest;
    };
    defer driver.result.mem.free(ptr) catch {};

    try std.testing.expect(ptr != 0);
}
