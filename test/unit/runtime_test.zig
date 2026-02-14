/// zCUDA Unit Tests: Runtime API
const std = @import("std");
const cuda = @import("zcuda");
const RuntimeContext = cuda.runtime.RuntimeContext;

test "Runtime device count" {
    const count = RuntimeContext.deviceCount() catch return error.SkipZigTest;
    try std.testing.expect(count > 0);
}

test "Runtime context creation" {
    const ctx = RuntimeContext.init(0) catch return error.SkipZigTest;
    try ctx.synchronize();
}

test "Runtime typed allocation and memcpy" {
    const ctx = RuntimeContext.init(0) catch return error.SkipZigTest;

    var slice = try ctx.alloc(f32, 64);
    defer slice.deinit();

    const host_data = [_]f32{1.0} ** 64;
    try ctx.copyToDevice(f32, slice, &host_data);

    var result_data: [64]f32 = undefined;
    try ctx.copyToHost(f32, &result_data, slice);
    try std.testing.expectApproxEqAbs(result_data[0], 1.0, 1e-6);
    try std.testing.expectApproxEqAbs(result_data[63], 1.0, 1e-6);
}

test "Runtime stream and event" {
    const ctx = RuntimeContext.init(0) catch return error.SkipZigTest;

    var stream = try ctx.newStream();
    defer stream.deinit();

    var event = try ctx.newEvent();
    defer event.deinit();

    try event.record(stream);
    try stream.synchronize();
    try event.synchronize();
}

test "Runtime device properties" {
    const ctx = RuntimeContext.init(0) catch return error.SkipZigTest;
    const name = try ctx.deviceName();
    try std.testing.expect(name.len > 0);
}

test "Runtime stream query" {
    const ctx = RuntimeContext.init(0) catch return error.SkipZigTest;
    var stream = try ctx.newStream();
    defer stream.deinit();
    const done = try stream.query();
    try std.testing.expect(done);
}

test "Runtime stream with flags (non-blocking)" {
    const ctx = RuntimeContext.init(0) catch return error.SkipZigTest;

    // 0x1 = cudaStreamNonBlocking
    var stream = try ctx.newStreamWithFlags(0x1);
    defer stream.deinit();

    try stream.synchronize();
    const done = try stream.query();
    try std.testing.expect(done);
}

test "Runtime event with flags (disable timing)" {
    const ctx = RuntimeContext.init(0) catch return error.SkipZigTest;

    // 0x2 = cudaEventDisableTiming
    var event = try ctx.newEventWithFlags(0x2);
    defer event.deinit();

    var stream = try ctx.newStream();
    defer stream.deinit();

    try event.record(stream);
    try stream.synchronize();
    try event.synchronize();
}

test "Runtime mallocPitch — 2D allocation" {
    const ctx = RuntimeContext.init(0) catch return error.SkipZigTest;

    // Allocate 256x128 bytes (pitched)
    const pitched = try ctx.mallocPitch(256, 128);
    try std.testing.expect(pitched.pitch >= 256); // pitch is >= width
    try cuda.runtime.result.free(pitched.ptr);
}
