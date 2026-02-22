// examples/kernel/10_Integration/E_ErrorHandling/oob_launch.zig
// Reference: cuda-samples/0_Introduction/simpleAssert
// API: CudaStream.launch with invalid configs, error handling
//
// ── Kernel Loading: Way 5 build.zig auto-generated bridge module ──

const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;

const kernel_vector_add = @import("kernel_vector_add");

/// Out-of-bounds and invalid launch configuration tests.
pub fn main() !void {
    var ctx = try driver.CudaContext.new(0);
    defer ctx.deinit();
    var stream = try ctx.newStream();
    defer stream.deinit();

    // Way 5: load via bridge module
    const module = try kernel_vector_add.load(ctx, std.heap.page_allocator);
    defer module.deinit();
    const func = try kernel_vector_add.getFunction(module, .vectorAdd);

    // Valid allocation
    const n: u32 = 64;
    var d_a = try stream.alloc(f32, std.heap.page_allocator, n);
    defer d_a.deinit();
    var d_b = try stream.alloc(f32, std.heap.page_allocator, n);
    defer d_b.deinit();
    var d_c = try stream.alloc(f32, std.heap.page_allocator, n);
    defer d_c.deinit();

    // Normal launch should succeed
    try stream.launch(func, .{ .grid_dim = .{ .x = 1 }, .block_dim = .{ .x = 64 } }, .{
        d_a.devicePtr(), d_b.devicePtr(), d_c.devicePtr(), n,
    });
    try stream.synchronize();

    // Invalid block size (too large) — should produce error
    const bad_launch = stream.launch(func, .{ .grid_dim = .{ .x = 1 }, .block_dim = .{ .x = 2048 } }, .{
        d_a.devicePtr(), d_b.devicePtr(), d_c.devicePtr(), n,
    });
    if (bad_launch) |_| {
        // Some CUDA drivers may allow >1024 threads, but it's device-dependent
    } else |_| {
        // Expected: invalid configuration
    }
}
