// examples/kernel/10_Integration/B_StreamsAndEvents/stream_callback.zig
// Reference: cuda-samples/0_Introduction/simpleCallback
// API: CudaStream, synchronize, launch, memcpyDtoHAsync plus host-side verification
//
// ── Kernel Loading: Way 5 (enhanced) build.zig auto-generated bridge module ──
// Uses: @import("kernel_vector_add") — compile-time type-safe kernel reference

const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;

// Way 5: import kernel as a Zig module — type-safe, zero-overhead
const kernel_vector_add = @import("kernel_vector_add");

/// Stream with kernel + memcpy + host verification callback pattern.
pub fn main() !void {
    var ctx = try driver.CudaContext.new(0);
    defer ctx.deinit();
    var stream = try ctx.newStream();
    defer stream.deinit();

    // Way 5 (enhanced): load module, getFunction with compile-time checked Fn enum
    const module = try kernel_vector_add.load(ctx, std.heap.page_allocator);
    defer module.deinit();
    const func = try kernel_vector_add.getFunction(module, .vectorAdd);

    const n: u32 = 256;
    var d_a = try stream.alloc(f32, std.heap.page_allocator, n);
    defer d_a.deinit();
    var d_b = try stream.alloc(f32, std.heap.page_allocator, n);
    defer d_b.deinit();
    var d_c = try stream.alloc(f32, std.heap.page_allocator, n);
    defer d_c.deinit();

    // Init data
    var h_a: [256]f32 = undefined;
    var h_b: [256]f32 = undefined;
    for (0..n) |i| {
        h_a[i] = 1.0;
        h_b[i] = 2.0;
    }
    try stream.memcpyHtoDAsync(f32, d_a, &h_a);
    try stream.memcpyHtoDAsync(f32, d_b, &h_b);

    // Launch
    try stream.launch(func, .{ .grid_dim = .{ .x = 1 }, .block_dim = .{ .x = 256 } }, .{
        d_a.devicePtr(), d_b.devicePtr(), d_c.devicePtr(), n,
    });

    // Read back
    var h_c: [256]f32 = undefined;
    try stream.memcpyDtoHAsync(f32, &h_c, d_c);
    try stream.synchronize();

    // Verify (host callback equivalent)
    for (h_c) |v| {
        if (v != 3.0) return error.VerificationFailed;
    }
}
