// examples/kernel/10_Integration/G_EndToEnd/saxpy_e2e.zig
// Reference: End-to-end SAXPY: compile → allocate → transfer → compute → verify
// API: Full zcuda pipeline
//
// ── Kernel Loading: Way 5 build.zig auto-generated bridge module ──
// Uses: @import("kernel_saxpy") — type-safe PTX loading via bridge module

const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;

const kernel_saxpy = @import("kernel_saxpy");

/// End-to-end SAXPY: bridge load → alloc → H2D → launch → D2H → verify.
pub fn main() !void {
    var ctx = try driver.CudaContext.new(0);
    defer ctx.deinit();
    var stream = try ctx.newStream();
    defer stream.deinit();

    const module = try kernel_saxpy.load(ctx, std.heap.page_allocator);
    defer module.deinit();
    const func = try kernel_saxpy.getFunction(module, .saxpy);

    const n: u32 = 8192;
    const a: f32 = 2.5;

    var h_x: [8192]f32 = undefined;
    var h_y: [8192]f32 = undefined;
    for (0..n) |i| {
        h_x[i] = @floatFromInt(i);
        h_y[i] = @as(f32, @floatFromInt(i)) * 0.5;
    }

    var d_x = try stream.cloneHtoD(f32, &h_x);
    defer d_x.deinit();
    var d_y = try stream.cloneHtoD(f32, &h_y);
    defer d_y.deinit();

    const block_size: u32 = 256;
    const grid_size: u32 = (n + block_size - 1) / block_size;
    try stream.launch(func, .{
        .grid_dim = .{ .x = grid_size },
        .block_dim = .{ .x = block_size },
    }, .{
        d_x.devicePtr(), d_y.devicePtr(), a, n,
    });

    var h_result: [8192]f32 = undefined;
    try stream.memcpyDtoHAsync(f32, &h_result, d_y);
    try stream.synchronize();

    for (0..n) |i| {
        const expected = a * h_x[i] + @as(f32, @floatFromInt(i)) * 0.5;
        const diff = @abs(h_result[i] - expected);
        if (diff > 1e-4) return error.VerificationFailed;
    }
}
