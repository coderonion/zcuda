// examples/kernel/10_Integration/G_EndToEnd/reduction_e2e.zig
// Reference: End-to-end reduction: compile → launch → verify against CPU
// API: Full zcuda pipeline with reduction kernel
//
// ── Kernel Loading: Way 5 build.zig auto-generated bridge module ──
// Uses: @import("kernel_reduce_sum") — type-safe PTX loading via bridge module

const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;

const kernel_reduce_sum = @import("kernel_reduce_sum");

/// End-to-end reduction: load kernel → compute GPU sum → compare with CPU sum.
pub fn main() !void {
    var ctx = try driver.CudaContext.new(0);
    defer ctx.deinit();
    var stream = try ctx.newStream();
    defer stream.deinit();

    const module = try kernel_reduce_sum.load(ctx, std.heap.page_allocator);
    defer module.deinit();
    const func = try kernel_reduce_sum.getFunction(module, .reduceSum);

    const n: u32 = 4096;

    var h_input: [4096]f32 = undefined;
    var cpu_sum: f64 = 0.0;
    for (0..n) |i| {
        h_input[i] = @as(f32, @floatFromInt(i % 100)) * 0.01;
        cpu_sum += @as(f64, h_input[i]);
    }

    var d_input = try stream.cloneHtoD(f32, &h_input);
    defer d_input.deinit();
    var d_result = try stream.allocZeros(f32, std.heap.page_allocator, 1);
    defer d_result.deinit();

    const block_size: u32 = 256;
    const grid_size: u32 = (n + block_size - 1) / block_size;
    try stream.launch(func, .{
        .grid_dim = .{ .x = grid_size },
        .block_dim = .{ .x = block_size },
    }, .{
        d_input.devicePtr(), d_result.devicePtr(), n,
    });

    var h_result: [1]f32 = undefined;
    try stream.memcpyDtoHAsync(f32, &h_result, d_result);
    try stream.synchronize();

    const gpu_sum: f64 = @floatCast(h_result[0]);
    const rel_error = @abs(gpu_sum - cpu_sum) / @max(@abs(cpu_sum), 1e-8);
    if (rel_error > 0.01) return error.ReductionMismatch;
}
