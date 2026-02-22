// examples/kernel/10_Integration/C_CudaGraphs/graph_replay_update.zig
// Reference: cuda-samples/3_CUDA_Features/simpleCudaGraphs (graph update)
// API: CudaStream.beginCapture, endCapture → CudaGraph.launch (replay)
//
// ── Kernel Loading: Way 5 build.zig auto-generated bridge module ──

const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;

const kernel_vector_add = @import("kernel_vector_add");

/// CUDA Graph replay: capture once, launch many times for low-latency replay.
pub fn main() !void {
    var ctx = try driver.CudaContext.new(0);
    defer ctx.deinit();
    var stream = try ctx.newStream();
    defer stream.deinit();

    // Way 5: load via bridge module
    const module = try kernel_vector_add.load(ctx, std.heap.page_allocator);
    defer module.deinit();
    const func = try kernel_vector_add.getFunction(module, .vectorAdd);

    const n: u32 = 512;
    var d_a = try stream.alloc(f32, std.heap.page_allocator, n);
    defer d_a.deinit();
    var d_b = try stream.alloc(f32, std.heap.page_allocator, n);
    defer d_b.deinit();
    var d_c = try stream.alloc(f32, std.heap.page_allocator, n);
    defer d_c.deinit();

    // Capture graph
    try stream.beginCapture();
    try stream.launch(func, .{ .grid_dim = .{ .x = 2 }, .block_dim = .{ .x = 256 } }, .{
        d_a.devicePtr(), d_b.devicePtr(), d_c.devicePtr(), n,
    });
    var exec = try stream.endCapture();
    defer if (exec) |*g| g.deinit();

    // Replay the same graph 10 times (fast: no re-capture overhead)
    if (exec) |g| {
        for (0..10) |_| {
            try g.launch();
        }
    }
    try stream.synchronize();
}
