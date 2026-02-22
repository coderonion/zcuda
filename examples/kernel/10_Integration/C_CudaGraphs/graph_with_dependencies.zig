// examples/kernel/10_Integration/C_CudaGraphs/graph_with_dependencies.zig
// Reference: cuda-samples/3_CUDA_Features/simpleCudaGraphs (multi-node)
// API: CudaStream.beginCapture, endCapture — sequential captured ops create dependency chain
//
// ── Kernel Loading: Way 5 build.zig auto-generated bridge module ──

const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;

const kernel_vector_add = @import("kernel_vector_add");

/// CUDA Graph with implicit dependencies via stream capture.
/// Operations captured in order create a dependency chain automatically.
pub fn main() !void {
    var ctx = try driver.CudaContext.new(0);
    defer ctx.deinit();
    var stream = try ctx.newStream();
    defer stream.deinit();

    // Way 5: load via bridge module
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

    // Begin capture — sequential stream ops form a dependency chain
    try stream.beginCapture();

    // Node 1: kernel A
    try stream.launch(func, .{ .grid_dim = .{ .x = 1 }, .block_dim = .{ .x = 256 } }, .{
        d_a.devicePtr(), d_b.devicePtr(), d_c.devicePtr(), n,
    });

    // Node 2: kernel B (automatically depends on node 1 via stream ordering)
    try stream.launch(func, .{ .grid_dim = .{ .x = 1 }, .block_dim = .{ .x = 256 } }, .{
        d_a.devicePtr(), d_c.devicePtr(), d_b.devicePtr(), n,
    });

    var exec_graph = try stream.endCapture();
    defer if (exec_graph) |*g| g.deinit();

    if (exec_graph) |g| {
        try g.launch();
    }
    try stream.synchronize();
}
