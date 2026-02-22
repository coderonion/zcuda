// examples/kernel/10_Integration/C_CudaGraphs/basic_graph.zig
// Reference: cuda-samples/3_CUDA_Features/simpleCudaGraphs
// API: CudaStream.beginCapture, endCapture → CudaGraph.launch
//
// ── Kernel Loading: Way 5 (enhanced) build.zig auto-generated bridge module ──
// Bridge provides: .load(), .getFunction(mod, .enum), .source_path, .source

const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;

// Way 5: import kernel as a Zig module
const kernel_vector_add = @import("kernel_vector_add");

/// Basic CUDA Graph: capture a linear workflow as a graph and replay it.
pub fn main() !void {
    var ctx = try driver.CudaContext.new(0);
    defer ctx.deinit();
    var stream = try ctx.newStream();
    defer stream.deinit();

    // Way 5 (enhanced): compile-time checked function name
    const module = try kernel_vector_add.load(ctx, std.heap.page_allocator);
    defer module.deinit();
    const func = try kernel_vector_add.getFunction(module, .vectorAdd);

    const n: u32 = 1024;
    var d_a = try stream.alloc(f32, std.heap.page_allocator, n);
    defer d_a.deinit();
    var d_b = try stream.alloc(f32, std.heap.page_allocator, n);
    defer d_b.deinit();
    var d_c = try stream.alloc(f32, std.heap.page_allocator, n);
    defer d_c.deinit();

    // Begin graph capture — all subsequent stream ops become graph nodes
    try stream.beginCapture();

    // Captured operations become graph nodes
    try stream.launch(func, .{ .grid_dim = .{ .x = 4 }, .block_dim = .{ .x = 256 } }, .{
        d_a.devicePtr(), d_b.devicePtr(), d_c.devicePtr(), n,
    });

    // End capture → returns an executable CudaGraph
    var exec_graph = try stream.endCapture();
    defer if (exec_graph) |*g| g.deinit();

    // Launch graph (can be replayed many times)
    if (exec_graph) |g| {
        try g.launch();
    }
    try stream.synchronize();
}
