// examples/kernel/10_Integration/B_StreamsAndEvents/stream_concurrency.zig
// Reference: cuda-samples/0_Introduction/simpleStreams
// API: multiple CudaStreams, concurrent launches, synchronize
//
// ── Kernel Loading: Way 5 (enhanced) build.zig auto-generated bridge module ──
// Bridge provides: .load(), .getFunction(mod, .enum), .source_path, .source

const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;

// Way 5: import kernel as a Zig module
const kernel_vector_add = @import("kernel_vector_add");

/// Launch same kernel on multiple streams concurrently.
pub fn main() !void {
    var ctx = try driver.CudaContext.new(0);
    defer ctx.deinit();

    // Way 5 (enhanced): Fn enum ensures function name is valid at compile time
    const module = try kernel_vector_add.load(ctx, std.heap.page_allocator);
    defer module.deinit();
    const func = try kernel_vector_add.getFunction(module, .vectorAdd);

    const NUM_STREAMS = 4;
    const n: u32 = 1024;

    var streams: [NUM_STREAMS]driver.CudaStream = undefined;
    var d_bufs: [NUM_STREAMS]driver.CudaSlice(f32) = undefined;

    // Create streams and allocate per-stream buffers
    for (0..NUM_STREAMS) |s| {
        streams[s] = try ctx.newStream();
        d_bufs[s] = try streams[s].alloc(f32, std.heap.page_allocator, n);
    }
    defer for (0..NUM_STREAMS) |s| {
        d_bufs[s].deinit();
        streams[s].deinit();
    };

    // Concurrent launches across all streams
    for (0..NUM_STREAMS) |s| {
        try streams[s].launch(func, .{ .grid_dim = .{ .x = 4 }, .block_dim = .{ .x = 256 } }, .{
            d_bufs[s].devicePtr(), d_bufs[s].devicePtr(), d_bufs[s].devicePtr(), n,
        });
    }

    // Sync all
    for (0..NUM_STREAMS) |s| {
        try streams[s].synchronize();
    }
}
