/// Streams Example
///
/// Demonstrates CUDA streams for concurrent execution:
/// 1. Default stream vs custom non-blocking streams
/// 2. Overlap of compute and memory copy across streams
/// 3. Event-based inter-stream synchronization
/// 4. Stream synchronization
///
/// Reference: cuda-samples/simpleStreams + simpleMultiCopy + cudarc/04-streams
const std = @import("std");
const cuda = @import("zcuda");

const kernel_src =
    \\extern "C" __global__ void fill_pattern(float *data, float value, int n) {
    \\    int i = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (i < n) {
    \\        // Simulate compute work
    \\        float result = value;
    \\        for (int j = 0; j < 50; j++) {
    \\            result = result * 1.001f + 0.001f;
    \\        }
    \\        data[i] = result;
    \\    }
    \\}
;

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("=== CUDA Streams Example ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    std.debug.print("Device: {s}\n\n", .{ctx.name()});

    // Check async engine count
    const sys = cuda.driver.sys;
    const async_engines = try ctx.attribute(sys.CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT);
    std.debug.print("Async copy engines: {}\n", .{async_engines});

    // Compile kernel
    const ptx = try cuda.nvrtc.compilePtx(allocator, kernel_src);
    defer allocator.free(ptx);
    const module = try ctx.loadModule(ptx);
    defer module.deinit();
    const kernel = try module.getFunction("fill_pattern");

    // --- Part 1: Sequential execution on default stream ---
    std.debug.print("\n─── Part 1: Sequential (Default Stream) ───\n", .{});

    const default_stream = ctx.defaultStream();
    const n: usize = 256 * 1024; // 256K elements per chunk
    const n_i32: i32 = @intCast(n);

    const start_all = try default_stream.createEvent(0);
    defer start_all.deinit();
    const stop_all = try default_stream.createEvent(0);
    defer stop_all.deinit();

    // Time sequential work
    try start_all.record(default_stream);
    {
        const d1 = try default_stream.allocZeros(f32, allocator, n);
        defer d1.deinit();
        const d2 = try default_stream.allocZeros(f32, allocator, n);
        defer d2.deinit();

        const config = cuda.LaunchConfig.forNumElems(@intCast(n));

        // Process chunk 1 then chunk 2 sequentially
        try default_stream.launch(kernel, config, .{ &d1, @as(f32, 1.0), n_i32 });
        try default_stream.launch(kernel, config, .{ &d2, @as(f32, 2.0), n_i32 });

        var h_result1: [256 * 1024]f32 = undefined;
        var h_result2: [256 * 1024]f32 = undefined;
        try default_stream.memcpyDtoh(f32, &h_result1, d1);
        try default_stream.memcpyDtoh(f32, &h_result2, d2);

        std.debug.print("  Chunk 1: d[0]={d:.6}, d[last]={d:.6}\n", .{ h_result1[0], h_result1[n - 1] });
        std.debug.print("  Chunk 2: d[0]={d:.6}, d[last]={d:.6}\n", .{ h_result2[0], h_result2[n - 1] });
    }
    try stop_all.record(default_stream);
    try stop_all.synchronize();

    const seq_ms = try start_all.elapsedTime(stop_all);
    std.debug.print("  Sequential time: {d:.3} ms\n", .{seq_ms});

    // --- Part 2: Concurrent execution on multiple streams ---
    std.debug.print("\n─── Part 2: Concurrent (Two Streams) ───\n", .{});

    const stream1 = try ctx.newStream();
    defer stream1.deinit();
    const stream2 = try ctx.newStream();
    defer stream2.deinit();

    try start_all.record(default_stream);
    {
        const d1 = try default_stream.allocZeros(f32, allocator, n);
        defer d1.deinit();
        const d2 = try default_stream.allocZeros(f32, allocator, n);
        defer d2.deinit();

        const config = cuda.LaunchConfig.forNumElems(@intCast(n));

        // Launch on different streams for potential overlap
        try stream1.launch(kernel, config, .{ &d1, @as(f32, 1.0), n_i32 });
        try stream2.launch(kernel, config, .{ &d2, @as(f32, 2.0), n_i32 });

        // Synchronize both streams
        try stream1.synchronize();
        try stream2.synchronize();

        var h_result1: [256 * 1024]f32 = undefined;
        var h_result2: [256 * 1024]f32 = undefined;
        try default_stream.memcpyDtoh(f32, &h_result1, d1);
        try default_stream.memcpyDtoh(f32, &h_result2, d2);

        std.debug.print("  Chunk 1: d[0]={d:.6}, d[last]={d:.6}\n", .{ h_result1[0], h_result1[n - 1] });
        std.debug.print("  Chunk 2: d[0]={d:.6}, d[last]={d:.6}\n", .{ h_result2[0], h_result2[n - 1] });
    }
    try stop_all.record(default_stream);
    try stop_all.synchronize();

    const conc_ms = try start_all.elapsedTime(stop_all);
    std.debug.print("  Concurrent time: {d:.3} ms\n", .{conc_ms});

    // --- Summary ---
    std.debug.print("\n━━━ Results ━━━━━━━━━━━━━━━━━━━━━\n", .{});
    std.debug.print("  Sequential:  {d:.3} ms\n", .{seq_ms});
    std.debug.print("  Concurrent:  {d:.3} ms\n", .{conc_ms});
    if (seq_ms > 0.001) {
        std.debug.print("  Speedup:     {d:.2}x\n", .{seq_ms / conc_ms});
    }

    std.debug.print("\n✓ Stream concurrency demo complete\n", .{});
}
