/// Event Timing Example
///
/// Demonstrates CUDA event-based timing:
/// 1. Measure host-to-device memory copy bandwidth (GB/s)
/// 2. Measure device-to-host memory copy bandwidth (GB/s)
/// 3. Measure kernel execution time
/// 4. Event creation, recording, synchronization, and elapsed time query
///
/// Reference: cuda-samples/asyncAPI + cuda-samples/clock
const std = @import("std");
const cuda = @import("zcuda");

const kernel_src =
    \\extern "C" __global__ void scale_add(float *data, float scale, float offset, int n) {
    \\    int i = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (i < n) {
    \\        // Do some work to make timing measurable
    \\        float val = data[i];
    \\        for (int j = 0; j < 100; j++) {
    \\            val = val * scale + offset;
    \\        }
    \\        data[i] = val;
    \\    }
    \\}
;

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("=== CUDA Event Timing Example ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    std.debug.print("Device: {s}\n\n", .{ctx.name()});

    const stream = ctx.defaultStream();

    // Compile kernel
    const ptx = try cuda.nvrtc.compilePtx(allocator, kernel_src);
    defer allocator.free(ptx);
    const module = try ctx.loadModule(ptx);
    defer module.deinit();
    const kernel = try module.getFunction("scale_add");

    // Create timing events
    const start = try stream.createEvent(0);
    defer start.deinit();
    const stop = try stream.createEvent(0);
    defer stop.deinit();

    // --- Benchmark parameters ---
    const n: usize = 1024 * 1024; // 1M elements = 4 MB
    const size_bytes = n * @sizeOf(f32);
    const size_mb: f64 = @as(f64, @floatFromInt(size_bytes)) / (1024.0 * 1024.0);

    std.debug.print("Data size: {d:.1} MB ({} elements)\n\n", .{ size_mb, n });

    // Prepare host data
    var h_data: [1024 * 1024]f32 = undefined;
    for (&h_data, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 1000)) * 0.001;
    }

    // --- Benchmark 1: Host→Device bandwidth ---
    std.debug.print("─── Host → Device Copy ───\n", .{});
    try start.record(stream);
    const d_data = try stream.cloneHtoD(f32, &h_data);
    defer d_data.deinit();
    try stop.record(stream);
    try stop.synchronize();

    const htod_ms = try start.elapsedTime(stop);
    const htod_gbps = size_mb / 1024.0 / (@as(f64, htod_ms) / 1000.0);
    std.debug.print("  Time:      {d:.3} ms\n", .{htod_ms});
    std.debug.print("  Bandwidth: {d:.2} GB/s\n\n", .{htod_gbps});

    // --- Benchmark 2: Kernel execution ---
    std.debug.print("─── Kernel Execution ───\n", .{});
    const config = cuda.LaunchConfig.forNumElems(@intCast(n));
    const n_i32: i32 = @intCast(n);
    const scale: f32 = 1.001;
    const offset: f32 = 0.0001;

    // Warm-up run
    try stream.launch(kernel, config, .{ &d_data, scale, offset, n_i32 });
    try stream.synchronize();

    // Timed run
    try start.record(stream);
    try stream.launch(kernel, config, .{ &d_data, scale, offset, n_i32 });
    try stop.record(stream);
    try stop.synchronize();

    const kernel_ms = try start.elapsedTime(stop);
    std.debug.print("  Time:      {d:.3} ms\n", .{kernel_ms});
    std.debug.print("  Throughput: {d:.2} GElements/s\n\n", .{
        @as(f64, @floatFromInt(n)) / 1.0e9 / (@as(f64, kernel_ms) / 1000.0),
    });

    // --- Benchmark 3: Device→Host bandwidth ---
    std.debug.print("─── Device → Host Copy ───\n", .{});
    var h_result: [1024 * 1024]f32 = undefined;

    try start.record(stream);
    try stream.memcpyDtoH(f32, &h_result, d_data);
    try stop.record(stream);
    try stop.synchronize();

    const dtoh_ms = try start.elapsedTime(stop);
    const dtoh_gbps = size_mb / 1024.0 / (@as(f64, dtoh_ms) / 1000.0);
    std.debug.print("  Time:      {d:.3} ms\n", .{dtoh_ms});
    std.debug.print("  Bandwidth: {d:.2} GB/s\n\n", .{dtoh_gbps});

    // --- Summary ---
    std.debug.print("━━━ Summary ━━━━━━━━━━━━━━━━━━━━━\n", .{});
    std.debug.print("  HtoD Bandwidth:  {d:.2} GB/s\n", .{htod_gbps});
    std.debug.print("  DtoH Bandwidth:  {d:.2} GB/s\n", .{dtoh_gbps});
    std.debug.print("  Kernel Time:     {d:.3} ms\n", .{kernel_ms});
    std.debug.print("\n✓ Timing complete\n", .{});
}
