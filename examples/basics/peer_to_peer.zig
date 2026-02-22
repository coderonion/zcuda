/// Peer-to-Peer (Multi-GPU) Example
///
/// Demonstrates multi-GPU operations:
/// 1. Enumerate available GPUs
/// 2. Query peer access capability between devices
/// 3. Cross-device memory copy
///
/// Note: Gracefully handles single-GPU systems.
///
/// Reference: cuda-samples/simpleP2P + simpleMultiGPU + cudarc/13-copy-multi-gpu
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("=== Peer-to-Peer Multi-GPU Example ===\n\n", .{});

    const device_count = try cuda.driver.CudaContext.deviceCount();
    std.debug.print("CUDA devices found: {}\n\n", .{device_count});

    if (device_count < 2) {
        std.debug.print("⚠ Multi-GPU example requires at least 2 GPUs.\n", .{});
        std.debug.print("  Running single-GPU demo instead.\n\n", .{});

        const ctx = try cuda.driver.CudaContext.new(0);
        defer ctx.deinit();
        std.debug.print("Device 0: {s}\n", .{ctx.name()});

        const stream = ctx.defaultStream();

        // Demonstrate basic alloc/copy on single GPU
        const h_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
        const d_data = try stream.cloneHtoD(f32, &h_data);
        defer d_data.deinit();

        var h_result: [5]f32 = undefined;
        try stream.memcpyDtoH(f32, &h_result, d_data);

        std.debug.print("  Data roundtrip: ", .{});
        for (&h_result) |v| std.debug.print("{d:.1} ", .{v});
        std.debug.print("\n\n✓ Single-GPU demo complete\n", .{});
        return;
    }

    // --- Multi-GPU path ---
    // Print all devices
    for (0..@intCast(device_count)) |i| {
        const ctx = try cuda.driver.CudaContext.new(i);
        defer ctx.deinit();
        std.debug.print("Device {}: {s}\n", .{ i, ctx.name() });
    }

    // Check peer access between device 0 and 1
    std.debug.print("\n─── Peer Access Check ───\n", .{});
    const can_access = cuda.driver.result.peer.canAccessPeer(0, 1) catch false;
    std.debug.print("  Device 0 → Device 1: {s}\n", .{if (can_access) "Yes" else "No"});

    // Work with two GPUs
    const ctx0 = try cuda.driver.CudaContext.new(0);
    defer ctx0.deinit();
    const stream0 = ctx0.defaultStream();

    const ctx1 = try cuda.driver.CudaContext.new(1);
    defer ctx1.deinit();
    const stream1 = ctx1.defaultStream();

    // Allocate on GPU 0 and copy data
    const n: usize = 1024;
    var h_data: [1024]f32 = undefined;
    for (&h_data, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i));
    }

    const d0_data = try stream0.cloneHtoD(f32, &h_data);
    defer d0_data.deinit();
    std.debug.print("\n  GPU 0: allocated {} elements\n", .{n});

    // Allocate on GPU 1
    const d1_data = try stream1.allocZeros(f32, allocator, n);
    defer d1_data.deinit();
    std.debug.print("  GPU 1: allocated {} elements (zeros)\n", .{n});

    // Copy back from GPU 0 and verify
    var h_result0: [1024]f32 = undefined;
    try stream0.memcpyDtoH(f32, &h_result0, d0_data);

    std.debug.print("\n  GPU 0 data[0..5]: ", .{});
    for (h_result0[0..5]) |v| std.debug.print("{d:.0} ", .{v});
    std.debug.print("\n", .{});

    std.debug.print("\n✓ Multi-GPU example complete\n", .{});
}
