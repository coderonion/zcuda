// examples/kernel/10_Integration/D_MemoryManagement/unified_memory.zig
// Reference: cuda-samples/0_Introduction/UnifiedMemoryStreams
// API: driver.allocManaged, freeManaged, prefetchAsync

const cuda = @import("zcuda");
const driver = cuda.driver;

/// Unified (managed) memory: single pointer accessible from host and device.
pub fn main() !void {
    var ctx = try driver.CudaContext.new(0);
    defer ctx.deinit();
    var stream = try ctx.newStream();
    defer stream.deinit();

    const module = try ctx.loadModule("zig-out/kernel/kernel_vector_add.ptx");
    defer module.deinit();
    const func = try module.getFunction("vectorAdd");

    const n: u32 = 1024;

    // Allocate managed memory (accessible from both host and device)
    var a = try driver.allocManaged(f32, n);
    defer driver.freeManaged(a);
    var b = try driver.allocManaged(f32, n);
    defer driver.freeManaged(b);
    var c = try driver.allocManaged(f32, n);
    defer driver.freeManaged(c);

    // Initialize on host (no explicit copy needed!)
    for (0..n) |i| {
        a[i] = @floatFromInt(i);
        b[i] = @floatFromInt(i * 2);
    }

    // Optional: prefetch to device for performance
    try stream.prefetchAsync(f32, a, n, 0);
    try stream.prefetchAsync(f32, b, n, 0);

    // Launch kernel (uses managed pointers directly)
    try stream.launch(func, .{ 4, 1, 1 }, .{ 256, 1, 1 }, .{
        @as([*]const f32, a.ptr()), @as([*]const f32, b.ptr()), @as([*]f32, c.ptr()), n,
    });
    try stream.sync();

    // Read back on host (no explicit copy needed!)
    for (0..n) |i| {
        if (c[i] != a[i] + b[i]) return error.ManagedMemoryFailed;
    }
}
