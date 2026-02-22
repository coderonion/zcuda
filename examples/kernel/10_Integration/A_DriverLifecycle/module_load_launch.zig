// examples/kernel/10_Integration/A_DriverLifecycle/module_load_launch.zig
// Reference: cuda-samples/0_Introduction/matrixMulDrv
// API: loadModule, getFunction, CudaStream.launch
//
// ── Kernel Loading: Way 5 build.zig auto-generated bridge module ──
// Bridge provides: .load(), .getFunction(mod, .enum), type-safe function names.

const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;

const kernel_vector_add = @import("kernel_vector_add");

/// Load a PTX module and launch a kernel function.
/// End-to-end: bridge → CudaModule → CudaFunction → launch.
pub fn main() !void {
    var ctx = try driver.CudaContext.new(0);
    defer ctx.deinit();

    // Way 5: load via bridge module (build system compiles + embeds PTX)
    const module = try kernel_vector_add.load(ctx, std.heap.page_allocator);
    defer module.deinit();

    // Get kernel function handle (type-safe enum)
    const func = try kernel_vector_add.getFunction(module, .vectorAdd);

    // Allocate device memory
    var stream = try ctx.newStream();
    defer stream.deinit();

    const n: u32 = 1024;
    var d_a = try stream.alloc(f32, std.heap.page_allocator, n);
    defer d_a.deinit();
    var d_b = try stream.alloc(f32, std.heap.page_allocator, n);
    defer d_b.deinit();
    var d_c = try stream.alloc(f32, std.heap.page_allocator, n);
    defer d_c.deinit();

    // Prepare host data
    var h_a: [1024]f32 = undefined;
    var h_b: [1024]f32 = undefined;
    for (0..n) |i| {
        h_a[i] = @floatFromInt(i);
        h_b[i] = @floatFromInt(i * 2);
    }

    // Transfer host → device
    try stream.memcpyHtoDAsync(f32, d_a, &h_a);
    try stream.memcpyHtoDAsync(f32, d_b, &h_b);

    // Launch kernel: vectorAdd(A, B, C, n)
    const block_size: u32 = 256;
    const grid_size: u32 = (n + block_size - 1) / block_size;
    try stream.launch(
        func,
        .{ .grid_dim = .{ .x = grid_size }, .block_dim = .{ .x = block_size } },
        .{ d_a.devicePtr(), d_b.devicePtr(), d_c.devicePtr(), n },
    );

    // Transfer device → host
    var h_c: [1024]f32 = undefined;
    try stream.memcpyDtoHAsync(f32, &h_c, d_c);
    try stream.synchronize();

    // Verify
    for (0..n) |i| {
        const expected = h_a[i] + h_b[i];
        if (h_c[i] != expected) return error.VerificationFailed;
    }
}
