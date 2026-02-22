// examples/kernel/10_Integration/A_DriverLifecycle/ptx_compile_execute.zig
// Reference: cuda-samples/0_Introduction/ptxjit
// API: bridge.load, CudaStream.launch
//
// ── Kernel Loading: Way 5 build.zig auto-generated bridge module ──
// Bridge provides: .load(), .getFunction(mod, .enum), type-safe function names.
// (Original Way 2/3 design used nvrtc JIT — converted to Way 5 for portability.)

const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;

const kernel_vector_add = @import("kernel_vector_add");

/// Load compiled PTX via bridge module → launch kernel.
/// Exercises the bridge loading pipeline (PTX is compiled at build time).
pub fn main() !void {
    var ctx = try driver.CudaContext.new(0);
    defer ctx.deinit();

    // Way 5: load via bridge module
    const module = try kernel_vector_add.load(ctx, std.heap.page_allocator);
    defer module.deinit();

    const func = try kernel_vector_add.getFunction(module, .vectorAdd);

    // Allocate and launch
    var stream = try ctx.newStream();
    defer stream.deinit();

    const n: u32 = 512;
    var d_a = try stream.alloc(f32, std.heap.page_allocator, n);
    defer d_a.deinit();
    var d_b = try stream.alloc(f32, std.heap.page_allocator, n);
    defer d_b.deinit();
    var d_out = try stream.allocZeros(f32, std.heap.page_allocator, n);
    defer d_out.deinit();

    // Fill host data
    var h_a: [512]f32 = undefined;
    var h_b: [512]f32 = undefined;
    for (0..n) |i| {
        h_a[i] = 42.0;
        h_b[i] = 0.0;
    }
    try stream.memcpyHtoDAsync(f32, d_a, &h_a);
    try stream.memcpyHtoDAsync(f32, d_b, &h_b);

    try stream.launch(func, .{ .grid_dim = .{ .x = 2 }, .block_dim = .{ .x = 256 } }, .{
        d_a.devicePtr(), d_b.devicePtr(), d_out.devicePtr(), n,
    });

    var h_out: [512]f32 = undefined;
    try stream.memcpyDtoHAsync(f32, &h_out, d_out);
    try stream.synchronize();

    for (h_out) |v| {
        if (v != 42.0) return error.VerificationFailed;
    }
}
