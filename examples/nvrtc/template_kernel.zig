/// NVRTC Template Kernel Example
///
/// Demonstrates runtime compilation with multiple kernels and name expressions.
/// Compiles a CUDA source with multiple kernels, then selects and runs each.
///
/// Reference: cuda-samples clock_nvrtc + cudarc matmul-kernel
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    std.debug.print("=== NVRTC Template Kernel Example ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    std.debug.print("Device: {s}\n\n", .{ctx.name()});

    const stream = ctx.defaultStream();

    // Source with multiple kernels
    const src =
        \\extern "C" __global__ void scale_kernel(float* data, float factor, int n) {
        \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
        \\    if (idx < n) data[idx] *= factor;
        \\}
        \\
        \\extern "C" __global__ void add_kernel(float* data, float value, int n) {
        \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
        \\    if (idx < n) data[idx] += value;
        \\}
        \\
        \\extern "C" __global__ void clamp_kernel(float* data, float lo, float hi, int n) {
        \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
        \\    if (idx < n) {
        \\        if (data[idx] < lo) data[idx] = lo;
        \\        if (data[idx] > hi) data[idx] = hi;
        \\    }
        \\}
    ;

    std.debug.print("--- Compiling CUDA source with 3 kernels ---\n", .{});
    const ptx = cuda.nvrtc.compilePtx(allocator, src) catch |err| {
        std.debug.print("Compilation failed: {}\n", .{err});
        return err;
    };
    defer allocator.free(ptx);
    std.debug.print("  PTX size: {} bytes\n\n", .{ptx.len});

    const module = try ctx.loadModule(ptx);
    defer module.deinit();

    // Prepare data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    const n: usize = 8;
    const n_i32: i32 = @intCast(n);
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const d_data = try stream.cloneHtoD(f32, &input);
    defer d_data.deinit();

    std.debug.print("Input: [ ", .{});
    for (&input) |v| std.debug.print("{d:.1} ", .{v});
    std.debug.print("]\n\n", .{});

    const config = cuda.LaunchConfig.forNumElems(@intCast(n));

    // Step 1: Scale by 2.0
    std.debug.print("--- Step 1: scale_kernel(x2.0) ---\n", .{});
    const scale_fn = try module.getFunction("scale_kernel");
    try stream.launch(scale_fn, config, .{ &d_data, @as(f32, 2.0), n_i32 });

    var h_result: [8]f32 = undefined;
    try stream.memcpyDtoH(f32, &h_result, d_data);
    std.debug.print("  Result: [ ", .{});
    for (&h_result) |v| std.debug.print("{d:.1} ", .{v});
    std.debug.print("]\n\n", .{});

    // Step 2: Add 10.0
    std.debug.print("--- Step 2: add_kernel(+10.0) ---\n", .{});
    const add_fn = try module.getFunction("add_kernel");
    try stream.launch(add_fn, config, .{ &d_data, @as(f32, 10.0), n_i32 });

    try stream.memcpyDtoH(f32, &h_result, d_data);
    std.debug.print("  Result: [ ", .{});
    for (&h_result) |v| std.debug.print("{d:.1} ", .{v});
    std.debug.print("]\n\n", .{});

    // Step 3: Clamp to [15.0, 22.0]
    std.debug.print("--- Step 3: clamp_kernel([15, 22]) ---\n", .{});
    const clamp_fn = try module.getFunction("clamp_kernel");
    try stream.launch(clamp_fn, config, .{ &d_data, @as(f32, 15.0), @as(f32, 22.0), n_i32 });

    try stream.memcpyDtoH(f32, &h_result, d_data);
    std.debug.print("  Result: [ ", .{});
    for (&h_result) |v| std.debug.print("{d:.1} ", .{v});
    std.debug.print("]\n\n", .{});

    // Verify final result: clamp(input * 2.0 + 10.0, 15, 22)
    const expected = [_]f32{ 15.0, 15.0, 16.0, 18.0, 20.0, 22.0, 22.0, 22.0 };
    for (&h_result, &expected) |got, exp| {
        if (@abs(got - exp) > 1e-5) {
            std.debug.print("FAILED: got {d:.1}, expected {d:.1}\n", .{ got, exp });
            return error.ValidationFailed;
        }
    }
    std.debug.print("✓ All 3 kernel pipeline verified\n", .{});
}
