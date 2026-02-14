/// NVRTC JIT Compilation Example
///
/// Demonstrates runtime compilation of CUDA C++ to PTX:
/// 1. Compile a kernel from source string using NVRTC
/// 2. Pass compilation options (architecture target)
/// 3. Load the PTX module and execute the kernel
/// 4. Multiple kernels in one compilation unit
///
/// Reference: cudarc/nvrtc-compile + cudarc/matmul-kernel
const std = @import("std");
const cuda = @import("zcuda");

const kernel_src =
    \\// Vector operations - two kernels in one source
    \\extern "C" __global__ void vec_scale(float *data, float scalar, int n) {
    \\    int i = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (i < n) {
    \\        data[i] *= scalar;
    \\    }
    \\}
    \\
    \\extern "C" __global__ void vec_add_scalar(float *data, float value, int n) {
    \\    int i = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (i < n) {
    \\        data[i] += value;
    \\    }
    \\}
;

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("=== NVRTC JIT Compilation Example ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    std.debug.print("Device: {s}\n", .{ctx.name()});

    const cap = try ctx.computeCapability();
    std.debug.print("Compute Capability: {}.{}\n\n", .{ cap.major, cap.minor });

    const stream = ctx.defaultStream();

    // --- Compile ---
    std.debug.print("Compiling CUDA source ({} bytes)...\n", .{kernel_src.len});
    const ptx = try cuda.nvrtc.compilePtx(allocator, kernel_src);
    defer allocator.free(ptx);
    std.debug.print("✓ Compiled to {} bytes of PTX\n", .{ptx.len});

    // --- Load module with two kernels ---
    const module = try ctx.loadModule(ptx);
    defer module.deinit();

    const scale_kernel = try module.getFunction("vec_scale");
    const add_kernel = try module.getFunction("vec_add_scalar");
    std.debug.print("✓ Loaded 2 kernels: vec_scale, vec_add_scalar\n\n", .{});

    // --- Execute chained operations ---
    const n: usize = 10;
    const n_i32: i32 = @intCast(n);
    var h_data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

    std.debug.print("Input:  [ ", .{});
    for (&h_data) |v| std.debug.print("{d:.0} ", .{v});
    std.debug.print("]\n", .{});

    const d_data = try stream.cloneHtod(f32, &h_data);
    defer d_data.deinit();

    const config = cuda.LaunchConfig.forNumElems(@intCast(n));

    // Step 1: Scale by 2.0
    try stream.launch(scale_kernel, config, .{ &d_data, @as(f32, 2.0), n_i32 });

    // Step 2: Add 10.0
    try stream.launch(add_kernel, config, .{ &d_data, @as(f32, 10.0), n_i32 });
    try stream.synchronize();

    var h_result: [10]f32 = undefined;
    try stream.memcpyDtoh(f32, &h_result, d_data);

    std.debug.print("Output: [ ", .{});
    for (&h_result) |v| std.debug.print("{d:.0} ", .{v});
    std.debug.print("]\n", .{});

    // Verify: result should be data * 2.0 + 10.0
    std.debug.print("Expected: [ ", .{});
    for (&h_data) |v| std.debug.print("{d:.0} ", .{v * 2.0 + 10.0});
    std.debug.print("]\n", .{});

    for (&h_data, &h_result) |orig, actual| {
        const expected = orig * 2.0 + 10.0;
        if (@abs(expected - actual) > 1e-5) return error.ValidationFailed;
    }

    std.debug.print("\n✓ NVRTC JIT compilation with chained kernels complete\n", .{});
}
