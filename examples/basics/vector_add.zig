/// Vector Addition Example
///
/// The "Hello World" of CUDA programming. Demonstrates:
/// 1. NVRTC runtime compilation of CUDA C++ to PTX
/// 2. Device memory allocation (alloc, allocZeros, cloneHtoD)
/// 3. Kernel launch with grid/block configuration
/// 4. Device-to-host data transfer and result verification
///
/// Reference: cuda-samples/vectorAdd, cudarc/01-allocate + 02-copy + 03-launch-kernel
const std = @import("std");
const cuda = @import("zcuda");

const kernel_src =
    \\extern "C" __global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    \\    int i = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (i < n) {
    \\        C[i] = A[i] + B[i];
    \\    }
    \\}
;

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("=== Vector Addition Example ===\n\n", .{});

    // --- Device setup ---
    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    std.debug.print("Device: {s}\n", .{ctx.name()});

    const stream = ctx.defaultStream();

    // --- Compile CUDA kernel at runtime ---
    std.debug.print("Compiling CUDA kernel via NVRTC...\n", .{});
    const ptx = try cuda.nvrtc.compilePtx(allocator, kernel_src);
    defer allocator.free(ptx);

    const module = try ctx.loadModule(ptx);
    defer module.deinit();

    const kernel = try module.getFunction("vectorAdd");
    std.debug.print("✓ Kernel compiled and loaded\n\n", .{});

    // --- Prepare host data ---
    const n: usize = 50000;
    const n_i32: i32 = @intCast(n);

    var h_A: [50000]f32 = undefined;
    var h_B: [50000]f32 = undefined;

    for (&h_A, &h_B, 0..) |*a, *b, i| {
        a.* = @as(f32, @floatFromInt(i)) * 0.001;
        b.* = @as(f32, @floatFromInt(n - i)) * 0.001;
    }

    std.debug.print("Host data prepared: {} elements\n", .{n});
    std.debug.print("  A[0..5] = ", .{});
    for (h_A[0..5]) |v| std.debug.print("{d:.3} ", .{v});
    std.debug.print("\n  B[0..5] = ", .{});
    for (h_B[0..5]) |v| std.debug.print("{d:.3} ", .{v});
    std.debug.print("\n\n", .{});

    // --- Allocate device memory and copy data ---
    const d_A = try stream.cloneHtoD(f32, &h_A);
    defer d_A.deinit();

    const d_B = try stream.cloneHtoD(f32, &h_B);
    defer d_B.deinit();

    const d_C = try stream.allocZeros(f32, allocator, n);
    defer d_C.deinit();

    std.debug.print("Device memory allocated and copied\n", .{});

    // --- Launch kernel ---
    const config = cuda.LaunchConfig.forNumElems(@intCast(n));
    std.debug.print("Launch config: grid({},{},{}), block({},{},{})\n", .{
        config.grid_dim.x,  config.grid_dim.y,  config.grid_dim.z,
        config.block_dim.x, config.block_dim.y, config.block_dim.z,
    });

    try stream.launch(kernel, config, .{ &d_A, &d_B, &d_C, n_i32 });
    try stream.synchronize();
    std.debug.print("✓ Kernel executed\n\n", .{});

    // --- Copy results back and verify ---
    var h_C: [50000]f32 = undefined;
    try stream.memcpyDtoH(f32, &h_C, d_C);

    std.debug.print("Verifying results...\n", .{});
    var max_error: f32 = 0.0;
    for (&h_A, &h_B, &h_C) |a, b, c| {
        const expected = a + b;
        const err = @abs(c - expected);
        max_error = @max(max_error, err);
    }

    std.debug.print("  Max error: {e}\n", .{max_error});
    std.debug.print("  C[0..5]  = ", .{});
    for (h_C[0..5]) |v| std.debug.print("{d:.3} ", .{v});
    std.debug.print("\n", .{});

    if (max_error > 1e-5) {
        std.debug.print("\n✗ FAILED: error too large\n", .{});
        return error.ValidationFailed;
    }

    std.debug.print("\n✓ Vector addition successful! ({} elements verified)\n", .{n});
}
