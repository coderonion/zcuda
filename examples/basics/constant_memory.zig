/// Constant Memory Example
///
/// Demonstrates GPU constant memory for read-only data:
/// 1. Load coefficients into constant memory via kernel parameters
/// 2. Polynomial evaluation using constant coefficients: f(x) = a₀ + a₁x + a₂x² + a₃x³
/// 3. Verify results against CPU computation
///
/// Constant memory is ideal for small, read-only data accessed by all threads
/// (coefficients, lookup tables, configuration). It's cached and broadcast-optimized.
///
/// Reference: cudarc/09-constant-memory + cuda-samples/simpleTexture
const std = @import("std");
const cuda = @import("zcuda");

const kernel_src =
    \\extern "C" __global__ void polynomial_eval(
    \\    const float *input, float *output,
    \\    float c0, float c1, float c2, float c3,
    \\    int n
    \\) {
    \\    int i = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (i < n) {
    \\        float x = input[i];
    \\        // Horner's method: c0 + x*(c1 + x*(c2 + x*c3))
    \\        output[i] = c0 + x * (c1 + x * (c2 + x * c3));
    \\    }
    \\}
;

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("=== Constant Memory Example ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    std.debug.print("Device: {s}\n\n", .{ctx.name()});

    const stream = ctx.defaultStream();

    // Compile kernel
    const ptx = try cuda.nvrtc.compilePtx(allocator, kernel_src);
    defer allocator.free(ptx);
    const module = try ctx.loadModule(ptx);
    defer module.deinit();
    const kernel = try module.getFunction("polynomial_eval");

    // Polynomial coefficients: f(x) = 1.0 + 2.0x + 3.0x² + 4.0x³
    const c0: f32 = 1.0;
    const c1: f32 = 2.0;
    const c2: f32 = 3.0;
    const c3: f32 = 4.0;
    std.debug.print("Polynomial: f(x) = {d} + {d}x + {d}x² + {d}x³\n\n", .{ c0, c1, c2, c3 });

    // Prepare input data: x = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, ...]
    const n: usize = 100;
    const n_i32: i32 = @intCast(n);
    var h_input: [100]f32 = undefined;
    for (&h_input, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i)) * 0.5;
    }

    // Copy input to device
    const d_input = try stream.cloneHtoD(f32, &h_input);
    defer d_input.deinit();
    const d_output = try stream.allocZeros(f32, allocator, n);
    defer d_output.deinit();

    // Launch kernel — coefficients passed as kernel parameters (like constant memory)
    const config = cuda.LaunchConfig.forNumElems(@intCast(n));
    try stream.launch(kernel, config, .{ &d_input, &d_output, c0, c1, c2, c3, n_i32 });
    try stream.synchronize();

    // Copy results back
    var h_output: [100]f32 = undefined;
    try stream.memcpyDtoH(f32, &h_output, d_output);

    // Verify results against CPU computation
    std.debug.print("─── Results ───\n", .{});
    std.debug.print("{s:>8}  {s:>12}  {s:>12}  {s:>10}\n", .{ "x", "GPU", "Expected", "Error" });
    std.debug.print("{s:->8}  {s:->12}  {s:->12}  {s:->10}\n", .{ "", "", "", "" });

    var max_error: f32 = 0.0;
    for (&h_input, &h_output, 0..) |x, gpu_result, i| {
        const expected = c0 + x * (c1 + x * (c2 + x * c3));
        const err = @abs(gpu_result - expected);
        max_error = @max(max_error, err);

        // Print first 10 and last 2
        if (i < 10 or i >= n - 2) {
            std.debug.print("{d:8.2}  {d:12.4}  {d:12.4}  {e:10}\n", .{ x, gpu_result, expected, err });
        } else if (i == 10) {
            std.debug.print("{s:>8}  {s:>12}  {s:>12}  {s:>10}\n", .{ "...", "...", "...", "..." });
        }
    }

    std.debug.print("\nMax error: {e}\n", .{max_error});

    if (max_error > 1e-3) {
        std.debug.print("✗ FAILED\n", .{});
        return error.ValidationFailed;
    }

    std.debug.print("\n✓ Polynomial evaluation with constant coefficients successful!\n", .{});
}
