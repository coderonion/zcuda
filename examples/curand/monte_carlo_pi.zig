/// Monte Carlo Pi Estimation Example
///
/// Estimates π using GPU-accelerated Monte Carlo simulation:
/// 1. Generate random (x,y) points in [0,1] × [0,1] using cuRAND
/// 2. Count points inside the unit quarter-circle (x²+y²≤1)
/// 3. π ≈ 4 × (points inside circle) / (total points)
///
/// Demonstrates cuRAND + custom CUDA kernel integration.
///
/// Reference: CUDALibrarySamples/cuRAND (Monte Carlo applications)
const std = @import("std");
const cuda = @import("zcuda");

const count_kernel_src =
    \\extern "C" __global__ void count_inside(
    \\    const float *x, const float *y,
    \\    unsigned int *count, int n
    \\) {
    \\    __shared__ unsigned int block_count;
    \\    if (threadIdx.x == 0) block_count = 0;
    \\    __syncthreads();
    \\
    \\    int i = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (i < n) {
    \\        float xi = x[i];
    \\        float yi = y[i];
    \\        if (xi * xi + yi * yi <= 1.0f) {
    \\            atomicAdd(&block_count, 1);
    \\        }
    \\    }
    \\    __syncthreads();
    \\
    \\    if (threadIdx.x == 0) {
    \\        atomicAdd(count, block_count);
    \\    }
    \\}
;

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("=== Monte Carlo Pi Estimation ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    std.debug.print("Device: {s}\n\n", .{ctx.name()});

    const stream = ctx.defaultStream();

    // Compile counting kernel
    const ptx = try cuda.nvrtc.compilePtx(allocator, count_kernel_src);
    defer allocator.free(ptx);
    const module = try ctx.loadModule(ptx);
    defer module.deinit();
    const kernel = try module.getFunction("count_inside");

    // Create RNG
    const rng = try cuda.curand.CurandContext.init(ctx, .philox4_32_10);
    defer rng.deinit();
    try rng.setStream(stream);

    // Run multiple sample sizes
    const sample_sizes = [_]usize{ 10_000, 100_000, 1_000_000, 10_000_000 };

    std.debug.print("{s:>12}  {s:>12}  {s:>10}\n", .{ "Samples", "π estimate", "Error" });
    std.debug.print("{s:->12}  {s:->12}  {s:->10}\n", .{ "", "", "" });

    for (&sample_sizes) |n| {
        try rng.setSeed(42);

        // Generate random x, y coordinates
        const d_x = try stream.alloc(f32, allocator, n);
        defer d_x.deinit();
        const d_y = try stream.alloc(f32, allocator, n);
        defer d_y.deinit();

        try rng.fillUniform(d_x);
        try rng.fillUniform(d_y);

        // Allocate counter (initialize to 0)
        const d_count = try stream.allocZeros(u32, allocator, 1);
        defer d_count.deinit();

        // Count points inside quarter circle
        const n_i32: i32 = @intCast(n);
        const config = cuda.LaunchConfig.forNumElems(@intCast(n));
        try stream.launch(kernel, config, .{ &d_x, &d_y, &d_count, n_i32 });
        try stream.synchronize();

        var count: [1]u32 = undefined;
        try stream.memcpyDtoh(u32, &count, d_count);

        const pi_estimate = 4.0 * @as(f64, @floatFromInt(count[0])) / @as(f64, @floatFromInt(n));
        const err = @abs(pi_estimate - std.math.pi);

        std.debug.print("{:12}  {d:12.8}  {d:10.8}\n", .{ n, pi_estimate, err });
    }

    std.debug.print("\nπ (actual) = {d:.10}\n", .{std.math.pi});
    std.debug.print("\n✓ Monte Carlo simulation complete\n", .{});
}
