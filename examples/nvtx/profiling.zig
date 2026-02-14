/// NVTX Profiling Annotations Example
///
/// Demonstrates NVIDIA Tools Extension markers for profiling:
/// 1. Range push/pop for named code sections
/// 2. ScopedRange for RAII-style automatic pop
/// 3. Domain-based isolation
/// 4. Instant markers for events
///
/// Visible in NVIDIA Nsight Systems (nsys profile ./profiling)
///
/// Reference: NVTX documentation
const std = @import("std");
const cuda = @import("zcuda");

const kernel_src =
    \\extern "C" __global__ void compute(float *data, int n) {
    \\    int i = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (i < n) {
    \\        float val = data[i];
    \\        for (int j = 0; j < 100; j++) {
    \\            val = sinf(val) + cosf(val);
    \\        }
    \\        data[i] = val;
    \\    }
    \\}
;

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("=== NVTX Profiling Example ===\n\n", .{});
    std.debug.print("Run with: nsys profile ./zig-out/bin/nvtx-profiling\n\n", .{});

    // --- Setup ---
    cuda.nvtx.rangePush("CUDA Setup");

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    std.debug.print("Device: {s}\n", .{ctx.name()});

    const stream = ctx.defaultStream();

    // Compile kernel
    const ptx = try cuda.nvrtc.compilePtx(allocator, kernel_src);
    defer allocator.free(ptx);
    const module = try ctx.loadModule(ptx);
    defer module.deinit();
    const kernel = try module.getFunction("compute");

    cuda.nvtx.rangePop();
    std.debug.print("✓ Setup complete (NVTX range: CUDA Setup)\n", .{});

    // --- Using ScopedRange for automatic cleanup ---
    {
        const range = cuda.nvtx.ScopedRange.init("Data Preparation");
        defer range.deinit();

        const n: usize = 100_000;
        var h_data: [100_000]f32 = undefined;
        for (&h_data, 0..) |*v, i| {
            v.* = @as(f32, @floatFromInt(i)) * 0.001;
        }
        std.debug.print("✓ Data prepared (NVTX: Data Preparation)\n", .{});

        // --- Mark important events ---
        cuda.nvtx.mark("HtoD Transfer Start");

        const d_data = try stream.cloneHtod(f32, &h_data);
        defer d_data.deinit();

        cuda.nvtx.mark("HtoD Transfer End");
        std.debug.print("✓ Data transferred to GPU\n", .{});

        // --- Annotate kernel execution ---
        {
            const kernel_range = cuda.nvtx.ScopedRange.init("Kernel Execution");
            defer kernel_range.deinit();

            const config = cuda.LaunchConfig.forNumElems(@intCast(n));
            const n_i32: i32 = @intCast(n);

            // Multiple iterations
            for (0..5) |iter| {
                cuda.nvtx.rangePush("Iteration");
                try stream.launch(kernel, config, .{ &d_data, n_i32 });
                try stream.synchronize();
                cuda.nvtx.rangePop();
                _ = iter;
            }

            std.debug.print("✓ Kernel executed 5 iterations (NVTX: Kernel Execution)\n", .{});
        }

        // --- DtoH transfer ---
        cuda.nvtx.rangePush("DtoH Transfer");
        try stream.memcpyDtoh(f32, &h_data, d_data);
        cuda.nvtx.rangePop();

        std.debug.print("✓ Results transferred back\n", .{});
        std.debug.print("  First 3 results: {d:.4} {d:.4} {d:.4}\n", .{
            h_data[0], h_data[1], h_data[2],
        });
    }

    // --- Domain-based profiling ---
    std.debug.print("\n─── Domain Isolation ───\n", .{});
    const domain = cuda.nvtx.Domain.create("zcuda_example");
    defer domain.destroy();
    std.debug.print("✓ Created domain: 'zcuda_example'\n", .{});

    std.debug.print("\n✓ NVTX profiling example complete\n", .{});
    std.debug.print("  Tip: Use 'nsys profile' to visualize the annotations\n", .{});
}
