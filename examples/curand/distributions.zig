/// cuRAND Distributions Example
///
/// Demonstrates GPU random number generation with various distributions:
/// 1. Uniform distribution (0, 1]
/// 2. Normal distribution (Gaussian)
/// 3. Log-normal distribution
/// 4. Poisson distribution
/// 5. Seed control for reproducibility
///
/// Reference: CUDALibrarySamples/cuRAND
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("=== cuRAND Distributions Example ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    std.debug.print("Device: {s}\n\n", .{ctx.name()});

    const stream = ctx.defaultStream();

    const rng = try cuda.curand.CurandContext.init(ctx, .default);
    defer rng.deinit();
    try rng.setSeed(42);
    try rng.setStream(stream);

    const n: usize = 10000;

    // --- Uniform distribution ---
    std.debug.print("─── Uniform Distribution (0, 1] ───\n", .{});
    {
        const d_data = try stream.alloc(f32, allocator, n);
        defer d_data.deinit();
        try rng.fillUniform(d_data);

        var h_data: [10000]f32 = undefined;
        try stream.memcpyDtoh(f32, &h_data, d_data);

        var sum: f64 = 0.0;
        var min_val: f32 = std.math.floatMax(f32);
        var max_val: f32 = -std.math.floatMax(f32);
        for (&h_data) |v| {
            sum += @as(f64, v);
            min_val = @min(min_val, v);
            max_val = @max(max_val, v);
        }
        const mean = sum / @as(f64, @floatFromInt(n));

        std.debug.print("  Samples: {}\n", .{n});
        std.debug.print("  Mean:    {d:.4} (expected ~0.5)\n", .{mean});
        std.debug.print("  Range:   [{d:.4}, {d:.4}]\n", .{ min_val, max_val });
        std.debug.print("  First 5: ", .{});
        for (h_data[0..5]) |v| std.debug.print("{d:.4} ", .{v});
        std.debug.print("\n\n", .{});
    }

    // --- Normal distribution ---
    std.debug.print("─── Normal Distribution (μ=0, σ=1) ───\n", .{});
    {
        const d_data = try stream.alloc(f32, allocator, n);
        defer d_data.deinit();
        try rng.fillNormal(d_data, 0.0, 1.0);

        var h_data: [10000]f32 = undefined;
        try stream.memcpyDtoh(f32, &h_data, d_data);

        var sum: f64 = 0.0;
        for (&h_data) |v| sum += @as(f64, v);
        const mean = sum / @as(f64, @floatFromInt(n));

        var var_sum: f64 = 0.0;
        for (&h_data) |v| {
            const diff = @as(f64, v) - mean;
            var_sum += diff * diff;
        }
        const stddev = @sqrt(var_sum / @as(f64, @floatFromInt(n)));

        std.debug.print("  Samples: {}\n", .{n});
        std.debug.print("  Mean:    {d:.4} (expected ~0.0)\n", .{mean});
        std.debug.print("  Stddev:  {d:.4} (expected ~1.0)\n", .{stddev});
        std.debug.print("  First 5: ", .{});
        for (h_data[0..5]) |v| std.debug.print("{d:.4} ", .{v});
        std.debug.print("\n\n", .{});
    }

    // --- Log-Normal distribution ---
    std.debug.print("─── Log-Normal Distribution (μ=0, σ=0.5) ───\n", .{});
    {
        const d_data = try stream.alloc(f32, allocator, n);
        defer d_data.deinit();
        try rng.fillLogNormal(d_data, 0.0, 0.5);

        var h_data: [10000]f32 = undefined;
        try stream.memcpyDtoh(f32, &h_data, d_data);

        var sum: f64 = 0.0;
        var min_val: f32 = std.math.floatMax(f32);
        for (&h_data) |v| {
            sum += @as(f64, v);
            min_val = @min(min_val, v);
        }
        const mean = sum / @as(f64, @floatFromInt(n));

        std.debug.print("  Samples: {}\n", .{n});
        std.debug.print("  Mean:    {d:.4} (expected ~{d:.4})\n", .{
            mean,
            @exp(@as(f64, 0.0) + 0.5 * 0.25),
        });
        std.debug.print("  Min:     {d:.4} (all positive)\n\n", .{min_val});
    }

    // --- Poisson distribution ---
    std.debug.print("─── Poisson Distribution (λ=5.0) ───\n", .{});
    {
        const d_data = try stream.alloc(u32, allocator, n);
        defer d_data.deinit();
        try rng.fillPoisson(d_data, 5.0);

        var h_data: [10000]u32 = undefined;
        try stream.memcpyDtoh(u32, &h_data, d_data);

        var sum: u64 = 0;
        for (&h_data) |v| sum += v;
        const mean = @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(n));

        std.debug.print("  Samples: {}\n", .{n});
        std.debug.print("  Mean:    {d:.4} (expected ~5.0)\n", .{mean});
        std.debug.print("  First 10: ", .{});
        for (h_data[0..10]) |v| std.debug.print("{} ", .{v});
        std.debug.print("\n\n", .{});
    }

    // --- Reproducibility test ---
    std.debug.print("─── Seed Reproducibility ───\n", .{});
    {
        const d1 = try stream.alloc(f32, allocator, 5);
        defer d1.deinit();
        const d2 = try stream.alloc(f32, allocator, 5);
        defer d2.deinit();

        try rng.setSeed(12345);
        try rng.fillUniform(d1);

        try rng.setSeed(12345);
        try rng.fillUniform(d2);

        var h1: [5]f32 = undefined;
        var h2: [5]f32 = undefined;
        try stream.memcpyDtoh(f32, &h1, d1);
        try stream.memcpyDtoh(f32, &h2, d2);

        std.debug.print("  Run 1: ", .{});
        for (&h1) |v| std.debug.print("{d:.6} ", .{v});
        std.debug.print("\n  Run 2: ", .{});
        for (&h2) |v| std.debug.print("{d:.6} ", .{v});
        std.debug.print("\n", .{});

        var match = true;
        for (&h1, &h2) |a, b| {
            if (a != b) {
                match = false;
                break;
            }
        }
        std.debug.print("  Reproducible: {s}\n", .{if (match) "Yes ✓" else "No ✗"});
    }

    std.debug.print("\n✓ cuRAND distributions example complete\n", .{});
}
