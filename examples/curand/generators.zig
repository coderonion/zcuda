/// cuRAND Generator Comparison
///
/// Compares XORWOW, Philox, and MRG32k3a generators on quality.
///
/// Reference: cuda-samples MersenneTwisterGP11213
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    std.debug.print("=== cuRAND Generator Comparison ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    std.debug.print("Device: {s}\n\n", .{ctx.name()});

    const stream = ctx.defaultStream();
    const allocator = std.heap.page_allocator;
    const n: usize = 1000;

    const generators = [_]struct { rng: cuda.curand.RngType, name: []const u8 }{
        .{ .rng = .xorwow, .name = "XORWOW" },
        .{ .rng = .philox4_32_10, .name = "Philox4x32-10" },
        .{ .rng = .mrg32k3a, .name = "MRG32k3a" },
    };

    for (generators) |gen_info| {
        std.debug.print("--- {s} ---\n", .{gen_info.name});

        const rng = try cuda.curand.CurandContext.init(ctx, gen_info.rng);
        defer rng.deinit();
        try rng.setSeed(42);
        try rng.setStream(stream);

        // Generate uniform random numbers
        const d_data = try stream.alloc(f32, allocator, n);
        defer d_data.deinit();
        try rng.fillUniform(d_data);

        // Copy back and compute statistics
        var h_data: [n]f32 = undefined;
        try stream.memcpyDtoH(f32, &h_data, d_data);

        var sum: f64 = 0.0;
        var min_v: f32 = std.math.floatMax(f32);
        var max_v: f32 = -std.math.floatMax(f32);
        for (&h_data) |v| {
            sum += @as(f64, v);
            min_v = @min(min_v, v);
            max_v = @max(max_v, v);
        }
        const mean = sum / @as(f64, @floatFromInt(n));

        var var_sum: f64 = 0.0;
        for (&h_data) |v| {
            const diff = @as(f64, v) - mean;
            var_sum += diff * diff;
        }
        const stddev = @sqrt(var_sum / @as(f64, @floatFromInt(n)));

        std.debug.print("  Samples:  {}\n", .{n});
        std.debug.print("  Mean:     {d:.6} (ideal: 0.5)\n", .{mean});
        std.debug.print("  Stddev:   {d:.6} (ideal: {d:.6})\n", .{ stddev, @as(f64, 1.0) / @sqrt(12.0) });
        std.debug.print("  Range:    [{d:.6}, {d:.6}]\n", .{ min_v, max_v });
        std.debug.print("  First 5:  ", .{});
        for (h_data[0..5]) |v| std.debug.print("{d:.4} ", .{v});
        std.debug.print("\n\n", .{});
    }

    // --- Normal distribution comparison ---
    std.debug.print("--- Normal Distribution Quality ---\n", .{});
    for (generators) |gen_info| {
        const rng = try cuda.curand.CurandContext.init(ctx, gen_info.rng);
        defer rng.deinit();
        try rng.setSeed(123);
        try rng.setStream(stream);

        const d_normal = try stream.alloc(f32, allocator, n);
        defer d_normal.deinit();
        try rng.fillNormal(d_normal, 0.0, 1.0);

        var h_normal: [n]f32 = undefined;
        try stream.memcpyDtoH(f32, &h_normal, d_normal);

        var sum: f64 = 0.0;
        for (&h_normal) |v| sum += @as(f64, v);
        const mean = sum / @as(f64, @floatFromInt(n));

        var var_sum: f64 = 0.0;
        for (&h_normal) |v| {
            const diff = @as(f64, v) - mean;
            var_sum += diff * diff;
        }
        const stddev = @sqrt(var_sum / @as(f64, @floatFromInt(n)));

        std.debug.print("  {s}  mean={d:.4}  stddev={d:.4}\n", .{ gen_info.name, mean, stddev });
    }

    std.debug.print("\n✓ cuRAND generator comparison complete\n", .{});
}
