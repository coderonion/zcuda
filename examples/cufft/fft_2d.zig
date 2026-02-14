/// cuFFT 2D Example
///
/// Performs a 2D complex-to-complex FFT on a small grid.
///
/// Reference: cuda-samples simpleCUFFT
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    std.debug.print("=== cuFFT 2D Example ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();

    const stream = ctx.defaultStream();
    const allocator = std.heap.page_allocator;

    const nx: usize = 4;
    const ny: usize = 4;
    const total = nx * ny;
    const complex_total = total * 2;

    // Create a 2D signal: delta at (1,1) → flat spectrum
    var h_input: [complex_total]f32 = undefined;
    @memset(&h_input, 0);
    // Set (1,1) to 1.0: row 1, col 1 → index = 1*ny + 1 = 5
    h_input[5 * 2] = 1.0; // real part

    std.debug.print("Input 4×4 grid (real part, delta at (1,1)):\n", .{});
    for (0..nx) |r| {
        std.debug.print("  [", .{});
        for (0..ny) |c| std.debug.print(" {d:.0}", .{h_input[(r * ny + c) * 2]});
        std.debug.print(" ]\n", .{});
    }
    std.debug.print("\n", .{});

    const d_data = try stream.cloneHtod(f32, &h_input);
    defer d_data.deinit();
    const d_out = try stream.alloc(f32, allocator, complex_total);
    defer d_out.deinit();

    // Forward 2D FFT
    const plan = try cuda.cufft.CufftPlan.plan2d(@intCast(nx), @intCast(ny), .c2c_f32);
    defer plan.deinit();
    try plan.setStream(stream);
    try plan.execC2C(d_data, d_out, .forward);

    var h_fft: [complex_total]f32 = undefined;
    try stream.memcpyDtoh(f32, &h_fft, d_out);

    std.debug.print("2D FFT magnitude:\n", .{});
    for (0..nx) |r| {
        std.debug.print("  [", .{});
        for (0..ny) |c| {
            const re = h_fft[(r * ny + c) * 2];
            const im = h_fft[(r * ny + c) * 2 + 1];
            const mag = @sqrt(re * re + im * im);
            std.debug.print(" {d:.2}", .{mag});
        }
        std.debug.print(" ]\n", .{});
    }
    std.debug.print("\n", .{});

    // A shifted delta produces all magnitudes = 1.0
    // (uniform magnitude, varying phase)
    var all_unit = true;
    for (0..total) |i| {
        const re = h_fft[i * 2];
        const im = h_fft[i * 2 + 1];
        const mag = @sqrt(re * re + im * im);
        if (@abs(mag - 1.0) > 0.01) all_unit = false;
    }
    std.debug.print("All magnitudes ≈ 1.0: {}\n\n", .{all_unit});

    // Inverse FFT
    try plan.execC2C(d_out, d_data, .inverse);

    var h_result: [complex_total]f32 = undefined;
    try stream.memcpyDtoh(f32, &h_result, d_data);

    // Normalize
    const nf: f32 = @floatFromInt(total);
    for (&h_result) |*v| v.* /= nf;

    std.debug.print("Roundtrip (real part):\n", .{});
    for (0..nx) |r| {
        std.debug.print("  [", .{});
        for (0..ny) |c| std.debug.print(" {d:.0}", .{h_result[(r * ny + c) * 2]});
        std.debug.print(" ]\n", .{});
    }

    // Verify roundtrip
    for (0..complex_total) |i| {
        if (@abs(h_result[i] - h_input[i]) > 1e-4) return error.ValidationFailed;
    }
    std.debug.print("\n✓ 2D FFT roundtrip verified\n", .{});
}
