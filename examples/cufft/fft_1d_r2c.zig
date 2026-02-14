/// cuFFT 1D Real-to-Complex Example
///
/// Performs R2C forward and C2R inverse with spectral filtering.
///
/// Reference: cuda-samples simpleCUFFT
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    std.debug.print("=== cuFFT 1D R2C / C2R Example ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();

    const stream = ctx.defaultStream();
    const allocator = std.heap.page_allocator;

    // Real signal: sum of two frequencies
    // f(k) = sin(2π·k/N) + 0.5*sin(2π·3·k/N)
    const n: usize = 16;
    var h_input: [n]f32 = undefined;
    for (0..n) |k| {
        const t = @as(f32, @floatFromInt(k)) / @as(f32, @floatFromInt(n));
        h_input[k] = @sin(2.0 * std.math.pi * t) + 0.5 * @sin(2.0 * std.math.pi * 3.0 * t);
    }

    std.debug.print("Input signal (sin freq-1 + 0.5·sin freq-3):\n  ", .{});
    for (0..8) |k| std.debug.print("{d:.3} ", .{h_input[k]});
    std.debug.print("...\n\n", .{});

    // R2C output has n/2+1 complex values = (n/2+1)*2 floats
    const complex_out_n = (n / 2 + 1) * 2;

    const d_input = try stream.cloneHtod(f32, &h_input);
    defer d_input.deinit();
    const d_freq = try stream.alloc(f32, allocator, complex_out_n);
    defer d_freq.deinit();

    // R2C forward
    const plan_r2c = try cuda.cufft.CufftPlan.plan1d(@intCast(n), .r2c_f32, 1);
    defer plan_r2c.deinit();
    try plan_r2c.setStream(stream);
    try plan_r2c.execR2C(d_input, d_freq);

    var h_freq: [complex_out_n]f32 = undefined;
    try stream.memcpyDtoh(f32, &h_freq, d_freq);

    std.debug.print("Frequency spectrum magnitudes:\n", .{});
    for (0..n / 2 + 1) |k| {
        const re = h_freq[2 * k];
        const im = h_freq[2 * k + 1];
        const mag = @sqrt(re * re + im * im);
        if (mag > 0.1) {
            std.debug.print("  bin[{}] = {d:.2} (re={d:.2}, im={d:.2})\n", .{ k, mag, re, im });
        }
    }
    std.debug.print("\n", .{});

    // Filter: zero out frequency bin 3 (remove the higher frequency)
    std.debug.print("─── Filtering: zero out bin 3 ───\n", .{});
    h_freq[6] = 0; // bin 3 real
    h_freq[7] = 0; // bin 3 imag
    try stream.memcpyHtod(f32, d_freq, &h_freq);

    // C2R inverse
    const d_output = try stream.alloc(f32, allocator, n);
    defer d_output.deinit();

    const plan_c2r = try cuda.cufft.CufftPlan.plan1d(@intCast(n), .c2r_f32, 1);
    defer plan_c2r.deinit();
    try plan_c2r.setStream(stream);
    try plan_c2r.execC2R(d_freq, d_output);

    var h_output: [n]f32 = undefined;
    try stream.memcpyDtoh(f32, &h_output, d_output);

    // Normalize
    for (&h_output) |*v| v.* /= @as(f32, @floatFromInt(n));

    std.debug.print("Filtered signal (only freq-1 remains):\n  ", .{});
    for (0..8) |k| std.debug.print("{d:.3} ", .{h_output[k]});
    std.debug.print("...\n\n", .{});

    // Verify: should match sin(2π·k/N) only
    var max_err: f32 = 0;
    for (0..n) |k| {
        const t = @as(f32, @floatFromInt(k)) / @as(f32, @floatFromInt(n));
        const expected = @sin(2.0 * std.math.pi * t);
        const err = @abs(h_output[k] - expected);
        max_err = @max(max_err, err);
    }

    std.debug.print("Max error vs pure sin wave: {d:.6}\n", .{max_err});
    if (max_err > 1e-3) return error.ValidationFailed;
    std.debug.print("✓ R2C → filter → C2R roundtrip verified\n", .{});
}
