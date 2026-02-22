/// cuFFT 1D Complex-to-Complex Example
///
/// Performs forward C2C transform and inverse to verify roundtrip.
///
/// Reference: cuda-samples simpleCUFFT
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    std.debug.print("=== cuFFT 1D C2C Example ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();

    const stream = ctx.defaultStream();
    const allocator = std.heap.page_allocator;

    // Each complex number stored as 2 floats (re, im)
    const n: usize = 8;
    const complex_n = n * 2; // float pairs

    // Input: 8 complex numbers representing a simple signal
    // f(k) = cos(2π·k/8) = known DFT pair
    var h_input: [complex_n]f32 = undefined;
    for (0..n) |k| {
        const angle = 2.0 * std.math.pi * @as(f32, @floatFromInt(k)) / @as(f32, @floatFromInt(n));
        h_input[2 * k] = @cos(angle); // real
        h_input[2 * k + 1] = 0.0; // imaginary
    }

    std.debug.print("Input signal (cos(2πk/8)):\n  ", .{});
    for (0..n) |k| {
        std.debug.print("({d:.2},{d:.2}) ", .{ h_input[2 * k], h_input[2 * k + 1] });
    }
    std.debug.print("\n\n", .{});

    // Allocate device memory
    const d_data = try stream.cloneHtoD(f32, &h_input);
    defer d_data.deinit();
    const d_out = try stream.alloc(f32, allocator, complex_n);
    defer d_out.deinit();

    // Create 1D C2C FFT plan
    const plan = try cuda.cufft.CufftPlan.plan1d(@intCast(n), .c2c_f32, 1);
    defer plan.deinit();
    try plan.setStream(stream);

    // Forward FFT
    try plan.execC2C(d_data, d_out, .forward);

    var h_fft: [complex_n]f32 = undefined;
    try stream.memcpyDtoH(f32, &h_fft, d_out);

    std.debug.print("FFT output (frequency domain):\n  ", .{});
    for (0..n) |k| {
        std.debug.print("({d:.1},{d:.1}) ", .{ h_fft[2 * k], h_fft[2 * k + 1] });
    }
    std.debug.print("\n", .{});

    // For cos signal, expect peaks at k=1 and k=7 (N-1) with magnitude N/2=4
    const mag_1 = @sqrt(h_fft[2] * h_fft[2] + h_fft[3] * h_fft[3]);
    const mag_7 = @sqrt(h_fft[14] * h_fft[14] + h_fft[15] * h_fft[15]);
    std.debug.print("  |F[1]| = {d:.2}, |F[7]| = {d:.2} (expected ~4.0)\n\n", .{ mag_1, mag_7 });

    // Inverse FFT (in-place)
    const plan_inv = try cuda.cufft.CufftPlan.plan1d(@intCast(n), .c2c_f32, 1);
    defer plan_inv.deinit();
    try plan_inv.setStream(stream);
    try plan_inv.execC2C(d_out, d_data, .inverse);

    var h_result: [complex_n]f32 = undefined;
    try stream.memcpyDtoH(f32, &h_result, d_data);

    // cuFFT inverse is unnormalized — divide by N
    std.debug.print("Roundtrip (after IFFT/N):\n  ", .{});
    for (0..n) |k| {
        h_result[2 * k] /= @as(f32, @floatFromInt(n));
        h_result[2 * k + 1] /= @as(f32, @floatFromInt(n));
        std.debug.print("({d:.2},{d:.2}) ", .{ h_result[2 * k], h_result[2 * k + 1] });
    }
    std.debug.print("\n\n", .{});

    // Verify roundtrip matches input
    for (0..complex_n) |i| {
        if (@abs(h_result[i] - h_input[i]) > 1e-4) {
            std.debug.print("✗ Mismatch at index {}: {d:.6} vs {d:.6}\n", .{ i, h_result[i], h_input[i] });
            return error.ValidationFailed;
        }
    }
    std.debug.print("✓ C2C roundtrip verified (max error < 1e-4)\n", .{});
}
