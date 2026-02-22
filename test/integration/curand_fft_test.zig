/// zCUDA Integration Test: cuRAND → cuFFT spectral analysis pipeline
///
/// Generates random data with cuRAND, applies FFT, verifies spectral properties.
const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;
const curand = cuda.curand;
const cufft = cuda.cufft;

test "cuRAND → cuFFT: random signal spectral analysis" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const stream = ctx.defaultStream();

    // Generate random signal with cuRAND
    const n: usize = 256;
    const rng = curand.CurandContext.init(ctx, .philox4_32_10) catch return error.SkipZigTest;
    defer rng.deinit();
    try rng.setSeed(42);
    try rng.setStream(stream);

    const d_signal = try stream.alloc(f32, allocator, n);
    defer d_signal.deinit();
    try rng.fillNormal(d_signal, 0.0, 1.0);

    // R2C FFT
    const complex_n = (n / 2 + 1) * 2;
    var d_freq = try stream.allocZeros(f32, allocator, complex_n);
    defer d_freq.deinit();

    const plan = cufft.CufftPlan.plan1d(@intCast(n), .r2c_f32, 1) catch return error.SkipZigTest;
    defer plan.deinit();
    try plan.setStream(stream);
    try plan.execR2C(d_signal, d_freq);
    try ctx.synchronize();

    // Read back and verify spectral properties
    var h_freq: [258]f32 = undefined; // (128+1)*2
    try stream.memcpyDtoH(f32, h_freq[0..complex_n], d_freq);

    // For white noise, energy should be spread across bins (not concentrated)
    var total_energy: f64 = 0;
    var max_energy: f64 = 0;
    for (0..n / 2 + 1) |k| {
        const re = h_freq[2 * k];
        const im = h_freq[2 * k + 1];
        const e = @as(f64, re) * @as(f64, re) + @as(f64, im) * @as(f64, im);
        total_energy += e;
        if (k > 0) max_energy = @max(max_energy, e); // Skip DC
    }

    // No single bin (except DC) should have more than 20% of total energy for white noise
    try std.testing.expect(total_energy > 0);
    try std.testing.expect(max_energy < total_energy * 0.2);
}
