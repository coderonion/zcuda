/// cuFFT Pipeline: Custom Kernel → FFT → Custom Kernel
///
/// Pipeline: Zig kernel (generate sine wave) → cuFFT forward (R2C interleaved as C2C) →
///           Zig kernel (bandpass filter on separate re/im) → cuFFT inverse
///
/// Reference: cuda-samples/convolutionFFT2D, cuda-samples/simpleCUFFT_callback
//
// ── Kernel Loading: Way 5 (enhanced) build.zig auto-generated bridge module ──
const std = @import("std");
const cuda = @import("zcuda");

// kernel: generateSineWave(output, frequency, sample_rate, n)
const kernel_signal_gen = @import("kernel_signal_gen");

// kernel: bandpassFilter(data_re, data_im, low_bin, high_bin, n)
const kernel_freq_filter = @import("kernel_freq_filter");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    std.debug.print("=== cuFFT Pipeline: Signal → FFT → BandPass → IFFT ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    const stream = ctx.defaultStream();

    // ── Load custom kernels ──
    const mod_sig = try kernel_signal_gen.load(ctx, allocator);
    defer mod_sig.deinit();
    const sig_fn = try kernel_signal_gen.getFunction(mod_sig, .generateSineWave);

    const mod_flt = try kernel_freq_filter.load(ctx, allocator);
    defer mod_flt.deinit();
    const filter_fn = try kernel_freq_filter.getFunction(mod_flt, .bandpassFilter);

    const n: u32 = 1024;
    const frequency: f32 = 10.0; // 10 Hz
    const sample_rate: f32 = @as(f32, @floatFromInt(n)); // 1 sample/unit

    // ── Stage 1: Custom kernel — generate test sine wave signal ──
    // generateSineWave(output: [*]f32, frequency: f32, sample_rate: f32, n: u32)
    const d_signal = try stream.alloc(f32, allocator, n);
    defer d_signal.deinit();

    const sig_config = cuda.LaunchConfig.forNumElems(n);
    try stream.launch(sig_fn, sig_config, .{
        d_signal.devicePtr(), frequency, sample_rate, n,
    });
    std.debug.print("Stage 1: Generated {}-sample sine wave (freq={d:.0}Hz) on GPU\n", .{ n, frequency });

    // ── Stage 2: cuFFT forward — time domain → frequency domain ──
    // Use C2C plan on real data (imaginary part = 0)
    // We use two separate real arrays for re/im (matchng bandpassFilter's signature)
    const d_freq_re = try stream.alloc(f32, allocator, n);
    defer d_freq_re.deinit();
    const d_freq_im = try stream.allocZeros(f32, allocator, n);
    defer d_freq_im.deinit();

    // Copy signal to freq_re, then do in-place C2C (treating re=signal, im=0)
    // For simplicity: copy signal → d_freq_re and run DFT via C2C on interleaved buffer
    const d_complex = try stream.allocZeros(f32, allocator, n * 2); // interleaved re,im
    defer d_complex.deinit();

    // Upload signal as real parts of complex input
    const h_signal = try allocator.alloc(f32, n);
    defer allocator.free(h_signal);
    try stream.synchronize();
    try stream.memcpyDtoH(f32, h_signal, d_signal);
    var h_complex = try allocator.alloc(f32, n * 2);
    defer allocator.free(h_complex);
    for (0..n) |i| {
        h_complex[i * 2] = h_signal[i]; // real
        h_complex[i * 2 + 1] = 0.0; // imaginary
    }
    const d_complex_in = try stream.cloneHtoD(f32, h_complex);
    defer d_complex_in.deinit();

    const plan = try cuda.cufft.CufftPlan.plan1d(@intCast(n), .c2c_f32, 1);
    defer plan.deinit();
    try plan.setStream(stream);
    try plan.execC2C(d_complex_in, d_complex_in, .forward);
    std.debug.print("Stage 2: Forward FFT done\n", .{});

    // Extract re/im parts back to separate buffers for bandpassFilter
    var h_complex_out = try allocator.alloc(f32, n * 2);
    defer allocator.free(h_complex_out);
    try stream.synchronize();
    try stream.memcpyDtoH(f32, h_complex_out, d_complex_in);

    var h_re = try allocator.alloc(f32, n);
    defer allocator.free(h_re);
    var h_im = try allocator.alloc(f32, n);
    defer allocator.free(h_im);
    for (0..n) |i| {
        h_re[i] = h_complex_out[i * 2];
        h_im[i] = h_complex_out[i * 2 + 1];
    }

    const d_re = try stream.cloneHtoD(f32, h_re);
    defer d_re.deinit();
    const d_im = try stream.cloneHtoD(f32, h_im);
    defer d_im.deinit();

    // ── Stage 3: Custom kernel — bandpass filter ──
    // bandpassFilter(data_re, data_im, low_bin, high_bin, n)
    const low_bin: u32 = 5; // keep bins 5..20
    const high_bin: u32 = 20;
    try stream.launch(filter_fn, sig_config, .{
        d_re.devicePtr(), d_im.devicePtr(), low_bin, high_bin, n,
    });
    std.debug.print("Stage 3: Bandpass filter applied (bins {}..{})\n", .{ low_bin, high_bin });

    // ── Stage 4: IFFT — reconstruct filtered signal ──
    var h_re2 = try allocator.alloc(f32, n);
    defer allocator.free(h_re2);
    var h_im2 = try allocator.alloc(f32, n);
    defer allocator.free(h_im2);
    try stream.synchronize();
    try stream.memcpyDtoH(f32, h_re2, d_re);
    try stream.memcpyDtoH(f32, h_im2, d_im);

    // Pack back into interleaved buffer for IFFT
    for (0..n) |i| {
        h_complex[i * 2] = h_re2[i];
        h_complex[i * 2 + 1] = h_im2[i];
    }
    const d_ifft_in = try stream.cloneHtoD(f32, h_complex);
    defer d_ifft_in.deinit();
    try plan.execC2C(d_ifft_in, d_ifft_in, .inverse);
    std.debug.print("Stage 4: Inverse FFT done\n", .{});

    // ── Read back ──
    var h_result = try allocator.alloc(f32, n * 2);
    defer allocator.free(h_result);
    try stream.synchronize();
    try stream.memcpyDtoH(f32, h_result, d_ifft_in);

    const nf: f32 = @floatFromInt(n);
    std.debug.print("\nFirst 8 filtered samples (normalized real part):\n  [", .{});
    for (0..8) |i| std.debug.print(" {d:.4}", .{h_result[i * 2] / nf});
    std.debug.print(" ]\n", .{});

    std.debug.print("\n✓ Pipeline complete: SineGen → FFT → BandPass → IFFT\n", .{});
}
