/// cuFFT Pipeline: 2D Convolution via FFT
///
/// Pipeline: custom kernel (pad kernel) → cuFFT 2D forward (signal + kernel) →
///           custom kernel (pointwise complex multiply) → cuFFT 2D inverse
/// Classic frequency‑domain convolution: conv(f, g) = IFFT(FFT(f) · FFT(g))
///
/// Reference: cuda-samples/convolutionFFT2D
//
// ── Kernel Loading: Way 5 (enhanced) build.zig auto-generated bridge module ──
const std = @import("std");
const cuda = @import("zcuda");

// kernel: zeroPad2d(src, dst, src_rows, src_cols, dst_cols)
const kernel_pad_2d = @import("kernel_pad_2d");

// kernel: complexMul(a_re, a_im, b_re, b_im, out_re, out_im, n)
const kernel_complex_mul = @import("kernel_complex_mul");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    std.debug.print("=== cuFFT Pipeline: 2D Convolution ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    const stream = ctx.defaultStream();

    // ── Load custom kernels ──
    const mod_pad = try kernel_pad_2d.load(ctx, allocator);
    defer mod_pad.deinit();
    const pad_fn = try kernel_pad_2d.getFunction(mod_pad, .zeroPad2d);

    const mod_mul = try kernel_complex_mul.load(ctx, allocator);
    defer mod_mul.deinit();
    const mul_fn = try kernel_complex_mul.getFunction(mod_mul, .complexMul);

    // Image and convolution kernel dimensions
    const img_h: u32 = 32;
    const img_w: u32 = 32;
    const kern_h: u32 = 5;
    const kern_w: u32 = 5;
    // FFT dimensions (power of 2 ≥ image dimensions)
    const fft_h: u32 = 64;
    const fft_w: u32 = 64;
    const fft_n: u32 = fft_h * fft_w;

    // ── Stage 1: Pad image and convolution kernel to FFT size ──
    // Create gradient test image on host
    var h_img: [img_h * img_w]f32 = undefined;
    for (0..img_h) |r| for (0..img_w) |c| {
        h_img[r * img_w + c] = @as(f32, @floatFromInt(r + c)) / 64.0;
    };
    const d_img = try stream.cloneHtoD(f32, &h_img);
    defer d_img.deinit();

    // Gaussian-like 5x5 convolution kernel (normalized)
    var h_kern: [kern_h * kern_w]f32 = .{
        1, 4,  6,  4,  1, 4, 16, 24, 16, 4, 6, 24, 36, 24, 6,
        4, 16, 24, 16, 4, 1, 4,  6,  4,  1,
    };
    for (&h_kern) |*v| v.* /= 256.0;
    const d_kern_src = try stream.cloneHtoD(f32, &h_kern);
    defer d_kern_src.deinit();

    // Padded output buffers (real-valued, FFT will produce complex output)
    const d_img_pad = try stream.allocZeros(f32, allocator, fft_n);
    defer d_img_pad.deinit();
    const d_kern_pad = try stream.allocZeros(f32, allocator, fft_n);
    defer d_kern_pad.deinit();

    // zeroPad2d(src, dst, src_rows, src_cols, dst_cols)
    const img_grid_x = @divTrunc(img_w + 31, 32);
    const img_grid_y = @divTrunc(img_h + 7, 8);
    const img_pad_config = cuda.LaunchConfig{
        .grid_dim = .{ .x = img_grid_x, .y = img_grid_y },
        .block_dim = .{ .x = 32, .y = 8 },
    };
    try stream.launch(pad_fn, img_pad_config, .{
        d_img.devicePtr(), d_img_pad.devicePtr(), img_h, img_w, fft_w,
    });

    const kern_grid_x = @divTrunc(kern_w + 31, 32);
    const kern_grid_y = @divTrunc(kern_h + 7, 8);
    const kern_pad_config = cuda.LaunchConfig{
        .grid_dim = .{ .x = kern_grid_x, .y = kern_grid_y },
        .block_dim = .{ .x = 32, .y = 8 },
    };
    try stream.launch(pad_fn, kern_pad_config, .{
        d_kern_src.devicePtr(), d_kern_pad.devicePtr(), kern_h, kern_w, fft_w,
    });
    std.debug.print("Stage 1: Image and kernel padded to {}×{}\n", .{ fft_h, fft_w });

    // ── Stage 2: cuFFT 2D R2C forward on padded real buffers ──
    // We use C2C with zero imaginary parts
    // Allocate separate re/im buffers for complexMul kernel
    const d_img_re = try stream.allocZeros(f32, allocator, fft_n);
    defer d_img_re.deinit();
    const d_img_im = try stream.allocZeros(f32, allocator, fft_n);
    defer d_img_im.deinit();
    const d_kern_re = try stream.allocZeros(f32, allocator, fft_n);
    defer d_kern_re.deinit();
    const d_kern_im = try stream.allocZeros(f32, allocator, fft_n);
    defer d_kern_im.deinit();
    const d_out_re = try stream.allocZeros(f32, allocator, fft_n);
    defer d_out_re.deinit();
    const d_out_im = try stream.allocZeros(f32, allocator, fft_n);
    defer d_out_im.deinit();

    // Copy padded real data into re arrays
    try stream.memcpyDtoD(f32, d_img_re, d_img_pad);
    try stream.memcpyDtoD(f32, d_kern_re, d_kern_pad);

    // Perform C2C FFT using interleaved complex buffers (allocate for plan)
    const d_img_complex = try stream.allocZeros(f32, allocator, fft_n * 2);
    defer d_img_complex.deinit();
    const d_kern_complex = try stream.allocZeros(f32, allocator, fft_n * 2);
    defer d_kern_complex.deinit();

    // Read image and kernel re back to host to interleave
    try stream.synchronize();
    const h_img_pad = try allocator.alloc(f32, fft_n);
    defer allocator.free(h_img_pad);
    const h_kern_pad = try allocator.alloc(f32, fft_n);
    defer allocator.free(h_kern_pad);
    try stream.memcpyDtoH(f32, h_img_pad, d_img_pad);
    try stream.memcpyDtoH(f32, h_kern_pad, d_kern_pad);

    const h_img_complex = try allocator.alloc(f32, fft_n * 2);
    defer allocator.free(h_img_complex);
    const h_kern_complex = try allocator.alloc(f32, fft_n * 2);
    defer allocator.free(h_kern_complex);
    for (0..fft_n) |i| {
        h_img_complex[i * 2] = h_img_pad[i];
        h_img_complex[i * 2 + 1] = 0.0;
        h_kern_complex[i * 2] = h_kern_pad[i];
        h_kern_complex[i * 2 + 1] = 0.0;
    }
    const d_img_c = try stream.cloneHtoD(f32, h_img_complex);
    defer d_img_c.deinit();
    const d_kern_c = try stream.cloneHtoD(f32, h_kern_complex);
    defer d_kern_c.deinit();

    const plan = try cuda.cufft.CufftPlan.plan2d(@intCast(fft_h), @intCast(fft_w), .c2c_f32);
    defer plan.deinit();
    try plan.setStream(stream);
    try plan.execC2C(d_img_c, d_img_c, .forward);
    try plan.execC2C(d_kern_c, d_kern_c, .forward);
    std.debug.print("Stage 2: 2D Forward FFT done (image + kernel)\n", .{});

    // Extract re/im from interleaved complex for complexMul kernel
    try stream.synchronize();
    const h_img_freq = try allocator.alloc(f32, fft_n * 2);
    defer allocator.free(h_img_freq);
    const h_kern_freq = try allocator.alloc(f32, fft_n * 2);
    defer allocator.free(h_kern_freq);
    try stream.memcpyDtoH(f32, h_img_freq, d_img_c);
    try stream.memcpyDtoH(f32, h_kern_freq, d_kern_c);

    const h_img_re = try allocator.alloc(f32, fft_n);
    defer allocator.free(h_img_re);
    const h_img_im = try allocator.alloc(f32, fft_n);
    defer allocator.free(h_img_im);
    const h_kern_re = try allocator.alloc(f32, fft_n);
    defer allocator.free(h_kern_re);
    const h_kern_im = try allocator.alloc(f32, fft_n);
    defer allocator.free(h_kern_im);
    for (0..fft_n) |i| {
        h_img_re[i] = h_img_freq[i * 2];
        h_img_im[i] = h_img_freq[i * 2 + 1];
        h_kern_re[i] = h_kern_freq[i * 2];
        h_kern_im[i] = h_kern_freq[i * 2 + 1];
    }
    const d_a_re = try stream.cloneHtoD(f32, h_img_re);
    defer d_a_re.deinit();
    const d_a_im = try stream.cloneHtoD(f32, h_img_im);
    defer d_a_im.deinit();
    const d_b_re = try stream.cloneHtoD(f32, h_kern_re);
    defer d_b_re.deinit();
    const d_b_im = try stream.cloneHtoD(f32, h_kern_im);
    defer d_b_im.deinit();

    // ── Stage 3: complexMul(a_re, a_im, b_re, b_im, out_re, out_im, n) ──
    const elem_config = cuda.LaunchConfig.forNumElems(fft_n);
    try stream.launch(mul_fn, elem_config, .{
        d_a_re.devicePtr(),   d_a_im.devicePtr(),
        d_b_re.devicePtr(),   d_b_im.devicePtr(),
        d_out_re.devicePtr(), d_out_im.devicePtr(),
        fft_n,
    });
    std.debug.print("Stage 3: Pointwise complex multiply done\n", .{});

    // ── Stage 4: IFFT — pack back to interleaved, run inverse ──
    try stream.synchronize();
    const h_out_re = try allocator.alloc(f32, fft_n);
    defer allocator.free(h_out_re);
    const h_out_im = try allocator.alloc(f32, fft_n);
    defer allocator.free(h_out_im);
    try stream.memcpyDtoH(f32, h_out_re, d_out_re);
    try stream.memcpyDtoH(f32, h_out_im, d_out_im);

    const h_conv_complex = try allocator.alloc(f32, fft_n * 2);
    defer allocator.free(h_conv_complex);
    for (0..fft_n) |i| {
        h_conv_complex[i * 2] = h_out_re[i];
        h_conv_complex[i * 2 + 1] = h_out_im[i];
    }
    const d_conv_c = try stream.cloneHtoD(f32, h_conv_complex);
    defer d_conv_c.deinit();

    try plan.execC2C(d_conv_c, d_conv_c, .inverse);
    std.debug.print("Stage 4: 2D Inverse FFT done\n", .{});

    // ── Read back and normalize ──
    try stream.synchronize();
    const h_result = try allocator.alloc(f32, fft_n * 2);
    defer allocator.free(h_result);
    try stream.memcpyDtoH(f32, h_result, d_conv_c);

    const nf: f32 = @floatFromInt(fft_n);
    std.debug.print("\nFirst 4×4 of convolved image (real parts, normalized):\n", .{});
    for (0..4) |r| {
        std.debug.print("  [", .{});
        for (0..4) |c| {
            const real = h_result[(r * fft_w + c) * 2] / nf;
            std.debug.print(" {d:7.4}", .{real});
        }
        std.debug.print(" ]\n", .{});
    }

    std.debug.print("\n✓ Pipeline complete: PadKernel → FFT2D → PointwiseMul → IFFT2D\n", .{});
}
