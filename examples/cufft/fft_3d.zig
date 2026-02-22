/// cuFFT 3D FFT Example
///
/// Performs a 3D complex-to-complex FFT on a small volume.
///
/// Reference: CUDALibrarySamples/cuFFT/3d_c2c
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    std.debug.print("=== cuFFT 3D Complex-to-Complex FFT ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    const stream = ctx.defaultStream();

    // 4x4x4 volume
    const nx: usize = 4;
    const ny: usize = 4;
    const nz: usize = 4;
    const n_elems = nx * ny * nz;

    // Initialize with a single non-zero element (impulse at origin)
    var h_data: [n_elems * 2]f32 = [_]f32{0} ** (n_elems * 2);
    h_data[0] = 1.0; // real part of (0,0,0)
    h_data[1] = 0.0; // imag part of (0,0,0)

    std.debug.print("Input: 4x4x4 volume, impulse at origin\n", .{});
    std.debug.print("  data[0,0,0] = 1.0 + 0.0i\n\n", .{});

    var d_data = try stream.cloneHtoD(f32, &h_data);
    defer d_data.deinit();

    // Create 3D FFT plan
    const plan = try cuda.cufft.CufftPlan.plan3d(@intCast(nx), @intCast(ny), @intCast(nz), .c2c_f32);
    defer plan.deinit();

    // Forward FFT
    try plan.execC2C(d_data, d_data, .forward);
    try ctx.synchronize();

    try stream.memcpyDtoH(f32, &h_data, d_data);

    // Impulse at origin → all frequency bins = 1+0i
    std.debug.print("After forward FFT (impulse → flat spectrum):\n", .{});
    var all_one = true;
    for (0..n_elems) |i| {
        const re = h_data[i * 2];
        const im = h_data[i * 2 + 1];
        if (@abs(re - 1.0) > 0.01 or @abs(im) > 0.01) all_one = false;
    }
    std.debug.print("  First 4 bins: [{d:.2}+{d:.2}i, {d:.2}+{d:.2}i, {d:.2}+{d:.2}i, {d:.2}+{d:.2}i]\n", .{
        h_data[0], h_data[1], h_data[2], h_data[3],
        h_data[4], h_data[5], h_data[6], h_data[7],
    });
    std.debug.print("  All bins = 1+0i: {}\n\n", .{all_one});

    // Inverse FFT
    try plan.execC2C(d_data, d_data, .inverse);
    try ctx.synchronize();

    try stream.memcpyDtoH(f32, &h_data, d_data);

    // After inverse, result is N * original (N = 64 for 4x4x4)
    const scale: f32 = @floatFromInt(n_elems);
    std.debug.print("After inverse FFT (scaled by N={}):\n", .{n_elems});
    std.debug.print("  data[0,0,0] = {d:.1} (expected {d:.0})\n", .{ h_data[0], scale });
    std.debug.print("  data[1,0,0] = {d:.4} (expected 0)\n\n", .{h_data[2]});

    if (@abs(h_data[0] - scale) > 0.1) return error.ValidationFailed;

    std.debug.print("✓ 3D FFT roundtrip verified\n", .{});
}
