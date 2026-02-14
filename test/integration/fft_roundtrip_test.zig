/// zCUDA Integration Test: FFT Roundtrip
/// Tests cuFFT forward → inverse roundtrip to verify reconstruction.
const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;
const cufft = cuda.cufft;
const CufftPlan = cufft.CufftPlan;

test "FFT roundtrip: R2C forward → C2R inverse → verify" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const stream = ctx.defaultStream();

    const n: i32 = 8;
    const n_complex = @divExact(n, 2) + 1; // 5 complex = 10 floats

    // Input signal: [1, 2, 3, 4, 5, 6, 7, 8]
    const input_data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };

    // Forward plan: R2C
    const fwd_plan = CufftPlan.plan1d(n, .r2c_f32, 1) catch return error.SkipZigTest;
    defer fwd_plan.deinit();

    // Inverse plan: C2R
    const inv_plan = CufftPlan.plan1d(n, .c2r_f32, 1) catch return error.SkipZigTest;
    defer inv_plan.deinit();

    // Upload input
    const d_input = try stream.cloneHtod(f32, &input_data);
    defer d_input.deinit();

    // Allocate complex buffer (10 floats for 5 complex numbers)
    var d_complex = try stream.allocZeros(f32, allocator, @intCast(n_complex * 2));
    defer d_complex.deinit();

    // Allocate output buffer
    var d_output = try stream.allocZeros(f32, allocator, @intCast(n));
    defer d_output.deinit();

    // Step 1: Forward FFT (R2C)
    try fwd_plan.execR2C(d_input, d_complex);
    try ctx.synchronize();

    // Step 2: Inverse FFT (C2R) — cuFFT inverse is un-normalized
    try inv_plan.execC2R(d_complex, d_output);
    try ctx.synchronize();

    // Step 3: Verify roundtrip (result = input * N due to un-normalized inverse)
    var result: [8]f32 = undefined;
    try stream.memcpyDtoh(f32, &result, d_output);
    for (0..8) |i| {
        const expected = input_data[i] * @as(f32, @floatFromInt(n));
        try std.testing.expectApproxEqAbs(expected, result[i], 1e-3);
    }
}
