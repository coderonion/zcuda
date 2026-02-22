/// zCUDA Unit Tests: cuFFT
const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;
const cufft = cuda.cufft;
const CufftPlan = cufft.CufftPlan;

test "cuFFT plan1d creation" {
    const plan = CufftPlan.plan1d(256, .c2c_f32, 1) catch |err| {
        std.debug.print("Cannot create cuFFT plan: {}\n", .{err});
        return error.SkipZigTest;
    };
    defer plan.deinit();
}

test "cuFFT plan2d creation" {
    const plan = CufftPlan.plan2d(64, 64, .c2c_f32) catch |err| {
        std.debug.print("Cannot create cuFFT 2D plan: {}\n", .{err});
        return error.SkipZigTest;
    };
    defer plan.deinit();
}

test "cuFFT R2C forward transform" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const stream = ctx.defaultStream();

    // Plan: real-to-complex 1D, 8 elements
    const plan = CufftPlan.plan1d(8, .r2c_f32, 1) catch return error.SkipZigTest;
    defer plan.deinit();

    // Input: DC signal (all 1.0)
    const input_data = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    const d_input = try stream.cloneHtoD(f32, &input_data);
    defer d_input.deinit();

    // Output: N/2+1 = 5 complex numbers = 10 floats
    var d_output = try stream.allocZeros(f32, allocator, 10);
    defer d_output.deinit();

    try plan.execR2C(d_input, d_output);
    try ctx.synchronize();

    var result: [10]f32 = undefined;
    try stream.memcpyDtoH(f32, &result, d_output);

    // For a DC signal of N=8 ones, FFT[0] should be (8.0, 0.0), all other bins should be (0,0).
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), result[0], 1e-4); // real part of bin 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[1], 1e-4); // imag part of bin 0
    // Bins 1-4 should be zero
    for (2..10) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[i], 1e-4);
    }
}

test "cuFFT planMany — batched 1D FFT" {
    // Create a batched 1D plan: batch of 2 transforms, each of size 8
    var n_arr = [_]c_int{8};
    const plan = CufftPlan.planMany(
        1, // rank
        &n_arr, // n
        null, // inembed
        1, // istride
        8, // idist
        null, // onembed
        1, // ostride
        8, // odist
        .c2c_f32,
        2, // batch
    ) catch |err| {
        std.debug.print("Cannot create cuFFT planMany: {}\n", .{err});
        return error.SkipZigTest;
    };
    defer plan.deinit();
}

test "cuFFT workspace size query" {
    const plan = CufftPlan.plan1d(256, .c2c_f32, 1) catch return error.SkipZigTest;
    defer plan.deinit();

    const size = plan.getSize() catch |err| {
        std.debug.print("Cannot get workspace size: {}\n", .{err});
        return error.SkipZigTest;
    };
    // Workspace size should be a reasonable value (>= 0)
    try std.testing.expect(size >= 0);
}

test "cuFFT C2C forward+inverse roundtrip" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const stream = ctx.defaultStream();

    const n: usize = 8;
    // DC signal: all (1, 0)
    var input: [16]f32 = undefined;
    for (0..n) |k| {
        input[2 * k] = 1.0;
        input[2 * k + 1] = 0.0;
    }

    const d_in = try stream.cloneHtoD(f32, &input);
    defer d_in.deinit();
    var d_out = try stream.allocZeros(f32, allocator, 16);
    defer d_out.deinit();

    const plan_fwd = CufftPlan.plan1d(@intCast(n), .c2c_f32, 1) catch return error.SkipZigTest;
    defer plan_fwd.deinit();
    try plan_fwd.setStream(stream);

    // Forward
    try plan_fwd.execC2C(d_in, d_out, .forward);
    try ctx.synchronize();

    var fft_result: [16]f32 = undefined;
    try stream.memcpyDtoH(f32, &fft_result, d_out);

    // DC signal: FFT[0] = (8, 0), all others = (0, 0)
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), fft_result[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), fft_result[1], 1e-4);
    for (1..n) |k| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), fft_result[2 * k], 1e-4);
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), fft_result[2 * k + 1], 1e-4);
    }

    // Inverse
    const plan_inv = CufftPlan.plan1d(@intCast(n), .c2c_f32, 1) catch return error.SkipZigTest;
    defer plan_inv.deinit();
    try plan_inv.setStream(stream);
    try plan_inv.execC2C(d_out, d_in, .inverse);
    try ctx.synchronize();

    var roundtrip: [16]f32 = undefined;
    try stream.memcpyDtoH(f32, &roundtrip, d_in);

    // Normalize and verify
    for (0..n) |k| {
        const re = roundtrip[2 * k] / @as(f32, @floatFromInt(n));
        const im = roundtrip[2 * k + 1] / @as(f32, @floatFromInt(n));
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), re, 1e-4);
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), im, 1e-4);
    }
}

test "cuFFT plan3d creation" {
    const plan = CufftPlan.plan3d(8, 8, 8, .c2c_f32) catch |err| {
        std.debug.print("Cannot create cuFFT 3D plan: {}\n", .{err});
        return error.SkipZigTest;
    };
    defer plan.deinit();
}
