/// zCUDA Unit Tests: cuRAND
const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;
const curand = cuda.curand;
const CurandContext = curand.CurandContext;

test "cuRAND generator creation and seed" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();

    const rng = CurandContext.init(ctx, .default) catch |err| {
        std.debug.print("Cannot create cuRAND generator: {}\n", .{err});
        return error.SkipZigTest;
    };
    defer rng.deinit();
    try rng.setSeed(42);
}

test "cuRAND fillUniform — values in [0,1)" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();

    const rng = CurandContext.init(ctx, .default) catch return error.SkipZigTest;
    defer rng.deinit();
    try rng.setSeed(42);

    const stream = ctx.defaultStream();
    const n: usize = 1024;
    const data = try stream.alloc(f32, allocator, n);
    defer data.deinit();

    try rng.fillUniform(data);
    try ctx.synchronize();

    // Read back and check range
    var host: [1024]f32 = undefined;
    try stream.memcpyDtoH(f32, &host, data);

    for (host) |val| {
        try std.testing.expect(val >= 0.0);
        try std.testing.expect(val <= 1.0);
    }
}

test "cuRAND fillNormal — non-zero output" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();

    const rng = CurandContext.init(ctx, .default) catch return error.SkipZigTest;
    defer rng.deinit();
    try rng.setSeed(123);

    const stream = ctx.defaultStream();
    const n: usize = 1024;
    const data = try stream.alloc(f32, allocator, n);
    defer data.deinit();

    try rng.fillNormal(data, 0.0, 1.0);
    try ctx.synchronize();

    var host: [1024]f32 = undefined;
    try stream.memcpyDtoH(f32, &host, data);

    // Check that we got some nonzero values (mean=0 but stdev=1, so most will be nonzero)
    var nonzero: usize = 0;
    for (host) |val| {
        if (@abs(val) > 1e-10) nonzero += 1;
    }
    try std.testing.expect(nonzero > 900); // at least 90%
}

test "cuRAND fillLogNormal — all positive values" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();

    const rng = CurandContext.init(ctx, .default) catch return error.SkipZigTest;
    defer rng.deinit();
    try rng.setSeed(99);

    const stream = ctx.defaultStream();
    const n: usize = 1024;
    const data = try stream.alloc(f32, allocator, n);
    defer data.deinit();

    try rng.fillLogNormal(data, 0.0, 1.0);
    try ctx.synchronize();

    var host: [1024]f32 = undefined;
    try stream.memcpyDtoH(f32, &host, data);

    for (host) |val| {
        try std.testing.expect(val > 0.0); // log-normal is always positive
    }
}

test "cuRAND fillPoisson — non-negative integer values" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();

    const rng = CurandContext.init(ctx, .default) catch return error.SkipZigTest;
    defer rng.deinit();
    try rng.setSeed(42);

    const stream = ctx.defaultStream();
    const n: usize = 1024;
    const data = try stream.alloc(u32, allocator, n);
    defer data.deinit();

    // Poisson distribution with lambda=5.0
    try rng.fillPoisson(data, 5.0);
    try ctx.synchronize();

    var host: [1024]u32 = undefined;
    try stream.memcpyDtoH(u32, &host, data);

    // Poisson values should be non-negative and reasonable (for lambda=5, most values < 20)
    var sum: u64 = 0;
    for (host) |val| {
        try std.testing.expect(val < 100); // sanity check
        sum += val;
    }
    // Mean should be close to lambda=5 (sum/n ≈ 5)
    const mean = @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(n));
    try std.testing.expect(mean > 3.0);
    try std.testing.expect(mean < 7.0);
}

test "cuRAND fillUniformDouble — f64 values in [0,1)" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();

    const rng = CurandContext.init(ctx, .default) catch return error.SkipZigTest;
    defer rng.deinit();
    try rng.setSeed(42);

    const stream = ctx.defaultStream();
    const n: usize = 512;
    const data = try stream.alloc(f64, allocator, n);
    defer data.deinit();

    try rng.fillUniformDouble(data);
    try ctx.synchronize();

    var host: [512]f64 = undefined;
    try stream.memcpyDtoH(f64, &host, data);

    for (host) |val| {
        try std.testing.expect(val >= 0.0);
        try std.testing.expect(val <= 1.0);
    }
}

test "cuRAND Philox generator" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();

    const rng = CurandContext.init(ctx, .philox4_32_10) catch return error.SkipZigTest;
    defer rng.deinit();
    try rng.setSeed(42);

    const allocator = std.testing.allocator;
    const stream = ctx.defaultStream();
    const data = try stream.alloc(f32, allocator, 1024);
    defer data.deinit();

    try rng.fillUniform(data);
    try ctx.synchronize();

    var host: [1024]f32 = undefined;
    try stream.memcpyDtoH(f32, &host, data);

    // Verify range and non-trivial output
    var sum: f64 = 0.0;
    for (host) |val| {
        try std.testing.expect(val >= 0.0);
        try std.testing.expect(val <= 1.0);
        sum += @as(f64, val);
    }
    const mean = sum / 1024.0;
    // Mean should be ~0.5
    try std.testing.expect(mean > 0.3);
    try std.testing.expect(mean < 0.7);
}
