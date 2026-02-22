/// zCUDA Integration Test: JIT Kernel Pipeline
/// Tests the NVRTC → PTX → Driver load → launch pipeline.
const std = @import("std");
const cuda = @import("zcuda");
const nvrtc = cuda.nvrtc;
const driver = cuda.driver;

test "JIT Kernel: NVRTC compile → load module → launch → verify" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const stream = ctx.defaultStream();

    // Step 1: Compile CUDA source to PTX
    const src =
        \\extern "C" __global__ void scale(float *data, float factor, int n) {
        \\    int i = blockIdx.x * blockDim.x + threadIdx.x;
        \\    if (i < n) data[i] *= factor;
        \\}
    ;
    const ptx = nvrtc.compilePtx(allocator, src) catch |err| {
        std.debug.print("NVRTC compile failed: {}\n", .{err});
        return error.SkipZigTest;
    };
    defer allocator.free(ptx);
    try std.testing.expect(ptx.len > 0);

    // Step 2: Load PTX module via driver safe API
    const module = ctx.loadModule(ptx) catch |err| {
        std.debug.print("Module load failed: {}\n", .{err});
        return error.SkipZigTest;
    };
    defer module.deinit();

    // Step 3: Get kernel function
    const func = module.getFunction("scale") catch |err| {
        std.debug.print("getFunction failed: {}\n", .{err});
        return error.SkipZigTest;
    };

    // Step 4: Prepare data
    const host_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var d_data = try stream.cloneHtoD(f32, &host_data);
    defer d_data.deinit();

    // Step 5: Launch kernel: data *= 3.0
    const factor: f32 = 3.0;
    const n: i32 = 4;
    try stream.launch(
        func,
        .{ .grid_dim = .{ .x = 1 }, .block_dim = .{ .x = 4 } },
        .{ &d_data, factor, n },
    );
    try ctx.synchronize();

    // Step 6: Verify results
    var result: [4]f32 = undefined;
    try stream.memcpyDtoH(f32, &result, d_data);
    const expected = [_]f32{ 3.0, 6.0, 9.0, 12.0 };
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(expected[i], result[i], 1e-5);
    }
}
