/// zCUDA Integration Test: Shared Memory GPU Operations
///
/// Tests shared memory correctness on GPU by running kernels that use
/// SharedArray for transpose, reduction, and inter-thread communication.
/// Requires: zig build compile-kernels && zig build test-integration
const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;
const h = @import("test_helpers");

// ============================================================================
// Shared memory reduction (uses shared_mem_test kernel)
// ============================================================================

test "shared mem — reduceSum via shared memory matches CPU" {
    const allocator = std.testing.allocator;
    const env = try h.initCuda();
    defer env.ctx.deinit();

    const ptx = h.readPtxFile(allocator, "shared_mem_test") catch return error.SkipZigTest;
    defer allocator.free(ptx);

    const module = try env.ctx.loadModule(ptx);
    defer module.deinit();

    const func = try module.getFunction("smem_reduce");
    const n: u32 = 256;

    // Input: 1, 2, 3, ..., 256
    var input: [256]f32 = undefined;
    for (0..n) |i| {
        input[i] = @as(f32, @floatFromInt(i + 1));
    }
    const expected: f32 = @as(f32, @floatFromInt(n * (n + 1) / 2));

    var d_input = try env.stream.cloneHtoD(f32, &input);
    defer d_input.deinit();
    var d_output = try env.stream.allocZeros(f32, allocator, 1);
    defer d_output.deinit();

    try env.stream.launch(func, .{
        .grid_dim = .{ .x = 1 },
        .block_dim = .{ .x = n },
    }, .{ &d_input, &d_output, n });

    var result: [1]f32 = undefined;
    try env.stream.memcpyDtoH(f32, &result, d_output);
    try env.stream.synchronize();

    try std.testing.expectApproxEqRel(expected, result[0], 1e-4);
}

// ============================================================================
// Shared memory transpose (uses shared_mem_test kernel)
// ============================================================================

test "shared mem — 32×32 transpose via shared memory matches CPU" {
    const allocator = std.testing.allocator;
    const env = try h.initCuda();
    defer env.ctx.deinit();

    const ptx = h.readPtxFile(allocator, "shared_mem_test") catch return error.SkipZigTest;
    defer allocator.free(ptx);

    const module = try env.ctx.loadModule(ptx);
    defer module.deinit();

    const func = try module.getFunction("smem_transpose");
    const dim: u32 = 32;
    const n = dim * dim;

    // Input: row-major matrix
    var input: [32 * 32]f32 = undefined;
    for (0..n) |i| {
        input[i] = @as(f32, @floatFromInt(i));
    }

    // CPU reference transpose
    var cpu_transpose: [32 * 32]f32 = undefined;
    for (0..dim) |row| {
        for (0..dim) |col| {
            cpu_transpose[col * dim + row] = input[row * dim + col];
        }
    }

    var d_input = try env.stream.cloneHtoD(f32, &input);
    defer d_input.deinit();
    var d_output = try env.stream.allocZeros(f32, allocator, n);
    defer d_output.deinit();

    try env.stream.launch(func, .{
        .grid_dim = .{ .x = dim / 16, .y = dim / 16 },
        .block_dim = .{ .x = 16, .y = 16 },
    }, .{ &d_input, &d_output, dim, dim });

    var gpu_transpose: [32 * 32]f32 = undefined;
    try env.stream.memcpyDtoH(f32, &gpu_transpose, d_output);
    try env.stream.synchronize();

    for (0..n) |i| {
        try std.testing.expectEqual(cpu_transpose[i], gpu_transpose[i]);
    }
}

// ============================================================================
// Tiled matmul (uses shared memory tiles for performance)
// ============================================================================

test "shared mem — tiled matmul 32×32 matches naive matmul" {
    const allocator = std.testing.allocator;
    const env = try h.initCuda();
    defer env.ctx.deinit();

    const ptx_naive = h.readPtxFile(allocator, "matmul") catch return error.SkipZigTest;
    defer allocator.free(ptx_naive);
    const ptx_tiled = h.readPtxFile(allocator, "tiled_matmul") catch return error.SkipZigTest;
    defer allocator.free(ptx_tiled);

    const mod_naive = try env.ctx.loadModule(ptx_naive);
    defer mod_naive.deinit();
    const mod_tiled = try env.ctx.loadModule(ptx_tiled);
    defer mod_tiled.deinit();

    const func_naive = try mod_naive.getFunction("matmulNaive");
    const func_tiled = try mod_tiled.getFunction("tiled_matmul");

    const M: u32 = 32;
    const size = M * M;

    var a: [32 * 32]f32 = undefined;
    var b: [32 * 32]f32 = undefined;
    for (0..size) |i| {
        a[i] = @sin(@as(f32, @floatFromInt(i)) * 0.1);
        b[i] = @cos(@as(f32, @floatFromInt(i)) * 0.07);
    }

    var d_a = try env.stream.cloneHtoD(f32, &a);
    defer d_a.deinit();
    var d_b = try env.stream.cloneHtoD(f32, &b);
    defer d_b.deinit();
    var d_c_naive = try env.stream.allocZeros(f32, allocator, size);
    defer d_c_naive.deinit();
    var d_c_tiled = try env.stream.allocZeros(f32, allocator, size);
    defer d_c_tiled.deinit();

    // Launch naive — 16×16 blocks, 2×2 grid
    try env.stream.launch(func_naive, .{
        .grid_dim = .{ .x = 2, .y = 2 },
        .block_dim = .{ .x = 16, .y = 16 },
    }, .{ &d_a, &d_b, &d_c_naive, M, M, M });

    // Launch tiled — must use TILE=16 block dims
    try env.stream.launch(func_tiled, .{
        .grid_dim = .{ .x = 2, .y = 2 },
        .block_dim = .{ .x = 16, .y = 16 },
    }, .{ &d_a, &d_b, &d_c_tiled, M, M, M });

    try env.ctx.synchronize();

    var c_naive: [32 * 32]f32 = undefined;
    var c_tiled: [32 * 32]f32 = undefined;
    try env.stream.memcpyDtoH(f32, &c_naive, d_c_naive);
    try env.stream.memcpyDtoH(f32, &c_tiled, d_c_tiled);

    // Tiled and naive should produce equivalent results
    for (0..size) |i| {
        try std.testing.expectApproxEqAbs(c_naive[i], c_tiled[i], 1e-3);
    }
}
