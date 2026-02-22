// examples/kernel/10_Integration/G_EndToEnd/matmul_e2e.zig
// Reference: End-to-end matrix multiplication: load → compute → verify
// API: Full zcuda pipeline with matmul kernel
//
// ── Kernel Loading: Way 5 build.zig auto-generated bridge module ──
// Uses: @import("kernel_matmul_naive") — type-safe PTX loading via bridge module

const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;

const kernel_matmul_naive = @import("kernel_matmul_naive");

/// End-to-end matmul: device kernel vs CPU reference.
pub fn main() !void {
    var ctx = try driver.CudaContext.new(0);
    defer ctx.deinit();
    var stream = try ctx.newStream();
    defer stream.deinit();

    const module = try kernel_matmul_naive.load(ctx, std.heap.page_allocator);
    defer module.deinit();
    const func = try kernel_matmul_naive.getFunction(module, .matmulNaive);

    const M: u32 = 32;
    const N: u32 = 32;
    const K: u32 = 32;

    var h_A: [M * K]f32 = undefined;
    var h_B: [K * N]f32 = undefined;
    for (0..M * K) |i| h_A[i] = @as(f32, @floatFromInt(i % 7)) * 0.1;
    for (0..K * N) |i| h_B[i] = @as(f32, @floatFromInt(i % 5)) * 0.2;

    var h_C_ref: [M * N]f32 = [_]f32{0.0} ** (M * N);
    for (0..M) |r| {
        for (0..N) |c| {
            var sum: f32 = 0.0;
            for (0..K) |k| {
                sum += h_A[r * K + k] * h_B[k * N + c];
            }
            h_C_ref[r * N + c] = sum;
        }
    }

    var d_A = try stream.cloneHtoD(f32, &h_A);
    defer d_A.deinit();
    var d_B = try stream.cloneHtoD(f32, &h_B);
    defer d_B.deinit();
    var d_C = try stream.allocZeros(f32, std.heap.page_allocator, M * N);
    defer d_C.deinit();

    try stream.launch(func, .{
        .grid_dim = .{ .x = (N + 15) / 16, .y = (M + 15) / 16 },
        .block_dim = .{ .x = 16, .y = 16 },
    }, .{
        d_A.devicePtr(), d_B.devicePtr(), d_C.devicePtr(), M, N, K,
    });

    var h_C_gpu: [M * N]f32 = undefined;
    try stream.memcpyDtoHAsync(f32, &h_C_gpu, d_C);
    try stream.synchronize();

    for (0..M * N) |i| {
        const diff = @abs(h_C_gpu[i] - h_C_ref[i]);
        if (diff > 1e-3) return error.MatmulMismatch;
    }
}
