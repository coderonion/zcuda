/// zCUDA Integration Test: GEMM end-to-end pipeline
const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;
const cublas = cuda.cublas;
const CublasContext = cublas.CublasContext;

test "GEMM end-to-end: 4x4 identity multiply" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const cb = CublasContext.init(ctx) catch return error.SkipZigTest;
    defer cb.deinit();
    const stream = ctx.defaultStream();

    const n: i32 = 4;

    // A = identity 4x4 (col-major)
    var a_data: [16]f32 = .{0} ** 16;
    a_data[0] = 1;
    a_data[5] = 1;
    a_data[10] = 1;
    a_data[15] = 1;

    // B = sequential (col-major)
    var b_data: [16]f32 = undefined;
    for (0..4) |col| {
        for (0..4) |row| {
            b_data[col * 4 + row] = @floatFromInt(row * 4 + col + 1);
        }
    }

    const c_data = [_]f32{0} ** 16;

    const d_a = try stream.cloneHtoD(f32, &a_data);
    defer d_a.deinit();
    const d_b = try stream.cloneHtoD(f32, &b_data);
    defer d_b.deinit();
    var d_c = try stream.cloneHtoD(f32, &c_data);
    defer d_c.deinit();

    // C = I * B = B
    try cb.sgemm(.no_transpose, .no_transpose, n, n, n, 1.0, d_a, n, d_b, n, 0.0, d_c, n);
    try ctx.synchronize();

    var result: [16]f32 = undefined;
    try stream.memcpyDtoH(f32, &result, d_c);

    for (0..16) |i| {
        try std.testing.expectApproxEqAbs(b_data[i], result[i], 1e-5);
    }
}
