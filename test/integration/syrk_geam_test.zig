/// zCUDA Integration Test: cuBLAS SYRK → GEAM pipeline
///
/// Computes A*A^T (via SYRK), then adds identity (via GEAM) → regularized Gram matrix.
const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;
const cublas = cuda.cublas;
const CublasContext = cublas.CublasContext;

test "cuBLAS SYRK → GEAM: regularized Gram matrix" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const blas = CublasContext.init(ctx) catch return error.SkipZigTest;
    defer blas.deinit();
    const stream = ctx.defaultStream();

    const n: i32 = 2;
    const k: i32 = 3;

    // A = [[1,3,5],[2,4,6]] (2x3 col-major)
    const a_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    // Identity (2x2)
    const eye_data = [_]f32{ 1, 0, 0, 1 };

    const d_a = try stream.cloneHtod(f32, &a_data);
    defer d_a.deinit();
    var d_c = try stream.allocZeros(f32, allocator, 4);
    defer d_c.deinit();
    const d_eye = try stream.cloneHtod(f32, &eye_data);
    defer d_eye.deinit();
    var d_result = try stream.allocZeros(f32, allocator, 4);
    defer d_result.deinit();

    // Step 1: C = A * A^T (SYRK)
    try blas.ssyrk(.lower, .no_transpose, n, k, 1.0, d_a, n, 0.0, d_c, n);
    try ctx.synchronize();

    // Step 2: Result = C + 0.1 * I (GEAM for regularization)
    try blas.sgeam(.no_transpose, .no_transpose, n, n, 1.0, d_c, n, 0.1, d_eye, n, d_result, n);
    try ctx.synchronize();

    var result: [4]f32 = undefined;
    try stream.memcpyDtoh(f32, &result, d_result);

    // A*A^T = [[35, 44], [44, 56]]
    // + 0.1*I = [[35.1, 44], [44, 56.1]]
    // Note: SYRK only fills lower triangle, GEAM with no_transpose operates on full matrix.
    // Lower triangle values:
    try std.testing.expectApproxEqAbs(@as(f32, 35.1), result[0], 0.01); // (0,0)
    try std.testing.expectApproxEqAbs(@as(f32, 44.0), result[1], 0.01); // (1,0)
    try std.testing.expectApproxEqAbs(@as(f32, 56.1), result[3], 0.01); // (1,1)
}
