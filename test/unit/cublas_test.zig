/// zCUDA Unit Tests: cuBLAS
/// Tests all cuBLAS L1/L2/L3 operations with numerical verification.
const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;
const cublas = cuda.cublas;
const CublasContext = cublas.CublasContext;

// ============================================================================
// L1 Tests
// ============================================================================

test "cuBLAS SAXPY: y = alpha*x + y" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const blas = CublasContext.init(ctx) catch return error.SkipZigTest;
    defer blas.deinit();
    const stream = ctx.defaultStream();

    const x_host = [_]f32{ 1, 2, 3, 4 };
    const y_host = [_]f32{ 10, 20, 30, 40 };

    const x_dev = try stream.cloneHtod(f32, &x_host);
    defer x_dev.deinit();
    var y_dev = try stream.cloneHtod(f32, &y_host);
    defer y_dev.deinit();

    try blas.saxpy(4, 2.0, x_dev, y_dev);
    try ctx.synchronize();

    var result: [4]f32 = undefined;
    try stream.memcpyDtoh(f32, &result, y_dev);

    // y = 2*x + y = [12, 24, 36, 48]
    const expected = [_]f32{ 12, 24, 36, 48 };
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(expected[i], result[i], 1e-5);
    }
}

test "cuBLAS SDOT: dot product" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const blas = CublasContext.init(ctx) catch return error.SkipZigTest;
    defer blas.deinit();
    const stream = ctx.defaultStream();

    const x_host = [_]f32{ 1, 2, 3 };
    const y_host = [_]f32{ 4, 5, 6 };

    const x_dev = try stream.cloneHtod(f32, &x_host);
    defer x_dev.deinit();
    const y_dev = try stream.cloneHtod(f32, &y_host);
    defer y_dev.deinit();

    try ctx.synchronize();
    const dot = try blas.sdot(3, x_dev, y_dev);
    try std.testing.expectApproxEqAbs(@as(f32, 32), dot, 1e-5);
}

test "cuBLAS SNRM2: Euclidean norm" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const blas = CublasContext.init(ctx) catch return error.SkipZigTest;
    defer blas.deinit();
    const stream = ctx.defaultStream();

    const x_host = [_]f32{ 3, 4 };
    const x_dev = try stream.cloneHtod(f32, &x_host);
    defer x_dev.deinit();
    try ctx.synchronize();

    const nrm = try blas.snrm2(2, x_dev);
    try std.testing.expectApproxEqAbs(@as(f32, 5), nrm, 1e-5);
}

test "cuBLAS SASUM: sum of absolute values" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const blas = CublasContext.init(ctx) catch return error.SkipZigTest;
    defer blas.deinit();
    const stream = ctx.defaultStream();

    const x_host = [_]f32{ -1, 2, -3, 4 };
    const x_dev = try stream.cloneHtod(f32, &x_host);
    defer x_dev.deinit();
    try ctx.synchronize();

    const asum = try blas.sasum(4, x_dev);
    try std.testing.expectApproxEqAbs(@as(f32, 10), asum, 1e-5);
}

test "cuBLAS SSCAL: x = alpha * x" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const blas = CublasContext.init(ctx) catch return error.SkipZigTest;
    defer blas.deinit();
    const stream = ctx.defaultStream();

    const x_host = [_]f32{ 1, 2, 3 };
    var x_dev = try stream.cloneHtod(f32, &x_host);
    defer x_dev.deinit();

    try blas.sscal(3, 3.0, x_dev);
    try ctx.synchronize();

    var result: [3]f32 = undefined;
    try stream.memcpyDtoh(f32, &result, x_dev);

    const expected = [_]f32{ 3, 6, 9 };
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i], result[i], 1e-5);
    }
}

test "cuBLAS SCOPY: y = x" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const blas = CublasContext.init(ctx) catch return error.SkipZigTest;
    defer blas.deinit();
    const stream = ctx.defaultStream();

    const x_host = [_]f32{ 10, 20, 30, 40 };
    const y_host = [_]f32{ 0, 0, 0, 0 };

    const x_dev = try stream.cloneHtod(f32, &x_host);
    defer x_dev.deinit();
    var y_dev = try stream.cloneHtod(f32, &y_host);
    defer y_dev.deinit();

    try blas.scopy(4, x_dev, y_dev);
    try ctx.synchronize();

    var result: [4]f32 = undefined;
    try stream.memcpyDtoh(f32, &result, y_dev);

    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(x_host[i], result[i], 1e-5);
    }
}

test "cuBLAS SSWAP: swap x and y" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const blas = CublasContext.init(ctx) catch return error.SkipZigTest;
    defer blas.deinit();
    const stream = ctx.defaultStream();

    const x_host = [_]f32{ 1, 2, 3 };
    const y_host = [_]f32{ 10, 20, 30 };

    var x_dev = try stream.cloneHtod(f32, &x_host);
    defer x_dev.deinit();
    var y_dev = try stream.cloneHtod(f32, &y_host);
    defer y_dev.deinit();

    try blas.sswap(3, x_dev, y_dev);
    try ctx.synchronize();

    var x_result: [3]f32 = undefined;
    var y_result: [3]f32 = undefined;
    try stream.memcpyDtoh(f32, &x_result, x_dev);
    try stream.memcpyDtoh(f32, &y_result, y_dev);

    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(y_host[i], x_result[i], 1e-5);
        try std.testing.expectApproxEqAbs(x_host[i], y_result[i], 1e-5);
    }
}

test "cuBLAS ISAMAX: index of max absolute value" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const blas = CublasContext.init(ctx) catch return error.SkipZigTest;
    defer blas.deinit();
    const stream = ctx.defaultStream();

    const x_host = [_]f32{ 1, -5, 3, 2 };
    const x_dev = try stream.cloneHtod(f32, &x_host);
    defer x_dev.deinit();
    try ctx.synchronize();

    const idx = try blas.isamax(4, x_dev);
    try std.testing.expectEqual(@as(i32, 2), idx);
}

test "cuBLAS ISAMIN: index of min absolute value" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const blas = CublasContext.init(ctx) catch return error.SkipZigTest;
    defer blas.deinit();
    const stream = ctx.defaultStream();

    const x_host = [_]f32{ 5, -1, 3, 2 };
    const x_dev = try stream.cloneHtod(f32, &x_host);
    defer x_dev.deinit();
    try ctx.synchronize();

    const idx = try blas.isamin(4, x_dev);
    try std.testing.expectEqual(@as(i32, 2), idx);
}

// ============================================================================
// L2 Tests
// ============================================================================

test "cuBLAS SGEMV: y = alpha*A*x + beta*y" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const blas = CublasContext.init(ctx) catch return error.SkipZigTest;
    defer blas.deinit();
    const stream = ctx.defaultStream();

    // A = [[1,2],[3,4]], x = [1,1], y = A*x = [3,7]
    // Col-major: A = [1,3,2,4]
    const a_host = [_]f32{ 1, 3, 2, 4 };
    const x_host = [_]f32{ 1, 1 };
    const y_host = [_]f32{ 0, 0 };

    const a_dev = try stream.cloneHtod(f32, &a_host);
    defer a_dev.deinit();
    const x_dev = try stream.cloneHtod(f32, &x_host);
    defer x_dev.deinit();
    var y_dev = try stream.cloneHtod(f32, &y_host);
    defer y_dev.deinit();

    try blas.sgemv(.no_transpose, 2, 2, 1.0, a_dev, 2, x_dev, 0.0, y_dev);
    try ctx.synchronize();

    var result: [2]f32 = undefined;
    try stream.memcpyDtoh(f32, &result, y_dev);
    try std.testing.expectApproxEqAbs(@as(f32, 3), result[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 7), result[1], 1e-5);
}

// ============================================================================
// L3 Tests
// ============================================================================

test "cuBLAS SGEMM: C = alpha*A*B + beta*C" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const blas = CublasContext.init(ctx) catch return error.SkipZigTest;
    defer blas.deinit();
    const stream = ctx.defaultStream();

    // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    // C = A*B = [[19,22],[43,50]]
    const a_host = [_]f32{ 1, 3, 2, 4 };
    const b_host = [_]f32{ 5, 7, 6, 8 };
    const c_host = [_]f32{ 0, 0, 0, 0 };

    const a_dev = try stream.cloneHtod(f32, &a_host);
    defer a_dev.deinit();
    const b_dev = try stream.cloneHtod(f32, &b_host);
    defer b_dev.deinit();
    var c_dev = try stream.cloneHtod(f32, &c_host);
    defer c_dev.deinit();

    try blas.sgemm(.no_transpose, .no_transpose, 2, 2, 2, 1.0, a_dev, 2, b_dev, 2, 0.0, c_dev, 2);
    try ctx.synchronize();

    var result: [4]f32 = undefined;
    try stream.memcpyDtoh(f32, &result, c_dev);
    try std.testing.expectApproxEqAbs(@as(f32, 19), result[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 43), result[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 22), result[2], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 50), result[3], 1e-4);
}

test "cuBLAS DGEMM: double-precision matrix multiply" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const blas = CublasContext.init(ctx) catch return error.SkipZigTest;
    defer blas.deinit();
    const stream = ctx.defaultStream();

    const a_host = [_]f64{ 1, 0, 0, 1 }; // col-major I
    const b_host = [_]f64{ 3, 5, 4, 6 };
    const c_host = [_]f64{ 0, 0, 0, 0 };

    const a_dev = try stream.cloneHtod(f64, &a_host);
    defer a_dev.deinit();
    const b_dev = try stream.cloneHtod(f64, &b_host);
    defer b_dev.deinit();
    var c_dev = try stream.cloneHtod(f64, &c_host);
    defer c_dev.deinit();

    try blas.dgemm(.no_transpose, .no_transpose, 2, 2, 2, 1.0, a_dev, 2, b_dev, 2, 0.0, c_dev, 2);
    try ctx.synchronize();

    var result: [4]f64 = undefined;
    try stream.memcpyDtoh(f64, &result, c_dev);
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(b_host[i], result[i], 1e-10);
    }
}

test "cuBLAS SGEMM with transpose: C = A^T * A" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const blas = CublasContext.init(ctx) catch return error.SkipZigTest;
    defer blas.deinit();
    const stream = ctx.defaultStream();

    const a_host = [_]f32{ 1, 3, 2, 4 };
    const c_host = [_]f32{ 0, 0, 0, 0 };

    const a_dev = try stream.cloneHtod(f32, &a_host);
    defer a_dev.deinit();
    var c_dev = try stream.cloneHtod(f32, &c_host);
    defer c_dev.deinit();

    try blas.sgemm(.transpose, .no_transpose, 2, 2, 2, 1.0, a_dev, 2, a_dev, 2, 0.0, c_dev, 2);
    try ctx.synchronize();

    var result: [4]f32 = undefined;
    try stream.memcpyDtoh(f32, &result, c_dev);
    // A^T*A = [[10,14],[14,20]] col-major
    try std.testing.expectApproxEqAbs(@as(f32, 10), result[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 14), result[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 14), result[2], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 20), result[3], 1e-4);
}

test "cuBLAS SGEAM: C = alpha*A + beta*B (matrix addition)" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const blas = CublasContext.init(ctx) catch return error.SkipZigTest;
    defer blas.deinit();
    const stream = ctx.defaultStream();

    // A = [[1,2],[3,4]], B = [[10,20],[30,40]]  (col-major: A=[1,3,2,4] B=[10,30,20,40])
    // C = 2*A + 1*B = [[12,24],[36,48]] (col-major: [12,36,24,48])
    const a_host = [_]f32{ 1, 3, 2, 4 };
    const b_host = [_]f32{ 10, 30, 20, 40 };
    const c_host = [_]f32{ 0, 0, 0, 0 };

    const a_dev = try stream.cloneHtod(f32, &a_host);
    defer a_dev.deinit();
    const b_dev = try stream.cloneHtod(f32, &b_host);
    defer b_dev.deinit();
    var c_dev = try stream.cloneHtod(f32, &c_host);
    defer c_dev.deinit();

    // Use safe layer sgeam
    try blas.sgeam(.no_transpose, .no_transpose, 2, 2, 2.0, a_dev, 2, 1.0, b_dev, 2, c_dev, 2);
    try ctx.synchronize();

    var result: [4]f32 = undefined;
    try stream.memcpyDtoh(f32, &result, c_dev);
    try std.testing.expectApproxEqAbs(@as(f32, 12), result[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 36), result[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 24), result[2], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 48), result[3], 1e-4);
}

test "cuBLAS SSYRK: symmetric rank-k update (safe)" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const blas = CublasContext.init(ctx) catch return error.SkipZigTest;
    defer blas.deinit();
    const stream = ctx.defaultStream();

    // C = alpha * A * A^T + beta * C (2x2 result from 2x3 A)
    const n: i32 = 2;
    const k: i32 = 3;
    const a_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const c_data = [_]f32{ 0, 0, 0, 0 };

    const a_dev = try stream.cloneHtod(f32, &a_data);
    defer a_dev.deinit();
    var c_dev = try stream.cloneHtod(f32, &c_data);
    defer c_dev.deinit();

    try blas.ssyrk(.lower, .no_transpose, n, k, 1.0, a_dev, n, 0.0, c_dev, n);
    try ctx.synchronize();

    var result: [4]f32 = undefined;
    try stream.memcpyDtoh(f32, &result, c_dev);
    try std.testing.expectApproxEqAbs(@as(f32, 35), result[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 44), result[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 56), result[3], 1e-4);
}

// ============================================================================
// New L1/L2 Safe Layer Tests
// ============================================================================

test "cuBLAS SROT: Givens rotation preserves norm" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const blas = CublasContext.init(ctx) catch return error.SkipZigTest;
    defer blas.deinit();
    const stream = ctx.defaultStream();

    var x_data = [_]f32{ 3.0, 4.0 };
    var y_data = [_]f32{ 0.0, 0.0 };

    const d_x = try stream.cloneHtod(f32, &x_data);
    defer d_x.deinit();
    const d_y = try stream.cloneHtod(f32, &y_data);
    defer d_y.deinit();

    // 90° rotation: cos=0, sin=1 → x'=s*y=0, y'=-s*x+c*y=-x → x'=0, y'=-x
    // Actually: x'=c*x+s*y, y'=-s*x+c*y, so cos=0, sin=1 → x'=y, y'=-x
    try blas.srot(2, d_x, 1, d_y, 1, 0.0, 1.0);
    try ctx.synchronize();

    var x_result: [2]f32 = undefined;
    var y_result: [2]f32 = undefined;
    try stream.memcpyDtoh(f32, &x_result, d_x);
    try stream.memcpyDtoh(f32, &y_result, d_y);

    // x' = 0*x + 1*y = [0, 0], y' = -1*x + 0*y = [-3, -4]
    try std.testing.expectApproxEqAbs(@as(f32, 0), x_result[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -3), y_result[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -4), y_result[1], 1e-5);
}

test "cuBLAS SSYMV: symmetric matrix-vector multiply" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const blas = CublasContext.init(ctx) catch return error.SkipZigTest;
    defer blas.deinit();
    const stream = ctx.defaultStream();

    // Symmetric 2x2: [[4, 2], [2, 3]], x = [1, 2]
    const a_data = [_]f32{ 4, 2, 2, 3 };
    const x_data = [_]f32{ 1, 2 };

    const d_a = try stream.cloneHtod(f32, &a_data);
    defer d_a.deinit();
    const d_x = try stream.cloneHtod(f32, &x_data);
    defer d_x.deinit();
    var d_y = try stream.allocZeros(f32, allocator, 2);
    defer d_y.deinit();

    try blas.ssymv(.lower, 2, 1.0, d_a, 2, d_x, 1, 0.0, d_y, 1);
    try ctx.synchronize();

    var result: [2]f32 = undefined;
    try stream.memcpyDtoh(f32, &result, d_y);
    // y = [[4,2],[2,3]] * [1,2] = [4+4, 2+6] = [8, 8]
    try std.testing.expectApproxEqAbs(@as(f32, 8), result[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 8), result[1], 1e-5);
}

test "cuBLAS SSYR: symmetric rank-1 update" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const blas = CublasContext.init(ctx) catch return error.SkipZigTest;
    defer blas.deinit();
    const stream = ctx.defaultStream();

    // Start with zero matrix, x = [1, 2]
    var a_data = [_]f32{ 0, 0, 0, 0 };
    const x_data = [_]f32{ 1, 2 };

    var d_a = try stream.cloneHtod(f32, &a_data);
    defer d_a.deinit();
    const d_x = try stream.cloneHtod(f32, &x_data);
    defer d_x.deinit();

    // A = 0 + 1.0 * [1,2] * [1,2]^T = [[1,2],[2,4]]
    try blas.ssyr(.lower, 2, 1.0, d_x, 1, d_a, 2);
    try ctx.synchronize();

    try stream.memcpyDtoh(f32, &a_data, d_a);
    try std.testing.expectApproxEqAbs(@as(f32, 1), a_data[0], 1e-5); // (0,0)
    try std.testing.expectApproxEqAbs(@as(f32, 2), a_data[1], 1e-5); // (1,0)
    try std.testing.expectApproxEqAbs(@as(f32, 4), a_data[3], 1e-5); // (1,1)
}

test "cuBLAS STRMV: triangular matrix-vector multiply" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const blas = CublasContext.init(ctx) catch return error.SkipZigTest;
    defer blas.deinit();
    const stream = ctx.defaultStream();

    // Lower triangular L = [[2,0],[3,4]], x = [1,1]
    const l_data = [_]f32{ 2, 3, 0, 4 };
    var x_data = [_]f32{ 1, 1 };

    const d_l = try stream.cloneHtod(f32, &l_data);
    defer d_l.deinit();
    var d_x = try stream.cloneHtod(f32, &x_data);
    defer d_x.deinit();

    // x = L*x = [2*1, 3*1+4*1] = [2, 7]
    try blas.strmv(.lower, .no_transpose, .non_unit, 2, d_l, 2, d_x, 1);
    try ctx.synchronize();

    try stream.memcpyDtoh(f32, &x_data, d_x);
    try std.testing.expectApproxEqAbs(@as(f32, 2), x_data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 7), x_data[1], 1e-5);
}

test "cuBLAS STRSV: triangular solve" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const blas = CublasContext.init(ctx) catch return error.SkipZigTest;
    defer blas.deinit();
    const stream = ctx.defaultStream();

    // Lower triangular L = [[2,0],[3,4]], solve L*x = b for x where b = [2, 7]
    // Solution: x0 = 2/2 = 1, x1 = (7 - 3*1)/4 = 1
    const l_data = [_]f32{ 2, 3, 0, 4 };
    var b_data = [_]f32{ 2, 7 };

    const d_l = try stream.cloneHtod(f32, &l_data);
    defer d_l.deinit();
    var d_b = try stream.alloc(f32, allocator, 2);
    defer d_b.deinit();
    try stream.memcpyHtod(f32, d_b, &b_data);

    try blas.strsv(.lower, .no_transpose, .non_unit, 2, d_l, 2, d_b, 1);
    try ctx.synchronize();

    try stream.memcpyDtoh(f32, &b_data, d_b);
    try std.testing.expectApproxEqAbs(@as(f32, 1), b_data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1), b_data[1], 1e-5);
}

test "cuBLAS SGEAM: matrix transpose (safe)" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const blas = CublasContext.init(ctx) catch return error.SkipZigTest;
    defer blas.deinit();
    const stream = ctx.defaultStream();

    // Transpose a 2x3 matrix
    // A = [[1,3,5],[2,4,6]] (col-major: [1,2,3,4,5,6])
    const a_data = [_]f32{ 1, 2, 3, 4, 5, 6 };

    const d_a = try stream.cloneHtod(f32, &a_data);
    defer d_a.deinit();
    var d_c = try stream.allocZeros(f32, allocator, 6);
    defer d_c.deinit();

    // C = 1.0 * A^T + 0.0 * A^T  (m_out=3, n_out=2)
    try blas.sgeam(.transpose, .transpose, 3, 2, 1.0, d_a, 2, 0.0, d_a, 2, d_c, 3);
    try ctx.synchronize();

    var result: [6]f32 = undefined;
    try stream.memcpyDtoh(f32, &result, d_c);
    // A^T (3x2 col-major): [[1,2],[3,4],[5,6]] → [1,3,5,2,4,6]
    try std.testing.expectApproxEqAbs(@as(f32, 1), result[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3), result[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5), result[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2), result[3], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4), result[4], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 6), result[5], 1e-5);
}

test "cuBLAS SDGMM: diagonal matrix multiply (safe)" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const blas = CublasContext.init(ctx) catch return error.SkipZigTest;
    defer blas.deinit();
    const stream = ctx.defaultStream();

    // A = [[1,3],[2,4]], x = [10, 20]
    const a_data = [_]f32{ 1, 2, 3, 4 };
    const x_data = [_]f32{ 10, 20 };

    const d_a = try stream.cloneHtod(f32, &a_data);
    defer d_a.deinit();
    const d_x = try stream.cloneHtod(f32, &x_data);
    defer d_x.deinit();
    var d_c = try stream.allocZeros(f32, allocator, 4);
    defer d_c.deinit();

    // Right: C = A * diag(x), each col j scaled by x[j]
    try blas.sdgmm(.right, 2, 2, d_a, 2, d_x, d_c, 2);
    try ctx.synchronize();

    var result: [4]f32 = undefined;
    try stream.memcpyDtoh(f32, &result, d_c);
    // Col 0: [1,2]*10=[10,20], Col 1: [3,4]*20=[60,80]
    try std.testing.expectApproxEqAbs(@as(f32, 10), result[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 20), result[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 60), result[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 80), result[3], 1e-5);
}
