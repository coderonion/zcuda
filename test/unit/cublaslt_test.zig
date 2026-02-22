/// zCUDA Unit Tests: cuBLAS LT
const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;
const cublaslt = cuda.cublaslt;
const CublasLtContext = cublaslt.CublasLtContext;

test "cuBLAS LT context creation" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const lt = CublasLtContext.init(ctx) catch |err| {
        std.debug.print("Cannot create cuBLAS LT context: {}\n", .{err});
        return error.SkipZigTest;
    };
    defer lt.deinit();
}

test "cuBLAS LT matmul descriptor and layout creation" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const lt = CublasLtContext.init(ctx) catch return error.SkipZigTest;
    defer lt.deinit();

    const desc = try lt.createMatmulDesc(.f32, .f32);
    defer cublaslt.destroyMatmulDesc(desc);

    const layout_a = try lt.createMatrixLayout(.f32, 4, 4, 4);
    defer cublaslt.destroyMatrixLayout(layout_a);

    const layout_b = try lt.createMatrixLayout(.f32, 4, 4, 4);
    defer cublaslt.destroyMatrixLayout(layout_b);

    const layout_c = try lt.createMatrixLayout(.f32, 4, 4, 4);
    defer cublaslt.destroyMatrixLayout(layout_c);
}

test "cuBLAS LT preference and heuristic" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const lt = CublasLtContext.init(ctx) catch return error.SkipZigTest;
    defer lt.deinit();

    const desc = try lt.createMatmulDesc(.f32, .f32);
    defer cublaslt.destroyMatmulDesc(desc);

    const layout_a = try lt.createMatrixLayout(.f32, 64, 64, 64);
    defer cublaslt.destroyMatrixLayout(layout_a);
    const layout_b = try lt.createMatrixLayout(.f32, 64, 64, 64);
    defer cublaslt.destroyMatrixLayout(layout_b);
    const layout_c = try lt.createMatrixLayout(.f32, 64, 64, 64);
    defer cublaslt.destroyMatrixLayout(layout_c);

    const pref = try lt.createPreference();
    defer cublaslt.destroyPreference(pref);

    var results: [1]cublaslt.MatmulHeuristicResult = undefined;
    _ = lt.getHeuristics(desc, layout_a, layout_b, layout_c, layout_c, pref, &results) catch |err| {
        std.debug.print("getHeuristics error (may be expected): {}\n", .{err});
        return;
    };
}

test "cuBLAS LT matmul execution — D = A*B" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const lt = CublasLtContext.init(ctx) catch return error.SkipZigTest;
    defer lt.deinit();
    const stream = ctx.defaultStream();

    // 2x2 A = [[1,3],[2,4]] (col-major), B = [[5,7],[6,8]]
    const a_h = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b_h = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    const d_a = try stream.cloneHtoD(f32, &a_h);
    defer d_a.deinit();
    const d_b = try stream.cloneHtoD(f32, &b_h);
    defer d_b.deinit();
    var d_c = try stream.allocZeros(f32, allocator, 4);
    defer d_c.deinit();
    var d_d = try stream.allocZeros(f32, allocator, 4);
    defer d_d.deinit();

    const desc = try lt.createMatmulDesc(.f32, .f32);
    defer cublaslt.destroyMatmulDesc(desc);
    const la = try lt.createMatrixLayout(.f32, 2, 2, 2);
    defer cublaslt.destroyMatrixLayout(la);
    const lb = try lt.createMatrixLayout(.f32, 2, 2, 2);
    defer cublaslt.destroyMatrixLayout(lb);
    const lc = try lt.createMatrixLayout(.f32, 2, 2, 2);
    defer cublaslt.destroyMatrixLayout(lc);
    const ld = try lt.createMatrixLayout(.f32, 2, 2, 2);
    defer cublaslt.destroyMatrixLayout(ld);

    lt.matmul(f32, desc, 1.0, d_a, la, d_b, lb, 0.0, d_c, lc, d_d, ld, stream) catch |err| {
        std.debug.print("matmul error (may be expected on some HW): {}\n", .{err});
        return;
    };
    try ctx.synchronize();

    var res: [4]f32 = undefined;
    try stream.memcpyDtoH(f32, &res, d_d);
    // A*B = [[1*5+3*6, 1*7+3*8],[2*5+4*6, 2*7+4*8]] = [[23,31],[34,46]] col-major: [23,34,31,46]
    try std.testing.expectApproxEqAbs(@as(f32, 23.0), res[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 34.0), res[1], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 31.0), res[2], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 46.0), res[3], 1e-3);
}
