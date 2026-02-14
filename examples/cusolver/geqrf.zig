/// cuSOLVER QR Factorization Example
///
/// Computes QR factorization A = Q * R and extracts Q.
///
/// Reference: CUDALibrarySamples/cuSOLVER/geqrf
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    std.debug.print("=== cuSOLVER QR Factorization ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    const stream = ctx.defaultStream();
    const allocator = std.heap.page_allocator;

    const sol = try cuda.cusolver.CusolverDnContext.init(ctx);
    defer sol.deinit();
    const ext = cuda.cusolver.CusolverDnExt.init(&sol);

    // 3x2 matrix A (col-major)
    // A = | 1  4 |
    //     | 2  5 |
    //     | 3  6 |
    const m: i32 = 3;
    const n: i32 = 2;
    var A_data = [_]f32{ 1, 2, 3, 4, 5, 6 };

    std.debug.print("A (3x2):\n", .{});
    for (0..3) |r| {
        std.debug.print("  [{d:.0}, {d:.0}]\n", .{ A_data[r], A_data[3 + r] });
    }
    std.debug.print("\n", .{});

    var d_A = try stream.cloneHtod(f32, &A_data);
    defer d_A.deinit();
    var d_tau = try stream.allocZeros(f32, allocator, @intCast(n));
    defer d_tau.deinit();

    // QR factorization
    const buf_size = try sol.sgeqrf_bufferSize(m, n, d_A, m);
    const d_ws = try stream.alloc(f32, allocator, @intCast(buf_size));
    defer d_ws.deinit();

    var info: i32 = -1;
    try sol.sgeqrf(m, n, d_A, m, d_tau, d_ws, buf_size, &info);
    try ctx.synchronize();

    if (info != 0) {
        std.debug.print("QR factorization failed: info = {}\n", .{info});
        return error.FactorizationFailed;
    }

    // R is upper triangle of d_A
    try stream.memcpyDtoh(f32, &A_data, d_A);

    std.debug.print("R (upper triangle):\n", .{});
    for (0..3) |r| {
        std.debug.print("  [", .{});
        for (0..2) |c| {
            if (c >= r) {
                std.debug.print(" {d:7.3}", .{A_data[c * 3 + r]});
            } else {
                std.debug.print("       0", .{});
            }
        }
        std.debug.print(" ]\n", .{});
    }
    std.debug.print("\n", .{});

    // Extract Q using sorgqr
    const qr_buf = try ext.sorgqr_bufferSize(m, n, n, d_A, m, d_tau);
    const d_qr_ws = try stream.alloc(f32, allocator, @intCast(qr_buf));
    defer d_qr_ws.deinit();

    try ext.sorgqr(m, n, n, d_A, m, d_tau, d_qr_ws, qr_buf, &info);
    try ctx.synchronize();

    if (info != 0) {
        std.debug.print("Q extraction failed: info = {}\n", .{info});
        return error.ExtractionFailed;
    }

    try stream.memcpyDtoh(f32, &A_data, d_A);
    std.debug.print("Q (orthonormal columns):\n", .{});
    for (0..3) |r| {
        std.debug.print("  [", .{});
        for (0..2) |c| std.debug.print(" {d:7.4}", .{A_data[c * 3 + r]});
        std.debug.print(" ]\n", .{});
    }
    std.debug.print("\n", .{});

    // Verify Q^T * Q = I (orthonormality)
    var dot0: f64 = 0;
    var dot1: f64 = 0;
    var cross: f64 = 0;
    for (0..3) |r| {
        dot0 += @as(f64, A_data[r]) * @as(f64, A_data[r]);
        dot1 += @as(f64, A_data[3 + r]) * @as(f64, A_data[3 + r]);
        cross += @as(f64, A_data[r]) * @as(f64, A_data[3 + r]);
    }
    std.debug.print("Orthonormality: Q0.Q0={d:.4}, Q1.Q1={d:.4}, Q0.Q1={d:.4}\n\n", .{ dot0, dot1, cross });

    if (@abs(dot0 - 1.0) > 0.01 or @abs(dot1 - 1.0) > 0.01 or @abs(cross) > 0.01) {
        return error.ValidationFailed;
    }

    std.debug.print("✓ QR factorization verified (Q is orthonormal)\n", .{});
}
