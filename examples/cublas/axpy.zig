/// cuBLAS AXPY Example: y = α·x + y
///
/// Demonstrates the fundamental BLAS Level-1 operation.
/// Both single (SAXPY) and double precision (DAXPY) variants.
///
/// Reference: CUDALibrarySamples/cuBLAS/Level-1/axpy
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    std.debug.print("=== cuBLAS AXPY Example ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    std.debug.print("Device: {s}\n\n", .{ctx.name()});

    const stream = ctx.defaultStream();
    const blas = try cuda.cublas.CublasContext.init(ctx);
    defer blas.deinit();

    // --- SAXPY (single precision) ---
    std.debug.print("─── SAXPY: y = 2.0 * x + y ───\n", .{});
    const n: i32 = 8;
    const alpha: f32 = 2.0;

    const x_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const y_data = [_]f32{ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0 };

    const d_x = try stream.cloneHtoD(f32, &x_data);
    defer d_x.deinit();
    const d_y = try stream.cloneHtoD(f32, &y_data);
    defer d_y.deinit();

    std.debug.print("  x = [ ", .{});
    for (&x_data) |v| std.debug.print("{d:.0} ", .{v});
    std.debug.print("]\n  y = [ ", .{});
    for (&y_data) |v| std.debug.print("{d:.0} ", .{v});
    std.debug.print("]\n  α = {d:.1}\n\n", .{alpha});

    try blas.saxpy(n, alpha, d_x, d_y);

    var h_result: [8]f32 = undefined;
    try stream.memcpyDtoH(f32, &h_result, d_y);

    std.debug.print("  Result y = [ ", .{});
    for (&h_result) |v| std.debug.print("{d:.0} ", .{v});
    std.debug.print("]\n", .{});

    // Verify
    std.debug.print("  Expected  = [ ", .{});
    for (&x_data, &y_data) |x, y| {
        const expected = alpha * x + y;
        std.debug.print("{d:.0} ", .{expected});
    }
    std.debug.print("]\n", .{});

    // Check correctness
    for (&x_data, &y_data, &h_result) |x, y, r| {
        const expected = alpha * x + y;
        if (@abs(r - expected) > 1e-5) {
            std.debug.print("  ✗ FAILED\n", .{});
            return error.ValidationFailed;
        }
    }
    std.debug.print("  ✓ Verified\n", .{});

    std.debug.print("\n✓ cuBLAS AXPY complete\n", .{});
}
