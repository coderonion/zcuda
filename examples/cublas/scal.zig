/// cuBLAS SCAL Example: x = α·x
///
/// Scales a vector by a scalar value in-place.
///
/// Reference: CUDALibrarySamples/cuBLAS/Level-1/scal
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    std.debug.print("=== cuBLAS SCAL Example ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();

    const stream = ctx.defaultStream();
    const blas = try cuda.cublas.CublasContext.init(ctx);
    defer blas.deinit();

    const n: i32 = 8;
    const x_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const alpha: f32 = 0.5;

    std.debug.print("Before: x = [ ", .{});
    for (&x_data) |v| std.debug.print("{d:.1} ", .{v});
    std.debug.print("]\nα = {d:.1}\n\n", .{alpha});

    const d_x = try stream.cloneHtoD(f32, &x_data);
    defer d_x.deinit();

    try blas.sscal(n, alpha, d_x);

    var h_result: [8]f32 = undefined;
    try stream.memcpyDtoH(f32, &h_result, d_x);

    std.debug.print("After:  x = [ ", .{});
    for (&h_result) |v| std.debug.print("{d:.1} ", .{v});
    std.debug.print("]\n", .{});

    // Verify
    for (&x_data, &h_result) |orig, scaled| {
        if (@abs(scaled - orig * alpha) > 1e-5) {
            std.debug.print("✗ FAILED\n", .{});
            return error.ValidationFailed;
        }
    }

    std.debug.print("\n✓ cuBLAS SCAL complete\n", .{});
}
