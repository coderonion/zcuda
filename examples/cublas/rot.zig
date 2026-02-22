/// cuBLAS Givens Rotation Example
///
/// Applies Givens rotation to vectors x and y using the safe layer.
///
/// Reference: CUDALibrarySamples/cuBLAS/Level-1/rot
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    std.debug.print("=== cuBLAS Givens Rotation Example ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();

    const stream = ctx.defaultStream();
    const blas = try cuda.cublas.CublasContext.init(ctx);
    defer blas.deinit();

    // Input vectors
    const n: usize = 5;
    var x_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var y_data = [_]f32{ 5.0, 4.0, 3.0, 2.0, 1.0 };

    std.debug.print("Before rotation:\n", .{});
    std.debug.print("  x = [", .{});
    for (&x_data) |v| std.debug.print(" {d:.1}", .{v});
    std.debug.print(" ]\n", .{});
    std.debug.print("  y = [", .{});
    for (&y_data) |v| std.debug.print(" {d:.1}", .{v});
    std.debug.print(" ]\n\n", .{});

    const d_x = try stream.cloneHtoD(f32, &x_data);
    defer d_x.deinit();
    const d_y = try stream.cloneHtoD(f32, &y_data);
    defer d_y.deinit();

    // Apply rotation: c = cos(45°), s = sin(45°)
    const angle = std.math.pi / 4.0; // 45 degrees
    const c: f32 = @cos(angle);
    const s: f32 = @sin(angle);

    std.debug.print("Rotation angle: 45° (π/4)\n", .{});
    std.debug.print("  cos = {d:.6}, sin = {d:.6}\n\n", .{ c, s });

    // Use safe layer srot
    try blas.srot(@intCast(n), d_x, 1, d_y, 1, c, s);

    // Copy back
    try stream.memcpyDtoH(f32, &x_data, d_x);
    try stream.memcpyDtoH(f32, &y_data, d_y);

    std.debug.print("After rotation:\n", .{});
    std.debug.print("  x' = [", .{});
    for (&x_data) |v| std.debug.print(" {d:.4}", .{v});
    std.debug.print(" ]\n", .{});
    std.debug.print("  y' = [", .{});
    for (&y_data) |v| std.debug.print(" {d:.4}", .{v});
    std.debug.print(" ]\n\n", .{});

    // Verify: rotation preserves L2 norm
    var norm_before: f64 = 0;
    var norm_after: f64 = 0;
    for (0..n) |i| {
        const ox: f64 = @as(f64, @as(f32, @floatFromInt(i + 1)));
        const oy: f64 = @as(f64, @as(f32, @floatFromInt(5 - @as(i32, @intCast(i)))));
        norm_before += ox * ox + oy * oy;
        norm_after += @as(f64, x_data[i]) * @as(f64, x_data[i]) + @as(f64, y_data[i]) * @as(f64, y_data[i]);
    }

    std.debug.print("Norm preservation (should be equal):\n", .{});
    std.debug.print("  ||[x,y]||² before = {d:.4}\n", .{norm_before});
    std.debug.print("  ||[x,y]||² after  = {d:.4}\n", .{norm_after});

    if (@abs(norm_before - norm_after) > 1e-2) return error.ValidationFailed;
    std.debug.print("\n✓ Norm preserved — Givens rotation verified\n", .{});
}
