/// cuBLAS DOT Product Example
///
/// Computes the inner (dot) product of two vectors: result = x · y = Σ(xᵢ * yᵢ)
///
/// Reference: CUDALibrarySamples/cuBLAS/Level-1/dot
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    std.debug.print("=== cuBLAS DOT Product Example ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    std.debug.print("Device: {s}\n\n", .{ctx.name()});

    const stream = ctx.defaultStream();
    const blas = try cuda.cublas.CublasContext.init(ctx);
    defer blas.deinit();

    const n: i32 = 6;

    // Orthogonal-ish vectors
    const x_data = [_]f32{ 1.0, 0.0, 3.0, 0.0, 5.0, 0.0 };
    const y_data = [_]f32{ 0.0, 2.0, 0.0, 4.0, 0.0, 6.0 };

    const d_x = try stream.cloneHtod(f32, &x_data);
    defer d_x.deinit();
    const d_y = try stream.cloneHtod(f32, &y_data);
    defer d_y.deinit();

    std.debug.print("x = [ ", .{});
    for (&x_data) |v| std.debug.print("{d:.1} ", .{v});
    std.debug.print("]\ny = [ ", .{});
    for (&y_data) |v| std.debug.print("{d:.1} ", .{v});
    std.debug.print("]\n\n", .{});

    const dot_result = try blas.sdot(n, d_x, d_y);

    var expected: f32 = 0.0;
    for (&x_data, &y_data) |x, y| expected += x * y;

    std.debug.print("x · y  = {d:.1}\n", .{dot_result});
    std.debug.print("Expected: {d:.1}\n", .{expected});

    if (@abs(dot_result - expected) > 1e-5) {
        std.debug.print("✗ FAILED\n", .{});
        return error.ValidationFailed;
    }

    // Non-orthogonal vectors
    std.debug.print("\n─── Cosine similarity building block ───\n", .{});
    const a_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const b_data = [_]f32{ 6.0, 5.0, 4.0, 3.0, 2.0, 1.0 };

    const d_a = try stream.cloneHtod(f32, &a_data);
    defer d_a.deinit();
    const d_b = try stream.cloneHtod(f32, &b_data);
    defer d_b.deinit();

    const dot_ab = try blas.sdot(n, d_a, d_b);
    std.debug.print("a · b = {d:.1}\n", .{dot_ab});

    var expected_ab: f32 = 0.0;
    for (&a_data, &b_data) |a, b| expected_ab += a * b;
    std.debug.print("Expected: {d:.1}\n", .{expected_ab});

    std.debug.print("\n✓ cuBLAS DOT product complete\n", .{});
}
