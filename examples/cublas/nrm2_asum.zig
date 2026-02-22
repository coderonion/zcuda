/// cuBLAS NRM2 + ASUM Example
///
/// Demonstrates vector norm operations:
/// - SNRM2: Euclidean norm ||x||₂ = √(Σ xᵢ²)
/// - SASUM: Sum of absolute values ||x||₁ = Σ |xᵢ|
///
/// Reference: CUDALibrarySamples/cuBLAS/Level-1/nrm2 + Level-1/asum
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    std.debug.print("=== cuBLAS NRM2 + ASUM Example ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();

    const stream = ctx.defaultStream();
    const blas = try cuda.cublas.CublasContext.init(ctx);
    defer blas.deinit();

    const n: i32 = 5;
    const x_data = [_]f32{ 3.0, -4.0, 5.0, -12.0, 8.0 };

    const d_x = try stream.cloneHtoD(f32, &x_data);
    defer d_x.deinit();

    std.debug.print("x = [ ", .{});
    for (&x_data) |v| std.debug.print("{d:.1} ", .{v});
    std.debug.print("]\n\n", .{});

    // SNRM2: L2 norm
    const l2_norm = try blas.snrm2(n, d_x);
    var expected_l2: f32 = 0.0;
    for (&x_data) |v| expected_l2 += v * v;
    expected_l2 = @sqrt(expected_l2);

    std.debug.print("─── L2 Norm (SNRM2) ───\n", .{});
    std.debug.print("  ||x||₂ = {d:.6}\n", .{l2_norm});
    std.debug.print("  Expected: {d:.6}\n", .{expected_l2});

    if (@abs(l2_norm - expected_l2) > 1e-4) {
        std.debug.print("  ✗ FAILED\n", .{});
        return error.ValidationFailed;
    }
    std.debug.print("  ✓ Verified\n\n", .{});

    // SASUM: L1 norm
    const l1_norm = try blas.sasum(n, d_x);
    var expected_l1: f32 = 0.0;
    for (&x_data) |v| expected_l1 += @abs(v);

    std.debug.print("─── L1 Norm (SASUM) ───\n", .{});
    std.debug.print("  ||x||₁ = {d:.6}\n", .{l1_norm});
    std.debug.print("  Expected: {d:.6}\n", .{expected_l1});

    if (@abs(l1_norm - expected_l1) > 1e-4) {
        std.debug.print("  ✗ FAILED\n", .{});
        return error.ValidationFailed;
    }
    std.debug.print("  ✓ Verified\n", .{});

    std.debug.print("\n✓ cuBLAS NRM2 + ASUM complete\n", .{});
}
