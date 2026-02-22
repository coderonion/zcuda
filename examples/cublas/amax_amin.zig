/// cuBLAS AMAX/AMIN Example
///
/// Finds the index of the element with the maximum/minimum absolute value.
/// Note: cuBLAS uses 1-based indexing for AMAX/AMIN results.
///
/// Reference: CUDALibrarySamples/cuBLAS/Level-1/amax + Level-1/amin
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    std.debug.print("=== cuBLAS AMAX/AMIN Example ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();

    const stream = ctx.defaultStream();
    const blas = try cuda.cublas.CublasContext.init(ctx);
    defer blas.deinit();

    const n: i32 = 7;
    const x_data = [_]f32{ -2.0, 5.0, -10.0, 3.0, 7.0, -1.0, 8.0 };

    std.debug.print("x = [ ", .{});
    for (&x_data, 0..) |v, i| {
        if (i > 0) std.debug.print(", ", .{});
        std.debug.print("{d:.1}", .{v});
    }
    std.debug.print(" ]\n\n", .{});

    const d_x = try stream.cloneHtoD(f32, &x_data);
    defer d_x.deinit();

    // ISAMAX: index of max |x|
    const max_idx = try blas.isamax(n, d_x);
    std.debug.print("ISAMAX: index = {} (1-based), x[{}] = {d:.1}\n", .{
        max_idx,
        max_idx - 1,
        x_data[@intCast(max_idx - 1)],
    });

    // Find expected max
    var exp_max_idx: usize = 0;
    var exp_max_val: f32 = 0.0;
    for (&x_data, 0..) |v, i| {
        if (@abs(v) > exp_max_val) {
            exp_max_val = @abs(v);
            exp_max_idx = i;
        }
    }
    std.debug.print("Expected: index = {} (0-based), |x| = {d:.1}\n", .{ exp_max_idx, exp_max_val });

    if (max_idx - 1 != @as(i32, @intCast(exp_max_idx))) {
        std.debug.print("✗ FAILED\n", .{});
        return error.ValidationFailed;
    }
    std.debug.print("✓ Verified\n", .{});

    // ISAMIN: index of min |x|
    std.debug.print("\n", .{});
    const min_idx = try blas.isamin(n, d_x);
    std.debug.print("ISAMIN: index = {} (1-based), x[{}] = {d:.1}\n", .{
        min_idx,
        min_idx - 1,
        x_data[@intCast(min_idx - 1)],
    });

    var exp_min_idx: usize = 0;
    var exp_min_val: f32 = std.math.floatMax(f32);
    for (&x_data, 0..) |v, i| {
        if (@abs(v) < exp_min_val) {
            exp_min_val = @abs(v);
            exp_min_idx = i;
        }
    }
    std.debug.print("Expected: index = {} (0-based), |x| = {d:.1}\n", .{ exp_min_idx, exp_min_val });

    if (min_idx - 1 != @as(i32, @intCast(exp_min_idx))) {
        std.debug.print("✗ FAILED\n", .{});
        return error.ValidationFailed;
    }
    std.debug.print("✓ Verified\n", .{});

    std.debug.print("\n✓ cuBLAS AMAX/AMIN complete\n", .{});
}
