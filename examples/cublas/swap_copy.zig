/// cuBLAS SWAP + COPY Example
///
/// SWAP: Exchange two vectors in-place
/// COPY: Copy one vector to another
///
/// Reference: CUDALibrarySamples/cuBLAS/Level-1/swap + Level-1/copy
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    std.debug.print("=== cuBLAS SWAP + COPY Example ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();

    const stream = ctx.defaultStream();
    const blas = try cuda.cublas.CublasContext.init(ctx);
    defer blas.deinit();

    const n: i32 = 5;
    const a_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const b_data = [_]f32{ 10.0, 20.0, 30.0, 40.0, 50.0 };

    // --- SCOPY: y = x ---
    std.debug.print("─── SCOPY: y = x ───\n", .{});
    {
        const d_x = try stream.cloneHtod(f32, &a_data);
        defer d_x.deinit();
        const d_y = try stream.cloneHtod(f32, &b_data);
        defer d_y.deinit();

        std.debug.print("  Before: x = [ ", .{});
        for (&a_data) |v| std.debug.print("{d:.0} ", .{v});
        std.debug.print("]  y = [ ", .{});
        for (&b_data) |v| std.debug.print("{d:.0} ", .{v});
        std.debug.print("]\n", .{});

        try blas.scopy(n, d_x, d_y);

        var h_y: [5]f32 = undefined;
        try stream.memcpyDtoh(f32, &h_y, d_y);

        std.debug.print("  After:  y = [ ", .{});
        for (&h_y) |v| std.debug.print("{d:.0} ", .{v});
        std.debug.print("]\n", .{});

        for (&a_data, &h_y) |expected, actual| {
            if (@abs(expected - actual) > 1e-5) return error.ValidationFailed;
        }
        std.debug.print("  ✓ y now contains a copy of x\n\n", .{});
    }

    // --- SSWAP: swap(x, y) ---
    std.debug.print("─── SSWAP: swap(x, y) ───\n", .{});
    {
        const d_x = try stream.cloneHtod(f32, &a_data);
        defer d_x.deinit();
        const d_y = try stream.cloneHtod(f32, &b_data);
        defer d_y.deinit();

        std.debug.print("  Before: x = [ ", .{});
        for (&a_data) |v| std.debug.print("{d:.0} ", .{v});
        std.debug.print("]  y = [ ", .{});
        for (&b_data) |v| std.debug.print("{d:.0} ", .{v});
        std.debug.print("]\n", .{});

        try blas.sswap(n, d_x, d_y);

        var h_x: [5]f32 = undefined;
        var h_y: [5]f32 = undefined;
        try stream.memcpyDtoh(f32, &h_x, d_x);
        try stream.memcpyDtoh(f32, &h_y, d_y);

        std.debug.print("  After:  x = [ ", .{});
        for (&h_x) |v| std.debug.print("{d:.0} ", .{v});
        std.debug.print("]  y = [ ", .{});
        for (&h_y) |v| std.debug.print("{d:.0} ", .{v});
        std.debug.print("]\n", .{});

        // x should now have b's data, y should have a's data
        for (&b_data, &h_x) |expected, actual| {
            if (@abs(expected - actual) > 1e-5) return error.ValidationFailed;
        }
        for (&a_data, &h_y) |expected, actual| {
            if (@abs(expected - actual) > 1e-5) return error.ValidationFailed;
        }
        std.debug.print("  ✓ Vectors swapped successfully\n", .{});
    }

    std.debug.print("\n✓ cuBLAS SWAP + COPY complete\n", .{});
}
