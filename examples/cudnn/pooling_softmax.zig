/// cuDNN Pooling + Softmax Example
///
/// Demonstrates max pooling followed by softmax on a small feature map.
///
/// Reference: CUDALibrarySamples/cuDNN/pooling + softmax
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    std.debug.print("=== cuDNN Pooling + Softmax ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    const stream = ctx.defaultStream();
    const allocator = std.heap.page_allocator;

    const dnn = try cuda.cudnn.CudnnContext.init(ctx);
    defer dnn.deinit();

    // Input: 1 batch, 1 channel, 4x4 feature map
    const h_input = [_]f32{
        1, 5, 2, 3,
        4, 6, 7, 8,
        9, 3, 1, 0,
        2, 4, 5, 6,
    };
    std.debug.print("Input (1x1x4x4):\n", .{});
    for (0..4) |r| {
        std.debug.print("  [", .{});
        for (0..4) |c| std.debug.print(" {d:3.0}", .{h_input[r * 4 + c]});
        std.debug.print(" ]\n", .{});
    }
    std.debug.print("\n", .{});

    const d_input = try stream.cloneHtod(f32, &h_input);
    defer d_input.deinit();

    const x_desc = try dnn.createTensor4d(.nchw, .float, 1, 1, 4, 4);
    defer x_desc.deinit();

    // 2x2 max pooling with stride 2 → output is 1x1x2x2
    const pool_desc = try cuda.cudnn.PoolingDescriptor.init2d(.max, 2, 2, 0, 0, 2, 2);
    defer pool_desc.deinit();

    const pool_y_desc = try dnn.createTensor4d(.nchw, .float, 1, 1, 2, 2);
    defer pool_y_desc.deinit();

    var d_pool = try stream.allocZeros(f32, allocator, 4);
    defer d_pool.deinit();

    try dnn.poolingForward(f32, pool_desc, 1.0, x_desc, d_input, 0.0, pool_y_desc, d_pool);
    try ctx.synchronize();

    var h_pool: [4]f32 = undefined;
    try stream.memcpyDtoh(f32, &h_pool, d_pool);

    std.debug.print("After 2x2 max pool (1x1x2x2):\n  [", .{});
    for (&h_pool) |v| std.debug.print(" {d:.0}", .{v});
    std.debug.print(" ]\n\n", .{});

    // Expected: max of each 2x2 block
    // [max(1,5,4,6), max(2,3,7,8), max(9,3,2,4), max(1,0,5,6)] = [6, 8, 9, 6]
    const expected_pool = [_]f32{ 6, 8, 9, 6 };
    for (&h_pool, &expected_pool) |got, exp| {
        if (@abs(got - exp) > 0.01) return error.PoolingFailed;
    }

    // Softmax on pooling output (1x4x1x1 channel-wise)
    const sm_desc = try dnn.createTensor4d(.nchw, .float, 1, 4, 1, 1);
    defer sm_desc.deinit();

    var d_softmax = try stream.allocZeros(f32, allocator, 4);
    defer d_softmax.deinit();

    try dnn.softmaxForward(f32, .accurate, .channel, 1.0, sm_desc, d_pool, 0.0, sm_desc, d_softmax);
    try ctx.synchronize();

    var h_softmax: [4]f32 = undefined;
    try stream.memcpyDtoh(f32, &h_softmax, d_softmax);

    std.debug.print("After softmax:\n  [", .{});
    for (&h_softmax) |v| std.debug.print(" {d:.4}", .{v});
    std.debug.print(" ]\n\n", .{});

    // Verify softmax sums to 1
    var sum: f64 = 0;
    for (&h_softmax) |v| sum += @as(f64, v);
    std.debug.print("Softmax sum = {d:.6} (expected 1.0)\n\n", .{sum});
    if (@abs(sum - 1.0) > 0.01) return error.SoftmaxFailed;

    std.debug.print("✓ Pooling + Softmax verified\n", .{});
}
