/// zCUDA Unit Tests: cuDNN
const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;
const cudnn = cuda.cudnn;
const CudnnContext = cudnn.CudnnContext;

test "cuDNN version" {
    const ver = CudnnContext.version();
    try std.testing.expect(ver > 0);
}

test "cuDNN context creation" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();

    const dnn = CudnnContext.init(ctx) catch |err| {
        std.debug.print("Cannot create cuDNN context: {}\n", .{err});
        return error.SkipZigTest;
    };
    defer dnn.deinit();
}

test "cuDNN tensor and activation descriptors" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const dnn = CudnnContext.init(ctx) catch return error.SkipZigTest;
    defer dnn.deinit();

    const tensor = try dnn.createTensor4d(.nchw, .float, 1, 3, 224, 224);
    defer tensor.deinit();

    const act = try cudnn.ActivationDescriptor.init(.relu, 0.0);
    defer act.deinit();
}

test "cuDNN pooling descriptor" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const dnn = CudnnContext.init(ctx) catch return error.SkipZigTest;
    defer dnn.deinit();

    const pool = try cudnn.PoolingDescriptor.init2d(.max, 2, 2, 0, 0, 2, 2);
    defer pool.deinit();
}

test "cuDNN filter descriptor" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const dnn = CudnnContext.init(ctx) catch return error.SkipZigTest;
    defer dnn.deinit();

    const filter = try dnn.createFilter4d(.float, .nchw, 32, 3, 3, 3);
    defer filter.deinit();
}

test "cuDNN convolution descriptor" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const dnn = CudnnContext.init(ctx) catch return error.SkipZigTest;
    defer dnn.deinit();

    const conv = try dnn.createConv2d(1, 1, 1, 1, 1, 1, .cross_correlation, .float);
    defer conv.deinit();
}

test "cuDNN activation forward — ReLU" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const dnn = CudnnContext.init(ctx) catch return error.SkipZigTest;
    defer dnn.deinit();
    const stream = ctx.defaultStream();

    const desc = try dnn.createTensor4d(.nchw, .float, 1, 4, 1, 1);
    defer desc.deinit();

    const act = try cudnn.ActivationDescriptor.init(.relu, 0.0);
    defer act.deinit();

    // Input: [-1, 2, -3, 4]  Expected ReLU: [0, 2, 0, 4]
    const input_data = [_]f32{ -1.0, 2.0, -3.0, 4.0 };
    const d_input = try stream.cloneHtod(f32, &input_data);
    defer d_input.deinit();

    var d_output = try stream.allocZeros(f32, allocator, 4);
    defer d_output.deinit();

    try dnn.activationForward(f32, act, 1.0, desc, d_input, 0.0, desc, d_output);
    try ctx.synchronize();

    var result: [4]f32 = undefined;
    try stream.memcpyDtoh(f32, &result, d_output);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), result[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), result[3], 1e-5);
}

test "cuDNN softmax forward — probabilities sum to 1" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const dnn = CudnnContext.init(ctx) catch return error.SkipZigTest;
    defer dnn.deinit();
    const stream = ctx.defaultStream();

    const desc = try dnn.createTensor4d(.nchw, .float, 1, 4, 1, 1);
    defer desc.deinit();

    const input_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const d_input = try stream.cloneHtod(f32, &input_data);
    defer d_input.deinit();

    var d_output = try stream.allocZeros(f32, allocator, 4);
    defer d_output.deinit();

    try dnn.softmaxForward(f32, .accurate, .channel, 1.0, desc, d_input, 0.0, desc, d_output);
    try ctx.synchronize();

    var result: [4]f32 = undefined;
    try stream.memcpyDtoh(f32, &result, d_output);

    var sum: f32 = 0.0;
    for (result) |val| {
        try std.testing.expect(val > 0.0);
        try std.testing.expect(val < 1.0);
        sum += val;
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
}

test "cuDNN conv2d getConvOutputDim" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const dnn = CudnnContext.init(ctx) catch return error.SkipZigTest;
    defer dnn.deinit();

    const input_desc = try dnn.createTensor4d(.nchw, .float, 1, 3, 32, 32);
    defer input_desc.deinit();
    const filter_desc = try dnn.createFilter4d(.float, .nchw, 16, 3, 3, 3);
    defer filter_desc.deinit();
    const conv_desc = try dnn.createConv2d(1, 1, 1, 1, 1, 1, .cross_correlation, .float);
    defer conv_desc.deinit();

    const dim = try dnn.getConvOutputDim(conv_desc, input_desc, filter_desc);
    try std.testing.expectEqual(@as(i32, 1), dim.n);
    try std.testing.expectEqual(@as(i32, 16), dim.c);
    try std.testing.expectEqual(@as(i32, 32), dim.h);
    try std.testing.expectEqual(@as(i32, 32), dim.w);
}

test "cuDNN Nd tensor descriptor" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const dnn = CudnnContext.init(ctx) catch return error.SkipZigTest;
    defer dnn.deinit();

    const dims = [_]i32{ 1, 3, 8, 16, 16 };
    const strides = [_]i32{ 3 * 8 * 16 * 16, 8 * 16 * 16, 16 * 16, 16, 1 };
    const tensor = try dnn.createTensorNd(.float, &dims, &strides);
    defer tensor.deinit();
}

test "cuDNN addTensor — bias addition" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const dnn = CudnnContext.init(ctx) catch return error.SkipZigTest;
    defer dnn.deinit();
    const stream = ctx.defaultStream();

    const desc = try dnn.createTensor4d(.nchw, .float, 1, 4, 1, 1);
    defer desc.deinit();

    // src = [1, 2, 3, 4], add to output
    const src_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const d_src = try stream.cloneHtod(f32, &src_data);
    defer d_src.deinit();

    // dst starts at [10, 20, 30, 40]
    const dst_data = [_]f32{ 10.0, 20.0, 30.0, 40.0 };
    var d_dst = try stream.cloneHtod(f32, &dst_data);
    defer d_dst.deinit();

    // dst = 1.0 * src + 1.0 * dst
    try dnn.addTensor(f32, 1.0, desc, d_src, 1.0, desc, d_dst);
    try ctx.synchronize();

    var result_buf: [4]f32 = undefined;
    try stream.memcpyDtoh(f32, &result_buf, d_dst);
    _ = allocator;
    try std.testing.expectApproxEqAbs(@as(f32, 11.0), result_buf[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 22.0), result_buf[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 33.0), result_buf[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 44.0), result_buf[3], 1e-5);
}

test "cuDNN scaleTensor — in-place scaling" {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const dnn = CudnnContext.init(ctx) catch return error.SkipZigTest;
    defer dnn.deinit();
    const stream = ctx.defaultStream();

    const desc = try dnn.createTensor4d(.nchw, .float, 1, 4, 1, 1);
    defer desc.deinit();

    const data = [_]f32{ 2.0, 4.0, 6.0, 8.0 };
    var d_data = try stream.cloneHtod(f32, &data);
    defer d_data.deinit();

    // Scale by 0.5
    try dnn.scaleTensor(f32, desc, d_data, 0.5);
    try ctx.synchronize();

    var result_buf: [4]f32 = undefined;
    try stream.memcpyDtoh(f32, &result_buf, d_data);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result_buf[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), result_buf[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), result_buf[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), result_buf[3], 1e-5);
}

test "cuDNN opTensor — element-wise add" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const dnn = CudnnContext.init(ctx) catch return error.SkipZigTest;
    defer dnn.deinit();
    const stream = ctx.defaultStream();

    const desc = try dnn.createTensor4d(.nchw, .float, 1, 4, 1, 1);
    defer desc.deinit();

    // Create OpTensor descriptor for ADD using safe layer
    const op_desc = cudnn.OpTensorDescriptor.init(.add, .float) catch return error.SkipZigTest;
    defer op_desc.deinit();

    const a_data = [_]f32{ 1, 2, 3, 4 };
    const b_data = [_]f32{ 10, 20, 30, 40 };
    const d_a = try stream.cloneHtod(f32, &a_data);
    defer d_a.deinit();
    const d_b = try stream.cloneHtod(f32, &b_data);
    defer d_b.deinit();
    var d_c = try stream.allocZeros(f32, allocator, 4);
    defer d_c.deinit();

    try dnn.opTensor(f32, op_desc, 1.0, desc, d_a, 1.0, desc, d_b, 0.0, desc, d_c);
    try ctx.synchronize();

    var result_buf: [4]f32 = undefined;
    try stream.memcpyDtoh(f32, &result_buf, d_c);
    try std.testing.expectApproxEqAbs(@as(f32, 11.0), result_buf[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 22.0), result_buf[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 33.0), result_buf[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 44.0), result_buf[3], 1e-5);
}

test "cuDNN pooling forward — 2x2 max pool" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const dnn = CudnnContext.init(ctx) catch return error.SkipZigTest;
    defer dnn.deinit();
    const stream = ctx.defaultStream();

    // 1x1x4x4 input
    const input = [_]f32{
        1, 5, 2, 3,
        4, 6, 7, 8,
        9, 3, 1, 0,
        2, 4, 5, 6,
    };
    const d_in = try stream.cloneHtod(f32, &input);
    defer d_in.deinit();
    var d_out = try stream.allocZeros(f32, allocator, 4);
    defer d_out.deinit();

    const x_desc = try dnn.createTensor4d(.nchw, .float, 1, 1, 4, 4);
    defer x_desc.deinit();
    const y_desc = try dnn.createTensor4d(.nchw, .float, 1, 1, 2, 2);
    defer y_desc.deinit();
    const pool = try cudnn.PoolingDescriptor.init2d(.max, 2, 2, 0, 0, 2, 2);
    defer pool.deinit();

    try dnn.poolingForward(f32, pool, 1.0, x_desc, d_in, 0.0, y_desc, d_out);
    try ctx.synchronize();

    var res: [4]f32 = undefined;
    try stream.memcpyDtoh(f32, &res, d_out);
    // max of 2x2 blocks: [6, 8, 9, 6]
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), res[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), res[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), res[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), res[3], 1e-5);
}

test "cuDNN activation forward — sigmoid" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const dnn = CudnnContext.init(ctx) catch return error.SkipZigTest;
    defer dnn.deinit();
    const stream = ctx.defaultStream();

    const desc = try dnn.createTensor4d(.nchw, .float, 1, 4, 1, 1);
    defer desc.deinit();
    const act = try cudnn.ActivationDescriptor.init(.sigmoid, 0.0);
    defer act.deinit();

    const input = [_]f32{ 0.0, 1.0, -1.0, 10.0 };
    const d_in = try stream.cloneHtod(f32, &input);
    defer d_in.deinit();
    var d_out = try stream.allocZeros(f32, allocator, 4);
    defer d_out.deinit();

    try dnn.activationForward(f32, act, 1.0, desc, d_in, 0.0, desc, d_out);
    try ctx.synchronize();

    var res: [4]f32 = undefined;
    try stream.memcpyDtoh(f32, &res, d_out);
    // sigmoid(0) = 0.5, sigmoid(1) ≈ 0.731, sigmoid(-1) ≈ 0.269, sigmoid(10) ≈ 1.0
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), res[0], 1e-4);
    try std.testing.expect(res[1] > 0.7 and res[1] < 0.8);
    try std.testing.expect(res[2] > 0.2 and res[2] < 0.3);
    try std.testing.expect(res[3] > 0.99);
}

test "cuDNN reduce tensor — sum" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const dnn = CudnnContext.init(ctx) catch return error.SkipZigTest;
    defer dnn.deinit();
    const stream = ctx.defaultStream();

    const x_desc = try dnn.createTensor4d(.nchw, .float, 1, 4, 1, 1);
    defer x_desc.deinit();
    const y_desc = try dnn.createTensor4d(.nchw, .float, 1, 1, 1, 1);
    defer y_desc.deinit();

    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const d_in = try stream.cloneHtod(f32, &input);
    defer d_in.deinit();
    var d_out = try stream.allocZeros(f32, allocator, 1);
    defer d_out.deinit();

    // Use safe layer reduceTensor
    const ws_size = dnn.getReductionWorkspaceSize(.add, x_desc, y_desc) catch return error.SkipZigTest;

    var d_ws: ?driver.CudaSlice(u8) = null;
    if (ws_size > 0) d_ws = try stream.alloc(u8, allocator, ws_size);
    defer if (d_ws) |ws| ws.deinit();

    try dnn.reduceTensor(f32, .add, 1.0, x_desc, d_in, 0.0, y_desc, d_out, d_ws);
    try ctx.synchronize();

    var res: [1]f32 = undefined;
    try stream.memcpyDtoh(f32, &res, d_out);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), res[0], 1e-4);
}

test "cuDNN addTensor — C = alpha*A + beta*C" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const dnn = CudnnContext.init(ctx) catch return error.SkipZigTest;
    defer dnn.deinit();
    const stream = ctx.defaultStream();

    const desc = try dnn.createTensor4d(.nchw, .float, 1, 4, 1, 1);
    defer desc.deinit();

    const a_data = [_]f32{ 1, 2, 3, 4 };
    const c_data = [_]f32{ 10, 20, 30, 40 };
    const d_a = try stream.cloneHtod(f32, &a_data);
    defer d_a.deinit();
    var d_c = try stream.cloneHtod(f32, &c_data);
    defer d_c.deinit();

    // C = 2.0 * A + 1.0 * C => [12, 24, 36, 44]
    try dnn.addTensor(f32, 2.0, desc, d_a, 1.0, desc, d_c);
    try ctx.synchronize();

    var result_buf: [4]f32 = undefined;
    try stream.memcpyDtoh(f32, &result_buf, d_c);
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), result_buf[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 24.0), result_buf[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 36.0), result_buf[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 48.0), result_buf[3], 1e-5);
    _ = allocator;
}

test "cuDNN scaleTensor — Y = alpha * Y" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const dnn = CudnnContext.init(ctx) catch return error.SkipZigTest;
    defer dnn.deinit();
    const stream = ctx.defaultStream();

    const desc = try dnn.createTensor4d(.nchw, .float, 1, 4, 1, 1);
    defer desc.deinit();

    const data = [_]f32{ 2, 4, 6, 8 };
    var d_y = try stream.cloneHtod(f32, &data);
    defer d_y.deinit();

    // Y = 3.0 * Y => [6, 12, 18, 24]
    try dnn.scaleTensor(f32, desc, d_y, 3.0);
    try ctx.synchronize();

    var result_buf: [4]f32 = undefined;
    try stream.memcpyDtoh(f32, &result_buf, d_y);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), result_buf[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), result_buf[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 18.0), result_buf[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 24.0), result_buf[3], 1e-5);
    _ = allocator;
}
