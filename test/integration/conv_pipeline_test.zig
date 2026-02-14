/// zCUDA Integration Test: cuDNN Conv + Activation pipeline
const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;
const cudnn = cuda.cudnn;
const CudnnContext = cudnn.CudnnContext;

test "cuDNN conv2d forward + relu pipeline" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const dnn = CudnnContext.init(ctx) catch return error.SkipZigTest;
    defer dnn.deinit();
    const stream = ctx.defaultStream();

    // 1x1x4x4 input, 1x1x3x3 filter, pad=1, stride=1, cross-correlation
    const input_desc = try dnn.createTensor4d(.nchw, .float, 1, 1, 4, 4);
    defer input_desc.deinit();

    const filter_desc = try dnn.createFilter4d(.float, .nchw, 1, 1, 3, 3);
    defer filter_desc.deinit();

    const conv_desc = try dnn.createConv2d(1, 1, 1, 1, 1, 1, .cross_correlation, .float);
    defer conv_desc.deinit();

    const dim = try dnn.getConvOutputDim(conv_desc, input_desc, filter_desc);
    try std.testing.expectEqual(@as(i32, 1), dim.n);
    try std.testing.expectEqual(@as(i32, 1), dim.c);
    try std.testing.expectEqual(@as(i32, 4), dim.h);
    try std.testing.expectEqual(@as(i32, 4), dim.w);

    const output_desc = try dnn.createTensor4d(.nchw, .float, dim.n, dim.c, dim.h, dim.w);
    defer output_desc.deinit();

    // Input: all 1.0
    const input_data = [_]f32{1.0} ** 16;
    const d_input = try stream.cloneHtod(f32, &input_data);
    defer d_input.deinit();

    // Filter: all 1.0 (3x3 sum filter)
    const filter_data = [_]f32{1.0} ** 9;
    const d_filter = try stream.cloneHtod(f32, &filter_data);
    defer d_filter.deinit();

    const out_size: usize = @intCast(dim.n * dim.c * dim.h * dim.w);
    var d_output = try stream.allocZeros(f32, allocator, out_size);
    defer d_output.deinit();

    // Use implicit_gemm as a safe default algorithm
    const algo: cudnn.ConvFwdAlgo = .implicit_gemm;

    // Get workspace size
    const ws_size = try dnn.convForwardWorkspaceSize(input_desc, filter_desc, conv_desc, output_desc, algo);

    var d_workspace: ?driver.CudaSlice(u8) = null;
    if (ws_size > 0) {
        d_workspace = try stream.alloc(u8, allocator, ws_size);
    }
    defer if (d_workspace) |ws| ws.deinit();

    // Conv forward
    try dnn.convForward(f32, 1.0, input_desc, d_input, filter_desc, d_filter, conv_desc, algo, d_workspace, 0.0, output_desc, d_output);
    try ctx.synchronize();

    var result: [16]f32 = undefined;
    try stream.memcpyDtoh(f32, &result, d_output);

    // All values should be > 0
    for (result) |val| {
        try std.testing.expect(val > 0.0);
    }

    // Corners = 4, center = 9
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), result[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), result[5], 1e-4);

    // Apply ReLU
    const act = try cudnn.ActivationDescriptor.init(.relu, 0.0);
    defer act.deinit();

    var d_act_output = try stream.allocZeros(f32, allocator, out_size);
    defer d_act_output.deinit();

    try dnn.activationForward(f32, act, 1.0, output_desc, d_output, 0.0, output_desc, d_act_output);
    try ctx.synchronize();

    var act_result: [16]f32 = undefined;
    try stream.memcpyDtoh(f32, &act_result, d_act_output);

    // After ReLU on positive values, should be unchanged
    for (0..16) |i| {
        try std.testing.expectApproxEqAbs(result[i], act_result[i], 1e-5);
    }
}
