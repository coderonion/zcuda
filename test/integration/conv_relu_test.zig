/// Integration Test: cuDNN Conv → Activation pipeline
///
/// Forward pass: conv2d → ReLU activation, mimicking a neural network layer.
const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;
const cudnn = cuda.cudnn;

test "conv2d → ReLU activation pipeline" {
    const allocator = std.testing.allocator;
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    defer ctx.deinit();
    const stream = ctx.defaultStream();
    const dnn = cudnn.CudnnContext.init(ctx) catch return error.SkipZigTest;
    defer dnn.deinit();

    // Simple 1x1x3x3 input
    const h_input = [_]f32{
        -1, 2, -1,
        0,  1, 0,
        -1, 2, -1,
    };
    const d_input = try stream.cloneHtod(f32, &h_input);
    defer d_input.deinit();

    // 1x1x3x3 averaging filter (each element = 1/9)
    var h_filter: [9]f32 = undefined;
    for (&h_filter) |*v| v.* = 1.0 / 9.0;
    const d_filter = try stream.cloneHtod(f32, &h_filter);
    defer d_filter.deinit();

    const x_desc = try dnn.createTensor4d(.nchw, .float, 1, 1, 3, 3);
    defer x_desc.deinit();
    const w_desc = try dnn.createFilter4d(.float, .nchw, 1, 1, 3, 3);
    defer w_desc.deinit();
    const conv_desc = try dnn.createConv2d(1, 1, 1, 1, 1, 1, .cross_correlation, .float);
    defer conv_desc.deinit();

    const out_dim = try dnn.getConvOutputDim(conv_desc, x_desc, w_desc);
    const out_n: usize = @intCast(out_dim.n * out_dim.c * out_dim.h * out_dim.w);
    const y_desc = try dnn.createTensor4d(.nchw, .float, out_dim.n, out_dim.c, out_dim.h, out_dim.w);
    defer y_desc.deinit();

    var d_conv_out = try stream.allocZeros(f32, allocator, out_n);
    defer d_conv_out.deinit();

    // Conv forward
    const ws_size = try dnn.convForwardWorkspaceSize(x_desc, w_desc, conv_desc, y_desc, .implicit_gemm);
    var workspace: ?driver.CudaSlice(u8) = null;
    if (ws_size > 0) workspace = try stream.alloc(u8, allocator, ws_size);
    defer if (workspace) |ws| ws.deinit();

    try dnn.convForward(f32, 1.0, x_desc, d_input, w_desc, d_filter, conv_desc, .implicit_gemm, workspace, 0.0, y_desc, d_conv_out);

    // ReLU activation
    const act = try cudnn.ActivationDescriptor.init(.relu, 0.0);
    defer act.deinit();

    var d_relu_out = try stream.allocZeros(f32, allocator, out_n);
    defer d_relu_out.deinit();

    try dnn.activationForward(f32, act, 1.0, y_desc, d_conv_out, 0.0, y_desc, d_relu_out);
    try ctx.synchronize();

    var h_result: [9]f32 = undefined;
    try stream.memcpyDtoh(f32, h_result[0..out_n], d_relu_out);

    // All values should be >= 0 after ReLU
    for (h_result[0..out_n]) |v| {
        try std.testing.expect(v >= 0.0);
    }
}
