/// cuDNN Conv2D Forward Example
///
/// Demonstrates 2D convolution forward pass with a 3x3 filter.
///
/// Reference: CUDALibrarySamples/cuDNN/conv_sample
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    std.debug.print("=== cuDNN Conv2D Forward ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    const stream = ctx.defaultStream();
    const allocator = std.heap.page_allocator;

    const dnn = try cuda.cudnn.CudnnContext.init(ctx);
    defer dnn.deinit();

    // Input: 1 batch, 1 channel, 5x5
    const in_n: i32 = 1;
    const in_c: i32 = 1;
    const in_h: i32 = 5;
    const in_w: i32 = 5;

    // 5x5 input with pattern
    const h_input = [_]f32{
        1, 0, 0, 0, 1,
        0, 1, 0, 1, 0,
        0, 0, 1, 0, 0,
        0, 1, 0, 1, 0,
        1, 0, 0, 0, 1,
    };

    std.debug.print("Input (1x1x5x5):\n", .{});
    for (0..5) |r| {
        std.debug.print("  [", .{});
        for (0..5) |c| std.debug.print(" {d:.0}", .{h_input[r * 5 + c]});
        std.debug.print(" ]\n", .{});
    }
    std.debug.print("\n", .{});

    // Filter: 1 output, 1 input, 3x3 edge detect
    const filt_k: i32 = 1;
    const h_filter = [_]f32{
        -1, -1, -1,
        -1, 8,  -1,
        -1, -1, -1,
    };

    std.debug.print("Filter (3x3 edge detect):\n  [-1 -1 -1]\n  [-1  8 -1]\n  [-1 -1 -1]\n\n", .{});

    const d_input = try stream.cloneHtod(f32, &h_input);
    defer d_input.deinit();
    const d_filter = try stream.cloneHtod(f32, &h_filter);
    defer d_filter.deinit();

    const x_desc = try dnn.createTensor4d(.nchw, .float, in_n, in_c, in_h, in_w);
    defer x_desc.deinit();
    const w_desc = try dnn.createFilter4d(.float, .nchw, filt_k, in_c, 3, 3);
    defer w_desc.deinit();

    // Conv2d: pad=1, stride=1, dilation=1 → same size output
    const conv_desc = try dnn.createConv2d(1, 1, 1, 1, 1, 1, .cross_correlation, .float);
    defer conv_desc.deinit();

    const out_dim = try dnn.getConvOutputDim(conv_desc, x_desc, w_desc);
    std.debug.print("Output dimensions: {}x{}x{}x{}\n\n", .{ out_dim.n, out_dim.c, out_dim.h, out_dim.w });

    const out_size: usize = @intCast(out_dim.n * out_dim.c * out_dim.h * out_dim.w);
    const y_desc = try dnn.createTensor4d(.nchw, .float, out_dim.n, out_dim.c, out_dim.h, out_dim.w);
    defer y_desc.deinit();

    var d_output = try stream.allocZeros(f32, allocator, out_size);
    defer d_output.deinit();

    // Get workspace
    const ws_size = try dnn.convForwardWorkspaceSize(x_desc, w_desc, conv_desc, y_desc, .implicit_gemm);
    var workspace: ?cuda.driver.CudaSlice(u8) = null;
    if (ws_size > 0) workspace = try stream.alloc(u8, allocator, ws_size);
    defer if (workspace) |ws| ws.deinit();

    try dnn.convForward(f32, 1.0, x_desc, d_input, w_desc, d_filter, conv_desc, .implicit_gemm, workspace, 0.0, y_desc, d_output);
    try ctx.synchronize();

    var h_output: [25]f32 = undefined;
    try stream.memcpyDtoh(f32, h_output[0..out_size], d_output);

    std.debug.print("Output (after 3x3 edge-detect conv):\n", .{});
    const oh: usize = @intCast(out_dim.h);
    const ow: usize = @intCast(out_dim.w);
    for (0..oh) |r| {
        std.debug.print("  [", .{});
        for (0..ow) |c| std.debug.print(" {d:5.1}", .{h_output[r * ow + c]});
        std.debug.print(" ]\n", .{});
    }
    std.debug.print("\n✓ Conv2D forward completed\n", .{});
}
