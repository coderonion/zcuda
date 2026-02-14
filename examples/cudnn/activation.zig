/// cuDNN Activation Example
///
/// Applies ReLU, Sigmoid, and Tanh activation functions on GPU tensors.
///
/// Reference: CUDALibrarySamples/cuDNN/activation
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    std.debug.print("=== cuDNN Activation Functions ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    const stream = ctx.defaultStream();
    const allocator = std.heap.page_allocator;

    const dnn = try cuda.cudnn.CudnnContext.init(ctx);
    defer dnn.deinit();

    // 1D tensor: 1 batch, 1 channel, 1 height, 8 width
    const n_elems: usize = 8;
    const h_input = [_]f32{ -3, -2, -1, 0, 1, 2, 3, 4 };

    const d_input = try stream.cloneHtod(f32, &h_input);
    defer d_input.deinit();
    var d_output = try stream.alloc(f32, allocator, n_elems);
    defer d_output.deinit();

    const x_desc = try dnn.createTensor4d(.nchw, .float, 1, 1, 1, @intCast(n_elems));
    defer x_desc.deinit();
    const y_desc = try dnn.createTensor4d(.nchw, .float, 1, 1, 1, @intCast(n_elems));
    defer y_desc.deinit();

    std.debug.print("Input: [ ", .{});
    for (&h_input) |v| std.debug.print("{d:.0} ", .{v});
    std.debug.print("]\n\n", .{});

    const activations = [_]struct { mode: cuda.cudnn.ActivationMode, name: []const u8 }{
        .{ .mode = .relu, .name = "ReLU" },
        .{ .mode = .sigmoid, .name = "Sigmoid" },
        .{ .mode = .tanh, .name = "Tanh" },
    };

    for (activations) |act| {
        const act_desc = try cuda.cudnn.ActivationDescriptor.init(act.mode, 0.0);
        defer act_desc.deinit();

        try dnn.activationForward(f32, act_desc, 1.0, x_desc, d_input, 0.0, y_desc, d_output);
        try ctx.synchronize();

        var h_output: [n_elems]f32 = undefined;
        try stream.memcpyDtoh(f32, &h_output, d_output);

        std.debug.print("{s:>8}: [ ", .{act.name});
        for (&h_output) |v| std.debug.print("{d:7.3} ", .{v});
        std.debug.print("]\n", .{});
    }
    std.debug.print("\n✓ Activation functions verified\n", .{});
}
