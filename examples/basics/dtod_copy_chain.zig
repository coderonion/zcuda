// examples/kernel/10_Integration/B_StreamsAndEvents/dtod_copy_chain.zig
// Reference: cuda-samples/0_Introduction/simpleMultiCopy (D2D variant)
// API: CudaStream.memcpyDtoDAsync, alloc, sync

const cuda = @import("zcuda");
const driver = cuda.driver;

/// Device-to-device copy chain: buf_a → buf_b → buf_c.
pub fn main() !void {
    var ctx = try driver.CudaContext.new(0);
    defer ctx.deinit();
    var stream = try ctx.newStream();
    defer stream.deinit();

    const n: u32 = 512;

    // Allocate three device buffers
    var d_a = try stream.cloneHtoD(f32, &([_]f32{42.0} ** 512));
    defer d_a.deinit();
    var d_b = try stream.allocZeros(f32, n);
    defer d_b.deinit();
    var d_c = try stream.allocZeros(f32, n);
    defer d_c.deinit();

    // Chain: a → b → c
    try stream.memcpyDtoDAsync(f32, d_b, d_a);
    try stream.memcpyDtoDAsync(f32, d_c, d_b);
    try stream.sync();

    // Verify final buffer
    var h_c: [512]f32 = undefined;
    try stream.memcpyDtoHAsync(f32, &h_c, d_c);
    try stream.sync();

    for (h_c) |v| {
        if (v != 42.0) return error.CopyChainFailed;
    }
}
