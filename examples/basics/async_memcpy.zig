// examples/kernel/10_Integration/B_StreamsAndEvents/async_memcpy.zig
// Reference: cuda-samples/0_Introduction/asyncAPI
// API: CudaStream, memcpyHtoDAsync, memcpyDtoHAsync, sync

const cuda = @import("zcuda");
const driver = cuda.driver;

/// Async memory copy: host→device→host with stream synchronization.
pub fn main() !void {
    var ctx = try driver.CudaContext.new(0);
    defer ctx.deinit();
    var stream = try ctx.newStream();
    defer stream.deinit();

    const n: u32 = 4096;
    var d_buf = try stream.alloc(f32, n);
    defer d_buf.deinit();

    // Prepare host data
    var h_src: [4096]f32 = undefined;
    for (0..n) |i| h_src[i] = @floatFromInt(i);

    // Async H→D
    try stream.memcpyHtoDAsync(f32, d_buf, &h_src);

    // Async D→H into a different buffer
    var h_dst: [4096]f32 = undefined;
    try stream.memcpyDtoHAsync(f32, &h_dst, d_buf);

    // Wait for completion
    try stream.sync();

    // Verify roundtrip
    for (0..n) |i| {
        if (h_dst[i] != h_src[i]) return error.RoundtripFailed;
    }
}
