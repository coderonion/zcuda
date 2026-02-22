// examples/kernel/10_Integration/D_MemoryManagement/pinned_memory.zig
// Reference: cuda-samples/0_Introduction/simpleZeroCopy
// API: driver.allocPinned, freePinned, memcpyHtoDAsync

const cuda = @import("zcuda");
const driver = cuda.driver;

/// Pinned (page-locked) memory for faster H2D/D2H transfers.
pub fn main() !void {
    var ctx = try driver.CudaContext.new(0);
    defer ctx.deinit();
    var stream = try ctx.newStream();
    defer stream.deinit();

    const n: u32 = 4096;

    // Allocate pinned host memory
    var h_pinned = try driver.allocPinned(f32, n);
    defer driver.freePinned(h_pinned);

    // Initialize
    for (0..n) |i| h_pinned[i] = @floatFromInt(i);

    // Allocate device memory
    var d_buf = try stream.alloc(f32, n);
    defer d_buf.deinit();

    // Pinned H→D transfer (faster than pageable memory)
    try stream.memcpyHtoDAsync(f32, d_buf, h_pinned[0..n]);

    // Pinned D→H transfer
    var h_result = try driver.allocPinned(f32, n);
    defer driver.freePinned(h_result);
    try stream.memcpyDtoHAsync(f32, h_result[0..n], d_buf);
    try stream.sync();

    for (0..n) |i| {
        if (h_result[i] != h_pinned[i]) return error.PinnedCopyFailed;
    }
}
