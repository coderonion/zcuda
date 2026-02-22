// examples/kernel/10_Integration/D_MemoryManagement/memset_patterns.zig
// Reference: cuda-samples/0_Introduction/asyncAPI (memset)
// API: CudaStream.allocZeros, memset, memcpyDtoHAsync

const cuda = @import("zcuda");
const driver = cuda.driver;

/// Memory set patterns: zero-fill, value-fill, pattern verification.
pub fn main() !void {
    var ctx = try driver.CudaContext.new(0);
    defer ctx.deinit();
    var stream = try ctx.newStream();
    defer stream.deinit();

    const n: u32 = 256;

    // Zero-fill via allocZeros
    var d_zeros = try stream.allocZeros(u32, n);
    defer d_zeros.deinit();

    // Verify zeros
    var h_check: [256]u32 = undefined;
    try stream.memcpyDtoHAsync(u32, &h_check, d_zeros);
    try stream.sync();
    for (h_check) |v| {
        if (v != 0) return error.ZeroFillFailed;
    }

    // Pattern fill: set all bytes to 0xAA
    var d_pattern = try stream.alloc(u8, n * @sizeOf(u32));
    defer d_pattern.deinit();
    try stream.memsetAsync(u8, d_pattern, 0xAA, n * @sizeOf(u32));
    try stream.sync();
}
