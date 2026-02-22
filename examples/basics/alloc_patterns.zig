// examples/kernel/10_Integration/D_MemoryManagement/alloc_patterns.zig
// Reference: cuda-samples/0_Introduction/simpleMultiGPU (allocation)
// API: CudaStream.alloc, allocZeros, deinit, devicePtr

const cuda = @import("zcuda");
const driver = cuda.driver;

/// Demonstrates various allocation patterns and error handling.
pub fn main() !void {
    var ctx = try driver.CudaContext.new(0);
    defer ctx.deinit();
    var stream = try ctx.newStream();
    defer stream.deinit();

    // Standard allocation
    var buf_f32 = try stream.alloc(f32, 1024);
    defer buf_f32.deinit();

    // Zero-initialized allocation
    var buf_zeros = try stream.allocZeros(f32, 512);
    defer buf_zeros.deinit();

    // Multi-type allocations
    var buf_u32 = try stream.alloc(u32, 256);
    defer buf_u32.deinit();

    var buf_f64 = try stream.alloc(f64, 128);
    defer buf_f64.deinit();

    // Clone from host data
    const host_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var buf_clone = try stream.cloneHtoD(f32, &host_data);
    defer buf_clone.deinit();

    // Verify clone roundtrip
    var h_back: [4]f32 = undefined;
    try stream.memcpyDtoHAsync(f32, &h_back, buf_clone);
    try stream.sync();
    for (0..4) |i| {
        if (h_back[i] != host_data[i]) return error.CloneFailed;
    }
}
