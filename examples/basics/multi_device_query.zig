// examples/kernel/10_Integration/A_DriverLifecycle/multi_device_query.zig
// Reference: cuda-samples/0_Introduction/simpleMultiGPU
// API: driver.CudaContext, deviceCount, computeCapability, totalMem

const cuda = @import("zcuda");
const driver = cuda.driver;

/// Query all available CUDA devices.
/// Demonstrates multi-device enumeration via zcuda driver API.
pub fn main() !void {
    const count = try driver.deviceCount();
    if (count == 0) return error.NoCudaDevices;

    // Query each device
    var dev: u32 = 0;
    while (dev < count) : (dev += 1) {
        var ctx = try driver.CudaContext.new(dev);
        defer ctx.deinit();

        const cc = try ctx.computeCapability();
        const total = try ctx.totalMem();
        const free = try ctx.freeMem();

        // Sanity checks
        if (cc.major < 3) continue; // Skip very old devices
        if (free > total) return error.InvalidMemoryReport;

        _ = cc;
        _ = total;
        _ = free;
    }
}
