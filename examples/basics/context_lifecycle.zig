// examples/kernel/10_Integration/A_DriverLifecycle/context_init_destroy.zig
// Reference: cuda-samples/0_Introduction/simpleDrvRuntime
// API: driver.CudaContext.new, deinit, computeCapability, totalMem, freeMem

const cuda = @import("zcuda");
const driver = cuda.driver;

/// Full context lifecycle: create → query → destroy.
/// Verifies zcuda driver API for context management.
pub fn main() !void {
    // Initialize CUDA context on device 0
    var ctx = try driver.CudaContext.new(0);
    defer ctx.deinit();

    // Query device properties
    const cc = try ctx.computeCapability();
    const total = try ctx.totalMem();
    const free = try ctx.freeMem();

    // Validate returned values
    if (cc.major < 3) return error.UnsupportedDevice;
    if (total == 0) return error.InvalidTotalMem;
    if (free > total) return error.FreeLargerThanTotal;

    // Create and destroy streams
    var stream = try ctx.newStream();
    defer stream.deinit();

    // Default stream
    const default = ctx.defaultStream();
    _ = default;
}
