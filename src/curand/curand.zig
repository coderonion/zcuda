/// zCUDA: cuRAND module — GPU random number generation.
///
/// ## Example
///
/// ```zig
/// const rng = try curand.CurandContext.init(cuda_ctx, .default);
/// defer rng.deinit();
/// try rng.setSeed(42);
/// try rng.fillUniform(dev_buffer);
/// ```
pub const sys = @import("sys.zig");
pub const result = @import("result.zig");
const safe = @import("safe.zig");

// Re-export safe layer
pub const CurandContext = safe.CurandContext;
pub const RngType = safe.RngType;
pub const CurandError = safe.CurandError;

test {
    _ = safe;
}
