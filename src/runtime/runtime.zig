/// zCUDA: Runtime module — CUDA Runtime API bindings.
///
/// The Runtime API provides a higher-level interface compared to the Driver API.
/// Most advanced users should prefer the Driver API (src/driver/).
pub const sys = @import("sys.zig");
pub const result = @import("result.zig");
const safe = @import("safe.zig");

pub const RuntimeContext = safe.RuntimeContext;
pub const RuntimeError = safe.RuntimeError;
pub const RuntimeSlice = safe.RuntimeSlice;
pub const RuntimeStream = safe.RuntimeStream;
pub const RuntimeEvent = safe.RuntimeEvent;

test {
    _ = safe;
}
