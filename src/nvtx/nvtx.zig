/// zCUDA: NVTX module — NVIDIA Tools Extension for annotations and markers.
pub const sys = @import("sys.zig");
const safe = @import("safe.zig");

pub const rangePush = safe.rangePush;
pub const rangePop = safe.rangePop;
pub const mark = safe.mark;
pub const ScopedRange = safe.ScopedRange;
pub const Domain = safe.Domain;

test {
    _ = safe;
}
