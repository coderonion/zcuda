/// zCUDA: NVRTC module — Runtime compilation of CUDA C++ to PTX.
///
/// Provides three layers of API:
/// - `sys`: Raw FFI bindings (@cImport of nvrtc.h)
/// - `result`: Error wrapping (nvrtcResult -> Zig errors)
/// - `safe`: Safe abstractions (recommended)
///
/// ## Example
///
/// ```zig
/// const ptx = try nvrtc.compilePtx(allocator,
///     \\extern "C" __global__ void kernel() { }
/// );
/// defer allocator.free(ptx);
/// ```
pub const sys = @import("sys.zig");
pub const result = @import("result.zig");
const safe = @import("safe.zig");

// Re-export safe layer (recommended API)
pub const compilePtx = safe.compilePtx;
pub const compilePtxWithOptions = safe.compilePtxWithOptions;
pub const CompileOptions = safe.CompileOptions;
pub const getVersion = safe.getVersion;
pub const NvrtcError = safe.NvrtcError;

test {
    _ = safe;
}
