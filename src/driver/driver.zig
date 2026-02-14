/// zCUDA: CUDA Driver API module.
///
/// Provides three layers of API:
/// - `sys`: Raw FFI bindings (@cImport of cuda.h)
/// - `result`: Error wrapping (CUresult -> Zig errors)
/// - `safe`: Safe abstractions (recommended for general use)
///
/// ## Example
///
/// ```zig
/// const ctx = try CudaContext.new(0);
/// defer ctx.deinit();
/// const stream = ctx.defaultStream();
/// const data = try stream.cloneHtod(f32, &[_]f32{1, 2, 3});
/// defer data.deinit();
/// ```
pub const sys = @import("sys.zig");
pub const result = @import("result.zig");
pub const safe = @import("safe.zig");

// Re-export safe layer types (recommended API)
pub const CudaContext = safe.CudaContext;
pub const CudaStream = safe.CudaStream;
pub const CudaModule = safe.CudaModule;
pub const CudaFunction = safe.CudaFunction;
pub const CudaSlice = safe.CudaSlice;
pub const CudaView = safe.CudaView;
pub const CudaViewMut = safe.CudaViewMut;
pub const CudaEvent = safe.CudaEvent;
pub const DriverError = safe.DriverError;

// Re-export tests
test {
    _ = safe;
}
