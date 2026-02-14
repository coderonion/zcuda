/// zCUDA: cuBLAS module — Basic Linear Algebra Subroutines on GPU.
///
/// Provides BLAS Level 1, 2, and 3 operations.
///
/// ## Example
///
/// ```zig
/// const cublas_ctx = try cublas.CublasContext.init(cuda_ctx);
/// defer cublas_ctx.deinit();
///
/// try cublas_ctx.sgemm(.no_transpose, .no_transpose, m, n, k,
///     1.0, a, m, b, k, 0.0, c, m);
/// ```
pub const sys = @import("sys.zig");
pub const result = @import("result.zig");
const safe = @import("safe.zig");

// Re-export safe layer
pub const CublasContext = safe.CublasContext;
pub const Operation = safe.Operation;
pub const CublasError = safe.CublasError;

test {
    _ = safe;
}
