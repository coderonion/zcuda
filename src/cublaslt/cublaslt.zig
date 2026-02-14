/// zCUDA: cuBLAS LT module — Lightweight BLAS with algorithm selection.
///
/// Provides advanced GEMM with mixed-precision, algorithm heuristics,
/// and workspace management.
pub const sys = @import("sys.zig");
pub const result = @import("result.zig");
const safe = @import("safe.zig");

pub const CublasLtContext = safe.CublasLtContext;
pub const DataType = safe.DataType;
pub const ComputeType = safe.ComputeType;
pub const Operation = safe.Operation;
pub const MatmulHeuristicResult = safe.MatmulHeuristicResult;
pub const CublasLtError = safe.CublasLtError;
pub const destroyMatmulDesc = safe.destroyMatmulDesc;
pub const destroyMatrixLayout = safe.destroyMatrixLayout;
pub const destroyPreference = safe.destroyPreference;

test {
    _ = safe;
}
