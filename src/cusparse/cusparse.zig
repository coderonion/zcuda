/// zCUDA: cuSPARSE module — Sparse matrix operations.
pub const sys = @import("sys.zig");
pub const result = @import("result.zig");
const safe = @import("safe.zig");

pub const CusparseContext = safe.CusparseContext;
pub const CusparseError = safe.CusparseError;

test {
    _ = safe;
}
