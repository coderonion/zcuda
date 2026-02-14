/// zCUDA: cuSOLVER module — Dense and sparse direct solvers.
///
/// Provides LU, QR, SVD, eigenvalue decomposition, and more.
/// Enable with `-Dcusolver=true`.
pub const sys = @import("sys.zig");
pub const result = @import("result.zig");
const safe = @import("safe.zig");

pub const CusolverDnContext = safe.CusolverDnContext;
pub const CusolverDnExt = safe.CusolverDnExt;
pub const CusolverError = safe.CusolverError;
pub const EigMode = safe.EigMode;
pub const FillMode = safe.FillMode;
pub const GesvdjInfo = safe.GesvdjInfo;

test {
    _ = safe;
}
