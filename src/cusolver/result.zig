/// zCUDA: cuSOLVER - Error wrapping layer.
///
/// Layer 2: Converts cuSOLVER status codes to Zig error unions.
const sys = @import("sys.zig");

pub const CusolverError = error{
    NotInitialized,
    AllocFailed,
    InvalidValue,
    ArchMismatch,
    InternalError,
    NotSupported,
    Unknown,
};

pub fn toError(status: sys.cusolverStatus_t) CusolverError!void {
    return switch (status) {
        sys.CUSOLVER_STATUS_SUCCESS => {},
        sys.CUSOLVER_STATUS_NOT_INITIALIZED => CusolverError.NotInitialized,
        sys.CUSOLVER_STATUS_ALLOC_FAILED => CusolverError.AllocFailed,
        sys.CUSOLVER_STATUS_INVALID_VALUE => CusolverError.InvalidValue,
        sys.CUSOLVER_STATUS_ARCH_MISMATCH => CusolverError.ArchMismatch,
        sys.CUSOLVER_STATUS_INTERNAL_ERROR => CusolverError.InternalError,
        sys.CUSOLVER_STATUS_NOT_SUPPORTED => CusolverError.NotSupported,
        else => CusolverError.Unknown,
    };
}

// ============================================================================
// Handle Management
// ============================================================================

pub fn create() CusolverError!sys.cusolverDnHandle_t {
    var handle: sys.cusolverDnHandle_t = undefined;
    try toError(sys.cusolverDnCreate(&handle));
    return handle;
}

pub fn destroy(handle: sys.cusolverDnHandle_t) CusolverError!void {
    try toError(sys.cusolverDnDestroy(handle));
}

pub fn setStream(handle: sys.cusolverDnHandle_t, stream: ?*anyopaque) CusolverError!void {
    try toError(sys.cusolverDnSetStream(handle, @ptrCast(stream)));
}

// ============================================================================
// LU Factorization (getrf / getrs)
// ============================================================================

/// Get workspace size for Sgetrf.
pub fn sgetrf_bufferSize(handle: sys.cusolverDnHandle_t, m: i32, n: i32, a: [*c]f32, lda: i32) CusolverError!i32 {
    var lwork: i32 = undefined;
    try toError(sys.cusolverDnSgetrf_bufferSize(handle, m, n, a, lda, &lwork));
    return lwork;
}

/// Get workspace size for Dgetrf.
pub fn dgetrf_bufferSize(handle: sys.cusolverDnHandle_t, m: i32, n: i32, a: [*c]f64, lda: i32) CusolverError!i32 {
    var lwork: i32 = undefined;
    try toError(sys.cusolverDnDgetrf_bufferSize(handle, m, n, a, lda, &lwork));
    return lwork;
}

/// LU factorization (float).
pub fn sgetrf(handle: sys.cusolverDnHandle_t, m: i32, n: i32, a: [*c]f32, lda: i32, workspace: [*c]f32, ipiv: [*c]i32, info: *i32) CusolverError!void {
    try toError(sys.cusolverDnSgetrf(handle, m, n, a, lda, workspace, ipiv, info));
}

/// LU factorization (double).
pub fn dgetrf(handle: sys.cusolverDnHandle_t, m: i32, n: i32, a: [*c]f64, lda: i32, workspace: [*c]f64, ipiv: [*c]i32, info: *i32) CusolverError!void {
    try toError(sys.cusolverDnDgetrf(handle, m, n, a, lda, workspace, ipiv, info));
}

/// Solve after LU factorization (float).
pub fn sgetrs(handle: sys.cusolverDnHandle_t, trans: sys.cublasOperation_t, n: i32, nrhs: i32, a: [*c]const f32, lda: i32, ipiv: [*c]const i32, b: [*c]f32, ldb: i32, info: *i32) CusolverError!void {
    try toError(sys.cusolverDnSgetrs(handle, trans, n, nrhs, a, lda, ipiv, b, ldb, info));
}

/// Solve after LU factorization (double).
pub fn dgetrs(handle: sys.cusolverDnHandle_t, trans: sys.cublasOperation_t, n: i32, nrhs: i32, a: [*c]const f64, lda: i32, ipiv: [*c]const i32, b: [*c]f64, ldb: i32, info: *i32) CusolverError!void {
    try toError(sys.cusolverDnDgetrs(handle, trans, n, nrhs, a, lda, ipiv, b, ldb, info));
}

// ============================================================================
// QR Factorization (geqrf / orgqr)
// ============================================================================

pub fn sgeqrf_bufferSize(handle: sys.cusolverDnHandle_t, m: i32, n: i32, a: [*c]f32, lda: i32) CusolverError!i32 {
    var lwork: i32 = undefined;
    try toError(sys.cusolverDnSgeqrf_bufferSize(handle, m, n, a, lda, &lwork));
    return lwork;
}

pub fn dgeqrf_bufferSize(handle: sys.cusolverDnHandle_t, m: i32, n: i32, a: [*c]f64, lda: i32) CusolverError!i32 {
    var lwork: i32 = undefined;
    try toError(sys.cusolverDnDgeqrf_bufferSize(handle, m, n, a, lda, &lwork));
    return lwork;
}

pub fn sgeqrf(handle: sys.cusolverDnHandle_t, m: i32, n: i32, a: [*c]f32, lda: i32, tau: [*c]f32, workspace: [*c]f32, lwork: i32, info: *i32) CusolverError!void {
    try toError(sys.cusolverDnSgeqrf(handle, m, n, a, lda, tau, workspace, lwork, info));
}

pub fn dgeqrf(handle: sys.cusolverDnHandle_t, m: i32, n: i32, a: [*c]f64, lda: i32, tau: [*c]f64, workspace: [*c]f64, lwork: i32, info: *i32) CusolverError!void {
    try toError(sys.cusolverDnDgeqrf(handle, m, n, a, lda, tau, workspace, lwork, info));
}

pub fn sorgqr_bufferSize(handle: sys.cusolverDnHandle_t, m: i32, n: i32, k: i32, a: [*c]const f32, lda: i32, tau: [*c]const f32) CusolverError!i32 {
    var lwork: i32 = undefined;
    try toError(sys.cusolverDnSorgqr_bufferSize(handle, m, n, k, a, lda, tau, &lwork));
    return lwork;
}

pub fn sorgqr(handle: sys.cusolverDnHandle_t, m: i32, n: i32, k: i32, a: [*c]f32, lda: i32, tau: [*c]const f32, workspace: [*c]f32, lwork: i32, info: *i32) CusolverError!void {
    try toError(sys.cusolverDnSorgqr(handle, m, n, k, a, lda, tau, workspace, lwork, info));
}

// ============================================================================
// SVD (gesvd)
// ============================================================================

pub fn sgesvd_bufferSize(handle: sys.cusolverDnHandle_t, m: i32, n: i32) CusolverError!i32 {
    var lwork: i32 = undefined;
    try toError(sys.cusolverDnSgesvd_bufferSize(handle, m, n, &lwork));
    return lwork;
}

pub fn dgesvd_bufferSize(handle: sys.cusolverDnHandle_t, m: i32, n: i32) CusolverError!i32 {
    var lwork: i32 = undefined;
    try toError(sys.cusolverDnDgesvd_bufferSize(handle, m, n, &lwork));
    return lwork;
}

pub fn sgesvd(
    handle: sys.cusolverDnHandle_t,
    jobu: u8,
    jobvt: u8,
    m: i32,
    n: i32,
    a: [*c]f32,
    lda: i32,
    s: [*c]f32,
    u: [*c]f32,
    ldu: i32,
    vt: [*c]f32,
    ldvt: i32,
    work: [*c]f32,
    lwork: i32,
    rwork: [*c]f32,
    info: *i32,
) CusolverError!void {
    try toError(sys.cusolverDnSgesvd(handle, @intCast(jobu), @intCast(jobvt), m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, info));
}

pub fn dgesvd(
    handle: sys.cusolverDnHandle_t,
    jobu: u8,
    jobvt: u8,
    m: i32,
    n: i32,
    a: [*c]f64,
    lda: i32,
    s: [*c]f64,
    u: [*c]f64,
    ldu: i32,
    vt: [*c]f64,
    ldvt: i32,
    work: [*c]f64,
    lwork: i32,
    rwork: [*c]f64,
    info: *i32,
) CusolverError!void {
    try toError(sys.cusolverDnDgesvd(handle, @intCast(jobu), @intCast(jobvt), m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, info));
}

// ============================================================================
// Cholesky Factorization (potrf / potrs)
// ============================================================================

pub fn spotrf_bufferSize(handle: sys.cusolverDnHandle_t, uplo: sys.cublasFillMode_t, n: i32, a: [*c]f32, lda: i32) CusolverError!i32 {
    var lwork: i32 = undefined;
    try toError(sys.cusolverDnSpotrf_bufferSize(handle, uplo, n, a, lda, &lwork));
    return lwork;
}

pub fn dpotrf_bufferSize(handle: sys.cusolverDnHandle_t, uplo: sys.cublasFillMode_t, n: i32, a: [*c]f64, lda: i32) CusolverError!i32 {
    var lwork: i32 = undefined;
    try toError(sys.cusolverDnDpotrf_bufferSize(handle, uplo, n, a, lda, &lwork));
    return lwork;
}

pub fn spotrf(handle: sys.cusolverDnHandle_t, uplo: sys.cublasFillMode_t, n: i32, a: [*c]f32, lda: i32, workspace: [*c]f32, lwork: i32, info: *i32) CusolverError!void {
    try toError(sys.cusolverDnSpotrf(handle, uplo, n, a, lda, workspace, lwork, info));
}

pub fn dpotrf(handle: sys.cusolverDnHandle_t, uplo: sys.cublasFillMode_t, n: i32, a: [*c]f64, lda: i32, workspace: [*c]f64, lwork: i32, info: *i32) CusolverError!void {
    try toError(sys.cusolverDnDpotrf(handle, uplo, n, a, lda, workspace, lwork, info));
}

pub fn spotrs(handle: sys.cusolverDnHandle_t, uplo: sys.cublasFillMode_t, n: i32, nrhs: i32, a: [*c]const f32, lda: i32, b: [*c]f32, ldb: i32, info: *i32) CusolverError!void {
    try toError(sys.cusolverDnSpotrs(handle, uplo, n, nrhs, a, lda, b, ldb, info));
}

pub fn dpotrs(handle: sys.cusolverDnHandle_t, uplo: sys.cublasFillMode_t, n: i32, nrhs: i32, a: [*c]const f64, lda: i32, b: [*c]f64, ldb: i32, info: *i32) CusolverError!void {
    try toError(sys.cusolverDnDpotrs(handle, uplo, n, nrhs, a, lda, b, ldb, info));
}

// ============================================================================
// Double QR Q-extraction (dorgqr)
// ============================================================================

pub fn dorgqr_bufferSize(handle: sys.cusolverDnHandle_t, m: i32, n: i32, k: i32, a: [*c]const f64, lda: i32, tau: [*c]const f64) CusolverError!i32 {
    var lwork: i32 = undefined;
    try toError(sys.cusolverDnDorgqr_bufferSize(handle, m, n, k, a, lda, tau, &lwork));
    return lwork;
}

pub fn dorgqr(handle: sys.cusolverDnHandle_t, m: i32, n: i32, k: i32, a: [*c]f64, lda: i32, tau: [*c]const f64, workspace: [*c]f64, lwork: i32, info: *i32) CusolverError!void {
    try toError(sys.cusolverDnDorgqr(handle, m, n, k, a, lda, tau, workspace, lwork, info));
}

// ============================================================================
// Eigenvalue Decomposition (syevd)
// ============================================================================

pub fn ssyevd_bufferSize(handle: sys.cusolverDnHandle_t, jobz: sys.cusolverEigMode_t, uplo: sys.cublasFillMode_t, n: i32, a: [*c]const f32, lda: i32, w: [*c]const f32) CusolverError!i32 {
    var lwork: i32 = undefined;
    try toError(sys.cusolverDnSsyevd_bufferSize(handle, jobz, uplo, n, a, lda, w, &lwork));
    return lwork;
}

pub fn dsyevd_bufferSize(handle: sys.cusolverDnHandle_t, jobz: sys.cusolverEigMode_t, uplo: sys.cublasFillMode_t, n: i32, a: [*c]const f64, lda: i32, w: [*c]const f64) CusolverError!i32 {
    var lwork: i32 = undefined;
    try toError(sys.cusolverDnDsyevd_bufferSize(handle, jobz, uplo, n, a, lda, w, &lwork));
    return lwork;
}

pub fn ssyevd(handle: sys.cusolverDnHandle_t, jobz: sys.cusolverEigMode_t, uplo: sys.cublasFillMode_t, n: i32, a: [*c]f32, lda: i32, w: [*c]f32, workspace: [*c]f32, lwork: i32, info: *i32) CusolverError!void {
    try toError(sys.cusolverDnSsyevd(handle, jobz, uplo, n, a, lda, w, workspace, lwork, info));
}

pub fn dsyevd(handle: sys.cusolverDnHandle_t, jobz: sys.cusolverEigMode_t, uplo: sys.cublasFillMode_t, n: i32, a: [*c]f64, lda: i32, w: [*c]f64, workspace: [*c]f64, lwork: i32, info: *i32) CusolverError!void {
    try toError(sys.cusolverDnDsyevd(handle, jobz, uplo, n, a, lda, w, workspace, lwork, info));
}

// ============================================================================
// Jacobi SVD (gesvdj) — batched-friendly SVD for small matrices
// ============================================================================

pub fn createGesvdjInfo() CusolverError!sys.gesvdjInfo_t {
    var info: sys.gesvdjInfo_t = undefined;
    try toError(sys.cusolverDnCreateGesvdjInfo(&info));
    return info;
}

pub fn destroyGesvdjInfo(info: sys.gesvdjInfo_t) CusolverError!void {
    try toError(sys.cusolverDnDestroyGesvdjInfo(info));
}

pub fn gesvdjSetTolerance(info: sys.gesvdjInfo_t, tolerance: f64) CusolverError!void {
    try toError(sys.cusolverDnXgesvdjSetTolerance(info, tolerance));
}

pub fn gesvdjSetMaxSweeps(info: sys.gesvdjInfo_t, max_sweeps: c_int) CusolverError!void {
    try toError(sys.cusolverDnXgesvdjSetMaxSweeps(info, max_sweeps));
}

pub fn sgesvdj_bufferSize(handle: sys.cusolverDnHandle_t, jobz: sys.cusolverEigMode_t, econ: c_int, m: i32, n: i32, a: [*c]const f32, lda: i32, s: [*c]const f32, u: [*c]const f32, ldu: i32, v: [*c]const f32, ldv: i32, params: sys.gesvdjInfo_t) CusolverError!i32 {
    var lwork: i32 = undefined;
    try toError(sys.cusolverDnSgesvdj_bufferSize(handle, jobz, econ, m, n, a, lda, s, u, ldu, v, ldv, &lwork, params));
    return lwork;
}

pub fn sgesvdj(handle: sys.cusolverDnHandle_t, jobz: sys.cusolverEigMode_t, econ: c_int, m: i32, n: i32, a: [*c]f32, lda: i32, s: [*c]f32, u: [*c]f32, ldu: i32, v: [*c]f32, ldv: i32, workspace: [*c]f32, lwork: i32, info: *i32, params: sys.gesvdjInfo_t) CusolverError!void {
    try toError(sys.cusolverDnSgesvdj(handle, jobz, econ, m, n, a, lda, s, u, ldu, v, ldv, workspace, lwork, info, params));
}

pub fn dgesvdj_bufferSize(handle: sys.cusolverDnHandle_t, jobz: sys.cusolverEigMode_t, econ: c_int, m: i32, n: i32, a: [*c]const f64, lda: i32, s: [*c]const f64, u: [*c]const f64, ldu: i32, v: [*c]const f64, ldv: i32, params: sys.gesvdjInfo_t) CusolverError!i32 {
    var lwork: i32 = undefined;
    try toError(sys.cusolverDnDgesvdj_bufferSize(handle, jobz, econ, m, n, a, lda, s, u, ldu, v, ldv, &lwork, params));
    return lwork;
}

pub fn dgesvdj(handle: sys.cusolverDnHandle_t, jobz: sys.cusolverEigMode_t, econ: c_int, m: i32, n: i32, a: [*c]f64, lda: i32, s: [*c]f64, u: [*c]f64, ldu: i32, v: [*c]f64, ldv: i32, workspace: [*c]f64, lwork: i32, info: *i32, params: sys.gesvdjInfo_t) CusolverError!void {
    try toError(sys.cusolverDnDgesvdj(handle, jobz, econ, m, n, a, lda, s, u, ldu, v, ldv, workspace, lwork, info, params));
}
