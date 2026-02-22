/// zCUDA: cuSOLVER - Safe abstraction layer.
///
/// Layer 3: High-level wrappers for cuSOLVER dense solvers (LU, QR, SVD).
const std = @import("std");
const sys = @import("sys.zig");
const result = @import("result.zig");
const driver = @import("../driver/driver.zig");

pub const CusolverError = result.CusolverError;

/// A cuSOLVER dense context.
pub const CusolverDnContext = struct {
    handle: sys.cusolverDnHandle_t,
    cuda_ctx: *const driver.CudaContext,

    const Self = @This();

    /// Create a new cuSOLVER dense context.
    pub fn init(cuda_ctx: *const driver.CudaContext) !Self {
        try cuda_ctx.bindToThread();
        const handle = try result.create();
        return Self{ .handle = handle, .cuda_ctx = cuda_ctx };
    }

    /// Destroy the cuSOLVER handle.
    pub fn deinit(self: Self) void {
        result.destroy(self.handle) catch {};
    }

    /// Set the CUDA stream for this handle.
    pub fn setStream(self: Self, stream: *const driver.CudaStream) CusolverError!void {
        try result.setStream(self.handle, stream.stream);
    }

    // --- LU Factorization ---

    /// Get workspace size for LU factorization (float).
    pub fn sgetrf_bufferSize(self: Self, m: i32, n: i32, a: driver.CudaSlice(f32), lda: i32) CusolverError!i32 {
        return result.sgetrf_bufferSize(self.handle, m, n, @ptrFromInt(a.ptr), lda);
    }

    /// Get workspace size for LU factorization (double).
    pub fn dgetrf_bufferSize(self: Self, m: i32, n: i32, a: driver.CudaSlice(f64), lda: i32) CusolverError!i32 {
        return result.dgetrf_bufferSize(self.handle, m, n, @ptrFromInt(a.ptr), lda);
    }

    /// LU factorization: PA = LU (float).
    pub fn sgetrf(self: Self, m: i32, n: i32, a: driver.CudaSlice(f32), lda: i32, workspace: driver.CudaSlice(f32), ipiv: driver.CudaSlice(i32), info: driver.CudaSlice(i32)) CusolverError!void {
        try result.sgetrf(self.handle, m, n, @ptrFromInt(a.ptr), lda, @ptrFromInt(workspace.ptr), @ptrFromInt(ipiv.ptr), @ptrFromInt(info.ptr));
    }

    /// LU factorization: PA = LU (double).
    pub fn dgetrf(self: Self, m: i32, n: i32, a: driver.CudaSlice(f64), lda: i32, workspace: driver.CudaSlice(f64), ipiv: driver.CudaSlice(i32), info: driver.CudaSlice(i32)) CusolverError!void {
        try result.dgetrf(self.handle, m, n, @ptrFromInt(a.ptr), lda, @ptrFromInt(workspace.ptr), @ptrFromInt(ipiv.ptr), @ptrFromInt(info.ptr));
    }

    /// Solve Ax = B after LU factorization (float).
    pub fn sgetrs(self: Self, n: i32, nrhs: i32, a: driver.CudaSlice(f32), lda: i32, ipiv: driver.CudaSlice(i32), b: driver.CudaSlice(f32), ldb: i32, info: driver.CudaSlice(i32)) CusolverError!void {
        try result.sgetrs(self.handle, sys.CUBLAS_OP_N, n, nrhs, @ptrFromInt(a.ptr), lda, @ptrFromInt(ipiv.ptr), @ptrFromInt(b.ptr), ldb, @ptrFromInt(info.ptr));
    }

    /// Solve Ax = B after LU factorization (double).
    pub fn dgetrs(self: Self, n: i32, nrhs: i32, a: driver.CudaSlice(f64), lda: i32, ipiv: driver.CudaSlice(i32), b: driver.CudaSlice(f64), ldb: i32, info: driver.CudaSlice(i32)) CusolverError!void {
        try result.dgetrs(self.handle, sys.CUBLAS_OP_N, n, nrhs, @ptrFromInt(a.ptr), lda, @ptrFromInt(ipiv.ptr), @ptrFromInt(b.ptr), ldb, @ptrFromInt(info.ptr));
    }

    // --- QR Factorization ---

    /// Get workspace size for QR factorization (float).
    pub fn sgeqrf_bufferSize(self: Self, m: i32, n: i32, a: driver.CudaSlice(f32), lda: i32) CusolverError!i32 {
        return result.sgeqrf_bufferSize(self.handle, m, n, @ptrFromInt(a.ptr), lda);
    }

    /// QR factorization (float).
    pub fn sgeqrf(self: Self, m: i32, n: i32, a: driver.CudaSlice(f32), lda: i32, tau: driver.CudaSlice(f32), workspace: driver.CudaSlice(f32), lwork: i32, info: driver.CudaSlice(i32)) CusolverError!void {
        try result.sgeqrf(self.handle, m, n, @ptrFromInt(a.ptr), lda, @ptrFromInt(tau.ptr), @ptrFromInt(workspace.ptr), lwork, @ptrFromInt(info.ptr));
    }

    // --- SVD ---

    /// Get workspace size for SVD (float).
    pub fn sgesvd_bufferSize(self: Self, m: i32, n: i32) CusolverError!i32 {
        return result.sgesvd_bufferSize(self.handle, m, n);
    }

    /// Get workspace size for SVD (double).
    pub fn dgesvd_bufferSize(self: Self, m: i32, n: i32) CusolverError!i32 {
        return result.dgesvd_bufferSize(self.handle, m, n);
    }

    /// SVD: A = U * S * V^T (float).
    pub fn sgesvd(
        self: Self,
        jobu: u8,
        jobvt: u8,
        m: i32,
        n: i32,
        a: driver.CudaSlice(f32),
        lda: i32,
        s: driver.CudaSlice(f32),
        u: driver.CudaSlice(f32),
        ldu: i32,
        vt: driver.CudaSlice(f32),
        ldvt: i32,
        work: driver.CudaSlice(f32),
        lwork: i32,
        info: driver.CudaSlice(i32),
    ) CusolverError!void {
        try result.sgesvd(
            self.handle,
            jobu,
            jobvt,
            m,
            n,
            @ptrFromInt(a.ptr),
            lda,
            @ptrFromInt(s.ptr),
            @ptrFromInt(u.ptr),
            ldu,
            @ptrFromInt(vt.ptr),
            ldvt,
            @ptrFromInt(work.ptr),
            lwork,
            null,
            @ptrFromInt(info.ptr),
        );
    }

    /// SVD: A = U * S * V^T (double).
    pub fn dgesvd(
        self: Self,
        jobu: u8,
        jobvt: u8,
        m: i32,
        n: i32,
        a: driver.CudaSlice(f64),
        lda: i32,
        s: driver.CudaSlice(f64),
        u: driver.CudaSlice(f64),
        ldu: i32,
        vt: driver.CudaSlice(f64),
        ldvt: i32,
        work: driver.CudaSlice(f64),
        lwork: i32,
        info: driver.CudaSlice(i32),
    ) CusolverError!void {
        try result.dgesvd(
            self.handle,
            jobu,
            jobvt,
            m,
            n,
            @ptrFromInt(a.ptr),
            lda,
            @ptrFromInt(s.ptr),
            @ptrFromInt(u.ptr),
            ldu,
            @ptrFromInt(vt.ptr),
            ldvt,
            @ptrFromInt(work.ptr),
            lwork,
            null,
            @ptrFromInt(info.ptr),
        );
    }
};

/// Fill mode for Cholesky factorization.
pub const FillMode = enum {
    lower,
    upper,

    fn toSys(self: FillMode) sys.cublasFillMode_t {
        return switch (self) {
            .lower => sys.CUBLAS_FILL_MODE_LOWER,
            .upper => sys.CUBLAS_FILL_MODE_UPPER,
        };
    }
};

/// Extended cuSOLVER dense context with Cholesky and double QR operations.
pub const CusolverDnExt = struct {
    base: *const CusolverDnContext,

    pub fn init(ctx: *const CusolverDnContext) CusolverDnExt {
        return .{ .base = ctx };
    }

    // --- Cholesky ---

    pub fn spotrf_bufferSize(self: CusolverDnExt, uplo: FillMode, n: i32, a: driver.CudaSlice(f32), lda: i32) CusolverError!i32 {
        return result.spotrf_bufferSize(self.base.handle, uplo.toSys(), n, @ptrFromInt(a.ptr), lda);
    }

    pub fn dpotrf_bufferSize(self: CusolverDnExt, uplo: FillMode, n: i32, a: driver.CudaSlice(f64), lda: i32) CusolverError!i32 {
        return result.dpotrf_bufferSize(self.base.handle, uplo.toSys(), n, @ptrFromInt(a.ptr), lda);
    }

    pub fn spotrf(self: CusolverDnExt, uplo: FillMode, n: i32, a: driver.CudaSlice(f32), lda: i32, workspace: driver.CudaSlice(f32), lwork: i32, info: driver.CudaSlice(i32)) CusolverError!void {
        try result.spotrf(self.base.handle, uplo.toSys(), n, @ptrFromInt(a.ptr), lda, @ptrFromInt(workspace.ptr), lwork, @ptrFromInt(info.ptr));
    }

    pub fn dpotrf(self: CusolverDnExt, uplo: FillMode, n: i32, a: driver.CudaSlice(f64), lda: i32, workspace: driver.CudaSlice(f64), lwork: i32, info: driver.CudaSlice(i32)) CusolverError!void {
        try result.dpotrf(self.base.handle, uplo.toSys(), n, @ptrFromInt(a.ptr), lda, @ptrFromInt(workspace.ptr), lwork, @ptrFromInt(info.ptr));
    }

    pub fn spotrs(self: CusolverDnExt, uplo: FillMode, n: i32, nrhs: i32, a: driver.CudaSlice(f32), lda: i32, b: driver.CudaSlice(f32), ldb: i32, info: driver.CudaSlice(i32)) CusolverError!void {
        try result.spotrs(self.base.handle, uplo.toSys(), n, nrhs, @ptrFromInt(a.ptr), lda, @ptrFromInt(b.ptr), ldb, @ptrFromInt(info.ptr));
    }

    pub fn dpotrs(self: CusolverDnExt, uplo: FillMode, n: i32, nrhs: i32, a: driver.CudaSlice(f64), lda: i32, b: driver.CudaSlice(f64), ldb: i32, info: driver.CudaSlice(i32)) CusolverError!void {
        try result.dpotrs(self.base.handle, uplo.toSys(), n, nrhs, @ptrFromInt(a.ptr), lda, @ptrFromInt(b.ptr), ldb, @ptrFromInt(info.ptr));
    }

    // --- Double QR ---

    pub fn dgeqrf_bufferSize(self: CusolverDnExt, m: i32, n: i32, a: driver.CudaSlice(f64), lda: i32) CusolverError!i32 {
        return result.dgeqrf_bufferSize(self.base.handle, m, n, @ptrFromInt(a.ptr), lda);
    }

    pub fn dgeqrf(self: CusolverDnExt, m: i32, n: i32, a: driver.CudaSlice(f64), lda: i32, tau: driver.CudaSlice(f64), workspace: driver.CudaSlice(f64), lwork: i32, info: driver.CudaSlice(i32)) CusolverError!void {
        try result.dgeqrf(self.base.handle, m, n, @ptrFromInt(a.ptr), lda, @ptrFromInt(tau.ptr), @ptrFromInt(workspace.ptr), lwork, @ptrFromInt(info.ptr));
    }

    pub fn sorgqr_bufferSize(self: CusolverDnExt, m: i32, n: i32, k: i32, a: driver.CudaSlice(f32), lda: i32, tau: driver.CudaSlice(f32)) CusolverError!i32 {
        return result.sorgqr_bufferSize(self.base.handle, m, n, k, @ptrFromInt(a.ptr), lda, @ptrFromInt(tau.ptr));
    }

    pub fn sorgqr(self: CusolverDnExt, m: i32, n: i32, k: i32, a: driver.CudaSlice(f32), lda: i32, tau: driver.CudaSlice(f32), workspace: driver.CudaSlice(f32), lwork: i32, info: driver.CudaSlice(i32)) CusolverError!void {
        try result.sorgqr(self.base.handle, m, n, k, @ptrFromInt(a.ptr), lda, @ptrFromInt(tau.ptr), @ptrFromInt(workspace.ptr), lwork, @ptrFromInt(info.ptr));
    }

    pub fn dorgqr_bufferSize(self: CusolverDnExt, m: i32, n: i32, k: i32, a: driver.CudaSlice(f64), lda: i32, tau: driver.CudaSlice(f64)) CusolverError!i32 {
        return result.dorgqr_bufferSize(self.base.handle, m, n, k, @ptrFromInt(a.ptr), lda, @ptrFromInt(tau.ptr));
    }

    pub fn dorgqr(self: CusolverDnExt, m: i32, n: i32, k: i32, a: driver.CudaSlice(f64), lda: i32, tau: driver.CudaSlice(f64), workspace: driver.CudaSlice(f64), lwork: i32, info: driver.CudaSlice(i32)) CusolverError!void {
        try result.dorgqr(self.base.handle, m, n, k, @ptrFromInt(a.ptr), lda, @ptrFromInt(tau.ptr), @ptrFromInt(workspace.ptr), lwork, @ptrFromInt(info.ptr));
    }

    // --- Eigenvalue Decomposition (syevd) ---

    pub fn ssyevd_bufferSize(self: CusolverDnExt, jobz: EigMode, uplo: FillMode, n: i32, a: driver.CudaSlice(f32), lda: i32, w: driver.CudaSlice(f32)) CusolverError!i32 {
        return result.ssyevd_bufferSize(self.base.handle, jobz.toSys(), uplo.toSys(), n, @ptrFromInt(a.ptr), lda, @ptrFromInt(w.ptr));
    }

    pub fn dsyevd_bufferSize(self: CusolverDnExt, jobz: EigMode, uplo: FillMode, n: i32, a: driver.CudaSlice(f64), lda: i32, w: driver.CudaSlice(f64)) CusolverError!i32 {
        return result.dsyevd_bufferSize(self.base.handle, jobz.toSys(), uplo.toSys(), n, @ptrFromInt(a.ptr), lda, @ptrFromInt(w.ptr));
    }

    pub fn ssyevd(self: CusolverDnExt, jobz: EigMode, uplo: FillMode, n: i32, a: driver.CudaSlice(f32), lda: i32, w: driver.CudaSlice(f32), workspace: driver.CudaSlice(f32), lwork: i32, info: driver.CudaSlice(i32)) CusolverError!void {
        try result.ssyevd(self.base.handle, jobz.toSys(), uplo.toSys(), n, @ptrFromInt(a.ptr), lda, @ptrFromInt(w.ptr), @ptrFromInt(workspace.ptr), lwork, @ptrFromInt(info.ptr));
    }

    pub fn dsyevd(self: CusolverDnExt, jobz: EigMode, uplo: FillMode, n: i32, a: driver.CudaSlice(f64), lda: i32, w: driver.CudaSlice(f64), workspace: driver.CudaSlice(f64), lwork: i32, info: driver.CudaSlice(i32)) CusolverError!void {
        try result.dsyevd(self.base.handle, jobz.toSys(), uplo.toSys(), n, @ptrFromInt(a.ptr), lda, @ptrFromInt(w.ptr), @ptrFromInt(workspace.ptr), lwork, @ptrFromInt(info.ptr));
    }

    // --- Jacobi SVD (gesvdj) ---

    /// Get workspace size for Jacobi SVD (float).
    pub fn sgesvdj_bufferSize(self: CusolverDnExt, jobz: EigMode, econ: c_int, m: i32, n: i32, a: driver.CudaSlice(f32), lda: i32, s: driver.CudaSlice(f32), u: driver.CudaSlice(f32), ldu: i32, v: driver.CudaSlice(f32), ldv: i32, params: GesvdjInfo) CusolverError!i32 {
        return result.sgesvdj_bufferSize(self.base.handle, jobz.toSys(), econ, m, n, @ptrFromInt(a.ptr), lda, @ptrFromInt(s.ptr), @ptrFromInt(u.ptr), ldu, @ptrFromInt(v.ptr), ldv, params.info);
    }

    /// Jacobi SVD: A = U * S * V^T (float).
    pub fn sgesvdj(self: CusolverDnExt, jobz: EigMode, econ: c_int, m: i32, n: i32, a: driver.CudaSlice(f32), lda: i32, s: driver.CudaSlice(f32), u: driver.CudaSlice(f32), ldu: i32, v: driver.CudaSlice(f32), ldv: i32, workspace: driver.CudaSlice(f32), lwork: i32, dev_info: driver.CudaSlice(i32), params: GesvdjInfo) CusolverError!void {
        try result.sgesvdj(self.base.handle, jobz.toSys(), econ, m, n, @ptrFromInt(a.ptr), lda, @ptrFromInt(s.ptr), @ptrFromInt(u.ptr), ldu, @ptrFromInt(v.ptr), ldv, @ptrFromInt(workspace.ptr), lwork, @ptrFromInt(dev_info.ptr), params.info);
    }

    /// Get workspace size for Jacobi SVD (double).
    pub fn dgesvdj_bufferSize(self: CusolverDnExt, jobz: EigMode, econ: c_int, m: i32, n: i32, a: driver.CudaSlice(f64), lda: i32, s: driver.CudaSlice(f64), u: driver.CudaSlice(f64), ldu: i32, v: driver.CudaSlice(f64), ldv: i32, params: GesvdjInfo) CusolverError!i32 {
        return result.dgesvdj_bufferSize(self.base.handle, jobz.toSys(), econ, m, n, @ptrFromInt(a.ptr), lda, @ptrFromInt(s.ptr), @ptrFromInt(u.ptr), ldu, @ptrFromInt(v.ptr), ldv, params.info);
    }

    /// Jacobi SVD: A = U * S * V^T (double).
    pub fn dgesvdj(self: CusolverDnExt, jobz: EigMode, econ: c_int, m: i32, n: i32, a: driver.CudaSlice(f64), lda: i32, s: driver.CudaSlice(f64), u: driver.CudaSlice(f64), ldu: i32, v: driver.CudaSlice(f64), ldv: i32, workspace: driver.CudaSlice(f64), lwork: i32, dev_info: driver.CudaSlice(i32), params: GesvdjInfo) CusolverError!void {
        try result.dgesvdj(self.base.handle, jobz.toSys(), econ, m, n, @ptrFromInt(a.ptr), lda, @ptrFromInt(s.ptr), @ptrFromInt(u.ptr), ldu, @ptrFromInt(v.ptr), ldv, @ptrFromInt(workspace.ptr), lwork, @ptrFromInt(dev_info.ptr), params.info);
    }
};

/// Eigenvalue computation mode.
pub const EigMode = enum {
    /// Compute eigenvalues only.
    no_vector,
    /// Compute eigenvalues and eigenvectors.
    vector,

    pub fn toSys(self: EigMode) sys.cusolverEigMode_t {
        return switch (self) {
            .no_vector => sys.CUSOLVER_EIG_MODE_NOVECTOR,
            .vector => sys.CUSOLVER_EIG_MODE_VECTOR,
        };
    }
};

/// Jacobi SVD parameters. Free with deinit().
pub const GesvdjInfo = struct {
    info: sys.gesvdjInfo_t,

    pub fn init() CusolverError!GesvdjInfo {
        return .{ .info = try result.createGesvdjInfo() };
    }

    pub fn deinit(self: GesvdjInfo) void {
        result.destroyGesvdjInfo(self.info) catch {};
    }

    /// Set convergence tolerance (default: machine epsilon).
    pub fn setTolerance(self: GesvdjInfo, tolerance: f64) CusolverError!void {
        try result.gesvdjSetTolerance(self.info, tolerance);
    }

    /// Set maximum number of sweeps (default: 100).
    pub fn setMaxSweeps(self: GesvdjInfo, max_sweeps: c_int) CusolverError!void {
        try result.gesvdjSetMaxSweeps(self.info, max_sweeps);
    }
};
