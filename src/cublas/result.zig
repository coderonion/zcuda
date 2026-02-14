/// zCUDA: cuBLAS API - Error wrapping layer.
///
/// Layer 2: Converts cuBLAS C-style status codes to Zig error unions.
const std = @import("std");
const sys = @import("sys.zig");

// ============================================================================
// Error Type
// ============================================================================

/// Represents a cuBLAS error.
pub const CublasError = error{
    NotInitialized,
    AllocFailed,
    InvalidValue,
    ArchMismatch,
    MappingError,
    ExecutionFailed,
    InternalError,
    NotSupported,
    LicenseError,
    Unknown,
};

/// Convert a cublasStatus_t to a Zig error.
pub fn toError(status: sys.cublasStatus_t) CublasError!void {
    return switch (status) {
        sys.CUBLAS_STATUS_SUCCESS => {},
        sys.CUBLAS_STATUS_NOT_INITIALIZED => CublasError.NotInitialized,
        sys.CUBLAS_STATUS_ALLOC_FAILED => CublasError.AllocFailed,
        sys.CUBLAS_STATUS_INVALID_VALUE => CublasError.InvalidValue,
        sys.CUBLAS_STATUS_ARCH_MISMATCH => CublasError.ArchMismatch,
        sys.CUBLAS_STATUS_MAPPING_ERROR => CublasError.MappingError,
        sys.CUBLAS_STATUS_EXECUTION_FAILED => CublasError.ExecutionFailed,
        sys.CUBLAS_STATUS_INTERNAL_ERROR => CublasError.InternalError,
        sys.CUBLAS_STATUS_NOT_SUPPORTED => CublasError.NotSupported,
        sys.CUBLAS_STATUS_LICENSE_ERROR => CublasError.LicenseError,
        else => CublasError.Unknown,
    };
}

// ============================================================================
// Handle Management
// ============================================================================

/// Create a cuBLAS handle.
pub fn create() CublasError!sys.cublasHandle_t {
    var handle: sys.cublasHandle_t = undefined;
    try toError(sys.cublasCreate_v2(&handle));
    return handle;
}

/// Destroy a cuBLAS handle.
pub fn destroy(handle: sys.cublasHandle_t) CublasError!void {
    try toError(sys.cublasDestroy_v2(handle));
}

/// Set the stream for a cuBLAS handle.
pub fn setStream(handle: sys.cublasHandle_t, stream: anyopaque) CublasError!void {
    try toError(sys.cublasSetStream(handle, @ptrCast(&stream)));
}

/// Set the pointer mode for a cuBLAS handle.
pub fn setPointerMode(handle: sys.cublasHandle_t, mode: sys.cublasPointerMode_t) CublasError!void {
    try toError(sys.cublasSetPointerMode(handle, mode));
}

// ============================================================================
// BLAS Level 1
// ============================================================================

/// SASUM: Sum of absolute values (float).
pub fn sasum(handle: sys.cublasHandle_t, n: i32, x: [*c]const f32, incx: i32, result_ptr: *f32) CublasError!void {
    try toError(sys.cublasSasum(handle, n, x, incx, result_ptr));
}

/// DASUM: Sum of absolute values (double).
pub fn dasum(handle: sys.cublasHandle_t, n: i32, x: [*c]const f64, incx: i32, result_ptr: *f64) CublasError!void {
    try toError(sys.cublasDasum(handle, n, x, incx, result_ptr));
}

/// SAXPY: y = a*x + y (float).
pub fn saxpy(handle: sys.cublasHandle_t, n: i32, alpha: *const f32, x: [*c]const f32, incx: i32, y: [*c]f32, incy: i32) CublasError!void {
    try toError(sys.cublasSaxpy(handle, n, alpha, x, incx, y, incy));
}

/// DAXPY: y = a*x + y (double).
pub fn daxpy(handle: sys.cublasHandle_t, n: i32, alpha: *const f64, x: [*c]const f64, incx: i32, y: [*c]f64, incy: i32) CublasError!void {
    try toError(sys.cublasDaxpy(handle, n, alpha, x, incx, y, incy));
}

/// SDOT: Dot product (float).
pub fn sdot(handle: sys.cublasHandle_t, n: i32, x: [*c]const f32, incx: i32, y: [*c]const f32, incy: i32, result_ptr: *f32) CublasError!void {
    try toError(sys.cublasSdot(handle, n, x, incx, y, incy, result_ptr));
}

/// SNRM2: Euclidean norm (float).
pub fn snrm2(handle: sys.cublasHandle_t, n: i32, x: [*c]const f32, incx: i32, result_ptr: *f32) CublasError!void {
    try toError(sys.cublasSnrm2(handle, n, x, incx, result_ptr));
}

/// SSCAL: x = a*x (float).
pub fn sscal(handle: sys.cublasHandle_t, n: i32, alpha: *const f32, x: [*c]f32, incx: i32) CublasError!void {
    try toError(sys.cublasSscal(handle, n, alpha, x, incx));
}

/// SCOPY: y = x (float copy).
pub fn scopy(handle: sys.cublasHandle_t, n: i32, x: [*c]const f32, incx: i32, y: [*c]f32, incy: i32) CublasError!void {
    try toError(sys.cublasScopy(handle, n, x, incx, y, incy));
}

/// DCOPY: y = x (double copy).
pub fn dcopy(handle: sys.cublasHandle_t, n: i32, x: [*c]const f64, incx: i32, y: [*c]f64, incy: i32) CublasError!void {
    try toError(sys.cublasDcopy(handle, n, x, incx, y, incy));
}

/// SSWAP: swap x and y (float).
pub fn sswap(handle: sys.cublasHandle_t, n: i32, x: [*c]f32, incx: i32, y: [*c]f32, incy: i32) CublasError!void {
    try toError(sys.cublasSswap(handle, n, x, incx, y, incy));
}

/// DSWAP: swap x and y (double).
pub fn dswap(handle: sys.cublasHandle_t, n: i32, x: [*c]f64, incx: i32, y: [*c]f64, incy: i32) CublasError!void {
    try toError(sys.cublasDswap(handle, n, x, incx, y, incy));
}

/// ISAMAX: index of max absolute value (float).
pub fn isamax(handle: sys.cublasHandle_t, n: i32, x: [*c]const f32, incx: i32, result_ptr: *i32) CublasError!void {
    try toError(sys.cublasIsamax(handle, n, x, incx, result_ptr));
}

/// IDAMAX: index of max absolute value (double).
pub fn idamax(handle: sys.cublasHandle_t, n: i32, x: [*c]const f64, incx: i32, result_ptr: *i32) CublasError!void {
    try toError(sys.cublasIdamax(handle, n, x, incx, result_ptr));
}

/// ISAMIN: index of min absolute value (float).
pub fn isamin(handle: sys.cublasHandle_t, n: i32, x: [*c]const f32, incx: i32, result_ptr: *i32) CublasError!void {
    try toError(sys.cublasIsamin(handle, n, x, incx, result_ptr));
}

/// IDAMIN: index of min absolute value (double).
pub fn idamin(handle: sys.cublasHandle_t, n: i32, x: [*c]const f64, incx: i32, result_ptr: *i32) CublasError!void {
    try toError(sys.cublasIdamin(handle, n, x, incx, result_ptr));
}

/// SROT: Givens rotation x = c*x + s*y, y = -s*x + c*y (float).
pub fn srot(handle: sys.cublasHandle_t, n: i32, x: [*c]f32, incx: i32, y: [*c]f32, incy: i32, c_val: *const f32, s: *const f32) CublasError!void {
    try toError(sys.cublasSrot_v2(handle, n, x, incx, y, incy, c_val, s));
}

/// DROT: Givens rotation x = c*x + s*y, y = -s*x + c*y (double).
pub fn drot(handle: sys.cublasHandle_t, n: i32, x: [*c]f64, incx: i32, y: [*c]f64, incy: i32, c_val: *const f64, s: *const f64) CublasError!void {
    try toError(sys.cublasDrot_v2(handle, n, x, incx, y, incy, c_val, s));
}

// ============================================================================
// BLAS Level 2
// ============================================================================

/// SSYMV: Symmetric matrix-vector multiply y = alpha*A*x + beta*y (float).
pub fn ssymv(
    handle: sys.cublasHandle_t,
    uplo: sys.cublasFillMode_t,
    n: i32,
    alpha: *const f32,
    a: [*c]const f32,
    lda: i32,
    x: [*c]const f32,
    incx: i32,
    beta: *const f32,
    y: [*c]f32,
    incy: i32,
) CublasError!void {
    try toError(sys.cublasSsymv_v2(handle, uplo, n, alpha, a, lda, x, incx, beta, y, incy));
}

/// DSYMV: Symmetric matrix-vector multiply y = alpha*A*x + beta*y (double).
pub fn dsymv(
    handle: sys.cublasHandle_t,
    uplo: sys.cublasFillMode_t,
    n: i32,
    alpha: *const f64,
    a: [*c]const f64,
    lda: i32,
    x: [*c]const f64,
    incx: i32,
    beta: *const f64,
    y: [*c]f64,
    incy: i32,
) CublasError!void {
    try toError(sys.cublasDsymv_v2(handle, uplo, n, alpha, a, lda, x, incx, beta, y, incy));
}

/// SSYR: Symmetric rank-1 update A = alpha*x*x^T + A (float).
pub fn ssyr(
    handle: sys.cublasHandle_t,
    uplo: sys.cublasFillMode_t,
    n: i32,
    alpha: *const f32,
    x: [*c]const f32,
    incx: i32,
    a: [*c]f32,
    lda: i32,
) CublasError!void {
    try toError(sys.cublasSsyr_v2(handle, uplo, n, alpha, x, incx, a, lda));
}

/// DSYR: Symmetric rank-1 update A = alpha*x*x^T + A (double).
pub fn dsyr(
    handle: sys.cublasHandle_t,
    uplo: sys.cublasFillMode_t,
    n: i32,
    alpha: *const f64,
    x: [*c]const f64,
    incx: i32,
    a: [*c]f64,
    lda: i32,
) CublasError!void {
    try toError(sys.cublasDsyr_v2(handle, uplo, n, alpha, x, incx, a, lda));
}

/// STRMV: Triangular matrix-vector multiply x = A*x (float).
pub fn strmv(
    handle: sys.cublasHandle_t,
    uplo: sys.cublasFillMode_t,
    trans: sys.cublasOperation_t,
    diag: sys.cublasDiagType_t,
    n: i32,
    a: [*c]const f32,
    lda: i32,
    x: [*c]f32,
    incx: i32,
) CublasError!void {
    try toError(sys.cublasStrmv_v2(handle, uplo, trans, diag, n, a, lda, x, incx));
}

/// DTRMV: Triangular matrix-vector multiply x = A*x (double).
pub fn dtrmv(
    handle: sys.cublasHandle_t,
    uplo: sys.cublasFillMode_t,
    trans: sys.cublasOperation_t,
    diag: sys.cublasDiagType_t,
    n: i32,
    a: [*c]const f64,
    lda: i32,
    x: [*c]f64,
    incx: i32,
) CublasError!void {
    try toError(sys.cublasDtrmv_v2(handle, uplo, trans, diag, n, a, lda, x, incx));
}

/// STRSV: Triangular solve A*x = b for x (float).
pub fn strsv(
    handle: sys.cublasHandle_t,
    uplo: sys.cublasFillMode_t,
    trans: sys.cublasOperation_t,
    diag: sys.cublasDiagType_t,
    n: i32,
    a: [*c]const f32,
    lda: i32,
    x: [*c]f32,
    incx: i32,
) CublasError!void {
    try toError(sys.cublasStrsv_v2(handle, uplo, trans, diag, n, a, lda, x, incx));
}

/// DTRSV: Triangular solve A*x = b for x (double).
pub fn dtrsv(
    handle: sys.cublasHandle_t,
    uplo: sys.cublasFillMode_t,
    trans: sys.cublasOperation_t,
    diag: sys.cublasDiagType_t,
    n: i32,
    a: [*c]const f64,
    lda: i32,
    x: [*c]f64,
    incx: i32,
) CublasError!void {
    try toError(sys.cublasDtrsv_v2(handle, uplo, trans, diag, n, a, lda, x, incx));
}
/// SGEMV: Matrix-vector multiply y = alpha*A*x + beta*y (float).
pub fn sgemv(
    handle: sys.cublasHandle_t,
    trans: sys.cublasOperation_t,
    m: i32,
    n: i32,
    alpha: *const f32,
    a: [*c]const f32,
    lda: i32,
    x: [*c]const f32,
    incx: i32,
    beta: *const f32,
    y: [*c]f32,
    incy: i32,
) CublasError!void {
    try toError(sys.cublasSgemv(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy));
}

/// DGEMV: Matrix-vector multiply y = alpha*A*x + beta*y (double).
pub fn dgemv(
    handle: sys.cublasHandle_t,
    trans: sys.cublasOperation_t,
    m: i32,
    n: i32,
    alpha: *const f64,
    a: [*c]const f64,
    lda: i32,
    x: [*c]const f64,
    incx: i32,
    beta: *const f64,
    y: [*c]f64,
    incy: i32,
) CublasError!void {
    try toError(sys.cublasDgemv(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy));
}

// ============================================================================
// BLAS Level 3
// ============================================================================

/// SGEMM: Matrix-matrix multiply C = alpha*A*B + beta*C (float).
pub fn sgemm(
    handle: sys.cublasHandle_t,
    transa: sys.cublasOperation_t,
    transb: sys.cublasOperation_t,
    m: i32,
    n: i32,
    k: i32,
    alpha: *const f32,
    a: [*c]const f32,
    lda: i32,
    b: [*c]const f32,
    ldb: i32,
    beta: *const f32,
    c_out: [*c]f32,
    ldc: i32,
) CublasError!void {
    try toError(sys.cublasSgemm_v2(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c_out, ldc));
}

/// DGEMM: Matrix-matrix multiply C = alpha*A*B + beta*C (double).
pub fn dgemm(
    handle: sys.cublasHandle_t,
    transa: sys.cublasOperation_t,
    transb: sys.cublasOperation_t,
    m: i32,
    n: i32,
    k: i32,
    alpha: *const f64,
    a: [*c]const f64,
    lda: i32,
    b: [*c]const f64,
    ldb: i32,
    beta: *const f64,
    c_out: [*c]f64,
    ldc: i32,
) CublasError!void {
    try toError(sys.cublasDgemm_v2(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c_out, ldc));
}

/// HGEMM: Matrix-matrix multiply C = alpha*A*B + beta*C (half-precision FP16).
/// Uses opaque pointers since Zig has no native __half type.
pub fn hgemm(
    handle: sys.cublasHandle_t,
    transa: sys.cublasOperation_t,
    transb: sys.cublasOperation_t,
    m: i32,
    n: i32,
    k: i32,
    alpha: *const anyopaque,
    a: *const anyopaque,
    lda: i32,
    b: *const anyopaque,
    ldb: i32,
    beta: *const anyopaque,
    c_out: *anyopaque,
    ldc: i32,
) CublasError!void {
    try toError(sys.cublasHgemm(handle, transa, transb, m, n, k, @ptrCast(alpha), @ptrCast(a), lda, @ptrCast(b), ldb, @ptrCast(beta), @ptrCast(c_out), ldc));
}

/// HGEMM Strided Batched (half-precision FP16).
pub fn hgemmStridedBatched(
    handle: sys.cublasHandle_t,
    transa: sys.cublasOperation_t,
    transb: sys.cublasOperation_t,
    m: i32,
    n: i32,
    k: i32,
    alpha: *const anyopaque,
    a: *const anyopaque,
    lda: i32,
    stride_a: i64,
    b: *const anyopaque,
    ldb: i32,
    stride_b: i64,
    beta: *const anyopaque,
    c_out: *anyopaque,
    ldc: i32,
    stride_c: i64,
    batch_count: i32,
) CublasError!void {
    try toError(sys.cublasHgemmStridedBatched(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        @ptrCast(alpha),
        @ptrCast(a),
        lda,
        stride_a,
        @ptrCast(b),
        ldb,
        stride_b,
        @ptrCast(beta),
        @ptrCast(c_out),
        ldc,
        stride_c,
        batch_count,
    ));
}

/// SGEMM Strided Batched (float).
pub fn sgemmStridedBatched(
    handle: sys.cublasHandle_t,
    transa: sys.cublasOperation_t,
    transb: sys.cublasOperation_t,
    m: i32,
    n: i32,
    k: i32,
    alpha: *const f32,
    a: [*c]const f32,
    lda: i32,
    stride_a: i64,
    b: [*c]const f32,
    ldb: i32,
    stride_b: i64,
    beta: *const f32,
    c_out: [*c]f32,
    ldc: i32,
    stride_c: i64,
    batch_count: i32,
) CublasError!void {
    try toError(sys.cublasSgemmStridedBatched(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        a,
        lda,
        stride_a,
        b,
        ldb,
        stride_b,
        beta,
        c_out,
        ldc,
        stride_c,
        batch_count,
    ));
}

/// DGEMM Strided Batched (double).
pub fn dgemmStridedBatched(
    handle: sys.cublasHandle_t,
    transa: sys.cublasOperation_t,
    transb: sys.cublasOperation_t,
    m: i32,
    n: i32,
    k: i32,
    alpha: *const f64,
    a: [*c]const f64,
    lda: i32,
    stride_a: i64,
    b: [*c]const f64,
    ldb: i32,
    stride_b: i64,
    beta: *const f64,
    c_out: [*c]f64,
    ldc: i32,
    stride_c: i64,
    batch_count: i32,
) CublasError!void {
    try toError(sys.cublasDgemmStridedBatched(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        a,
        lda,
        stride_a,
        b,
        ldb,
        stride_b,
        beta,
        c_out,
        ldc,
        stride_c,
        batch_count,
    ));
}

/// DSCAL: x = a*x (double).
pub fn dscal(handle: sys.cublasHandle_t, n: i32, alpha: *const f64, x: [*c]f64, incx: i32) CublasError!void {
    try toError(sys.cublasDscal(handle, n, alpha, x, incx));
}

/// DNRM2: Euclidean norm (double).
pub fn dnrm2(handle: sys.cublasHandle_t, n: i32, x: [*c]const f64, incx: i32, result_ptr: *f64) CublasError!void {
    try toError(sys.cublasDnrm2(handle, n, x, incx, result_ptr));
}

/// DDOT: Dot product (double).
pub fn ddot(handle: sys.cublasHandle_t, n: i32, x: [*c]const f64, incx: i32, y: [*c]const f64, incy: i32, result_ptr: *f64) CublasError!void {
    try toError(sys.cublasDdot(handle, n, x, incx, y, incy, result_ptr));
}

// ============================================================================
// Extended GEMM (mixed precision)
// ============================================================================

/// GemmEx: mixed-precision GEMM with explicit data/compute types.
pub fn gemmEx(
    handle: sys.cublasHandle_t,
    transa: sys.cublasOperation_t,
    transb: sys.cublasOperation_t,
    m: i32,
    n: i32,
    k: i32,
    alpha: *const anyopaque,
    a: *const anyopaque,
    a_type: sys.cudaDataType_t,
    lda: i32,
    b: *const anyopaque,
    b_type: sys.cudaDataType_t,
    ldb: i32,
    beta: *const anyopaque,
    c_out: *anyopaque,
    c_type: sys.cudaDataType_t,
    ldc: i32,
    compute_type: sys.cublasComputeType_t,
    algo: sys.cublasGemmAlgo_t,
) CublasError!void {
    try toError(sys.cublasGemmEx(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        a,
        a_type,
        lda,
        b,
        b_type,
        ldb,
        beta,
        c_out,
        c_type,
        ldc,
        compute_type,
        algo,
    ));
}

// ============================================================================
// Batched GEMM (pointer-array batch)
// ============================================================================

/// Single-precision batched GEMM with pointer arrays.
pub fn sgemmBatched(
    handle: sys.cublasHandle_t,
    transa: sys.cublasOperation_t,
    transb: sys.cublasOperation_t,
    m: i32,
    n: i32,
    k: i32,
    alpha: *const f32,
    a_array: *const [*c]const f32,
    lda: i32,
    b_array: *const [*c]const f32,
    ldb: i32,
    beta: *const f32,
    c_array: *const [*c]f32,
    ldc: i32,
    batch_count: i32,
) CublasError!void {
    try toError(sys.cublasSgemmBatched(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        a_array,
        lda,
        b_array,
        ldb,
        beta,
        c_array,
        ldc,
        batch_count,
    ));
}

/// Double-precision batched GEMM with pointer arrays.
pub fn dgemmBatched(
    handle: sys.cublasHandle_t,
    transa: sys.cublasOperation_t,
    transb: sys.cublasOperation_t,
    m: i32,
    n: i32,
    k: i32,
    alpha: *const f64,
    a_array: *const [*c]const f64,
    lda: i32,
    b_array: *const [*c]const f64,
    ldb: i32,
    beta: *const f64,
    c_array: *const [*c]f64,
    ldc: i32,
    batch_count: i32,
) CublasError!void {
    try toError(sys.cublasDgemmBatched(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        a_array,
        lda,
        b_array,
        ldb,
        beta,
        c_array,
        ldc,
        batch_count,
    ));
}

/// GemmStridedBatchedEx: mixed-precision strided batched GEMM.
pub fn gemmStridedBatchedEx(
    handle: sys.cublasHandle_t,
    transa: sys.cublasOperation_t,
    transb: sys.cublasOperation_t,
    m: i32,
    n: i32,
    k: i32,
    alpha: *const anyopaque,
    a: *const anyopaque,
    a_type: sys.cudaDataType_t,
    lda: i32,
    stride_a: i64,
    b: *const anyopaque,
    b_type: sys.cudaDataType_t,
    ldb: i32,
    stride_b: i64,
    beta: *const anyopaque,
    c_out: *anyopaque,
    c_type: sys.cudaDataType_t,
    ldc: i32,
    stride_c: i64,
    batch_count: i32,
    compute_type: sys.cublasComputeType_t,
    algo: sys.cublasGemmAlgo_t,
) CublasError!void {
    try toError(sys.cublasGemmStridedBatchedEx(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        a,
        a_type,
        lda,
        stride_a,
        b,
        b_type,
        ldb,
        stride_b,
        beta,
        c_out,
        c_type,
        ldc,
        stride_c,
        batch_count,
        compute_type,
        algo,
    ));
}

// ============================================================================
// Triangular Solve (TRSM)
// ============================================================================

/// Single-precision triangular solve: solves op(A) * X = alpha * B or X * op(A) = alpha * B.
pub fn strsm(
    handle: sys.cublasHandle_t,
    side: sys.cublasSideMode_t,
    uplo: sys.cublasFillMode_t,
    trans: sys.cublasOperation_t,
    diag: sys.cublasDiagType_t,
    m: i32,
    n: i32,
    alpha: *const f32,
    a: *const f32,
    lda: i32,
    b: *f32,
    ldb: i32,
) CublasError!void {
    try toError(sys.cublasStrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb));
}

/// Double-precision triangular solve.
pub fn dtrsm(
    handle: sys.cublasHandle_t,
    side: sys.cublasSideMode_t,
    uplo: sys.cublasFillMode_t,
    trans: sys.cublasOperation_t,
    diag: sys.cublasDiagType_t,
    m: i32,
    n: i32,
    alpha: *const f64,
    a: *const f64,
    lda: i32,
    b: *f64,
    ldb: i32,
) CublasError!void {
    try toError(sys.cublasDtrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb));
}

// ============================================================================
// Symmetric Rank-k Update (SYRK)
// ============================================================================

/// SSYRK: C = alpha * A * A^T + beta * C (float).
pub fn ssyrk(
    handle: sys.cublasHandle_t,
    uplo: sys.cublasFillMode_t,
    trans: sys.cublasOperation_t,
    n: i32,
    k: i32,
    alpha: *const f32,
    a: [*c]const f32,
    lda: i32,
    beta: *const f32,
    c_out: [*c]f32,
    ldc: i32,
) CublasError!void {
    try toError(sys.cublasSsyrk_v2(handle, uplo, trans, n, k, alpha, a, lda, beta, c_out, ldc));
}

/// DSYRK: C = alpha * A * A^T + beta * C (double).
pub fn dsyrk(
    handle: sys.cublasHandle_t,
    uplo: sys.cublasFillMode_t,
    trans: sys.cublasOperation_t,
    n: i32,
    k: i32,
    alpha: *const f64,
    a: [*c]const f64,
    lda: i32,
    beta: *const f64,
    c_out: [*c]f64,
    ldc: i32,
) CublasError!void {
    try toError(sys.cublasDsyrk_v2(handle, uplo, trans, n, k, alpha, a, lda, beta, c_out, ldc));
}

// ============================================================================
// Triangular Matrix Multiply (TRMM)
// ============================================================================

/// STRMM: B = alpha * op(A) * B (float, A is triangular).
pub fn strmm(
    handle: sys.cublasHandle_t,
    side: sys.cublasSideMode_t,
    uplo: sys.cublasFillMode_t,
    trans: sys.cublasOperation_t,
    diag: sys.cublasDiagType_t,
    m: i32,
    n: i32,
    alpha: *const f32,
    a: [*c]const f32,
    lda: i32,
    b: [*c]const f32,
    ldb: i32,
    c_out: [*c]f32,
    ldc: i32,
) CublasError!void {
    try toError(sys.cublasStrmm_v2(handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, c_out, ldc));
}

/// DTRMM: B = alpha * op(A) * B (double, A is triangular).
pub fn dtrmm(
    handle: sys.cublasHandle_t,
    side: sys.cublasSideMode_t,
    uplo: sys.cublasFillMode_t,
    trans: sys.cublasOperation_t,
    diag: sys.cublasDiagType_t,
    m: i32,
    n: i32,
    alpha: *const f64,
    a: [*c]const f64,
    lda: i32,
    b: [*c]const f64,
    ldb: i32,
    c_out: [*c]f64,
    ldc: i32,
) CublasError!void {
    try toError(sys.cublasDtrmm_v2(handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, c_out, ldc));
}

// ============================================================================
// Symmetric Multiply (SYMM)
// ============================================================================

/// Single-precision symmetric matrix multiply: C = alpha * A * B + beta * C (or B * A).
pub fn ssymm(
    handle: sys.cublasHandle_t,
    side: sys.cublasSideMode_t,
    uplo: sys.cublasFillMode_t,
    m: i32,
    n: i32,
    alpha: *const f32,
    a: *const f32,
    lda: i32,
    b: *const f32,
    ldb: i32,
    beta: *const f32,
    c_out: *f32,
    ldc: i32,
) CublasError!void {
    try toError(sys.cublasSsymm_v2(handle, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c_out, ldc));
}

/// Double-precision symmetric matrix multiply.
pub fn dsymm(
    handle: sys.cublasHandle_t,
    side: sys.cublasSideMode_t,
    uplo: sys.cublasFillMode_t,
    m: i32,
    n: i32,
    alpha: *const f64,
    a: *const f64,
    lda: i32,
    b: *const f64,
    ldb: i32,
    beta: *const f64,
    c_out: *f64,
    ldc: i32,
) CublasError!void {
    try toError(sys.cublasDsymm_v2(handle, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c_out, ldc));
}

// ============================================================================
// Grouped Batched GEMM (GemmGroupedBatchedEx)
// ============================================================================

/// Grouped batched mixed-precision GEMM.
/// Each group can have different m/n/k/transa/transb/lda/ldb/ldc.
/// group_count groups, with group_size[i] batches per group.
pub fn gemmGroupedBatchedEx(
    handle: sys.cublasHandle_t,
    transa_array: [*c]const sys.cublasOperation_t,
    transb_array: [*c]const sys.cublasOperation_t,
    m_array: [*c]const i32,
    n_array: [*c]const i32,
    k_array: [*c]const i32,
    alpha_array: *const anyopaque,
    a_array: [*c]const *const anyopaque,
    a_type: sys.cudaDataType_t,
    lda_array: [*c]const i32,
    b_array: [*c]const *const anyopaque,
    b_type: sys.cudaDataType_t,
    ldb_array: [*c]const i32,
    beta_array: *const anyopaque,
    c_array: [*c]const *anyopaque,
    c_type: sys.cudaDataType_t,
    ldc_array: [*c]const i32,
    group_count: i32,
    group_size: [*c]const i32,
    compute_type: sys.cublasComputeType_t,
) CublasError!void {
    try toError(sys.cublasGemmGroupedBatchedEx(
        handle,
        transa_array,
        transb_array,
        m_array,
        n_array,
        k_array,
        alpha_array,
        a_array,
        a_type,
        lda_array,
        b_array,
        b_type,
        ldb_array,
        beta_array,
        c_array,
        c_type,
        ldc_array,
        group_count,
        group_size,
        compute_type,
    ));
}

// ============================================================================
// BLAS Extensions — GEAM (Matrix Add/Transpose)
// ============================================================================

/// SGEAM: C = alpha * op(A) + beta * op(B) (float). Useful for matrix transpose.
pub fn sgeam(
    handle: sys.cublasHandle_t,
    transa: sys.cublasOperation_t,
    transb: sys.cublasOperation_t,
    m: i32,
    n: i32,
    alpha: *const f32,
    a: [*c]const f32,
    lda: i32,
    beta: *const f32,
    b: [*c]const f32,
    ldb: i32,
    c_out: [*c]f32,
    ldc: i32,
) CublasError!void {
    try toError(sys.cublasSgeam(handle, transa, transb, m, n, alpha, a, lda, beta, b, ldb, c_out, ldc));
}

/// DGEAM: C = alpha * op(A) + beta * op(B) (double).
pub fn dgeam(
    handle: sys.cublasHandle_t,
    transa: sys.cublasOperation_t,
    transb: sys.cublasOperation_t,
    m: i32,
    n: i32,
    alpha: *const f64,
    a: [*c]const f64,
    lda: i32,
    beta: *const f64,
    b: [*c]const f64,
    ldb: i32,
    c_out: [*c]f64,
    ldc: i32,
) CublasError!void {
    try toError(sys.cublasDgeam(handle, transa, transb, m, n, alpha, a, lda, beta, b, ldb, c_out, ldc));
}

// ============================================================================
// BLAS Extensions — DGMM (Diagonal Matrix Multiply)
// ============================================================================

/// SDGMM: C = A * diag(x) or diag(x) * A (float). `side` selects left or right.
pub fn sdgmm(
    handle: sys.cublasHandle_t,
    side: sys.cublasSideMode_t,
    m: i32,
    n: i32,
    a: [*c]const f32,
    lda: i32,
    x: [*c]const f32,
    incx: i32,
    c_out: [*c]f32,
    ldc: i32,
) CublasError!void {
    try toError(sys.cublasSdgmm(handle, side, m, n, a, lda, x, incx, c_out, ldc));
}

/// DDGMM: C = A * diag(x) or diag(x) * A (double).
pub fn ddgmm(
    handle: sys.cublasHandle_t,
    side: sys.cublasSideMode_t,
    m: i32,
    n: i32,
    a: [*c]const f64,
    lda: i32,
    x: [*c]const f64,
    incx: i32,
    c_out: [*c]f64,
    ldc: i32,
) CublasError!void {
    try toError(sys.cublasDdgmm(handle, side, m, n, a, lda, x, incx, c_out, ldc));
}
