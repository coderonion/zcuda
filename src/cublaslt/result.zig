/// zCUDA: cuBLAS LT API - Error wrapping layer.
///
/// Layer 2: Converts cuBLAS status codes to Zig error unions for LT operations.
const std = @import("std");
const sys = @import("sys.zig");

// ============================================================================
// Error Type (shared with cuBLAS)
// ============================================================================

pub const CublasLtError = error{
    NotInitialized,
    AllocFailed,
    InvalidValue,
    ArchMismatch,
    MappingError,
    ExecutionFailed,
    InternalError,
    NotSupported,
    Unknown,
};

pub fn toError(status: sys.cublasStatus_t) CublasLtError!void {
    return switch (status) {
        sys.CUBLAS_STATUS_SUCCESS => {},
        sys.CUBLAS_STATUS_NOT_INITIALIZED => CublasLtError.NotInitialized,
        sys.CUBLAS_STATUS_ALLOC_FAILED => CublasLtError.AllocFailed,
        sys.CUBLAS_STATUS_INVALID_VALUE => CublasLtError.InvalidValue,
        sys.CUBLAS_STATUS_ARCH_MISMATCH => CublasLtError.ArchMismatch,
        sys.CUBLAS_STATUS_MAPPING_ERROR => CublasLtError.MappingError,
        sys.CUBLAS_STATUS_EXECUTION_FAILED => CublasLtError.ExecutionFailed,
        sys.CUBLAS_STATUS_INTERNAL_ERROR => CublasLtError.InternalError,
        sys.CUBLAS_STATUS_NOT_SUPPORTED => CublasLtError.NotSupported,
        else => CublasLtError.Unknown,
    };
}

// ============================================================================
// Handle Management
// ============================================================================

pub fn create() CublasLtError!sys.cublasLtHandle_t {
    var handle: sys.cublasLtHandle_t = undefined;
    try toError(sys.cublasLtCreate(&handle));
    return handle;
}

pub fn destroy(handle: sys.cublasLtHandle_t) CublasLtError!void {
    try toError(sys.cublasLtDestroy(handle));
}

// ============================================================================
// Matmul Descriptor
// ============================================================================

pub fn matmulDescCreate(
    compute_type: sys.cublasComputeType_t,
    scale_type: sys.cudaDataType,
) CublasLtError!sys.cublasLtMatmulDesc_t {
    var desc: sys.cublasLtMatmulDesc_t = undefined;
    try toError(sys.cublasLtMatmulDescCreate(&desc, compute_type, scale_type));
    return desc;
}

pub fn matmulDescDestroy(desc: sys.cublasLtMatmulDesc_t) CublasLtError!void {
    try toError(sys.cublasLtMatmulDescDestroy(desc));
}

// ============================================================================
// Matrix Layout
// ============================================================================

pub fn matrixLayoutCreate(
    data_type: sys.cudaDataType,
    rows: u64,
    cols: u64,
    ld: i64,
) CublasLtError!sys.cublasLtMatrixLayout_t {
    var layout: sys.cublasLtMatrixLayout_t = undefined;
    try toError(sys.cublasLtMatrixLayoutCreate(&layout, data_type, rows, cols, ld));
    return layout;
}

pub fn matrixLayoutDestroy(layout: sys.cublasLtMatrixLayout_t) CublasLtError!void {
    try toError(sys.cublasLtMatrixLayoutDestroy(layout));
}

// ============================================================================
// Matmul Preference
// ============================================================================

pub fn matmulPreferenceCreate() CublasLtError!sys.cublasLtMatmulPreference_t {
    var pref: sys.cublasLtMatmulPreference_t = undefined;
    try toError(sys.cublasLtMatmulPreferenceCreate(&pref));
    return pref;
}

pub fn matmulPreferenceDestroy(pref: sys.cublasLtMatmulPreference_t) CublasLtError!void {
    try toError(sys.cublasLtMatmulPreferenceDestroy(pref));
}

// ============================================================================
// Matmul Desc Attribute
// ============================================================================

pub fn matmulDescSetAttribute(
    desc: sys.cublasLtMatmulDesc_t,
    attr: sys.cublasLtMatmulDescAttributes_t,
    buf: *const anyopaque,
    size: usize,
) CublasLtError!void {
    try toError(sys.cublasLtMatmulDescSetAttribute(desc, attr, buf, size));
}

/// Set a matmul preference attribute (e.g., max workspace bytes).
pub fn matmulPreferenceSetAttribute(
    pref: sys.cublasLtMatmulPreference_t,
    attr: sys.cublasLtMatmulPreferenceAttributes_t,
    buf: *const anyopaque,
    size: usize,
) CublasLtError!void {
    try toError(sys.cublasLtMatmulPreferenceSetAttribute(pref, attr, buf, size));
}

/// Set a matrix layout attribute (e.g., batch count, strided batch offset).
pub fn matrixLayoutSetAttribute(
    layout: sys.cublasLtMatrixLayout_t,
    attr: sys.cublasLtMatrixLayoutAttribute_t,
    buf: *const anyopaque,
    size: usize,
) CublasLtError!void {
    try toError(sys.cublasLtMatrixLayoutSetAttribute(layout, attr, buf, size));
}

// ============================================================================
// Algorithm Heuristic
// ============================================================================

pub fn matmulAlgoGetHeuristic(
    handle: sys.cublasLtHandle_t,
    operation_desc: sys.cublasLtMatmulDesc_t,
    a_desc: sys.cublasLtMatrixLayout_t,
    b_desc: sys.cublasLtMatrixLayout_t,
    c_desc: sys.cublasLtMatrixLayout_t,
    d_desc: sys.cublasLtMatrixLayout_t,
    preference: sys.cublasLtMatmulPreference_t,
    requested_algo_count: i32,
    results: [*c]sys.cublasLtMatmulHeuristicResult_t,
    return_algo_count: *i32,
) CublasLtError!void {
    try toError(sys.cublasLtMatmulAlgoGetHeuristic(
        handle,
        operation_desc,
        a_desc,
        b_desc,
        c_desc,
        d_desc,
        preference,
        requested_algo_count,
        results,
        return_algo_count,
    ));
}

// ============================================================================
// Matmul Execution
// ============================================================================

pub fn matmul(
    handle: sys.cublasLtHandle_t,
    compute_desc: sys.cublasLtMatmulDesc_t,
    alpha: *const anyopaque,
    a: *const anyopaque,
    a_desc: sys.cublasLtMatrixLayout_t,
    b: *const anyopaque,
    b_desc: sys.cublasLtMatrixLayout_t,
    beta: *const anyopaque,
    c: *const anyopaque,
    c_desc: sys.cublasLtMatrixLayout_t,
    d: *anyopaque,
    d_desc: sys.cublasLtMatrixLayout_t,
    algo: *const sys.cublasLtMatmulAlgo_t,
    workspace: ?*anyopaque,
    workspace_size: usize,
    stream: ?*anyopaque,
) CublasLtError!void {
    try toError(sys.cublasLtMatmul(
        handle,
        compute_desc,
        alpha,
        a,
        a_desc,
        b,
        b_desc,
        beta,
        c,
        c_desc,
        d,
        d_desc,
        algo,
        workspace,
        workspace_size,
        @ptrCast(stream),
    ));
}
