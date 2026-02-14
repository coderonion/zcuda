/// zCUDA: cuSPARSE - Error wrapping layer.
///
/// Layer 2: Converts cuSPARSE status codes to Zig error unions.
const sys = @import("sys.zig");

pub const CusparseError = error{
    NotInitialized,
    AllocFailed,
    InvalidValue,
    ArchMismatch,
    InternalError,
    NotSupported,
    Unknown,
};

pub fn toError(status: sys.cusparseStatus_t) CusparseError!void {
    return switch (status) {
        sys.CUSPARSE_STATUS_SUCCESS => {},
        sys.CUSPARSE_STATUS_NOT_INITIALIZED => CusparseError.NotInitialized,
        sys.CUSPARSE_STATUS_ALLOC_FAILED => CusparseError.AllocFailed,
        sys.CUSPARSE_STATUS_INVALID_VALUE => CusparseError.InvalidValue,
        sys.CUSPARSE_STATUS_ARCH_MISMATCH => CusparseError.ArchMismatch,
        sys.CUSPARSE_STATUS_INTERNAL_ERROR => CusparseError.InternalError,
        sys.CUSPARSE_STATUS_NOT_SUPPORTED => CusparseError.NotSupported,
        else => CusparseError.Unknown,
    };
}

// ============================================================================
// Handle Management
// ============================================================================

pub fn create() CusparseError!sys.cusparseHandle_t {
    var handle: sys.cusparseHandle_t = undefined;
    try toError(sys.cusparseCreate(&handle));
    return handle;
}

pub fn destroy(handle: sys.cusparseHandle_t) CusparseError!void {
    try toError(sys.cusparseDestroy(handle));
}

pub fn setStream(handle: sys.cusparseHandle_t, stream: ?*anyopaque) CusparseError!void {
    try toError(sys.cusparseSetStream(handle, @ptrCast(stream)));
}

// ============================================================================
// Sparse Matrix Descriptors (generic API)
// ============================================================================

/// Create a CSR sparse matrix descriptor.
pub fn createCsr(
    rows: i64,
    cols: i64,
    nnz: i64,
    row_offsets: *anyopaque,
    col_indices: *anyopaque,
    values: *anyopaque,
    row_offsets_type: sys.cusparseIndexType_t,
    col_indices_type: sys.cusparseIndexType_t,
    idx_base: sys.cusparseIndexBase_t,
    value_type: sys.cudaDataType,
) CusparseError!sys.cusparseSpMatDescr_t {
    var desc: sys.cusparseSpMatDescr_t = undefined;
    try toError(sys.cusparseCreateCsr(
        &desc,
        rows,
        cols,
        nnz,
        row_offsets,
        col_indices,
        values,
        row_offsets_type,
        col_indices_type,
        idx_base,
        value_type,
    ));
    return desc;
}

/// Create a COO sparse matrix descriptor.
pub fn createCoo(
    rows: i64,
    cols: i64,
    nnz: i64,
    row_indices: *anyopaque,
    col_indices: *anyopaque,
    values: *anyopaque,
    idx_type: sys.cusparseIndexType_t,
    idx_base: sys.cusparseIndexBase_t,
    value_type: sys.cudaDataType,
) CusparseError!sys.cusparseSpMatDescr_t {
    var desc: sys.cusparseSpMatDescr_t = undefined;
    try toError(sys.cusparseCreateCoo(
        &desc,
        rows,
        cols,
        nnz,
        row_indices,
        col_indices,
        values,
        idx_type,
        idx_base,
        value_type,
    ));
    return desc;
}

/// Destroy a sparse matrix descriptor.
pub fn destroySpMat(desc: sys.cusparseSpMatDescr_t) CusparseError!void {
    try toError(sys.cusparseDestroySpMat(desc));
}

/// Create a CSC (Compressed Sparse Column) matrix descriptor.
pub fn createCsc(
    rows: i64,
    cols: i64,
    nnz: i64,
    col_offsets: *anyopaque,
    row_indices: *anyopaque,
    values: *anyopaque,
    col_offset_type: sys.cusparseIndexType_t,
    row_idx_type: sys.cusparseIndexType_t,
    idx_base: sys.cusparseIndexBase_t,
    value_type: sys.cudaDataType,
) CusparseError!sys.cusparseSpMatDescr_t {
    var desc: sys.cusparseSpMatDescr_t = undefined;
    try toError(sys.cusparseCreateCsc(
        &desc,
        rows,
        cols,
        nnz,
        col_offsets,
        row_indices,
        values,
        col_offset_type,
        row_idx_type,
        idx_base,
        value_type,
    ));
    return desc;
}

// ============================================================================
// Dense Vector Descriptor
// ============================================================================

pub fn createDnVec(
    size: i64,
    values: *anyopaque,
    value_type: sys.cudaDataType,
) CusparseError!sys.cusparseDnVecDescr_t {
    var desc: sys.cusparseDnVecDescr_t = undefined;
    try toError(sys.cusparseCreateDnVec(&desc, size, values, value_type));
    return desc;
}

pub fn destroyDnVec(desc: sys.cusparseDnVecDescr_t) CusparseError!void {
    try toError(sys.cusparseDestroyDnVec(desc));
}

// ============================================================================
// Dense Matrix Descriptor
// ============================================================================

pub fn createDnMat(
    rows: i64,
    cols: i64,
    ld: i64,
    values: *anyopaque,
    value_type: sys.cudaDataType,
    order: sys.cusparseOrder_t,
) CusparseError!sys.cusparseDnMatDescr_t {
    var desc: sys.cusparseDnMatDescr_t = undefined;
    try toError(sys.cusparseCreateDnMat(&desc, rows, cols, ld, values, value_type, order));
    return desc;
}

pub fn destroyDnMat(desc: sys.cusparseDnMatDescr_t) CusparseError!void {
    try toError(sys.cusparseDestroyDnMat(desc));
}

// ============================================================================
// SpMV (Sparse Matrix-Vector Multiply)
// ============================================================================

pub fn spMV_bufferSize(
    handle: sys.cusparseHandle_t,
    op: sys.cusparseOperation_t,
    alpha: *const anyopaque,
    mat_a: sys.cusparseSpMatDescr_t,
    vec_x: sys.cusparseDnVecDescr_t,
    beta: *const anyopaque,
    vec_y: sys.cusparseDnVecDescr_t,
    compute_type: sys.cudaDataType,
    alg: sys.cusparseSpMVAlg_t,
) CusparseError!usize {
    var buffer_size: usize = undefined;
    try toError(sys.cusparseSpMV_bufferSize(
        handle,
        op,
        alpha,
        mat_a,
        vec_x,
        beta,
        vec_y,
        compute_type,
        alg,
        &buffer_size,
    ));
    return buffer_size;
}

pub fn spMV(
    handle: sys.cusparseHandle_t,
    op: sys.cusparseOperation_t,
    alpha: *const anyopaque,
    mat_a: sys.cusparseSpMatDescr_t,
    vec_x: sys.cusparseDnVecDescr_t,
    beta: *const anyopaque,
    vec_y: sys.cusparseDnVecDescr_t,
    compute_type: sys.cudaDataType,
    alg: sys.cusparseSpMVAlg_t,
    buffer: ?*anyopaque,
) CusparseError!void {
    try toError(sys.cusparseSpMV(
        handle,
        op,
        alpha,
        mat_a,
        vec_x,
        beta,
        vec_y,
        compute_type,
        alg,
        buffer,
    ));
}

// ============================================================================
// SpMM (Sparse Matrix-Matrix Multiply)
// ============================================================================

pub fn spMM_bufferSize(
    handle: sys.cusparseHandle_t,
    op_a: sys.cusparseOperation_t,
    op_b: sys.cusparseOperation_t,
    alpha: *const anyopaque,
    mat_a: sys.cusparseSpMatDescr_t,
    mat_b: sys.cusparseDnMatDescr_t,
    beta: *const anyopaque,
    mat_c: sys.cusparseDnMatDescr_t,
    compute_type: sys.cudaDataType,
    alg: sys.cusparseSpMMAlg_t,
) CusparseError!usize {
    var buffer_size: usize = undefined;
    try toError(sys.cusparseSpMM_bufferSize(
        handle,
        op_a,
        op_b,
        alpha,
        mat_a,
        mat_b,
        beta,
        mat_c,
        compute_type,
        alg,
        &buffer_size,
    ));
    return buffer_size;
}

pub fn spMM(
    handle: sys.cusparseHandle_t,
    op_a: sys.cusparseOperation_t,
    op_b: sys.cusparseOperation_t,
    alpha: *const anyopaque,
    mat_a: sys.cusparseSpMatDescr_t,
    mat_b: sys.cusparseDnMatDescr_t,
    beta: *const anyopaque,
    mat_c: sys.cusparseDnMatDescr_t,
    compute_type: sys.cudaDataType,
    alg: sys.cusparseSpMMAlg_t,
    buffer: ?*anyopaque,
) CusparseError!void {
    try toError(sys.cusparseSpMM(
        handle,
        op_a,
        op_b,
        alpha,
        mat_a,
        mat_b,
        beta,
        mat_c,
        compute_type,
        alg,
        buffer,
    ));
}

// ============================================================================
// SpGEMM (Sparse × Sparse)
// ============================================================================

pub fn spGEMM_createDescr() CusparseError!sys.cusparseSpGEMMDescr_t {
    var descr: sys.cusparseSpGEMMDescr_t = undefined;
    try toError(sys.cusparseSpGEMM_createDescr(&descr));
    return descr;
}

pub fn spGEMM_destroyDescr(descr: sys.cusparseSpGEMMDescr_t) CusparseError!void {
    try toError(sys.cusparseSpGEMM_destroyDescr(descr));
}

/// Estimate workspace for SpGEMM. Pass null buffer to query size, then allocate and call again.
pub fn spGEMM_workEstimation(
    handle: sys.cusparseHandle_t,
    op_a: sys.cusparseOperation_t,
    op_b: sys.cusparseOperation_t,
    alpha: *const anyopaque,
    mat_a: sys.cusparseSpMatDescr_t,
    mat_b: sys.cusparseSpMatDescr_t,
    beta: *const anyopaque,
    mat_c: sys.cusparseSpMatDescr_t,
    compute_type: sys.cudaDataType,
    alg: sys.cusparseSpGEMMAlg_t,
    spgemm_descr: sys.cusparseSpGEMMDescr_t,
    buffer_size: *usize,
    buffer: ?*anyopaque,
) CusparseError!void {
    try toError(sys.cusparseSpGEMM_workEstimation(
        handle,
        op_a,
        op_b,
        alpha,
        mat_a,
        mat_b,
        beta,
        mat_c,
        compute_type,
        alg,
        spgemm_descr,
        buffer_size,
        buffer,
    ));
}

/// Compute SpGEMM. Pass null buffer to query size, then allocate and call again.
pub fn spGEMM_compute(
    handle: sys.cusparseHandle_t,
    op_a: sys.cusparseOperation_t,
    op_b: sys.cusparseOperation_t,
    alpha: *const anyopaque,
    mat_a: sys.cusparseSpMatDescr_t,
    mat_b: sys.cusparseSpMatDescr_t,
    beta: *const anyopaque,
    mat_c: sys.cusparseSpMatDescr_t,
    compute_type: sys.cudaDataType,
    alg: sys.cusparseSpGEMMAlg_t,
    spgemm_descr: sys.cusparseSpGEMMDescr_t,
    buffer_size: *usize,
    buffer: ?*anyopaque,
) CusparseError!void {
    try toError(sys.cusparseSpGEMM_compute(
        handle,
        op_a,
        op_b,
        alpha,
        mat_a,
        mat_b,
        beta,
        mat_c,
        compute_type,
        alg,
        spgemm_descr,
        buffer_size,
        buffer,
    ));
}

/// Copy SpGEMM result into matC.
pub fn spGEMM_copy(
    handle: sys.cusparseHandle_t,
    op_a: sys.cusparseOperation_t,
    op_b: sys.cusparseOperation_t,
    alpha: *const anyopaque,
    mat_a: sys.cusparseSpMatDescr_t,
    mat_b: sys.cusparseSpMatDescr_t,
    beta: *const anyopaque,
    mat_c: sys.cusparseSpMatDescr_t,
    compute_type: sys.cudaDataType,
    alg: sys.cusparseSpGEMMAlg_t,
    spgemm_descr: sys.cusparseSpGEMMDescr_t,
) CusparseError!void {
    try toError(sys.cusparseSpGEMM_copy(
        handle,
        op_a,
        op_b,
        alpha,
        mat_a,
        mat_b,
        beta,
        mat_c,
        compute_type,
        alg,
        spgemm_descr,
    ));
}

// ============================================================================
// SpSV (Sparse Triangular Solve)
// ============================================================================

pub fn spSV_createDescr() CusparseError!sys.cusparseSpSVDescr_t {
    var descr: sys.cusparseSpSVDescr_t = undefined;
    try toError(sys.cusparseSpSV_createDescr(&descr));
    return descr;
}

pub fn spSV_destroyDescr(descr: sys.cusparseSpSVDescr_t) CusparseError!void {
    try toError(sys.cusparseSpSV_destroyDescr(descr));
}

pub fn spSV_bufferSize(
    handle: sys.cusparseHandle_t,
    op: sys.cusparseOperation_t,
    alpha: *const anyopaque,
    mat_a: sys.cusparseSpMatDescr_t,
    vec_x: sys.cusparseDnVecDescr_t,
    vec_y: sys.cusparseDnVecDescr_t,
    compute_type: sys.cudaDataType,
    alg: sys.cusparseSpSVAlg_t,
    spsv_descr: sys.cusparseSpSVDescr_t,
) CusparseError!usize {
    var buffer_size: usize = undefined;
    try toError(sys.cusparseSpSV_bufferSize(handle, op, alpha, mat_a, vec_x, vec_y, compute_type, alg, spsv_descr, &buffer_size));
    return buffer_size;
}

pub fn spSV_analysis(
    handle: sys.cusparseHandle_t,
    op: sys.cusparseOperation_t,
    alpha: *const anyopaque,
    mat_a: sys.cusparseSpMatDescr_t,
    vec_x: sys.cusparseDnVecDescr_t,
    vec_y: sys.cusparseDnVecDescr_t,
    compute_type: sys.cudaDataType,
    alg: sys.cusparseSpSVAlg_t,
    spsv_descr: sys.cusparseSpSVDescr_t,
    buffer: *anyopaque,
) CusparseError!void {
    try toError(sys.cusparseSpSV_analysis(handle, op, alpha, mat_a, vec_x, vec_y, compute_type, alg, spsv_descr, buffer));
}

pub fn spSV_solve(
    handle: sys.cusparseHandle_t,
    op: sys.cusparseOperation_t,
    alpha: *const anyopaque,
    mat_a: sys.cusparseSpMatDescr_t,
    vec_x: sys.cusparseDnVecDescr_t,
    vec_y: sys.cusparseDnVecDescr_t,
    compute_type: sys.cudaDataType,
    alg: sys.cusparseSpSVAlg_t,
    spsv_descr: sys.cusparseSpSVDescr_t,
) CusparseError!void {
    try toError(sys.cusparseSpSV_solve(handle, op, alpha, mat_a, vec_x, vec_y, compute_type, alg, spsv_descr));
}

// ============================================================================
// Sparse Matrix Attributes
// ============================================================================

pub fn spMatSetAttribute(
    mat: sys.cusparseSpMatDescr_t,
    attribute: c_uint,
    data: *const anyopaque,
    data_size: usize,
) CusparseError!void {
    try toError(sys.cusparseSpMatSetAttribute(mat, attribute, data, data_size));
}
