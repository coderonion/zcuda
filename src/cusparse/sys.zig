/// zCUDA: cuSPARSE - Raw FFI bindings.
const std = @import("std");
pub const c = @cImport({
    @cInclude("cusparse.h");
});

// Core types
pub const cusparseStatus_t = c.cusparseStatus_t;
pub const cusparseHandle_t = c.cusparseHandle_t;
pub const cusparseSpMatDescr_t = c.cusparseSpMatDescr_t;
pub const cusparseDnVecDescr_t = c.cusparseDnVecDescr_t;
pub const cusparseDnMatDescr_t = c.cusparseDnMatDescr_t;
pub const cusparseIndexType_t = c.cusparseIndexType_t;
pub const cusparseIndexBase_t = c.cusparseIndexBase_t;
pub const cusparseOperation_t = c.cusparseOperation_t;
pub const cusparseOrder_t = c.cusparseOrder_t;
pub const cusparseSpMVAlg_t = c.cusparseSpMVAlg_t;
pub const cusparseSpMMAlg_t = c.cusparseSpMMAlg_t;
pub const cudaDataType = c.cudaDataType;

// Status codes
pub const CUSPARSE_STATUS_SUCCESS = c.CUSPARSE_STATUS_SUCCESS;
pub const CUSPARSE_STATUS_NOT_INITIALIZED = c.CUSPARSE_STATUS_NOT_INITIALIZED;
pub const CUSPARSE_STATUS_ALLOC_FAILED = c.CUSPARSE_STATUS_ALLOC_FAILED;
pub const CUSPARSE_STATUS_INVALID_VALUE = c.CUSPARSE_STATUS_INVALID_VALUE;
pub const CUSPARSE_STATUS_ARCH_MISMATCH = c.CUSPARSE_STATUS_ARCH_MISMATCH;
pub const CUSPARSE_STATUS_INTERNAL_ERROR = c.CUSPARSE_STATUS_INTERNAL_ERROR;
pub const CUSPARSE_STATUS_NOT_SUPPORTED = c.CUSPARSE_STATUS_NOT_SUPPORTED;

// Index types
pub const CUSPARSE_INDEX_32I = c.CUSPARSE_INDEX_32I;
pub const CUSPARSE_INDEX_64I = c.CUSPARSE_INDEX_64I;
pub const CUSPARSE_INDEX_BASE_ZERO = c.CUSPARSE_INDEX_BASE_ZERO;
pub const CUSPARSE_INDEX_BASE_ONE = c.CUSPARSE_INDEX_BASE_ONE;

// Data types
pub const CUDA_R_32F = c.CUDA_R_32F;
pub const CUDA_R_64F = c.CUDA_R_64F;

// Operation
pub const CUSPARSE_OPERATION_NON_TRANSPOSE = c.CUSPARSE_OPERATION_NON_TRANSPOSE;
pub const CUSPARSE_OPERATION_TRANSPOSE = c.CUSPARSE_OPERATION_TRANSPOSE;
pub const CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = c.CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;

// Order
pub const CUSPARSE_ORDER_COL = c.CUSPARSE_ORDER_COL;
pub const CUSPARSE_ORDER_ROW = c.CUSPARSE_ORDER_ROW;

// Algorithm defaults
pub const CUSPARSE_SPMV_ALG_DEFAULT = c.CUSPARSE_SPMV_ALG_DEFAULT;
pub const CUSPARSE_SPMM_ALG_DEFAULT = c.CUSPARSE_SPMM_ALG_DEFAULT;

// Handle management
pub const cusparseCreate = c.cusparseCreate;
pub const cusparseDestroy = c.cusparseDestroy;
pub const cusparseSetStream = c.cusparseSetStream;

// Sparse matrix descriptors
pub const cusparseCreateCsr = c.cusparseCreateCsr;
pub const cusparseCreateCoo = c.cusparseCreateCoo;
pub const cusparseCreateCsc = c.cusparseCreateCsc;
pub const cusparseDestroySpMat = c.cusparseDestroySpMat;

// Dense vector descriptor
pub const cusparseCreateDnVec = c.cusparseCreateDnVec;
pub const cusparseDestroyDnVec = c.cusparseDestroyDnVec;

// Dense matrix descriptor
pub const cusparseCreateDnMat = c.cusparseCreateDnMat;
pub const cusparseDestroyDnMat = c.cusparseDestroyDnMat;

// SpMV
pub const cusparseSpMV_bufferSize = c.cusparseSpMV_bufferSize;
pub const cusparseSpMV = c.cusparseSpMV;

// SpMM
pub const cusparseSpMM_bufferSize = c.cusparseSpMM_bufferSize;
pub const cusparseSpMM = c.cusparseSpMM;

// SpGEMM types
pub const cusparseSpGEMMDescr_t = c.cusparseSpGEMMDescr_t;
pub const cusparseSpGEMMAlg_t = c.cusparseSpGEMMAlg_t;
pub const CUSPARSE_SPGEMM_DEFAULT = c.CUSPARSE_SPGEMM_DEFAULT;
pub const CUSPARSE_SPGEMM_CSR_ALG_DETERMINITIC = c.CUSPARSE_SPGEMM_CSR_ALG_DETERMINITIC;
pub const CUSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC = c.CUSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC;

// SpGEMM functions
pub const cusparseSpGEMM_createDescr = c.cusparseSpGEMM_createDescr;
pub const cusparseSpGEMM_destroyDescr = c.cusparseSpGEMM_destroyDescr;
pub const cusparseSpGEMM_workEstimation = c.cusparseSpGEMM_workEstimation;
pub const cusparseSpGEMM_compute = c.cusparseSpGEMM_compute;
pub const cusparseSpGEMM_copy = c.cusparseSpGEMM_copy;

// SpSV (sparse triangular solve) types
pub const cusparseSpSVDescr_t = c.cusparseSpSVDescr_t;
pub const cusparseSpSVAlg_t = c.cusparseSpSVAlg_t;
pub const CUSPARSE_SPSV_ALG_DEFAULT = c.CUSPARSE_SPSV_ALG_DEFAULT;

// SpSV functions
pub const cusparseSpSV_createDescr = c.cusparseSpSV_createDescr;
pub const cusparseSpSV_destroyDescr = c.cusparseSpSV_destroyDescr;
pub const cusparseSpSV_bufferSize = c.cusparseSpSV_bufferSize;
pub const cusparseSpSV_analysis = c.cusparseSpSV_analysis;
pub const cusparseSpSV_solve = c.cusparseSpSV_solve;

// BSR (Block Sparse Row) format
pub const cusparseCreateBsr = c.cusparseCreateBsr;

// Sparse matrix attributes
pub const cusparseSpMatGetSize = c.cusparseSpMatGetSize;
pub const cusparseCsrGet = c.cusparseCsrGet;

// Fill mode for triangular solves
pub const cusparseFillMode_t = c.cusparseFillMode_t;
pub const CUSPARSE_FILL_MODE_LOWER = c.CUSPARSE_FILL_MODE_LOWER;
pub const CUSPARSE_FILL_MODE_UPPER = c.CUSPARSE_FILL_MODE_UPPER;
pub const cusparseDiagType_t = c.cusparseDiagType_t;
pub const CUSPARSE_DIAG_TYPE_NON_UNIT = c.CUSPARSE_DIAG_TYPE_NON_UNIT;
pub const CUSPARSE_DIAG_TYPE_UNIT = c.CUSPARSE_DIAG_TYPE_UNIT;
pub const cusparseSpMatSetAttribute = c.cusparseSpMatSetAttribute;
pub const CUSPARSE_SPMAT_FILL_MODE = c.CUSPARSE_SPMAT_FILL_MODE;
pub const CUSPARSE_SPMAT_DIAG_TYPE = c.CUSPARSE_SPMAT_DIAG_TYPE;
