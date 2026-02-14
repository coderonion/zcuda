/// zCUDA: cuBLAS LT API - Raw FFI bindings.
///
/// Layer 1: Direct @cImport of cublasLt.h for lightweight BLAS operations.
const std = @import("std");

pub const c = @cImport({
    @cInclude("cublasLt.h");
});

// Core types
pub const cublasLtHandle_t = c.cublasLtHandle_t;
pub const cublasLtMatmulDesc_t = c.cublasLtMatmulDesc_t;
pub const cublasLtMatrixLayout_t = c.cublasLtMatrixLayout_t;
pub const cublasLtMatmulPreference_t = c.cublasLtMatmulPreference_t;
pub const cublasLtMatmulAlgo_t = c.cublasLtMatmulAlgo_t;
pub const cublasLtMatmulHeuristicResult_t = c.cublasLtMatmulHeuristicResult_t;

pub const cublasStatus_t = c.cublasStatus_t;
pub const cublasOperation_t = c.cublasOperation_t;
pub const cudaDataType = c.cudaDataType;
pub const cublasComputeType_t = c.cublasComputeType_t;

// Status codes
pub const CUBLAS_STATUS_SUCCESS = c.CUBLAS_STATUS_SUCCESS;
pub const CUBLAS_STATUS_NOT_INITIALIZED = c.CUBLAS_STATUS_NOT_INITIALIZED;
pub const CUBLAS_STATUS_ALLOC_FAILED = c.CUBLAS_STATUS_ALLOC_FAILED;
pub const CUBLAS_STATUS_INVALID_VALUE = c.CUBLAS_STATUS_INVALID_VALUE;
pub const CUBLAS_STATUS_ARCH_MISMATCH = c.CUBLAS_STATUS_ARCH_MISMATCH;
pub const CUBLAS_STATUS_MAPPING_ERROR = c.CUBLAS_STATUS_MAPPING_ERROR;
pub const CUBLAS_STATUS_EXECUTION_FAILED = c.CUBLAS_STATUS_EXECUTION_FAILED;
pub const CUBLAS_STATUS_INTERNAL_ERROR = c.CUBLAS_STATUS_INTERNAL_ERROR;
pub const CUBLAS_STATUS_NOT_SUPPORTED = c.CUBLAS_STATUS_NOT_SUPPORTED;

// Operation types
pub const CUBLAS_OP_N = c.CUBLAS_OP_N;
pub const CUBLAS_OP_T = c.CUBLAS_OP_T;
pub const CUBLAS_OP_C = c.CUBLAS_OP_C;

// Data types
pub const CUDA_R_16F = c.CUDA_R_16F;
pub const CUDA_R_32F = c.CUDA_R_32F;
pub const CUDA_R_64F = c.CUDA_R_64F;
pub const CUDA_R_16BF = c.CUDA_R_16BF;

// Compute types
pub const CUBLAS_COMPUTE_16F = c.CUBLAS_COMPUTE_16F;
pub const CUBLAS_COMPUTE_32F = c.CUBLAS_COMPUTE_32F;
pub const CUBLAS_COMPUTE_64F = c.CUBLAS_COMPUTE_64F;
pub const CUBLAS_COMPUTE_32F_FAST_TF32 = c.CUBLAS_COMPUTE_32F_FAST_TF32;

// Core functions
pub const cublasLtCreate = c.cublasLtCreate;
pub const cublasLtDestroy = c.cublasLtDestroy;
pub const cublasLtMatmulDescCreate = c.cublasLtMatmulDescCreate;
pub const cublasLtMatmulDescDestroy = c.cublasLtMatmulDescDestroy;
pub const cublasLtMatmulDescSetAttribute = c.cublasLtMatmulDescSetAttribute;
pub const cublasLtMatrixLayoutCreate = c.cublasLtMatrixLayoutCreate;
pub const cublasLtMatrixLayoutDestroy = c.cublasLtMatrixLayoutDestroy;
pub const cublasLtMatrixLayoutSetAttribute = c.cublasLtMatrixLayoutSetAttribute;
pub const cublasLtMatmulPreferenceCreate = c.cublasLtMatmulPreferenceCreate;
pub const cublasLtMatmulPreferenceDestroy = c.cublasLtMatmulPreferenceDestroy;
pub const cublasLtMatmulPreferenceSetAttribute = c.cublasLtMatmulPreferenceSetAttribute;
pub const cublasLtMatmulAlgoGetHeuristic = c.cublasLtMatmulAlgoGetHeuristic;
pub const cublasLtMatmul = c.cublasLtMatmul;

// Matmul descriptor attributes
pub const cublasLtMatmulDescAttributes_t = c.cublasLtMatmulDescAttributes_t;
pub const CUBLASLT_MATMUL_DESC_TRANSA = c.CUBLASLT_MATMUL_DESC_TRANSA;
pub const CUBLASLT_MATMUL_DESC_TRANSB = c.CUBLASLT_MATMUL_DESC_TRANSB;
pub const CUBLASLT_MATMUL_DESC_EPILOGUE = c.CUBLASLT_MATMUL_DESC_EPILOGUE;
pub const CUBLASLT_MATMUL_DESC_BIAS_POINTER = c.CUBLASLT_MATMUL_DESC_BIAS_POINTER;

// Matmul preference attributes
pub const cublasLtMatmulPreferenceAttributes_t = c.cublasLtMatmulPreferenceAttributes_t;
pub const CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = c.CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES;

// Matrix layout attributes
pub const cublasLtMatrixLayoutAttribute_t = c.cublasLtMatrixLayoutAttribute_t;
pub const CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT = c.CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT;
pub const CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = c.CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET;

// Epilogue types
pub const cublasLtEpilogue_t = c.cublasLtEpilogue_t;
pub const CUBLASLT_EPILOGUE_DEFAULT = c.CUBLASLT_EPILOGUE_DEFAULT;
pub const CUBLASLT_EPILOGUE_RELU = c.CUBLASLT_EPILOGUE_RELU;
pub const CUBLASLT_EPILOGUE_BIAS = c.CUBLASLT_EPILOGUE_BIAS;
pub const CUBLASLT_EPILOGUE_RELU_BIAS = c.CUBLASLT_EPILOGUE_RELU_BIAS;
pub const CUBLASLT_EPILOGUE_GELU = c.CUBLASLT_EPILOGUE_GELU;
pub const CUBLASLT_EPILOGUE_GELU_BIAS = c.CUBLASLT_EPILOGUE_GELU_BIAS;
