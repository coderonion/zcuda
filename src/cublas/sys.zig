/// zCUDA: cuBLAS API - Raw FFI bindings.
///
/// Layer 1: Direct @cImport of cublas_v2.h for BLAS operations on GPU.
const std = @import("std");

pub const c = @cImport({
    @cInclude("cublas_v2.h");
});

// Core types
pub const cublasStatus_t = c.cublasStatus_t;
pub const cublasHandle_t = c.cublasHandle_t;
pub const cublasOperation_t = c.cublasOperation_t;
pub const cublasFillMode_t = c.cublasFillMode_t;
pub const cublasDiagType_t = c.cublasDiagType_t;
pub const cublasSideMode_t = c.cublasSideMode_t;
pub const cublasPointerMode_t = c.cublasPointerMode_t;

// Operation constants
pub const CUBLAS_OP_N = c.CUBLAS_OP_N;
pub const CUBLAS_OP_T = c.CUBLAS_OP_T;
pub const CUBLAS_OP_C = c.CUBLAS_OP_C;

// Fill mode
pub const CUBLAS_FILL_MODE_LOWER = c.CUBLAS_FILL_MODE_LOWER;
pub const CUBLAS_FILL_MODE_UPPER = c.CUBLAS_FILL_MODE_UPPER;

// Diag type
pub const CUBLAS_DIAG_NON_UNIT = c.CUBLAS_DIAG_NON_UNIT;
pub const CUBLAS_DIAG_UNIT = c.CUBLAS_DIAG_UNIT;

// Side mode
pub const CUBLAS_SIDE_LEFT = c.CUBLAS_SIDE_LEFT;
pub const CUBLAS_SIDE_RIGHT = c.CUBLAS_SIDE_RIGHT;

// Pointer mode
pub const CUBLAS_POINTER_MODE_HOST = c.CUBLAS_POINTER_MODE_HOST;
pub const CUBLAS_POINTER_MODE_DEVICE = c.CUBLAS_POINTER_MODE_DEVICE;

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
pub const CUBLAS_STATUS_LICENSE_ERROR = c.CUBLAS_STATUS_LICENSE_ERROR;

// Handle management
pub const cublasCreate_v2 = c.cublasCreate_v2;
pub const cublasDestroy_v2 = c.cublasDestroy_v2;
pub const cublasSetStream = c.cublasSetStream;
pub const cublasGetStream = c.cublasGetStream;
pub const cublasSetPointerMode = c.cublasSetPointerMode;

// BLAS Level 1
pub const cublasSasum = c.cublasSasum;
pub const cublasDasum = c.cublasDasum;
pub const cublasSaxpy = c.cublasSaxpy;
pub const cublasDaxpy = c.cublasDaxpy;
pub const cublasScopy = c.cublasScopy;
pub const cublasDcopy = c.cublasDcopy;
pub const cublasSdot = c.cublasSdot;
pub const cublasDdot = c.cublasDdot;
pub const cublasSnrm2 = c.cublasSnrm2;
pub const cublasDnrm2 = c.cublasDnrm2;
pub const cublasSscal = c.cublasSscal;
pub const cublasDscal = c.cublasDscal;

// BLAS Level 2
pub const cublasSgemv = c.cublasSgemv;
pub const cublasDgemv = c.cublasDgemv;

// BLAS Level 3
pub const cublasSgemm_v2 = c.cublasSgemm_v2;
pub const cublasDgemm_v2 = c.cublasDgemm_v2;
pub const cublasHgemm = c.cublasHgemm;
pub const cublasHgemmStridedBatched = c.cublasHgemmStridedBatched;
pub const cublasSgemmStridedBatched = c.cublasSgemmStridedBatched;
pub const cublasDgemmStridedBatched = c.cublasDgemmStridedBatched;

// Data and compute types for extended API
pub const cudaDataType_t = c.cudaDataType_t;
pub const cublasComputeType_t = c.cublasComputeType_t;
pub const CUDA_R_16F = c.CUDA_R_16F;
pub const CUDA_R_32F = c.CUDA_R_32F;
pub const CUDA_R_64F = c.CUDA_R_64F;
pub const CUDA_R_16BF = c.CUDA_R_16BF;
pub const CUBLAS_COMPUTE_16F = c.CUBLAS_COMPUTE_16F;
pub const CUBLAS_COMPUTE_32F = c.CUBLAS_COMPUTE_32F;
pub const CUBLAS_COMPUTE_64F = c.CUBLAS_COMPUTE_64F;
pub const cublasGemmAlgo_t = c.cublasGemmAlgo_t;
pub const CUBLAS_GEMM_DEFAULT = c.CUBLAS_GEMM_DEFAULT;

// Extended GEMM (mixed precision)
pub const cublasGemmEx = c.cublasGemmEx;
pub const cublasGemmStridedBatchedEx = c.cublasGemmStridedBatchedEx;

// BLAS Level 3 — batched (pointer arrays)
pub const cublasSgemmBatched = c.cublasSgemmBatched;
pub const cublasDgemmBatched = c.cublasDgemmBatched;

// BLAS Level 3 — Triangular solve (TRSM)
pub const cublasStrsm_v2 = c.cublasStrsm_v2;
pub const cublasDtrsm_v2 = c.cublasDtrsm_v2;

// BLAS Level 3 — Symmetric multiply (SYMM)
pub const cublasSsymm_v2 = c.cublasSsymm_v2;
pub const cublasDsymm_v2 = c.cublasDsymm_v2;

// Extended — Grouped Batched GEMM
pub const cublasGemmGroupedBatchedEx = c.cublasGemmGroupedBatchedEx;

// BLAS Level 1 — Swap
pub const cublasSswap = c.cublasSswap;
pub const cublasDswap = c.cublasDswap;

// BLAS Level 1 — Index of max/min absolute value
pub const cublasIsamax = c.cublasIsamax;
pub const cublasIdamax = c.cublasIdamax;
pub const cublasIsamin = c.cublasIsamin;
pub const cublasIdamin = c.cublasIdamin;

// BLAS Level 3 — Symmetric rank-k update (SYRK)
pub const cublasSsyrk_v2 = c.cublasSsyrk_v2;
pub const cublasDsyrk_v2 = c.cublasDsyrk_v2;

// BLAS Level 3 — Triangular matrix multiply (TRMM)
pub const cublasStrmm_v2 = c.cublasStrmm_v2;
pub const cublasDtrmm_v2 = c.cublasDtrmm_v2;

// BLAS Extensions — GEAM (Matrix Add/Transpose)
pub const cublasSgeam = c.cublasSgeam;
pub const cublasDgeam = c.cublasDgeam;

// BLAS Extensions — DGMM (Diagonal Matrix Multiply)
pub const cublasSdgmm = c.cublasSdgmm;
pub const cublasDdgmm = c.cublasDdgmm;

// BLAS Level 1 — Givens rotation (ROT)
pub const cublasSrot_v2 = c.cublasSrot_v2;
pub const cublasDrot_v2 = c.cublasDrot_v2;

// BLAS Level 2 — Symmetric matrix-vector multiply (SYMV)
pub const cublasSsymv_v2 = c.cublasSsymv_v2;
pub const cublasDsymv_v2 = c.cublasDsymv_v2;

// BLAS Level 2 — Symmetric rank-1 update (SYR)
pub const cublasSsyr_v2 = c.cublasSsyr_v2;
pub const cublasDsyr_v2 = c.cublasDsyr_v2;

// BLAS Level 2 — Triangular matrix-vector multiply (TRMV)
pub const cublasStrmv_v2 = c.cublasStrmv_v2;
pub const cublasDtrmv_v2 = c.cublasDtrmv_v2;

// BLAS Level 2 — Triangular solve (TRSV)
pub const cublasStrsv_v2 = c.cublasStrsv_v2;
pub const cublasDtrsv_v2 = c.cublasDtrsv_v2;

// BLAS Level 3 — SYR2K (Symmetric rank-2k update)
pub const cublasSsyr2k_v2 = c.cublasSsyr2k_v2;
pub const cublasDsyr2k_v2 = c.cublasDsyr2k_v2;
