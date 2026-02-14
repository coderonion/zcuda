/// zCUDA: cuSOLVER - Raw FFI bindings.
const std = @import("std");
pub const c = @cImport({
    @cInclude("cusolverDn.h");
});

// Core types
pub const cusolverStatus_t = c.cusolverStatus_t;
pub const cusolverDnHandle_t = c.cusolverDnHandle_t;
pub const cublasOperation_t = c.cublasOperation_t;

// Status codes
pub const CUSOLVER_STATUS_SUCCESS = c.CUSOLVER_STATUS_SUCCESS;
pub const CUSOLVER_STATUS_NOT_INITIALIZED = c.CUSOLVER_STATUS_NOT_INITIALIZED;
pub const CUSOLVER_STATUS_ALLOC_FAILED = c.CUSOLVER_STATUS_ALLOC_FAILED;
pub const CUSOLVER_STATUS_INVALID_VALUE = c.CUSOLVER_STATUS_INVALID_VALUE;
pub const CUSOLVER_STATUS_ARCH_MISMATCH = c.CUSOLVER_STATUS_ARCH_MISMATCH;
pub const CUSOLVER_STATUS_INTERNAL_ERROR = c.CUSOLVER_STATUS_INTERNAL_ERROR;
pub const CUSOLVER_STATUS_NOT_SUPPORTED = c.CUSOLVER_STATUS_NOT_SUPPORTED;

// Operation constants
pub const CUBLAS_OP_N = c.CUBLAS_OP_N;
pub const CUBLAS_OP_T = c.CUBLAS_OP_T;

// Handle management
pub const cusolverDnCreate = c.cusolverDnCreate;
pub const cusolverDnDestroy = c.cusolverDnDestroy;
pub const cusolverDnSetStream = c.cusolverDnSetStream;

// LU factorization
pub const cusolverDnSgetrf_bufferSize = c.cusolverDnSgetrf_bufferSize;
pub const cusolverDnDgetrf_bufferSize = c.cusolverDnDgetrf_bufferSize;
pub const cusolverDnSgetrf = c.cusolverDnSgetrf;
pub const cusolverDnDgetrf = c.cusolverDnDgetrf;
pub const cusolverDnSgetrs = c.cusolverDnSgetrs;
pub const cusolverDnDgetrs = c.cusolverDnDgetrs;

// QR factorization
pub const cusolverDnSgeqrf_bufferSize = c.cusolverDnSgeqrf_bufferSize;
pub const cusolverDnDgeqrf_bufferSize = c.cusolverDnDgeqrf_bufferSize;
pub const cusolverDnSgeqrf = c.cusolverDnSgeqrf;
pub const cusolverDnDgeqrf = c.cusolverDnDgeqrf;
pub const cusolverDnSorgqr_bufferSize = c.cusolverDnSorgqr_bufferSize;
pub const cusolverDnSorgqr = c.cusolverDnSorgqr;

// SVD
pub const cusolverDnSgesvd_bufferSize = c.cusolverDnSgesvd_bufferSize;
pub const cusolverDnDgesvd_bufferSize = c.cusolverDnDgesvd_bufferSize;
pub const cusolverDnSgesvd = c.cusolverDnSgesvd;
pub const cusolverDnDgesvd = c.cusolverDnDgesvd;

// Cholesky factorization
pub const cusolverDnSpotrf_bufferSize = c.cusolverDnSpotrf_bufferSize;
pub const cusolverDnDpotrf_bufferSize = c.cusolverDnDpotrf_bufferSize;
pub const cusolverDnSpotrf = c.cusolverDnSpotrf;
pub const cusolverDnDpotrf = c.cusolverDnDpotrf;
pub const cusolverDnSpotrs = c.cusolverDnSpotrs;
pub const cusolverDnDpotrs = c.cusolverDnDpotrs;

// Fill mode
pub const cublasFillMode_t = c.cublasFillMode_t;
pub const CUBLAS_FILL_MODE_LOWER = c.CUBLAS_FILL_MODE_LOWER;
pub const CUBLAS_FILL_MODE_UPPER = c.CUBLAS_FILL_MODE_UPPER;

// Double QR Q-extraction
pub const cusolverDnDorgqr_bufferSize = c.cusolverDnDorgqr_bufferSize;
pub const cusolverDnDorgqr = c.cusolverDnDorgqr;

// Eigenvalue decomposition (syevd)
pub const cusolverEigMode_t = c.cusolverEigMode_t;
pub const CUSOLVER_EIG_MODE_NOVECTOR = c.CUSOLVER_EIG_MODE_NOVECTOR;
pub const CUSOLVER_EIG_MODE_VECTOR = c.CUSOLVER_EIG_MODE_VECTOR;
pub const cusolverDnSsyevd_bufferSize = c.cusolverDnSsyevd_bufferSize;
pub const cusolverDnDsyevd_bufferSize = c.cusolverDnDsyevd_bufferSize;
pub const cusolverDnSsyevd = c.cusolverDnSsyevd;
pub const cusolverDnDsyevd = c.cusolverDnDsyevd;

// Jacobi SVD (gesvdj) - batched-friendly SVD for small matrices
pub const gesvdjInfo_t = c.gesvdjInfo_t;
pub const cusolverDnCreateGesvdjInfo = c.cusolverDnCreateGesvdjInfo;
pub const cusolverDnDestroyGesvdjInfo = c.cusolverDnDestroyGesvdjInfo;
pub const cusolverDnXgesvdjSetTolerance = c.cusolverDnXgesvdjSetTolerance;
pub const cusolverDnXgesvdjSetMaxSweeps = c.cusolverDnXgesvdjSetMaxSweeps;
pub const cusolverDnSgesvdj_bufferSize = c.cusolverDnSgesvdj_bufferSize;
pub const cusolverDnDgesvdj_bufferSize = c.cusolverDnDgesvdj_bufferSize;
pub const cusolverDnSgesvdj = c.cusolverDnSgesvdj;
pub const cusolverDnDgesvdj = c.cusolverDnDgesvdj;

// Batched Jacobi SVD (gesvdjBatched)
pub const cusolverDnSgesvdjBatched_bufferSize = c.cusolverDnSgesvdjBatched_bufferSize;
pub const cusolverDnDgesvdjBatched_bufferSize = c.cusolverDnDgesvdjBatched_bufferSize;
pub const cusolverDnSgesvdjBatched = c.cusolverDnSgesvdjBatched;
pub const cusolverDnDgesvdjBatched = c.cusolverDnDgesvdjBatched;

// Approximate SVD (gesvda) - strided batched
pub const cusolverDnSgesvdaStridedBatched_bufferSize = c.cusolverDnSgesvdaStridedBatched_bufferSize;
pub const cusolverDnDgesvdaStridedBatched_bufferSize = c.cusolverDnDgesvdaStridedBatched_bufferSize;
pub const cusolverDnSgesvdaStridedBatched = c.cusolverDnSgesvdaStridedBatched;
pub const cusolverDnDgesvdaStridedBatched = c.cusolverDnDgesvdaStridedBatched;
