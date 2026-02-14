/// zCUDA: cuFFT - Raw FFI bindings.
const std = @import("std");
pub const c = @cImport({
    @cInclude("cufft.h");
});

// Core types
pub const cufftResult = c.cufftResult;
pub const cufftHandle = c.cufftHandle;
pub const cufftType = c.cufftType;

// Status codes
pub const CUFFT_SUCCESS = c.CUFFT_SUCCESS;
pub const CUFFT_INVALID_PLAN = c.CUFFT_INVALID_PLAN;
pub const CUFFT_ALLOC_FAILED = c.CUFFT_ALLOC_FAILED;
pub const CUFFT_INVALID_TYPE = c.CUFFT_INVALID_TYPE;
pub const CUFFT_INVALID_VALUE = c.CUFFT_INVALID_VALUE;
pub const CUFFT_INTERNAL_ERROR = c.CUFFT_INTERNAL_ERROR;
pub const CUFFT_EXEC_FAILED = c.CUFFT_EXEC_FAILED;
pub const CUFFT_SETUP_FAILED = c.CUFFT_SETUP_FAILED;
pub const CUFFT_INVALID_SIZE = c.CUFFT_INVALID_SIZE;
pub const CUFFT_INCOMPLETE_PARAMETER_LIST = c.CUFFT_INCOMPLETE_PARAMETER_LIST;

// FFT types
pub const CUFFT_C2C = c.CUFFT_C2C;
pub const CUFFT_R2C = c.CUFFT_R2C;
pub const CUFFT_C2R = c.CUFFT_C2R;
pub const CUFFT_Z2Z = c.CUFFT_Z2Z;
pub const CUFFT_D2Z = c.CUFFT_D2Z;
pub const CUFFT_Z2D = c.CUFFT_Z2D;

// Direction
pub const CUFFT_FORWARD = c.CUFFT_FORWARD;
pub const CUFFT_INVERSE = c.CUFFT_INVERSE;

// Plan functions
pub const cufftPlan1d = c.cufftPlan1d;
pub const cufftPlan2d = c.cufftPlan2d;
pub const cufftPlan3d = c.cufftPlan3d;
pub const cufftDestroy = c.cufftDestroy;
pub const cufftSetStream = c.cufftSetStream;
pub const cufftPlanMany = c.cufftPlanMany;
pub const cufftGetSize = c.cufftGetSize;
pub const cufftSetWorkArea = c.cufftSetWorkArea;
pub const cufftMakePlanMany = c.cufftMakePlanMany;
pub const cufftEstimateMany = c.cufftEstimateMany;
pub const cufftSetAutoAllocation = c.cufftSetAutoAllocation;

// Execution
pub const cufftExecC2C = c.cufftExecC2C;
pub const cufftExecZ2Z = c.cufftExecZ2Z;
pub const cufftExecR2C = c.cufftExecR2C;
pub const cufftExecC2R = c.cufftExecC2R;
pub const cufftExecD2Z = c.cufftExecD2Z;
pub const cufftExecZ2D = c.cufftExecZ2D;
