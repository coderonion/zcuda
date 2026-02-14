/// zCUDA: NVRTC API - Raw FFI bindings.
///
/// Layer 1: Direct @cImport of nvrtc.h for runtime compilation of CUDA C++ to PTX.
const std = @import("std");

pub const c = @cImport({
    @cInclude("nvrtc.h");
});

// Core types
pub const nvrtcResult = c.nvrtcResult;
pub const nvrtcProgram = c.nvrtcProgram;

// Error codes
pub const NVRTC_SUCCESS = c.NVRTC_SUCCESS;
pub const NVRTC_ERROR_OUT_OF_MEMORY = c.NVRTC_ERROR_OUT_OF_MEMORY;
pub const NVRTC_ERROR_PROGRAM_CREATION_FAILURE = c.NVRTC_ERROR_PROGRAM_CREATION_FAILURE;
pub const NVRTC_ERROR_INVALID_INPUT = c.NVRTC_ERROR_INVALID_INPUT;
pub const NVRTC_ERROR_INVALID_PROGRAM = c.NVRTC_ERROR_INVALID_PROGRAM;
pub const NVRTC_ERROR_INVALID_OPTION = c.NVRTC_ERROR_INVALID_OPTION;
pub const NVRTC_ERROR_COMPILATION = c.NVRTC_ERROR_COMPILATION;
pub const NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = c.NVRTC_ERROR_BUILTIN_OPERATION_FAILURE;

// Core functions
pub const nvrtcGetErrorString = c.nvrtcGetErrorString;
pub const nvrtcVersion = c.nvrtcVersion;
pub const nvrtcCreateProgram = c.nvrtcCreateProgram;
pub const nvrtcDestroyProgram = c.nvrtcDestroyProgram;
pub const nvrtcCompileProgram = c.nvrtcCompileProgram;
pub const nvrtcGetPTXSize = c.nvrtcGetPTXSize;
pub const nvrtcGetPTX = c.nvrtcGetPTX;
pub const nvrtcGetProgramLogSize = c.nvrtcGetProgramLogSize;
pub const nvrtcGetProgramLog = c.nvrtcGetProgramLog;

// CUBIN output
pub const nvrtcGetCUBINSize = c.nvrtcGetCUBINSize;
pub const nvrtcGetCUBIN = c.nvrtcGetCUBIN;

// Named expressions (for template mangling)
pub const nvrtcAddNameExpression = c.nvrtcAddNameExpression;
pub const nvrtcGetLoweredName = c.nvrtcGetLoweredName;

// Supported architectures
pub const nvrtcGetNumSupportedArchs = c.nvrtcGetNumSupportedArchs;
pub const nvrtcGetSupportedArchs = c.nvrtcGetSupportedArchs;
