/// zCUDA: CUDA Runtime API - Raw FFI bindings.
///
/// Layer 1: Direct @cImport of cuda_runtime.h for Runtime API operations.
const std = @import("std");

pub const c = @cImport({
    @cInclude("cuda_runtime.h");
});

// Core types
pub const cudaError_t = c.cudaError_t;
pub const cudaStream_t = c.cudaStream_t;
pub const cudaEvent_t = c.cudaEvent_t;
pub const cudaDeviceProp = c.cudaDeviceProp;

// Error codes
pub const cudaSuccess = c.cudaSuccess;
pub const cudaErrorInvalidValue = c.cudaErrorInvalidValue;
pub const cudaErrorMemoryAllocation = c.cudaErrorMemoryAllocation;
pub const cudaErrorInitializationError = c.cudaErrorInitializationError;
pub const cudaErrorInvalidDevice = c.cudaErrorInvalidDevice;
pub const cudaErrorInvalidMemcpyDirection = c.cudaErrorInvalidMemcpyDirection;
pub const cudaErrorNotReady = c.cudaErrorNotReady;

// Memory copy direction
pub const cudaMemcpyHostToHost = c.cudaMemcpyHostToHost;
pub const cudaMemcpyHostToDevice = c.cudaMemcpyHostToDevice;
pub const cudaMemcpyDeviceToHost = c.cudaMemcpyDeviceToHost;
pub const cudaMemcpyDeviceToDevice = c.cudaMemcpyDeviceToDevice;

// Core functions
pub const cudaGetDeviceCount = c.cudaGetDeviceCount;
pub const cudaSetDevice = c.cudaSetDevice;
pub const cudaGetDevice = c.cudaGetDevice;
pub const cudaGetDeviceProperties = c.cudaGetDeviceProperties;
pub const cudaDeviceSynchronize = c.cudaDeviceSynchronize;
pub const cudaDeviceReset = c.cudaDeviceReset;

// Memory functions
pub const cudaMalloc = c.cudaMalloc;
pub const cudaFree = c.cudaFree;
pub const cudaMemcpy = c.cudaMemcpy;
pub const cudaMemcpyAsync = c.cudaMemcpyAsync;
pub const cudaMemset = c.cudaMemset;
pub const cudaMemsetAsync = c.cudaMemsetAsync;
pub const cudaMallocHost = c.cudaMallocHost;
pub const cudaFreeHost = c.cudaFreeHost;

// Stream functions
pub const cudaStreamCreate = c.cudaStreamCreate;
pub const cudaStreamDestroy = c.cudaStreamDestroy;
pub const cudaStreamSynchronize = c.cudaStreamSynchronize;
pub const cudaStreamWaitEvent = c.cudaStreamWaitEvent;
pub const cudaStreamQuery = c.cudaStreamQuery;

// Event functions
pub const cudaEventCreate = c.cudaEventCreate;
pub const cudaEventDestroy = c.cudaEventDestroy;
pub const cudaEventRecord = c.cudaEventRecord;
pub const cudaEventSynchronize = c.cudaEventSynchronize;
pub const cudaEventElapsedTime = c.cudaEventElapsedTime;
pub const cudaEventQuery = c.cudaEventQuery;

// Error functions
pub const cudaGetErrorString = c.cudaGetErrorString;
pub const cudaGetErrorName = c.cudaGetErrorName;
pub const cudaGetLastError = c.cudaGetLastError;
pub const cudaPeekAtLastError = c.cudaPeekAtLastError;

// 2D/3D Memory Operations
pub const cudaMemcpy2D = c.cudaMemcpy2D;
pub const cudaMemcpy2DAsync = c.cudaMemcpy2DAsync;
pub const cudaMallocPitch = c.cudaMallocPitch;

// Peer Access (Multi-GPU)
pub const cudaDeviceCanAccessPeer = c.cudaDeviceCanAccessPeer;
pub const cudaDeviceEnablePeerAccess = c.cudaDeviceEnablePeerAccess;
pub const cudaDeviceDisablePeerAccess = c.cudaDeviceDisablePeerAccess;

// Stream/Event with flags
pub const cudaStreamCreateWithFlags = c.cudaStreamCreateWithFlags;
pub const cudaEventCreateWithFlags = c.cudaEventCreateWithFlags;
