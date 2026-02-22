/// zCUDA: CUDA Driver API - Raw FFI bindings.
///
/// Layer 1: Direct @cImport of cuda.h, re-exporting commonly used types
/// and functions. No error handling or abstractions at this level.
const std = @import("std");

// Import the CUDA Driver API header
pub const c = @cImport({
    @cInclude("cuda.h");
});

// ============================================================================
// Core Types
// ============================================================================

pub const CUresult = c.CUresult;
pub const CUdevice = c.CUdevice;
pub const CUcontext = c.CUcontext;
pub const CUstream = c.CUstream;
pub const CUmodule = c.CUmodule;
pub const CUfunction = c.CUfunction;
pub const CUdeviceptr = c.CUdeviceptr;
pub const CUevent = c.CUevent;
pub const CUuuid = c.CUuuid;

// ============================================================================
// Device Attributes
// ============================================================================

pub const CUdevice_attribute = c.CUdevice_attribute;
pub const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = c.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK;
pub const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = c.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X;
pub const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = c.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y;
pub const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = c.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z;
pub const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = c.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X;
pub const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = c.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y;
pub const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = c.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = c.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = c.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR;
pub const CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = c.CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED;
pub const CU_DEVICE_ATTRIBUTE_TOTAL_MEMORY = c.CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY;
pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = c.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
pub const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = c.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK;
pub const CU_DEVICE_ATTRIBUTE_WARP_SIZE = c.CU_DEVICE_ATTRIBUTE_WARP_SIZE;
pub const CU_DEVICE_ATTRIBUTE_CLOCK_RATE = c.CU_DEVICE_ATTRIBUTE_CLOCK_RATE;
pub const CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = c.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE;
pub const CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = c.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH;
pub const CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = c.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE;
pub const CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = c.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK;
pub const CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = c.CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING;
pub const CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = c.CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY;
pub const CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = c.CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH;
pub const CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = c.CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS;
pub const CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = c.CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT;

// ============================================================================
// Context Limits
// ============================================================================

pub const CUlimit = c.CUlimit;
pub const CU_LIMIT_STACK_SIZE = c.CU_LIMIT_STACK_SIZE;
pub const CU_LIMIT_MALLOC_HEAP_SIZE = c.CU_LIMIT_MALLOC_HEAP_SIZE;
pub const CU_LIMIT_PRINTF_FIFO_SIZE = c.CU_LIMIT_PRINTF_FIFO_SIZE;

// ============================================================================
// Context Flags
// ============================================================================

pub const CUctx_flags = c.CUctx_flags;
pub const CU_CTX_SCHED_AUTO = c.CU_CTX_SCHED_AUTO;
pub const CU_CTX_SCHED_SPIN = c.CU_CTX_SCHED_SPIN;
pub const CU_CTX_SCHED_YIELD = c.CU_CTX_SCHED_YIELD;
pub const CU_CTX_SCHED_BLOCKING_SYNC = c.CU_CTX_SCHED_BLOCKING_SYNC;

// ============================================================================
// Stream Flags
// ============================================================================

pub const CU_STREAM_DEFAULT = c.CU_STREAM_DEFAULT;
pub const CU_STREAM_NON_BLOCKING = c.CU_STREAM_NON_BLOCKING;

// ============================================================================
// Event Flags
// ============================================================================

pub const CU_EVENT_DEFAULT = c.CU_EVENT_DEFAULT;
pub const CU_EVENT_BLOCKING_SYNC = c.CU_EVENT_BLOCKING_SYNC;
pub const CU_EVENT_DISABLE_TIMING = c.CU_EVENT_DISABLE_TIMING;

// ============================================================================
// Error Codes
// ============================================================================

pub const CUDA_SUCCESS = c.CUDA_SUCCESS;
pub const CUDA_ERROR_INVALID_VALUE = c.CUDA_ERROR_INVALID_VALUE;
pub const CUDA_ERROR_OUT_OF_MEMORY = c.CUDA_ERROR_OUT_OF_MEMORY;
pub const CUDA_ERROR_NOT_INITIALIZED = c.CUDA_ERROR_NOT_INITIALIZED;
pub const CUDA_ERROR_DEINITIALIZED = c.CUDA_ERROR_DEINITIALIZED;
pub const CUDA_ERROR_NO_DEVICE = c.CUDA_ERROR_NO_DEVICE;
pub const CUDA_ERROR_INVALID_DEVICE = c.CUDA_ERROR_INVALID_DEVICE;
pub const CUDA_ERROR_INVALID_CONTEXT = c.CUDA_ERROR_INVALID_CONTEXT;
pub const CUDA_ERROR_INVALID_HANDLE = c.CUDA_ERROR_INVALID_HANDLE;
pub const CUDA_ERROR_NOT_FOUND = c.CUDA_ERROR_NOT_FOUND;
pub const CUDA_ERROR_NOT_READY = c.CUDA_ERROR_NOT_READY;
pub const CUDA_ERROR_LAUNCH_FAILED = c.CUDA_ERROR_LAUNCH_FAILED;
pub const CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = c.CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
pub const CUDA_ERROR_INVALID_IMAGE = c.CUDA_ERROR_INVALID_IMAGE;
pub const CUDA_ERROR_INVALID_PTX = c.CUDA_ERROR_INVALID_PTX;
pub const CUDA_ERROR_UNSUPPORTED_PTX_VERSION = c.CUDA_ERROR_UNSUPPORTED_PTX_VERSION;

// ============================================================================
// Function Attributes
// ============================================================================

pub const CUfunction_attribute = c.CUfunction_attribute;
pub const CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = c.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK;
pub const CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = c.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES;
pub const CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = c.CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES;
pub const CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = c.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES;
pub const CU_FUNC_ATTRIBUTE_NUM_REGS = c.CU_FUNC_ATTRIBUTE_NUM_REGS;
pub const CU_FUNC_ATTRIBUTE_PTX_VERSION = c.CU_FUNC_ATTRIBUTE_PTX_VERSION;
pub const CU_FUNC_ATTRIBUTE_BINARY_VERSION = c.CU_FUNC_ATTRIBUTE_BINARY_VERSION;

// ============================================================================
// Core Functions: Initialization
// ============================================================================

pub const cuInit = c.cuInit;

// ============================================================================
// Core Functions: Device Management
// ============================================================================

pub const cuDeviceGet = c.cuDeviceGet;
pub const cuDeviceGetCount = c.cuDeviceGetCount;
pub const cuDeviceGetName = c.cuDeviceGetName;
pub const cuDeviceGetAttribute = c.cuDeviceGetAttribute;
pub const cuDeviceGetUuid = c.cuDeviceGetUuid;
pub const cuDeviceTotalMem_v2 = c.cuDeviceTotalMem_v2;

// ============================================================================
// Core Functions: Primary Context Management
// ============================================================================

pub const cuDevicePrimaryCtxRetain = c.cuDevicePrimaryCtxRetain;
pub const cuDevicePrimaryCtxRelease_v2 = c.cuDevicePrimaryCtxRelease_v2;
pub const cuDevicePrimaryCtxSetFlags_v2 = c.cuDevicePrimaryCtxSetFlags_v2;

// ============================================================================
// Core Functions: Context Management
// ============================================================================

pub const cuCtxGetCurrent = c.cuCtxGetCurrent;
pub const cuCtxSetCurrent = c.cuCtxSetCurrent;
pub const cuCtxSynchronize = c.cuCtxSynchronize;
pub const cuCtxGetDevice = c.cuCtxGetDevice;

// ============================================================================
// Core Functions: Stream Management
// ============================================================================

pub const cuStreamCreate = c.cuStreamCreate;
pub const cuStreamDestroy_v2 = c.cuStreamDestroy_v2;
pub const cuStreamSynchronize = c.cuStreamSynchronize;
pub const cuStreamQuery = c.cuStreamQuery;
pub const cuStreamWaitEvent = c.cuStreamWaitEvent;

// ============================================================================
// Core Functions: Memory Management
// ============================================================================

pub const cuMemAlloc_v2 = c.cuMemAlloc_v2;
pub const cuMemFree_v2 = c.cuMemFree_v2;
pub const cuMemcpyHtoD_v2 = c.cuMemcpyHtoD_v2;
pub const cuMemcpyDtoH_v2 = c.cuMemcpyDtoH_v2;
pub const cuMemcpyDtoD_v2 = c.cuMemcpyDtoD_v2;
pub const cuMemcpyHtoDAsync_v2 = c.cuMemcpyHtoDAsync_v2;
pub const cuMemcpyDtoHAsync_v2 = c.cuMemcpyDtoHAsync_v2;
pub const cuMemcpyDtoDAsync_v2 = c.cuMemcpyDtoDAsync_v2;
pub const cuMemsetD8_v2 = c.cuMemsetD8_v2;
pub const cuMemsetD32_v2 = c.cuMemsetD32_v2;
pub const cuMemsetD32Async = c.cuMemsetD32Async;
pub const cuMemsetD8Async = c.cuMemsetD8Async;

// ============================================================================
// Core Functions: Module Management
// ============================================================================

pub const cuModuleLoad = c.cuModuleLoad;
pub const cuModuleLoadData = c.cuModuleLoadData;
pub const cuModuleUnload = c.cuModuleUnload;
pub const cuModuleGetFunction = c.cuModuleGetFunction;

// ============================================================================
// Core Functions: Kernel Launch
// ============================================================================

pub const cuLaunchKernel = c.cuLaunchKernel;

// ============================================================================
// Core Functions: Function Attributes
// ============================================================================

pub const cuFuncGetAttribute = c.cuFuncGetAttribute;

// ============================================================================
// Core Functions: Event Management
// ============================================================================

pub const cuEventCreate = c.cuEventCreate;
pub const cuEventDestroy_v2 = c.cuEventDestroy_v2;
pub const cuEventRecord = c.cuEventRecord;
pub const cuEventSynchronize = c.cuEventSynchronize;
pub const cuEventQuery = c.cuEventQuery;
pub const cuEventElapsedTime = c.cuEventElapsedTime;

// ============================================================================
// Core Functions: Error Handling
// ============================================================================

pub const cuGetErrorString = c.cuGetErrorString;
pub const cuGetErrorName = c.cuGetErrorName;

// ============================================================================
// Core Functions: Memory Info
// ============================================================================

pub const cuMemGetInfo_v2 = c.cuMemGetInfo_v2;
pub const cuMemAllocHost_v2 = c.cuMemAllocHost_v2;
pub const cuMemFreeHost = c.cuMemFreeHost;

// ============================================================================
// Core Functions: Context Limits & Cache Config
// ============================================================================

pub const CUfunc_cache = c.CUfunc_cache;
pub const cuCtxGetLimit = c.cuCtxGetLimit;
pub const cuCtxSetLimit = c.cuCtxSetLimit;
pub const cuCtxGetCacheConfig = c.cuCtxGetCacheConfig;
pub const cuCtxSetCacheConfig = c.cuCtxSetCacheConfig;
pub const cuCtxSetFlags = c.cuCtxSetFlags;
pub const cuCtxGetFlags = c.cuCtxGetFlags;

// ============================================================================
// Core Functions: CUDA Graph API
// ============================================================================

pub const CUgraph = c.CUgraph;
pub const CUgraphExec = c.CUgraphExec;
pub const CUstreamCaptureMode = c.CUstreamCaptureMode;
pub const CUstreamCaptureStatus = c.CUstreamCaptureStatus;
pub const CU_STREAM_CAPTURE_MODE_GLOBAL = c.CU_STREAM_CAPTURE_MODE_GLOBAL;
pub const CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = c.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;
pub const CU_STREAM_CAPTURE_MODE_RELAXED = c.CU_STREAM_CAPTURE_MODE_RELAXED;
pub const CU_STREAM_CAPTURE_STATUS_NONE = c.CU_STREAM_CAPTURE_STATUS_NONE;
pub const CU_STREAM_CAPTURE_STATUS_ACTIVE = c.CU_STREAM_CAPTURE_STATUS_ACTIVE;

pub const cuStreamBeginCapture_v2 = c.cuStreamBeginCapture_v2;
pub const cuStreamEndCapture = c.cuStreamEndCapture;
pub const cuStreamIsCapturing = c.cuStreamIsCapturing;
pub const cuGraphInstantiate = c.cuGraphInstantiate;
pub const cuGraphDestroy = c.cuGraphDestroy;
pub const cuGraphExecDestroy = c.cuGraphExecDestroy;
pub const cuGraphLaunch = c.cuGraphLaunch;

// ============================================================================
// Core Functions: Unified Memory
// ============================================================================

pub const CU_MEM_ATTACH_GLOBAL = c.CU_MEM_ATTACH_GLOBAL;
pub const CU_MEM_ATTACH_HOST = c.CU_MEM_ATTACH_HOST;
pub const cuMemAllocManaged = c.cuMemAllocManaged;
pub const cuMemPrefetchAsync = c.cuMemPrefetchAsync;

// Peer Access (Multi-GPU)
pub const cuDeviceCanAccessPeer = c.cuDeviceCanAccessPeer;
pub const cuCtxEnablePeerAccess = c.cuCtxEnablePeerAccess;
pub const cuCtxDisablePeerAccess = c.cuCtxDisablePeerAccess;

// Occupancy Calculator
pub const cuOccupancyMaxActiveBlocksPerMultiprocessor = c.cuOccupancyMaxActiveBlocksPerMultiprocessor;
pub const cuOccupancyMaxPotentialBlockSize = c.cuOccupancyMaxPotentialBlockSize;

// Function Attributes (extended)
pub const cuFuncSetAttribute = c.cuFuncSetAttribute;

// Module Load (for JIT-compiled code)
pub const cuModuleLoadFatBinary = c.cuModuleLoadFatBinary;
pub const cuModuleGetGlobal_v2 = c.cuModuleGetGlobal_v2;
