/// zCUDA: CUDA Driver API - Error wrapping layer.
///
/// Layer 2: Thin wrapper around sys.zig that converts C-style error codes
/// to Zig error unions. Functions here are still low-level but provide
/// proper error handling through Zig's error system.
const std = @import("std");
const sys = @import("sys.zig");
const types = @import("../types.zig");

// ============================================================================
// Error Type
// ============================================================================

/// Represents a CUDA driver API error.
pub const DriverError = error{
    InvalidValue,
    OutOfMemory,
    NotInitialized,
    Deinitialized,
    ProfilerDisabled,
    ProfilerNotInitialized,
    ProfilerAlreadyStarted,
    ProfilerAlreadyStopped,
    NoDevice,
    InvalidDevice,
    InvalidImage,
    InvalidContext,
    ContextAlreadyCurrent,
    MapFailed,
    UnmapFailed,
    ArrayIsMapped,
    AlreadyMapped,
    NoBinaryForGpu,
    AlreadyAcquired,
    NotMapped,
    NotMappedAsArray,
    NotMappedAsPointer,
    EccUncorrectable,
    UnsupportedLimit,
    ContextAlreadyInUse,
    PeerAccessUnsupported,
    InvalidPtx,
    InvalidGraphicsContext,
    NvlinkUncorrectable,
    JitCompilerNotFound,
    InvalidSource,
    FileNotFound,
    SharedObjectSymbolNotFound,
    SharedObjectInitFailed,
    OperatingSystem,
    InvalidHandle,
    IllegalState,
    NotFound,
    NotReady,
    IllegalAddress,
    LaunchOutOfResources,
    LaunchTimeout,
    LaunchIncompatibleTexturing,
    PeerAccessAlreadyEnabled,
    PeerAccessNotEnabled,
    PrimaryContextActive,
    ContextIsDestroyed,
    Assert,
    TooManyPeers,
    HostMemoryAlreadyRegistered,
    HostMemoryNotRegistered,
    HardwareStackError,
    IllegalInstruction,
    MisalignedAddress,
    InvalidAddressSpace,
    InvalidPc,
    LaunchFailed,
    CooperativeLaunchTooLarge,
    NotPermitted,
    NotSupported,
    SystemNotReady,
    SystemDriverMismatch,
    CompatNotSupportedOnDevice,
    StreamCaptureUnsupported,
    StreamCaptureInvalidated,
    StreamCaptureMerge,
    StreamCaptureUnmatched,
    StreamCaptureUnjoined,
    StreamCaptureIsolation,
    StreamCaptureImplicit,
    CapturedEvent,
    StreamCaptureWrongThread,
    Timeout,
    GraphExecUpdateFailure,
    UnsupportedPtxVersion,
    Unknown,
};

/// Convert a CUDA result code to a Zig error.
pub fn toError(result: sys.CUresult) DriverError!void {
    if (result == sys.CUDA_SUCCESS) return;

    return switch (result) {
        sys.CUDA_ERROR_INVALID_VALUE => DriverError.InvalidValue,
        sys.CUDA_ERROR_OUT_OF_MEMORY => DriverError.OutOfMemory,
        sys.CUDA_ERROR_NOT_INITIALIZED => DriverError.NotInitialized,
        sys.CUDA_ERROR_DEINITIALIZED => DriverError.Deinitialized,
        sys.CUDA_ERROR_NO_DEVICE => DriverError.NoDevice,
        sys.CUDA_ERROR_INVALID_DEVICE => DriverError.InvalidDevice,
        sys.CUDA_ERROR_INVALID_IMAGE => DriverError.InvalidImage,
        sys.CUDA_ERROR_INVALID_CONTEXT => DriverError.InvalidContext,
        sys.CUDA_ERROR_INVALID_HANDLE => DriverError.InvalidHandle,
        sys.CUDA_ERROR_NOT_FOUND => DriverError.NotFound,
        sys.CUDA_ERROR_NOT_READY => DriverError.NotReady,
        sys.CUDA_ERROR_LAUNCH_FAILED => DriverError.LaunchFailed,
        sys.CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES => DriverError.LaunchOutOfResources,
        sys.CUDA_ERROR_INVALID_PTX => DriverError.InvalidPtx,
        sys.CUDA_ERROR_UNSUPPORTED_PTX_VERSION => DriverError.UnsupportedPtxVersion,
        else => DriverError.Unknown,
    };
}

/// Get the error name string for a CUDA result code.
pub fn errorName(result: sys.CUresult) []const u8 {
    var str: [*c]const u8 = undefined;
    _ = sys.cuGetErrorName(result, &str);
    return std.mem.span(str);
}

/// Get the error description string for a CUDA result code.
pub fn errorString(result: sys.CUresult) []const u8 {
    var str: [*c]const u8 = undefined;
    _ = sys.cuGetErrorString(result, &str);
    return std.mem.span(str);
}

// ============================================================================
// Initialization
// ============================================================================

/// Initialize the CUDA driver API.
/// **Must be called before any other CUDA driver API function.**
///
/// See [cuInit()](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__INITIALIZE.html)
pub fn init() DriverError!void {
    try toError(sys.cuInit(0));
}

// ============================================================================
// Device Functions
// ============================================================================

pub const device = struct {
    /// Get the number of available CUDA devices.
    pub fn getCount() DriverError!i32 {
        var count: i32 = undefined;
        try toError(sys.cuDeviceGetCount(&count));
        return count;
    }

    /// Get a device handle for the given ordinal.
    pub fn get(ordinal: i32) DriverError!sys.CUdevice {
        var dev: sys.CUdevice = undefined;
        try toError(sys.cuDeviceGet(&dev, ordinal));
        return dev;
    }

    /// Get the name of a device.
    pub fn getName(dev: sys.CUdevice) DriverError![256]u8 {
        var name: [256]u8 = @splat(0);
        try toError(sys.cuDeviceGetName(&name, 256, dev));
        return name;
    }

    /// Get a device attribute.
    pub fn getAttribute(dev: sys.CUdevice, attrib: sys.CUdevice_attribute) DriverError!i32 {
        var value: i32 = undefined;
        try toError(sys.cuDeviceGetAttribute(&value, attrib, dev));
        return value;
    }

    /// Get the UUID of a device.
    pub fn getUuid(dev: sys.CUdevice) DriverError!sys.CUuuid {
        var uuid: sys.CUuuid = undefined;
        try toError(sys.cuDeviceGetUuid(&uuid, dev));
        return uuid;
    }

    /// Get the total memory in bytes on the device.
    pub fn getTotalMem(dev: sys.CUdevice) DriverError!usize {
        var bytes: usize = undefined;
        try toError(sys.cuDeviceTotalMem_v2(&bytes, dev));
        return bytes;
    }
};

// ============================================================================
// Primary Context Functions
// ============================================================================

pub const primaryCtx = struct {
    /// Retain the primary context for a device.
    pub fn retain(dev: sys.CUdevice) DriverError!sys.CUcontext {
        var context: sys.CUcontext = undefined;
        try toError(sys.cuDevicePrimaryCtxRetain(&context, dev));
        return context;
    }

    /// Release the primary context for a device.
    pub fn release(dev: sys.CUdevice) DriverError!void {
        try toError(sys.cuDevicePrimaryCtxRelease_v2(dev));
    }

    /// Set flags for the primary context.
    pub fn setFlags(dev: sys.CUdevice, flags: u32) DriverError!void {
        try toError(sys.cuDevicePrimaryCtxSetFlags_v2(dev, flags));
    }
};

// ============================================================================
// Context Functions
// ============================================================================

pub const ctx = struct {
    /// Get the current context.
    pub fn getCurrent() DriverError!sys.CUcontext {
        var context: sys.CUcontext = undefined;
        try toError(sys.cuCtxGetCurrent(&context));
        return context;
    }

    /// Set the current context for the calling thread.
    pub fn setCurrent(context: sys.CUcontext) DriverError!void {
        try toError(sys.cuCtxSetCurrent(context));
    }

    /// Synchronize the current context (block until all work completes).
    pub fn synchronize() DriverError!void {
        try toError(sys.cuCtxSynchronize());
    }

    /// Get the device associated with the current context.
    pub fn getDevice() DriverError!sys.CUdevice {
        var d: sys.CUdevice = undefined;
        try toError(sys.cuCtxGetDevice(&d));
        return d;
    }

    /// Get a context limit value.
    pub fn getLimit(limit: sys.CUlimit) DriverError!usize {
        var value: usize = undefined;
        try toError(sys.cuCtxGetLimit(&value, limit));
        return value;
    }

    /// Set a context limit value.
    pub fn setLimit(limit: sys.CUlimit, value: usize) DriverError!void {
        try toError(sys.cuCtxSetLimit(limit, value));
    }

    /// Set primary context flags.
    pub fn setFlags(flags: u32) DriverError!void {
        try toError(sys.cuCtxSetFlags(flags));
    }

    /// Get primary context flags.
    pub fn getFlags() DriverError!u32 {
        var flags: u32 = undefined;
        try toError(sys.cuCtxGetFlags(&flags));
        return flags;
    }

    /// Get the L1/shared memory cache configuration.
    pub fn getCacheConfig() DriverError!sys.CUfunc_cache {
        var config: sys.CUfunc_cache = undefined;
        try toError(sys.cuCtxGetCacheConfig(&config));
        return config;
    }

    /// Set the L1/shared memory cache configuration.
    pub fn setCacheConfig(config: sys.CUfunc_cache) DriverError!void {
        try toError(sys.cuCtxSetCacheConfig(config));
    }
};

// ============================================================================
// Stream Functions
// ============================================================================

pub const stream = struct {
    /// Create a new stream with the specified flags.
    pub fn create(flags: u32) DriverError!sys.CUstream {
        var s: sys.CUstream = undefined;
        try toError(sys.cuStreamCreate(&s, flags));
        return s;
    }

    /// Destroy a stream.
    pub fn destroy(s: sys.CUstream) DriverError!void {
        try toError(sys.cuStreamDestroy_v2(s));
    }

    /// Synchronize a stream (wait for all operations to complete).
    pub fn synchronize(s: sys.CUstream) DriverError!void {
        try toError(sys.cuStreamSynchronize(s));
    }

    /// Make a stream wait on an event.
    pub fn waitEvent(s: sys.CUstream, e: sys.CUevent) DriverError!void {
        try toError(sys.cuStreamWaitEvent(s, e, 0));
    }

    /// Non-blocking check whether all operations in the stream have completed.
    /// Returns true if complete, false if still pending.
    pub fn query(s: sys.CUstream) DriverError!bool {
        const status = sys.cuStreamQuery(s);
        if (status == sys.CU_SUCCESS) return true;
        if (status == sys.CUDA_ERROR_NOT_READY) return false;
        try toError(status);
        unreachable;
    }
};

// ============================================================================
// Memory Functions
// ============================================================================

pub const mem = struct {
    pub fn alloc(size: usize) DriverError!sys.CUdeviceptr {
        var ptr: sys.CUdeviceptr = undefined;
        try toError(sys.cuMemAlloc_v2(&ptr, size));
        return ptr;
    }

    pub fn free(ptr: sys.CUdeviceptr) DriverError!void {
        try toError(sys.cuMemFree_v2(ptr));
    }

    /// Get free and total device memory.
    pub fn getInfo() DriverError!struct { free: usize, total: usize } {
        var free_bytes: usize = undefined;
        var total_bytes: usize = undefined;
        try toError(sys.cuMemGetInfo_v2(&free_bytes, &total_bytes));
        return .{ .free = free_bytes, .total = total_bytes };
    }

    /// Allocate page-locked (pinned) host memory.
    pub fn allocHost(size: usize) DriverError!*anyopaque {
        var ptr: ?*anyopaque = null;
        try toError(sys.cuMemAllocHost_v2(@ptrCast(&ptr), size));
        return ptr.?;
    }

    /// Free page-locked host memory.
    pub fn freeHost(ptr: *anyopaque) DriverError!void {
        try toError(sys.cuMemFreeHost(ptr));
    }

    /// Synchronous host-to-device memory copy.
    pub fn copyHtoD(dst: sys.CUdeviceptr, src: *const anyopaque, size: usize) DriverError!void {
        try toError(sys.cuMemcpyHtoD_v2(dst, src, size));
    }

    /// Synchronous device-to-host memory copy.
    pub fn copyDtoH(dst: *anyopaque, src: sys.CUdeviceptr, size: usize) DriverError!void {
        try toError(sys.cuMemcpyDtoH_v2(dst, src, size));
    }

    /// Synchronous device-to-device memory copy.
    pub fn copyDtoD(dst: sys.CUdeviceptr, src: sys.CUdeviceptr, size: usize) DriverError!void {
        try toError(sys.cuMemcpyDtoD_v2(dst, src, size));
    }

    /// Asynchronous host-to-device memory copy.
    pub fn copyHtoDAsync(dst: sys.CUdeviceptr, src: *const anyopaque, size: usize, s: sys.CUstream) DriverError!void {
        try toError(sys.cuMemcpyHtoDAsync_v2(dst, src, size, s));
    }

    /// Asynchronous device-to-host memory copy.
    pub fn copyDtoHAsync(dst: *anyopaque, src: sys.CUdeviceptr, size: usize, s: sys.CUstream) DriverError!void {
        try toError(sys.cuMemcpyDtoHAsync_v2(dst, src, size, s));
    }

    /// Asynchronous device-to-device memory copy.
    pub fn copyDtoDAsync(dst: sys.CUdeviceptr, src: sys.CUdeviceptr, size: usize, s: sys.CUstream) DriverError!void {
        try toError(sys.cuMemcpyDtoDAsync_v2(dst, src, size, s));
    }

    /// Set device memory to a 32-bit value.
    pub fn setD32(dst: sys.CUdeviceptr, value: u32, count: usize) DriverError!void {
        try toError(sys.cuMemsetD32_v2(dst, value, count));
    }

    /// Set device memory to an 8-bit value.
    pub fn setD8(dst: sys.CUdeviceptr, value: u8, count: usize) DriverError!void {
        try toError(sys.cuMemsetD8_v2(dst, value, count));
    }

    /// Asynchronous set device memory to a 32-bit value.
    pub fn setD32Async(dst: sys.CUdeviceptr, value: u32, count: usize, s: sys.CUstream) DriverError!void {
        try toError(sys.cuMemsetD32Async(dst, value, count, s));
    }

    /// Asynchronous set device memory to an 8-bit value.
    pub fn setD8Async(dst: sys.CUdeviceptr, value: u8, count: usize, s: sys.CUstream) DriverError!void {
        try toError(sys.cuMemsetD8Async(dst, value, count, s));
    }
};

// ============================================================================
// Module Functions
// ============================================================================

pub const module = struct {
    /// Load a module from a file path.
    pub fn load(path: [*:0]const u8) DriverError!sys.CUmodule {
        var mod: sys.CUmodule = undefined;
        try toError(sys.cuModuleLoad(&mod, path));
        return mod;
    }

    /// Load a module from in-memory data (e.g., PTX string).
    pub fn loadData(data: *const anyopaque) DriverError!sys.CUmodule {
        var mod: sys.CUmodule = undefined;
        try toError(sys.cuModuleLoadData(&mod, data));
        return mod;
    }

    /// Unload a module.
    pub fn unload(mod: sys.CUmodule) DriverError!void {
        try toError(sys.cuModuleUnload(mod));
    }

    /// Get a function handle from a module by name.
    pub fn getFunction(mod: sys.CUmodule, name: [*:0]const u8) DriverError!sys.CUfunction {
        var func: sys.CUfunction = undefined;
        try toError(sys.cuModuleGetFunction(&func, mod, name));
        return func;
    }
};

// ============================================================================
// Kernel Launch Functions
// ============================================================================

pub const launch = struct {
    /// Launch a kernel on the GPU.
    pub fn kernel(
        func: sys.CUfunction,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        block_x: u32,
        block_y: u32,
        block_z: u32,
        shared_mem_bytes: u32,
        s: sys.CUstream,
        kernel_params: ?[*]?*anyopaque,
        extra: ?[*]?*anyopaque,
    ) DriverError!void {
        try toError(sys.cuLaunchKernel(
            func,
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
            shared_mem_bytes,
            s,
            kernel_params,
            extra,
        ));
    }
};

// ============================================================================
// Function Attribute Functions
// ============================================================================

pub const function = struct {
    /// Get a specific attribute of a CUDA function.
    pub fn getAttribute(func: sys.CUfunction, attrib: sys.CUfunction_attribute) DriverError!i32 {
        var value: i32 = undefined;
        try toError(sys.cuFuncGetAttribute(&value, attrib, func));
        return value;
    }
};

// ============================================================================
// Event Functions
// ============================================================================

pub const event = struct {
    /// Create an event.
    pub fn create(flags: u32) DriverError!sys.CUevent {
        var e: sys.CUevent = undefined;
        try toError(sys.cuEventCreate(&e, flags));
        return e;
    }

    /// Destroy an event.
    pub fn destroy(e: sys.CUevent) DriverError!void {
        try toError(sys.cuEventDestroy_v2(e));
    }

    /// Record an event on a stream.
    pub fn record(e: sys.CUevent, s: sys.CUstream) DriverError!void {
        try toError(sys.cuEventRecord(e, s));
    }

    /// Wait for an event to complete.
    pub fn synchronize(e: sys.CUevent) DriverError!void {
        try toError(sys.cuEventSynchronize(e));
    }

    /// Calculate elapsed time in milliseconds between two events.
    pub fn elapsedTime(start: sys.CUevent, end: sys.CUevent) DriverError!f32 {
        var ms: f32 = undefined;
        try toError(sys.cuEventElapsedTime(&ms, start, end));
        return ms;
    }

    /// Non-blocking check whether the event has been recorded/completed.
    /// Returns true if complete, false if still pending.
    pub fn query(e: sys.CUevent) DriverError!bool {
        const status = sys.cuEventQuery(e);
        if (status == sys.CU_SUCCESS) return true;
        if (status == sys.CUDA_ERROR_NOT_READY) return false;
        try toError(status);
        unreachable;
    }
};

// ============================================================================
// CUDA Graph API
// ============================================================================

pub const graph = struct {
    /// Begin stream capture — all subsequent ops on this stream are recorded into a graph.
    pub fn beginCapture(s: sys.CUstream, mode: sys.CUstreamCaptureMode) DriverError!void {
        try toError(sys.cuStreamBeginCapture_v2(s, mode));
    }

    /// End stream capture and return the captured graph.
    pub fn endCapture(s: sys.CUstream) DriverError!sys.CUgraph {
        var g: sys.CUgraph = undefined;
        try toError(sys.cuStreamEndCapture(s, &g));
        return g;
    }

    /// Query the capture status of a stream.
    pub fn isCapturing(s: sys.CUstream) DriverError!sys.CUstreamCaptureStatus {
        var status: sys.CUstreamCaptureStatus = undefined;
        try toError(sys.cuStreamIsCapturing(s, &status));
        return status;
    }

    /// Instantiate an executable graph from a graph template.
    pub fn instantiate(g: sys.CUgraph) DriverError!sys.CUgraphExec {
        var exec: sys.CUgraphExec = undefined;
        try toError(sys.cuGraphInstantiate(&exec, g, 0));
        return exec;
    }

    /// Launch an executable graph on a stream.
    pub fn launch(exec: sys.CUgraphExec, s: sys.CUstream) DriverError!void {
        try toError(sys.cuGraphLaunch(exec, s));
    }

    /// Destroy a graph template.
    pub fn destroy(g: sys.CUgraph) DriverError!void {
        try toError(sys.cuGraphDestroy(g));
    }

    /// Destroy an executable graph.
    pub fn execDestroy(exec: sys.CUgraphExec) DriverError!void {
        try toError(sys.cuGraphExecDestroy(exec));
    }
};

// ============================================================================
// Unified Memory
// ============================================================================

pub const unified = struct {
    /// Allocate unified memory accessible from both host and device.
    pub fn allocManaged(size: usize, flags: c_uint) DriverError!sys.CUdeviceptr {
        var ptr: sys.CUdeviceptr = undefined;
        try toError(sys.cuMemAllocManaged(&ptr, size, flags));
        return ptr;
    }

    /// Prefetch unified memory to the specified device asynchronously.
    pub fn prefetchAsync(ptr: sys.CUdeviceptr, count: usize, dst_device: sys.CUdevice, s: sys.CUstream) DriverError!void {
        try toError(sys.cuMemPrefetchAsync(ptr, count, dst_device, s));
    }
};

// ============================================================================
// Peer Access (Multi-GPU)
// ============================================================================

pub const peer = struct {
    /// Check if a device can access another device's memory.
    pub fn canAccessPeer(dev: sys.CUdevice, peer_dev: sys.CUdevice) DriverError!bool {
        var can_access: c_int = undefined;
        try toError(sys.cuDeviceCanAccessPeer(&can_access, dev, peer_dev));
        return can_access != 0;
    }

    /// Enable peer access from the current context to the peer context.
    pub fn enableAccess(peer_ctx: sys.CUcontext, flags: c_uint) DriverError!void {
        try toError(sys.cuCtxEnablePeerAccess(peer_ctx, flags));
    }

    /// Disable peer access from the current context to the peer context.
    pub fn disableAccess(peer_ctx: sys.CUcontext) DriverError!void {
        try toError(sys.cuCtxDisablePeerAccess(peer_ctx));
    }
};

// ============================================================================
// Occupancy Calculator
// ============================================================================

pub const occupancy = struct {
    /// Get the maximum number of active blocks per SM for a given kernel.
    pub fn maxActiveBlocksPerMultiprocessor(func: sys.CUfunction, block_size: i32, dynamic_smem_size: usize) DriverError!i32 {
        var num_blocks: i32 = undefined;
        try toError(sys.cuOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, func, block_size, dynamic_smem_size));
        return num_blocks;
    }

    /// Suggest an optimal block size for maximum occupancy.
    /// Returns .{ .min_grid_size, .block_size }.
    pub fn maxPotentialBlockSize(func: sys.CUfunction, dynamic_smem_size: usize, block_size_limit: i32) DriverError!struct { min_grid_size: i32, block_size: i32 } {
        var min_grid: i32 = undefined;
        var block: i32 = undefined;
        try toError(sys.cuOccupancyMaxPotentialBlockSize(&min_grid, &block, func, null, dynamic_smem_size, block_size_limit));
        return .{ .min_grid_size = min_grid, .block_size = block };
    }
};

// ============================================================================
// Memory Info
// ============================================================================

pub const memInfo = struct {
    /// Get free and total memory available on the current device.
    pub fn get() DriverError!struct { free: usize, total: usize } {
        var free_bytes: usize = undefined;
        var total_bytes: usize = undefined;
        try toError(sys.cuMemGetInfo_v2(&free_bytes, &total_bytes));
        return .{ .free = free_bytes, .total = total_bytes };
    }
};
