/// zCUDA: CUDA Runtime API - Error wrapping layer.
///
/// Layer 2: Converts CUDA Runtime API status codes to Zig error unions.
const std = @import("std");
const sys = @import("sys.zig");

// ============================================================================
// Error Type
// ============================================================================

pub const RuntimeError = error{
    InvalidValue,
    MemoryAllocation,
    InitializationError,
    InvalidDevice,
    InvalidMemcpyDirection,
    NotReady,
    Unknown,
};

pub fn toError(status: sys.cudaError_t) RuntimeError!void {
    return switch (status) {
        sys.cudaSuccess => {},
        sys.cudaErrorInvalidValue => RuntimeError.InvalidValue,
        sys.cudaErrorMemoryAllocation => RuntimeError.MemoryAllocation,
        sys.cudaErrorInitializationError => RuntimeError.InitializationError,
        sys.cudaErrorInvalidDevice => RuntimeError.InvalidDevice,
        sys.cudaErrorInvalidMemcpyDirection => RuntimeError.InvalidMemcpyDirection,
        sys.cudaErrorNotReady => RuntimeError.NotReady,
        else => RuntimeError.Unknown,
    };
}

// ============================================================================
// Device Management
// ============================================================================

pub fn getDeviceCount() RuntimeError!i32 {
    var count: i32 = 0;
    try toError(sys.cudaGetDeviceCount(&count));
    return count;
}

pub fn setDevice(device: i32) RuntimeError!void {
    try toError(sys.cudaSetDevice(device));
}

pub fn getDevice() RuntimeError!i32 {
    var device: i32 = 0;
    try toError(sys.cudaGetDevice(&device));
    return device;
}

pub fn deviceSynchronize() RuntimeError!void {
    try toError(sys.cudaDeviceSynchronize());
}

pub fn deviceReset() RuntimeError!void {
    try toError(sys.cudaDeviceReset());
}

pub fn getDeviceProperties(device: i32) RuntimeError!sys.cudaDeviceProp {
    var props: sys.cudaDeviceProp = undefined;
    try toError(sys.cudaGetDeviceProperties(&props, device));
    return props;
}

// ============================================================================
// Memory Management
// ============================================================================

pub fn malloc(size: usize) RuntimeError!*anyopaque {
    var ptr: ?*anyopaque = null;
    try toError(sys.cudaMalloc(&ptr, size));
    return ptr orelse return RuntimeError.MemoryAllocation;
}

pub fn free(ptr: *anyopaque) RuntimeError!void {
    try toError(sys.cudaFree(ptr));
}

pub fn memcpyHtoD(dst: *anyopaque, src: *const anyopaque, size: usize) RuntimeError!void {
    try toError(sys.cudaMemcpy(dst, src, size, sys.cudaMemcpyHostToDevice));
}

pub fn memcpyDtoH(dst: *anyopaque, src: *const anyopaque, size: usize) RuntimeError!void {
    try toError(sys.cudaMemcpy(dst, src, size, sys.cudaMemcpyDeviceToHost));
}

pub fn memcpyDtoD(dst: *anyopaque, src: *const anyopaque, size: usize) RuntimeError!void {
    try toError(sys.cudaMemcpy(dst, src, size, sys.cudaMemcpyDeviceToDevice));
}

pub fn memset(ptr: *anyopaque, value: i32, count: usize) RuntimeError!void {
    try toError(sys.cudaMemset(ptr, value, count));
}

pub fn memsetAsync(ptr: *anyopaque, value: i32, count: usize, stream: sys.cudaStream_t) RuntimeError!void {
    try toError(sys.cudaMemsetAsync(ptr, value, count, stream));
}

pub fn memcpyHtoDAsync(dst: *anyopaque, src: *const anyopaque, size: usize, stream: sys.cudaStream_t) RuntimeError!void {
    try toError(sys.cudaMemcpyAsync(dst, src, size, sys.cudaMemcpyHostToDevice, stream));
}

pub fn memcpyDtoHAsync(dst: *anyopaque, src: *const anyopaque, size: usize, stream: sys.cudaStream_t) RuntimeError!void {
    try toError(sys.cudaMemcpyAsync(dst, src, size, sys.cudaMemcpyDeviceToHost, stream));
}

pub fn memcpyDtoDAsync(dst: *anyopaque, src: *const anyopaque, size: usize, stream: sys.cudaStream_t) RuntimeError!void {
    try toError(sys.cudaMemcpyAsync(dst, src, size, sys.cudaMemcpyDeviceToDevice, stream));
}

pub fn mallocHost(size: usize) RuntimeError!*anyopaque {
    var ptr: ?*anyopaque = null;
    try toError(sys.cudaMallocHost(&ptr, size));
    return ptr orelse return RuntimeError.MemoryAllocation;
}

pub fn freeHost(ptr: *anyopaque) RuntimeError!void {
    try toError(sys.cudaFreeHost(ptr));
}

// ============================================================================
// Stream Management
// ============================================================================

pub fn streamCreate() RuntimeError!sys.cudaStream_t {
    var stream: sys.cudaStream_t = null;
    try toError(sys.cudaStreamCreate(&stream));
    return stream;
}

pub fn streamDestroy(stream: sys.cudaStream_t) RuntimeError!void {
    try toError(sys.cudaStreamDestroy(stream));
}

pub fn streamSynchronize(stream: sys.cudaStream_t) RuntimeError!void {
    try toError(sys.cudaStreamSynchronize(stream));
}

pub fn streamWaitEvent(stream: sys.cudaStream_t, event: sys.cudaEvent_t) RuntimeError!void {
    try toError(sys.cudaStreamWaitEvent(stream, event, 0));
}

pub fn streamQuery(stream: sys.cudaStream_t) RuntimeError!bool {
    const status = sys.cudaStreamQuery(stream);
    if (status == sys.cudaSuccess) return true;
    if (status == sys.cudaErrorNotReady) return false;
    try toError(status);
    unreachable;
}

// ============================================================================
// Event Management
// ============================================================================

pub fn eventCreate() RuntimeError!sys.cudaEvent_t {
    var event: sys.cudaEvent_t = null;
    try toError(sys.cudaEventCreate(&event));
    return event;
}

pub fn eventDestroy(event: sys.cudaEvent_t) RuntimeError!void {
    try toError(sys.cudaEventDestroy(event));
}

pub fn eventRecord(event: sys.cudaEvent_t, stream: sys.cudaStream_t) RuntimeError!void {
    try toError(sys.cudaEventRecord(event, stream));
}

pub fn eventSynchronize(event: sys.cudaEvent_t) RuntimeError!void {
    try toError(sys.cudaEventSynchronize(event));
}

pub fn eventElapsedTime(start: sys.cudaEvent_t, end: sys.cudaEvent_t) RuntimeError!f32 {
    var ms: f32 = 0;
    try toError(sys.cudaEventElapsedTime(&ms, start, end));
    return ms;
}

pub fn eventQuery(event: sys.cudaEvent_t) RuntimeError!bool {
    const status = sys.cudaEventQuery(event);
    if (status == sys.cudaSuccess) return true;
    if (status == sys.cudaErrorNotReady) return false;
    try toError(status);
    unreachable;
}

// ============================================================================
// Error Utilities
// ============================================================================

pub fn getLastError() sys.cudaError_t {
    return sys.cudaGetLastError();
}

pub fn getErrorString(err: sys.cudaError_t) [*:0]const u8 {
    return sys.cudaGetErrorString(err);
}

// ============================================================================
// 2D Memory Operations (P0 Critical)
// ============================================================================

/// 2D memory copy — essential for image and matrix data with pitch.
pub fn memcpy2D(
    dst: *anyopaque,
    dpitch: usize,
    src: *const anyopaque,
    spitch: usize,
    width: usize,
    height: usize,
    kind: c_uint,
) RuntimeError!void {
    try toError(sys.cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind));
}

// ============================================================================
// Peer Access (Multi-GPU)
// ============================================================================

/// Check if a device can access another device's memory.
pub fn deviceCanAccessPeer(device: i32, peer_device: i32) RuntimeError!bool {
    var can_access: c_int = undefined;
    try toError(sys.cudaDeviceCanAccessPeer(&can_access, device, peer_device));
    return can_access != 0;
}

/// Enable peer access between devices.
pub fn deviceEnablePeerAccess(peer_device: i32, flags: c_uint) RuntimeError!void {
    try toError(sys.cudaDeviceEnablePeerAccess(peer_device, flags));
}

/// Disable peer access between devices.
pub fn deviceDisablePeerAccess(peer_device: i32) RuntimeError!void {
    try toError(sys.cudaDeviceDisablePeerAccess(peer_device));
}

// ============================================================================
// Pitched Memory Allocation
// ============================================================================

/// Allocate pitched memory (2D arrays with aligned rows).
pub fn mallocPitch(width: usize, height: usize) RuntimeError!struct { ptr: *anyopaque, pitch: usize } {
    var ptr: ?*anyopaque = null;
    var pitch: usize = undefined;
    try toError(sys.cudaMallocPitch(&ptr, &pitch, width, height));
    return .{ .ptr = ptr.?, .pitch = pitch };
}

// ============================================================================
// Stream/Event with Flags
// ============================================================================

/// Create a stream with flags (e.g., non-blocking).
pub fn streamCreateWithFlags(flags: c_uint) RuntimeError!sys.cudaStream_t {
    var stream: sys.cudaStream_t = null;
    try toError(sys.cudaStreamCreateWithFlags(&stream, flags));
    return stream;
}

/// Create an event with flags (e.g., disable timing for performance).
pub fn eventCreateWithFlags(flags: c_uint) RuntimeError!sys.cudaEvent_t {
    var event: sys.cudaEvent_t = null;
    try toError(sys.cudaEventCreateWithFlags(&event, flags));
    return event;
}
