/// zCUDA: CUDA Runtime API - Safe abstraction layer.
///
/// Layer 3: High-level safe wrappers for CUDA Runtime API.
///
/// Most users should use the Driver API (src/driver/) for production code.
/// The Runtime API is provided for convenience and compatibility.
const std = @import("std");
const sys = @import("sys.zig");
const result = @import("result.zig");

pub const RuntimeError = result.RuntimeError;

// ============================================================================
// RuntimeSlice — Type-safe device memory (Runtime API)
// ============================================================================

/// Type-safe wrapper around a device pointer allocated via the Runtime API.
pub fn RuntimeSlice(comptime T: type) type {
    return struct {
        ptr: *anyopaque,
        len: usize,

        const Self = @This();

        /// Number of bytes for this slice.
        pub fn byteSize(self: Self) usize {
            return self.len * @sizeOf(T);
        }

        /// Free device memory.
        pub fn deinit(self: *Self) void {
            result.free(self.ptr) catch {};
        }
    };
}

// ============================================================================
// RuntimeStream — RAII stream wrapper
// ============================================================================

pub const RuntimeStream = struct {
    stream: sys.cudaStream_t,

    const Self = @This();

    pub fn init() RuntimeError!Self {
        return Self{ .stream = try result.streamCreate() };
    }

    pub fn synchronize(self: Self) RuntimeError!void {
        try result.streamSynchronize(self.stream);
    }

    /// Make this stream wait until the given event completes.
    pub fn waitEvent(self: Self, event: RuntimeEvent) RuntimeError!void {
        try result.streamWaitEvent(self.stream, event.event);
    }

    /// Non-blocking check whether all operations on this stream have completed.
    pub fn query(self: Self) RuntimeError!bool {
        return result.streamQuery(self.stream);
    }

    pub fn deinit(self: *Self) void {
        result.streamDestroy(self.stream) catch {};
    }
};

// ============================================================================
// RuntimeEvent — RAII event wrapper
// ============================================================================

pub const RuntimeEvent = struct {
    event: sys.cudaEvent_t,

    const Self = @This();

    pub fn init() RuntimeError!Self {
        return Self{ .event = try result.eventCreate() };
    }

    pub fn record(self: Self, stream: RuntimeStream) RuntimeError!void {
        try result.eventRecord(self.event, stream.stream);
    }

    pub fn synchronize(self: Self) RuntimeError!void {
        try result.eventSynchronize(self.event);
    }

    pub fn elapsedTime(start: Self, end: Self) RuntimeError!f32 {
        return result.eventElapsedTime(start.event, end.event);
    }

    /// Non-blocking check whether this event has completed.
    pub fn query(self: Self) RuntimeError!bool {
        return result.eventQuery(self.event);
    }

    pub fn deinit(self: *Self) void {
        result.eventDestroy(self.event) catch {};
    }
};

// ============================================================================
// RuntimeContext — Device management wrapper
// ============================================================================

/// A CUDA Runtime context for simplified device management.
pub const RuntimeContext = struct {
    device_id: i32,

    const Self = @This();

    /// Create a Runtime context on the given device.
    pub fn init(device: i32) RuntimeError!Self {
        try result.setDevice(device);
        return Self{ .device_id = device };
    }

    /// Synchronize the device.
    pub fn synchronize(_: Self) RuntimeError!void {
        try result.deviceSynchronize();
    }

    /// Get the number of available CUDA devices.
    pub fn deviceCount() RuntimeError!i32 {
        return result.getDeviceCount();
    }

    /// Allocate typed device memory.
    pub fn alloc(_: Self, comptime T: type, len: usize) RuntimeError!RuntimeSlice(T) {
        const ptr = try result.malloc(len * @sizeOf(T));
        return RuntimeSlice(T){ .ptr = ptr, .len = len };
    }

    /// Copy data from host slice to device slice.
    pub fn copyToDevice(_: Self, comptime T: type, dst: RuntimeSlice(T), src: []const T) RuntimeError!void {
        std.debug.assert(dst.len >= src.len);
        try result.memcpyHtoD(dst.ptr, @ptrCast(src.ptr), src.len * @sizeOf(T));
    }

    /// Copy data from device slice to host slice.
    pub fn copyToHost(_: Self, comptime T: type, dst: []T, src: RuntimeSlice(T)) RuntimeError!void {
        std.debug.assert(dst.len >= src.len);
        try result.memcpyDtoH(@ptrCast(dst.ptr), src.ptr, src.len * @sizeOf(T));
    }

    /// Copy data from device to device.
    pub fn copyDtoD(_: Self, comptime T: type, dst: RuntimeSlice(T), src: RuntimeSlice(T)) RuntimeError!void {
        std.debug.assert(dst.len >= src.len);
        try result.memcpyDtoD(dst.ptr, src.ptr, src.len * @sizeOf(T));
    }

    /// Set device memory to a byte value.
    pub fn memset(_: Self, comptime T: type, slice: RuntimeSlice(T), value: i32) RuntimeError!void {
        try result.memset(slice.ptr, value, slice.byteSize());
    }

    /// Allocate pinned host memory.
    pub fn allocHost(_: Self, size: usize) RuntimeError!*anyopaque {
        return result.mallocHost(size);
    }

    /// Free pinned host memory.
    pub fn freeHost(_: Self, ptr: *anyopaque) RuntimeError!void {
        try result.freeHost(ptr);
    }

    /// Create a new stream.
    pub fn newStream(_: Self) RuntimeError!RuntimeStream {
        return RuntimeStream.init();
    }

    /// Create a new event.
    pub fn newEvent(_: Self) RuntimeError!RuntimeEvent {
        return RuntimeEvent.init();
    }

    /// Reset the device, cleaning up all resources.
    pub fn reset() RuntimeError!void {
        try result.deviceReset();
    }

    /// Get device properties for this device.
    pub fn getDeviceProperties(self: Self) RuntimeError!sys.cudaDeviceProp {
        return result.getDeviceProperties(self.device_id);
    }

    /// Get the device name as a Zig string slice.
    pub fn deviceName(self: Self) RuntimeError![]const u8 {
        const props = try self.getDeviceProperties();
        const name: [*:0]const u8 = @ptrCast(&props.name);
        return std.mem.sliceTo(name, 0);
    }

    /// Async host-to-device copy on a stream.
    pub fn copyToDeviceAsync(_: Self, comptime T: type, dst: RuntimeSlice(T), src: []const T, stream: RuntimeStream) RuntimeError!void {
        std.debug.assert(dst.len >= src.len);
        try result.memcpyHtoDAsync(dst.ptr, @ptrCast(src.ptr), src.len * @sizeOf(T), stream.stream);
    }

    /// Async device-to-host copy on a stream.
    pub fn copyToHostAsync(_: Self, comptime T: type, dst: []T, src: RuntimeSlice(T), stream: RuntimeStream) RuntimeError!void {
        std.debug.assert(dst.len >= src.len);
        try result.memcpyDtoHAsync(@ptrCast(dst.ptr), src.ptr, src.len * @sizeOf(T), stream.stream);
    }

    /// Async device-to-device copy on a stream.
    pub fn copyDtoDAsync(_: Self, comptime T: type, dst: RuntimeSlice(T), src: RuntimeSlice(T), stream: RuntimeStream) RuntimeError!void {
        std.debug.assert(dst.len >= src.len);
        try result.memcpyDtoDAsync(dst.ptr, src.ptr, src.len * @sizeOf(T), stream.stream);
    }

    /// Async memset on a stream.
    pub fn memsetAsync(_: Self, comptime T: type, slice: RuntimeSlice(T), value: i32, stream: RuntimeStream) RuntimeError!void {
        try result.memsetAsync(slice.ptr, value, slice.byteSize(), stream.stream);
    }

    // ========================================================================
    // 2D Memory Operations
    // ========================================================================

    /// Allocate pitched (2D) device memory.
    /// Returns a pointer and its pitch (row stride in bytes).
    pub fn mallocPitch(_: Self, width: usize, height: usize) RuntimeError!struct { ptr: *anyopaque, pitch: usize } {
        const res = try result.mallocPitch(width, height);
        return .{ .ptr = res.ptr, .pitch = res.pitch };
    }

    /// Copy a 2D block of memory.
    pub fn memcpy2D(
        _: Self,
        dst: *anyopaque,
        dpitch: usize,
        src: *const anyopaque,
        spitch: usize,
        width: usize,
        height: usize,
        kind: sys.cudaMemcpyKind,
    ) RuntimeError!void {
        try result.memcpy2D(dst, dpitch, src, spitch, width, height, kind);
    }

    // ========================================================================
    // Advanced Stream/Event Creation
    // ========================================================================

    /// Create a stream with flags (e.g. cudaStreamNonBlocking).
    pub fn newStreamWithFlags(_: Self, flags: c_uint) RuntimeError!RuntimeStream {
        const stream = try result.streamCreateWithFlags(flags);
        return RuntimeStream{ .stream = stream };
    }

    /// Create an event with flags (e.g. cudaEventBlockingSync, cudaEventDisableTiming).
    pub fn newEventWithFlags(_: Self, flags: c_uint) RuntimeError!RuntimeEvent {
        const event = try result.eventCreateWithFlags(flags);
        return RuntimeEvent{ .event = event };
    }
};

// ============================================================================
// Tests
// ============================================================================
