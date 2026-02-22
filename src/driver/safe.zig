/// zCUDA: CUDA Driver API - Safe abstraction layer.
///
/// Layer 3: High-level, type-safe abstractions over the CUDA Driver API.
/// This is the recommended API for general use. All types manage their
/// own resource lifetimes via Zig's `deinit` pattern.
///
/// ## Key Types
///
/// | Concept | CPU Equivalent | zCUDA Type |
/// | --- | --- | --- |
/// | Memory allocator | `std.mem.Allocator` | `CudaContext` |
/// | Values on heap | `[]T` | `CudaSlice(T)` |
/// | Immutable slice | `[]const T` | `CudaView(T)` |
/// | Mutable slice | `[]T` | `CudaViewMut(T)` |
/// | Function | `fn` | `CudaFunction` |
/// | Calling a function | `myFn(a, b, c)` | `stream.launch(func, cfg, .{a, b, c})` |
///
/// ## Example
///
/// ```zig
/// const ctx = try CudaContext.new(0);
/// defer ctx.deinit();
/// const stream = ctx.defaultStream();
/// const data = try stream.allocZeros(f32, allocator, 100);
/// defer data.deinit();
/// ```
const std = @import("std");
const sys = @import("sys.zig");
const result = @import("result.zig");
const types = @import("../types.zig");

pub const DriverError = result.DriverError;

// ============================================================================
// CudaSlice — Owns device memory
// ============================================================================

/// A typed slice of device memory.
///
/// Analogous to `Vec<T>` on the GPU. Owns the device memory and frees
/// it when `deinit()` is called. Obtain sub-slices via `slice()` and
/// `sliceMut()`.
pub fn CudaSlice(comptime T: type) type {
    return struct {
        ptr: sys.CUdeviceptr,
        len: usize,
        ctx: *const CudaContext,

        const Self = @This();

        /// Free the device memory.
        pub fn deinit(self: Self) void {
            result.mem.free(self.ptr) catch {};
        }

        /// Create an immutable view of a sub-range.
        pub fn slice(self: Self, start: usize, end: usize) CudaView(T) {
            std.debug.assert(start <= end);
            std.debug.assert(end <= self.len);
            return CudaView(T){
                .ptr = self.ptr + start * @sizeOf(T),
                .len = end - start,
                .ctx = self.ctx,
            };
        }

        /// Create a mutable view of a sub-range.
        pub fn sliceMut(self: Self, start: usize, end: usize) CudaViewMut(T) {
            std.debug.assert(start <= end);
            std.debug.assert(end <= self.len);
            return CudaViewMut(T){
                .ptr = self.ptr + start * @sizeOf(T),
                .len = end - start,
                .ctx = self.ctx,
            };
        }

        /// Get the device pointer as a typed pointer (for passing to cuBLAS, etc.)
        pub fn devicePtr(self: Self) types.DevicePtr(T) {
            return types.DevicePtr(T).init(self.ptr);
        }
    };
}

// ============================================================================
// CudaView — Immutable device memory view (non-owning)
// ============================================================================

/// An immutable, non-owning view into device memory.
/// Analogous to `[]const T` on the GPU.
pub fn CudaView(comptime T: type) type {
    return struct {
        ptr: sys.CUdeviceptr,
        len: usize,
        ctx: *const CudaContext,

        const Self = @This();

        pub fn devicePtr(self: Self) types.DevicePtr(T) {
            return types.DevicePtr(T).init(self.ptr);
        }

        /// Create a sub-view.
        pub fn subView(self: Self, start: usize, end: usize) Self {
            std.debug.assert(start <= end);
            std.debug.assert(end <= self.len);
            return Self{
                .ptr = self.ptr + start * @sizeOf(T),
                .len = end - start,
                .ctx = self.ctx,
            };
        }
    };
}

// ============================================================================
// CudaViewMut — Mutable device memory view (non-owning)
// ============================================================================

/// A mutable, non-owning view into device memory.
/// Analogous to `[]T` on the GPU.
pub fn CudaViewMut(comptime T: type) type {
    return struct {
        ptr: sys.CUdeviceptr,
        len: usize,
        ctx: *const CudaContext,

        const Self = @This();

        pub fn devicePtr(self: Self) types.DevicePtr(T) {
            return types.DevicePtr(T).init(self.ptr);
        }

        /// Create a mutable sub-view.
        pub fn subView(self: Self, start: usize, end: usize) Self {
            std.debug.assert(start <= end);
            std.debug.assert(end <= self.len);
            return Self{
                .ptr = self.ptr + start * @sizeOf(T),
                .len = end - start,
                .ctx = self.ctx,
            };
        }
    };
}

// ============================================================================
// CudaModule — Loaded GPU module (containing kernels)
// ============================================================================

/// A loaded CUDA module containing device code (functions/kernels).
/// Created from PTX or CUBIN data via `CudaContext.loadModule()`.
pub const CudaModule = struct {
    module: sys.CUmodule,
    ctx: *const CudaContext,

    const Self = @This();

    /// Unload the module from the GPU.
    pub fn deinit(self: Self) void {
        result.module.unload(self.module) catch {};
    }

    /// Get a function (kernel) handle by name from this module.
    pub fn getFunction(self: Self, name: [*:0]const u8) DriverError!CudaFunction {
        const func = try result.module.getFunction(self.module, name);
        return CudaFunction{
            .function = func,
            .module = self,
        };
    }
};

// ============================================================================
// CudaFunction — A callable kernel function
// ============================================================================

/// Represents a callable CUDA kernel function obtained from a `CudaModule`.
pub const CudaFunction = struct {
    function: sys.CUfunction,
    module: CudaModule,

    const Self = @This();

    /// Get a function attribute (e.g., max threads per block, register count).
    pub fn getAttribute(self: Self, attrib: sys.CUfunction_attribute) DriverError!i32 {
        return try result.function.getAttribute(self.function, attrib);
    }

    /// Suggest an optimal block size for maximum occupancy.
    /// Returns .{ .block_size, .min_grid_size }.
    pub fn optimalBlockSize(self: Self, opts: struct { shared_mem_bytes: usize = 0, block_size_limit: i32 = 0 }) DriverError!struct { block_size: i32, min_grid_size: i32 } {
        const r = try result.occupancy.maxPotentialBlockSize(self.function, opts.shared_mem_bytes, opts.block_size_limit);
        return .{ .block_size = r.block_size, .min_grid_size = r.min_grid_size };
    }

    /// Get the maximum number of active blocks per SM for a given block size.
    pub fn maxActiveBlocksPerSM(self: Self, block_size: i32, dynamic_smem_size: usize) DriverError!i32 {
        return try result.occupancy.maxActiveBlocksPerMultiprocessor(self.function, block_size, dynamic_smem_size);
    }
};

// ============================================================================
// CudaEvent — Synchronization primitive
// ============================================================================

/// A CUDA event used for timing and synchronization.
pub const CudaEvent = struct {
    event: sys.CUevent,
    ctx: *const CudaContext,

    const Self = @This();

    /// Destroy the event.
    pub fn deinit(self: Self) void {
        result.event.destroy(self.event) catch {};
    }

    /// Record this event on the given stream.
    pub fn record(self: Self, s: *const CudaStream) DriverError!void {
        try result.event.record(self.event, s.stream);
    }

    /// Block the CPU until this event completes.
    pub fn synchronize(self: Self) DriverError!void {
        try result.event.synchronize(self.event);
    }

    /// Calculate elapsed time in milliseconds between two events.
    /// Both events must have been recorded and completed.
    pub fn elapsedTime(start: Self, end: Self) DriverError!f32 {
        return try result.event.elapsedTime(start.event, end.event);
    }

    /// Non-blocking check whether this event has completed.
    /// Returns true if complete, false if still pending.
    pub fn query(self: Self) DriverError!bool {
        return result.event.query(self.event);
    }
};

// ============================================================================
// CudaStream — Execution stream
// ============================================================================

/// A CUDA stream representing a sequence of asynchronous operations.
///
/// All memory operations and kernel launches occur on a stream.
/// Use `CudaContext.defaultStream()` for the default stream, or
/// `CudaContext.newStream()` for an independent stream.
pub const CudaStream = struct {
    stream: sys.CUstream,
    ctx: *const CudaContext,
    is_default: bool,

    const Self = @This();

    /// Destroy the stream (no-op for the default stream).
    pub fn deinit(self: Self) void {
        if (!self.is_default) {
            result.stream.destroy(self.stream) catch {};
        }
    }

    /// Block the CPU until all operations on this stream complete.
    pub fn synchronize(self: Self) DriverError!void {
        try self.ctx.bindToThread();
        try result.stream.synchronize(self.stream);
    }

    /// Make this stream wait until the given event completes.
    pub fn waitEvent(self: Self, e: CudaEvent) DriverError!void {
        try result.stream.waitEvent(self.stream, e.event);
    }

    /// Non-blocking check whether all operations on this stream have completed.
    /// Returns true if complete, false if still pending.
    pub fn query(self: Self) DriverError!bool {
        return result.stream.query(self.stream);
    }

    // --- Memory Allocation ---

    /// Allocate device memory for `n` elements of type T (uninitialized).
    pub fn alloc(self: Self, comptime T: type, allocator: std.mem.Allocator, n: usize) DriverError!CudaSlice(T) {
        _ = allocator;
        try self.ctx.bindToThread();
        const ptr = try result.mem.alloc(n * @sizeOf(T));
        return CudaSlice(T){
            .ptr = ptr,
            .len = n,
            .ctx = self.ctx,
        };
    }

    /// Allocate device memory for `n` elements of type T, zeroed.
    pub fn allocZeros(self: Self, comptime T: type, allocator: std.mem.Allocator, n: usize) DriverError!CudaSlice(T) {
        const slice = try self.alloc(T, allocator, n);
        try result.mem.setD8(slice.ptr, 0, n * @sizeOf(T));
        return slice;
    }

    // --- Host <-> Device Memory Transfer ---

    /// Clone host data to a new device allocation (synchronous).
    pub fn cloneHtoD(self: Self, comptime T: type, data: []const T) DriverError!CudaSlice(T) {
        try self.ctx.bindToThread();
        const ptr = try result.mem.alloc(data.len * @sizeOf(T));
        try result.mem.copyHtoD(ptr, @ptrCast(data.ptr), data.len * @sizeOf(T));
        return CudaSlice(T){
            .ptr = ptr,
            .len = data.len,
            .ctx = self.ctx,
        };
    }

    /// Copy host data to an existing device allocation (synchronous).
    pub fn memcpyHtoD(self: Self, comptime T: type, dst: CudaSlice(T), src: []const T) DriverError!void {
        try self.ctx.bindToThread();
        std.debug.assert(dst.len >= src.len);
        try result.mem.copyHtoD(dst.ptr, @ptrCast(src.ptr), src.len * @sizeOf(T));
    }

    /// Copy device data to host (synchronous).
    /// The caller must ensure `dst` has enough space.
    pub fn memcpyDtoH(self: Self, comptime T: type, dst: []T, src: CudaSlice(T)) DriverError!void {
        try self.ctx.bindToThread();
        std.debug.assert(dst.len >= src.len);
        try result.mem.copyDtoH(@ptrCast(dst.ptr), src.ptr, src.len * @sizeOf(T));
    }

    /// Clone device data to a newly allocated host buffer.
    pub fn cloneDtoH(self: Self, comptime T: type, allocator: std.mem.Allocator, src: CudaSlice(T)) ![]T {
        try self.ctx.bindToThread();
        const host = try allocator.alloc(T, src.len);
        errdefer allocator.free(host);
        try result.mem.copyDtoH(@ptrCast(host.ptr), src.ptr, src.len * @sizeOf(T));
        return host;
    }

    /// Copy device data to device (synchronous).
    pub fn memcpyDtoD(self: Self, comptime T: type, dst: CudaSlice(T), src: CudaSlice(T)) DriverError!void {
        try self.ctx.bindToThread();
        std.debug.assert(dst.len >= src.len);
        try result.mem.copyDtoD(dst.ptr, src.ptr, src.len * @sizeOf(T));
    }

    // --- Kernel Launch ---

    /// Launch a kernel function on this stream.
    ///
    /// `args` is a tuple of kernel arguments. Each element must be a scalar
    /// value (i32, f32, etc.) or a pointer to a CudaSlice's device pointer.
    ///
    /// ## Example
    /// ```zig
    /// try stream.launch(kernel, .{
    ///     .grid_dim = .{ .x = 1 },
    ///     .block_dim = .{ .x = 256 },
    /// }, .{ 2.0, &x_dev, &y_dev, @as(i32, n) });
    /// ```
    pub fn launch(
        self: Self,
        func: CudaFunction,
        config: types.LaunchConfig,
        args: anytype,
    ) DriverError!void {
        try self.ctx.bindToThread();

        // Build kernel argument array at comptime
        const ArgsType = @TypeOf(args);
        const fields = @typeInfo(ArgsType).@"struct".fields;
        comptime var num_args = fields.len;
        _ = &num_args;

        var kernel_params: [fields.len]?*anyopaque = undefined;
        var arg_storage: [fields.len]usize = undefined;

        inline for (fields, 0..) |field, i| {
            const arg = @field(args, field.name);
            const ArgFieldType = @TypeOf(arg);

            if (comptime isDeviceSlicePtr(ArgFieldType)) {
                // Device pointer: pass the address of the device pointer value
                arg_storage[i] = arg.ptr;
                kernel_params[i] = @ptrCast(&arg_storage[i]);
            } else {
                // Scalar value: store and pass address
                const StorageType = if (@sizeOf(ArgFieldType) <= @sizeOf(usize))
                    usize
                else
                    @compileError("Kernel argument too large");

                var storage: StorageType = 0;
                const dst: *ArgFieldType = @ptrCast(@alignCast(&storage));
                dst.* = arg;
                arg_storage[i] = storage;
                kernel_params[i] = @ptrCast(&arg_storage[i]);
            }
        }

        try result.launch.kernel(
            func.function,
            config.grid_dim.x,
            config.grid_dim.y,
            config.grid_dim.z,
            config.block_dim.x,
            config.block_dim.y,
            config.block_dim.z,
            config.shared_mem_bytes,
            self.stream,
            @ptrCast(&kernel_params),
            null,
        );
    }

    // --- Event Operations ---

    /// Create a new event associated with this stream's context.
    pub fn createEvent(self: Self, flags: u32) DriverError!CudaEvent {
        try self.ctx.bindToThread();
        const e = try result.event.create(flags);
        return CudaEvent{
            .event = e,
            .ctx = self.ctx,
        };
    }

    /// Record an event on this stream.
    pub fn recordEvent(self: Self, e: CudaEvent) DriverError!void {
        try result.event.record(e.event, self.stream);
    }

    // --- Async Memory Operations ---

    /// Asynchronous host-to-device memory copy.
    pub fn memcpyHtoDAsync(self: Self, comptime T: type, dst: CudaSlice(T), src: []const T) DriverError!void {
        try self.ctx.bindToThread();
        std.debug.assert(dst.len >= src.len);
        try result.mem.copyHtoDAsync(dst.ptr, @ptrCast(src.ptr), src.len * @sizeOf(T), self.stream);
    }

    /// Asynchronous device-to-host memory copy.
    pub fn memcpyDtoHAsync(self: Self, comptime T: type, dst: []T, src: CudaSlice(T)) DriverError!void {
        try self.ctx.bindToThread();
        std.debug.assert(dst.len >= src.len);
        try result.mem.copyDtoHAsync(@ptrCast(dst.ptr), src.ptr, src.len * @sizeOf(T), self.stream);
    }

    /// Asynchronous device-to-device memory copy.
    pub fn memcpyDtoDAsync(self: Self, comptime T: type, dst: CudaSlice(T), src: CudaSlice(T)) DriverError!void {
        try self.ctx.bindToThread();
        std.debug.assert(dst.len >= src.len);
        try result.mem.copyDtoDAsync(dst.ptr, src.ptr, src.len * @sizeOf(T), self.stream);
    }

    // --- Unified Memory ---

    /// Prefetch unified memory to the device associated with this stream's context.
    pub fn prefetchAsync(self: Self, comptime T: type, slice: CudaSlice(T)) DriverError!void {
        try self.ctx.bindToThread();
        try result.unified.prefetchAsync(
            slice.ptr,
            slice.len * @sizeOf(T),
            self.ctx.device,
            self.stream,
        );
    }

    // --- Graph Capture ---

    /// Begin capturing operations into a graph.
    pub fn beginCapture(self: Self) DriverError!void {
        try self.ctx.bindToThread();
        try result.graph.beginCapture(self.stream, sys.CU_STREAM_CAPTURE_MODE_GLOBAL);
    }

    /// End capture and return an executable CudaGraph (null if capture was empty).
    pub fn endCapture(self: Self) DriverError!?CudaGraph {
        try self.ctx.bindToThread();
        const g = try result.graph.endCapture(self.stream);
        const exec = try result.graph.instantiate(g);
        return CudaGraph{
            .cu_graph = g,
            .cu_graph_exec = exec,
            .ctx = self.ctx,
            .cu_stream = self.stream,
        };
    }

    /// Query whether this stream is currently capturing.
    pub fn captureStatus(self: Self) DriverError!sys.CUstreamCaptureStatus {
        try self.ctx.bindToThread();
        return try result.graph.isCapturing(self.stream);
    }
};

/// Check if a type is a pointer to a CudaSlice (used in kernel argument marshaling).
fn isDeviceSlicePtr(comptime T: type) bool {
    const info = @typeInfo(T);
    return switch (info) {
        .pointer => |p| {
            const child_info = @typeInfo(p.child);
            return switch (child_info) {
                .@"struct" => @hasField(p.child, "ptr") and @hasField(p.child, "len") and @hasField(p.child, "ctx"),
                else => false,
            };
        },
        else => false,
    };
}

// ============================================================================
// CudaContext — Entry point for CUDA operations
// ============================================================================

/// Represents a CUDA context on a specific device.
///
/// This is the entry point for all CUDA operations. A context manages
/// the device, streams, memory, and modules for a single GPU.
///
/// # Thread Safety
///
/// The context can be used from multiple threads. All safe API methods
/// call `bindToThread()` to ensure the correct CUDA context is active
/// on the calling thread.
///
/// # Example
///
/// ```zig
/// const ctx = try CudaContext.new(0);
/// defer ctx.deinit();
///
/// const stream = ctx.defaultStream();
/// const data = try stream.cloneHtoD(f32, &[_]f32{1.0, 2.0, 3.0});
/// defer data.deinit();
/// ```
pub const CudaContext = struct {
    device: sys.CUdevice,
    context: sys.CUcontext,
    ordinal: usize,
    allocator: std.mem.Allocator,
    default_stream: ?CudaStream,
    name_buf: [256]u8,

    const Self = @This();

    /// Create a new CUDA context on the device with the given ordinal.
    /// Initializes the CUDA driver if not already initialized.
    pub fn new(ordinal: usize) !*Self {
        try result.init();

        const dev = try result.device.get(@intCast(ordinal));
        const context = try result.primaryCtx.retain(dev);
        try result.ctx.setCurrent(context);

        const name_raw = try result.device.getName(dev);

        const self = try std.heap.page_allocator.create(Self);
        self.* = .{
            .device = dev,
            .context = context,
            .ordinal = ordinal,
            .allocator = std.heap.page_allocator,
            .default_stream = null,
            .name_buf = name_raw,
        };
        return self;
    }

    /// Release the CUDA context and free resources.
    pub fn deinit(self: *const Self) void {
        result.primaryCtx.release(self.device) catch {};
        // Use page_allocator to free the struct
        const self_mut: *Self = @constCast(self);
        std.heap.page_allocator.destroy(self_mut);
    }

    /// Bind this context to the calling thread.
    /// Must be called before any CUDA operation if using multiple threads.
    pub fn bindToThread(self: *const Self) DriverError!void {
        const current = try result.ctx.getCurrent();
        if (current != self.context) {
            try result.ctx.setCurrent(self.context);
        }
    }

    // --- Device Info ---

    /// Get the number of available CUDA devices.
    pub fn deviceCount() DriverError!i32 {
        try result.init();
        return try result.device.getCount();
    }

    /// Get the ordinal index of the device.
    pub fn getOrdinal(self: *const Self) usize {
        return self.ordinal;
    }

    /// Get the device name as a printable string.
    pub fn name(self: *const Self) []const u8 {
        return std.mem.sliceTo(&self.name_buf, 0);
    }

    /// Get the UUID of the device.
    pub fn uuid(self: *const Self) DriverError!sys.CUuuid {
        try self.bindToThread();
        return try result.device.getUuid(self.device);
    }

    /// Get the compute capability as (major, minor).
    pub fn computeCapability(self: *const Self) DriverError!struct { major: i32, minor: i32 } {
        try self.bindToThread();
        const major = try result.device.getAttribute(self.device, sys.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
        const minor = try result.device.getAttribute(self.device, sys.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
        return .{ .major = major, .minor = minor };
    }

    /// Get the total memory on the device in bytes.
    pub fn totalMem(self: *const Self) DriverError!usize {
        try self.bindToThread();
        return try result.device.getTotalMem(self.device);
    }

    /// Get a device attribute value.
    pub fn attribute(self: *const Self, attrib: sys.CUdevice_attribute) DriverError!i32 {
        try self.bindToThread();
        return try result.device.getAttribute(self.device, attrib);
    }

    /// Get the underlying CUdevice handle (for advanced use).
    pub fn cuDevice(self: *const Self) sys.CUdevice {
        return self.device;
    }

    /// Get the underlying CUcontext handle (for advanced use).
    pub fn cuCtx(self: *const Self) sys.CUcontext {
        return self.context;
    }

    // --- Context Operations ---

    /// Synchronize the context (block CPU until all work completes).
    pub fn synchronize(self: *const Self) DriverError!void {
        try self.bindToThread();
        try result.ctx.synchronize();
    }

    /// Get free device memory in bytes.
    pub fn freeMem(self: *const Self) DriverError!usize {
        try self.bindToThread();
        const info = try result.mem.getInfo();
        return info.free;
    }

    /// Get free and total device memory in bytes.
    pub fn memInfo(self: *const Self) DriverError!struct { free: usize, total: usize } {
        try self.bindToThread();
        const info = try result.mem.getInfo();
        return .{ .free = info.free, .total = info.total };
    }

    /// Get a context limit value (e.g., stack size, printf FIFO, malloc heap).
    pub fn getLimit(self: *const Self, limit: sys.CUlimit) DriverError!usize {
        try self.bindToThread();
        return try result.ctx.getLimit(limit);
    }

    /// Set a context limit value.
    pub fn setLimit(self: *const Self, limit: sys.CUlimit, value: usize) DriverError!void {
        try self.bindToThread();
        try result.ctx.setLimit(limit, value);
    }

    /// Get the L1/shared memory cache configuration.
    pub fn getCacheConfig(self: *const Self) DriverError!sys.CUfunc_cache {
        try self.bindToThread();
        return try result.ctx.getCacheConfig();
    }

    /// Set the L1/shared memory cache configuration.
    pub fn setCacheConfig(self: *const Self, config: sys.CUfunc_cache) DriverError!void {
        try self.bindToThread();
        try result.ctx.setCacheConfig(config);
    }

    /// Ensures calls to synchronize() block the calling thread.
    pub fn setBlockingSynchronize(self: *const Self) DriverError!void {
        try self.bindToThread();
        const flags = try result.ctx.getFlags();
        const CU_CTX_SCHED_BLOCKING_SYNC: u32 = 0x04;
        if (flags & CU_CTX_SCHED_BLOCKING_SYNC == 0) {
            try result.primaryCtx.setFlags(self.device, flags | CU_CTX_SCHED_BLOCKING_SYNC);
        }
    }

    // --- Stream Management ---

    /// Get the default stream for this context.
    pub fn defaultStream(self: *Self) *const CudaStream {
        if (self.default_stream == null) {
            self.default_stream = CudaStream{
                .stream = null, // null represents the default stream
                .ctx = self,
                .is_default = true,
            };
        }
        return &self.default_stream.?;
    }

    /// Create a new non-blocking stream.
    pub fn newStream(self: *const Self) DriverError!CudaStream {
        try self.bindToThread();
        const s = try result.stream.create(sys.CU_STREAM_NON_BLOCKING);
        return CudaStream{
            .stream = s,
            .ctx = self,
            .is_default = false,
        };
    }

    // --- Module Management ---

    /// Load a module from PTX data (null-terminated byte string).
    pub fn loadModule(self: *const Self, ptx: []const u8) DriverError!CudaModule {
        try self.bindToThread();

        // Ensure the PTX data is null-terminated
        const ptx_z = if (ptx.len > 0 and ptx[ptx.len - 1] == 0)
            ptx
        else blk: {
            const z = self.allocator.alloc(u8, ptx.len + 1) catch return DriverError.OutOfMemory;
            @memcpy(z[0..ptx.len], ptx);
            z[ptx.len] = 0;
            break :blk z;
        };
        defer if (ptx_z.ptr != ptx.ptr) self.allocator.free(ptx_z);

        const mod = try result.module.loadData(ptx_z.ptr);
        return CudaModule{
            .module = mod,
            .ctx = self,
        };
    }

    // --- Event Management ---

    /// Create a new event with the given flags.
    pub fn createEvent(self: *const Self, flags: u32) DriverError!CudaEvent {
        try self.bindToThread();
        const e = try result.event.create(flags);
        return CudaEvent{
            .event = e,
            .ctx = self,
        };
    }

    // --- Unified Memory ---

    /// Allocate unified memory accessible by both host and device.
    pub fn allocManaged(self: *const Self, comptime T: type, len: usize) DriverError!CudaSlice(T) {
        try self.bindToThread();
        const ptr = try result.unified.allocManaged(len * @sizeOf(T), sys.CU_MEM_ATTACH_GLOBAL);
        return CudaSlice(T){
            .ptr = ptr,
            .len = len,
        };
    }
};

// ============================================================================
// CudaGraph — Re-playable recorded GPU operation graph
// ============================================================================

pub const CudaGraph = struct {
    cu_graph: sys.CUgraph,
    cu_graph_exec: sys.CUgraphExec,
    ctx: *const CudaContext, // stable pointer (not a copy)
    cu_stream: sys.CUstream, // raw handle (value, not pointer)

    const Self = @This();

    /// Launch (replay) this recorded graph.
    pub fn launch(self: Self) DriverError!void {
        try self.ctx.bindToThread();
        try result.graph.launch(self.cu_graph_exec, self.cu_stream);
    }

    /// Destroy the graph and its executable.
    pub fn deinit(self: *Self) void {
        result.graph.execDestroy(self.cu_graph_exec) catch {};
        result.graph.destroy(self.cu_graph) catch {};
    }
};

// ============================================================================
// Tests
// ============================================================================
