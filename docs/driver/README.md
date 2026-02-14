# Driver Module

Device management, memory allocation, kernel launch, streams, events, and graphs.

**Import:** `const driver = @import("zcuda").driver;`

## CudaContext

Entry point for all CUDA operations. Manages the device, streams, memory, and modules.

### Creation & Lifecycle

```zig
fn new(ordinal: usize) !*CudaContext;    // Create context on device N
fn deinit(self: *const Self) void;       // Release context
fn bindToThread(self) !void;             // Bind context to current thread
```

### Device Info

```zig
fn deviceCount() !i32;                           // Number of CUDA devices
fn name(self) []const u8;                         // Device name
fn uuid(self) !CUuuid;                            // Device UUID
fn computeCapability(self) !struct{major, minor}; // SM version
fn totalMem(self) !usize;                         // Total memory (bytes)
fn attribute(self, attr) !i32;                     // Query device attribute
fn getOrdinal(self) usize;                         // Device ordinal index
```

### Memory Info & Limits

```zig
fn freeMem(self) !usize;                          // Free memory (bytes)
fn memInfo(self) !struct{free, total};             // Free/total memory
fn getLimit(self, limit) !usize;                   // Query context limit
fn setLimit(self, limit, value) !void;             // Set context limit
fn getCacheConfig(self) !CUfunc_cache;             // L1/shared preference
fn setCacheConfig(self, config) !void;             // Set L1/shared preference
fn setBlockingSynchronize(self) !void;             // Enable blocking sync
fn synchronize(self) !void;                        // Synchronize context
```

### Stream & Module Management

```zig
fn defaultStream(self) *const CudaStream;          // Default stream
fn newStream(self) !CudaStream;                    // Create non-blocking stream
fn loadModule(self, ptx) !CudaModule;              // Load PTX module
fn createEvent(self, flags) !CudaEvent;            // Create event
fn allocManaged(self, T, len) !CudaSlice(T);       // Unified memory
```

## CudaStream

Asynchronous execution stream for memory operations and kernel launches.

### Memory Operations

```zig
fn alloc(T, allocator, n) !CudaSlice(T);          // Allocate device memory
fn allocZeros(T, allocator, n) !CudaSlice(T);     // Allocate + zero-fill
fn cloneHtod(T, host_slice) !CudaSlice(T);        // Host → Device copy
fn memcpyHtod(T, dst, src) !void;                  // Copy host → device
fn memcpyDtoh(T, dst, src) !void;                  // Copy device → host
fn cloneDtoh(T, allocator, src) ![]T;              // Clone device → new host buf
fn memcpyDtoD(T, dst, src) !void;                  // Copy device → device
fn memcpyHtodAsync(T, dst, src) !void;             // Async host → device
fn memcpyDtohAsync(T, dst, src) !void;             // Async device → host
fn memcpyDtodAsync(T, dst, src) !void;             // Async device → device
```

### Kernel Launch

```zig
fn launch(func, config, args) !void;               // Launch kernel
```

### Synchronization & Events

```zig
fn synchronize(self) !void;                        // Wait for all operations
fn waitEvent(self, event) !void;                    // Wait for event
fn query(self) !bool;                              // Non-blocking completion check
fn createEvent(self, flags) !CudaEvent;            // Create event
fn recordEvent(self, event) !void;                 // Record event
```

### Unified Memory & Graph Capture

```zig
fn prefetchAsync(T, slice) !void;                  // Prefetch to device
fn beginCapture(self) !void;                       // Begin graph capture
fn endCapture(self) !?CudaGraph;                   // End capture → executable graph
fn captureStatus(self) !CUstreamCaptureStatus;     // Query capture status
```

## CudaSlice(T)

Typed, owning device memory (analogous to `Vec<T>` on GPU).

```zig
fn deinit(self) void;                              // Free device memory
fn slice(self, start, end) CudaView(T);            // Immutable sub-view
fn sliceMut(self, start, end) CudaViewMut(T);      // Mutable sub-view
fn devicePtr(self) DevicePtr(T);                    // Get typed device pointer
```

## CudaView(T) / CudaViewMut(T)

Non-owning views into device memory (analogous to `[]const T` / `[]T`).

```zig
fn devicePtr(self) DevicePtr(T);                    // Get typed device pointer
fn subView(self, start, end) Self;                  // Create sub-view
```

## CudaModule / CudaFunction

```zig
// CudaModule
fn deinit(self) void;                              // Unload module
fn getFunction(self, name) !CudaFunction;          // Get kernel by name

// CudaFunction
fn getAttribute(self, attrib) !i32;                // Query function attribute
```

## CudaEvent

```zig
fn deinit(self) void;                              // Destroy event
fn record(self, stream) !void;                     // Record on stream
fn synchronize(self) !void;                        // Wait for event
fn elapsedTime(start, end) !f32;                   // Milliseconds between events
fn query(self) !bool;                              // Non-blocking completion check
```

## CudaGraph

```zig
fn launch(self) !void;                             // Replay recorded graph
fn deinit(self) void;                              // Destroy graph
```

## Shared Types

```zig
const Dim3 = struct { x: u32 = 1, y: u32 = 1, z: u32 = 1 };

const LaunchConfig = struct {
    grid_dim: Dim3,
    block_dim: Dim3,
    shared_mem_bytes: u32,

    fn forNumElems(n: u32) LaunchConfig;            // Auto-configure for N elements
    fn forNumElemsCustom(n: u32, tpb: u32) LaunchConfig;
};

const DevicePtr = fn(T: type) struct { ptr: usize };
```

## Example

```zig
const cuda = @import("zcuda");

const ctx = try cuda.driver.CudaContext.new(0);
defer ctx.deinit();

const stream = ctx.defaultStream();
const data = try stream.cloneHtod(f32, &[_]f32{ 1.0, 2.0, 3.0 });
defer data.deinit();

var result: [3]f32 = undefined;
try stream.memcpyDtoh(f32, &result, data);
```
