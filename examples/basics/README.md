# Basics — CUDA Driver API Examples

16 examples covering core GPU programming: contexts, streams, memory, events, and kernel launch.
All examples use the **safe API layer** exclusively (`CudaContext`, `CudaStream`, `CudaSlice`).

## Build & Run

```bash
# Run any example (no flags required — driver + nvrtc always enabled)
zig build run-basics-<name>

# Examples:
zig build run-basics-vector_add
zig build run-basics-device_info
```

---

## Examples

### Device & Context Management

| Example | File | Description |
|---------|------|-------------|
| `device_info` | [device_info.zig](device_info.zig) | Query GPU name, compute capability, memory, and device attributes |
| `multi_device_query` | [multi_device_query.zig](multi_device_query.zig) | Enumerate and query all available CUDA devices |
| `context_lifecycle` | [context_lifecycle.zig](context_lifecycle.zig) | Context creation, binding, switching, and destruction |
| `kernel_attributes` | [kernel_attributes.zig](kernel_attributes.zig) | Query kernel register count, shared memory, and occupancy |

### Memory Management

| Example | File | Description |
|---------|------|-------------|
| `alloc_patterns` | [alloc_patterns.zig](alloc_patterns.zig) | Device, host, pinned (page-locked), and unified memory allocation |
| `pinned_memory` | [pinned_memory.zig](pinned_memory.zig) | Pinned host memory for faster H2D/D2H bandwidth |
| `unified_memory` | [unified_memory.zig](unified_memory.zig) | Unified memory (UM) with automatic migration |
| `memset_patterns` | [memset_patterns.zig](memset_patterns.zig) | Device memset patterns and zero-initialization |
| `constant_memory` | [constant_memory.zig](constant_memory.zig) | GPU constant memory for read-only broadcast data (polynomial eval) |

### Transfers & Copies

| Example | File | Description |
|---------|------|-------------|
| `async_memcpy` | [async_memcpy.zig](async_memcpy.zig) | Asynchronous H2D/D2H transfers overlapping with computation |
| `dtod_copy_chain` | [dtod_copy_chain.zig](dtod_copy_chain.zig) | Device-to-device chained copy pipeline |

### Streams & Events

| Example | File | Description |
|---------|------|-------------|
| `streams` | [streams.zig](streams.zig) | Multi-stream concurrent kernel execution |
| `event_timing` | [event_timing.zig](event_timing.zig) | Event-based GPU timing and memory bandwidth measurement |

### Kernel Launch

| Example | File | Description |
|---------|------|-------------|
| `vector_add` | [vector_add.zig](vector_add.zig) | Vector addition via NVRTC JIT — classic "Hello, GPU" |
| `struct_kernel` | [struct_kernel.zig](struct_kernel.zig) | Pass a Zig `extern struct` as kernel argument |

### Multi-GPU

| Example | File | Description |
|---------|------|-------------|
| `peer_to_peer` | [peer_to_peer.zig](peer_to_peer.zig) | Multi-GPU peer access and direct D2D transfers |

---

## Key APIs Demonstrated

```zig
const driver = @import("zcuda").driver;

// Context
const ctx = try driver.CudaContext.new(0);
defer ctx.deinit();

// Memory
const stream = ctx.defaultStream();
const d_buf = try stream.alloc(f32, n);     // device alloc
const d_in  = try stream.cloneHtoD(h_data); // host → device

// Kernel launch (via NVRTC PTX)
const module = try ctx.loadModule(ptx_source);
const func   = try module.getFunction("my_kernel");
try stream.launch(func, .{ .grid = dim3(blocks), .block = dim3(threads) }, .{ d_out, d_in, n });

// Events
const ev = try ctx.createEvent(.{});
try ev.record(stream);
try stream.waitEvent(ev);
const ms = try ev.elapsedTime(ev_end);
```

→ Full API reference: [`docs/driver/README.md`](../../docs/driver/README.md)
