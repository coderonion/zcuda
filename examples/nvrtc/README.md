# NVRTC — Runtime Compilation Examples

2 examples demonstrating JIT compilation of CUDA C++ kernels to PTX/CUBIN at runtime.
No pre-compilation step required — kernels are compiled on the fly from source strings.

## Build & Run

```bash
# No flags required — NVRTC is always enabled
zig build run-nvrtc-<name>

zig build run-nvrtc-jit_compile
zig build run-nvrtc-template_kernel
```

---

## Examples

| Example | File | Description |
|---------|------|-------------|
| `jit_compile` | [jit_compile.zig](jit_compile.zig) | Compile a simple CUDA C++ vector-add kernel at runtime and launch it |
| `template_kernel` | [template_kernel.zig](template_kernel.zig) | Multi-kernel pipeline with templated type specialization |

---

## Key API

```zig
const nvrtc = @import("zcuda").nvrtc;

const src =
    \\extern "C" __global__ void add1(float* arr, int n) {
    \\    int i = blockDim.x * blockIdx.x + threadIdx.x;
    \\    if (i < n) arr[i] += 1.0f;
    \\}
;

// Compile to PTX
const ptx = try nvrtc.compilePtx(allocator, src, .{
    .arch = "sm_86",
    .options = &.{"--use_fast_math"},
});
defer allocator.free(ptx);

// Load and launch
const module = try ctx.loadModule(ptx);
const func   = try module.getFunction("add1");
try stream.launch(func, config, .{ d_arr, @as(i32, n) });
```

### `CompileOptions` fields

| Field | Type | Description |
|-------|------|-------------|
| `arch` | `?[]const u8` | Target architecture, e.g. `"sm_86"` |
| `include_paths` | `[]const []const u8` | Extra `-I` paths |
| `defines` | `[]const []const u8` | `-D` preprocessor defines |
| `options` | `[]const []const u8` | Arbitrary nvrtc flags |
| `name_expressions` | `[]const []const u8` | Name expressions for lowered name lookup |

→ Full API reference: [`docs/nvrtc/README.md`](../../docs/nvrtc/README.md)
