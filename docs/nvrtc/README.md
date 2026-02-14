# NVRTC Module

Runtime compilation of CUDA C++ source code to PTX or CUBIN.

**Import:** `const nvrtc = @import("zcuda").nvrtc;`

## Compilation Functions

```zig
fn compilePtx(allocator, src) ![]u8;                        // Compile to PTX (default options)
fn compilePtxWithOptions(allocator, src, options) ![]u8;    // Compile to PTX with options
fn compileCubin(allocator, src) ![]u8;                      // Compile to CUBIN (default options)
fn compileCubinWithOptions(allocator, src, options) ![]u8;  // Compile to CUBIN with options
fn getVersion() !NvrtcVersion;                              // Get NVRTC version
```

## CompileOptions

```zig
const CompileOptions = struct {
    arch: ?[]const u8 = null,           // Target architecture (e.g., "compute_89")
    debug: bool = false,                 // Device debug info
    lineinfo: bool = false,              // Line number info
    dopt: bool = false,                  // Device code optimization
    ftz: bool = false,                   // Flush-to-zero for denormals
    maxrregcount: ?u32 = null,           // Max registers per thread
    extra_options: []const []const u8,   // Additional compiler flags
};
```

## Example

```zig
const cuda = @import("zcuda");

// Simple compilation
const ptx = try cuda.nvrtc.compilePtx(allocator,
    \\extern "C" __global__ void add1(float *data, int n) {
    \\    int i = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (i < n) data[i] += 1.0f;
    \\}
);
defer allocator.free(ptx);

// Compilation with options
const ptx2 = try cuda.nvrtc.compilePtxWithOptions(allocator, src, .{
    .arch = "compute_89",
    .dopt = true,
    .maxrregcount = 32,
});
defer allocator.free(ptx2);
```
