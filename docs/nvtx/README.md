# NVTX Module

Profiling annotations for NVIDIA Nsight Systems and Nsight Compute.

**Import:** `const nvtx = @import("zcuda").nvtx;`
**Enable:** `-Dnvtx=true`

## Global Functions

```zig
fn rangePush(name: [*:0]const u8) void;     // Push named range marker
fn rangePop() void;                         // Pop top range marker
fn mark(name: [*:0]const u8) void;          // Place marker at current time
```

## ScopedRange

RAII-style scoped range — automatically pops when the scope exits.

```zig
const ScopedRange = struct {
    fn init(name: [*:0]const u8) ScopedRange;  // Push range
    fn deinit(self) void;                       // Pop range (via defer)
};
```

## Domain

Named domain for isolating profiling markers per module/library.

```zig
const Domain = struct {
    fn create(name: [*:0]const u8) Domain;     // Create domain
    fn destroy(self) void;                      // Destroy domain
};
```

## Example

```zig
const cuda = @import("zcuda");

// Simple range markers
cuda.nvtx.rangePush("data_loading");
// ... load data ...
cuda.nvtx.rangePop();

// Scoped range (auto-pops via defer)
{
    const range = cuda.nvtx.ScopedRange.init("kernel_execution");
    defer range.deinit();

    try stream.launch(kernel, config, args);
    try stream.synchronize();
}

// Point marker
cuda.nvtx.mark("checkpoint_reached");
```
