# NVTX — Profiling Annotation Examples

1 example showing how to annotate GPU workloads for NVIDIA Nsight Systems / Nsight Compute.
Enable with `-Dnvtx=true`.

## Build & Run

```bash
zig build run-nvtx-profiling -Dnvtx=true
```

---

## Examples

| Example | File | Description |
|---------|------|-------------|
| `profiling` | [profiling.zig](profiling.zig) | Named range push/pop and point mark annotations visible in Nsight |

---

## Key API

```zig
const nvtx = @import("zcuda").nvtx;

// Named range (push/pop)
nvtx.rangePush("H2D Transfer");
try stream.memcpyHtoD(f32, d_buf, h_buf);
nvtx.rangePop();

// RAII scoped range (auto-pops on scope exit)
{
    var range = nvtx.ScopedRange.init("Compute");
    defer range.deinit();
    // ... computation ...
}

// Point marker
nvtx.mark("Kernel launch");
try stream.launch(func, config, args);

// Domain-isolated ranges (per-module namespacing)
const domain = nvtx.Domain.create("MyApp");
defer domain.deinit();
domain.rangePush("Phase 1");
```

## Using with Nsight Systems

```bash
# Profile with nsys
nsys profile --trace=cuda,nvtx zig-out/bin/my_app

# Open the resulting .nsys-rep in Nsight Systems UI
# NVTX ranges appear as colored bands in the timeline
```

→ Full API reference: [`docs/nvtx/README.md`](../../docs/nvtx/README.md)
