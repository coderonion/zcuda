# zcuda Kernel DSL — API Reference

> **Module:** `const cuda = @import("zcuda_kernel");`
> **Target:** `nvptx64-cuda-none`
> **Default SM:** `sm_80` — override with `-Dgpu-arch=sm_XX`

Pure-Zig GPU kernel authoring. No `nvcc`, no CUDA C++.
All functions below are inline and zero-overhead.

---

## Quick Start

```zig
const cuda = @import("zcuda_kernel");

export fn saxpy(n: u32, alpha: f32, x: [*]f32, y: [*]f32) callconv(.Kernel) void {
    var iter = cuda.types.gridStrideLoop(n);
    while (iter.next()) |i| {
        y[i] = alpha * x[i] + y[i];
    }
}
```

Build:
```bash
zig build compile-kernels -Dgpu-arch=sm_86
```

---

## Module Map

| Import path | Contents |
|-------------|----------|
| `cuda` (root) | All intrinsics, `SM`, `Dim3`, `warpSize`, `FULL_MASK` |
| `cuda.types` | `globalThreadIdx`, `gridStride`, `gridStrideLoop`, `DeviceSlice(T)`, `DevicePtr(T)` |
| `cuda.shared_mem` | `SharedArray(T,N)`, `dynamicShared(T)`, `dynamicSharedBytes()`, cooperative utilities |
| `cuda.shared` | `Vec2/3/4`, `Int2/3`, `Matrix3x3/4x4`, `LaunchConfig` (host-device shared) |
| `cuda.debug` | `assertf`, `assertInBounds`, `safeGet`, `ErrorFlag`, `printf`, `CycleTimer`, `__trap` |
| `cuda.arch` | `SmVersion` enum, `requireSM`, `SmVersion.atLeast`, `SmVersion.codename` |
| `cuda.tensor_core` | WMMA, MMA PTX, cp.async, wgmma, TMA, cluster, tcgen05 |

---

## Thread Indexing

```zig
cuda.threadIdx() -> Dim3   // %tid.{x,y,z}
cuda.blockIdx()  -> Dim3   // %ctaid.{x,y,z}
cuda.blockDim()  -> Dim3   // %ntid.{x,y,z}
cuda.gridDim()   -> Dim3   // %nctaid.{x,y,z}
cuda.warpSize    -> u32    // constant 32
cuda.FULL_MASK   -> u32    // 0xFFFFFFFF
cuda.SM          -> arch.SmVersion  // compile-time target SM
```

`Dim3` is `extern struct { x: u32, y: u32, z: u32 }`.

**Convenience (in `cuda.types`):**
```zig
cuda.types.globalThreadIdx() -> u32  // blockIdx.x * blockDim.x + threadIdx.x
cuda.types.gridStride()      -> u32  // blockDim.x * gridDim.x
```

---

## Synchronization

```zig
cuda.__syncthreads()                          // bar.sync 0
cuda.__syncthreads_count(pred: bool) -> u32   // bar.red.popc — count true threads
cuda.__syncthreads_and(pred: bool)   -> u32   // bar.red.and  — non-zero if ALL true
cuda.__syncthreads_or(pred: bool)    -> u32   // bar.red.or   — non-zero if ANY true
cuda.__syncwarp(mask: u32)                    // bar.warp.sync (sm_70+)
cuda.__threadfence()                          // membar.gl  — global scope
cuda.__threadfence_block()                    // membar.cta — block scope
cuda.__threadfence_system()                   // membar.sys — system scope (CPU visible)
```

---

## Atomic Operations

All atomics return the **old** value.

| Function | Types | PTX |
|----------|-------|-----|
| `atomicAdd(ptr, val)` | `f32`, `u32`, `i32` | `atom.global.add` |
| `atomicAdd_f64(ptr, val)` | `f64` | `atom.global.add.f64` (sm_60+) |
| `atomicSub(ptr, val)` | `u32`, `i32`, `f32` | via `atomicAdd(-val)` |
| `atomicMax(ptr, val)` | `u32`, `i32` | `atom.global.max` |
| `atomicMin(ptr, val)` | `u32`, `i32` | `atom.global.min` |
| `atomicCAS(ptr, compare, val)` | `u32` | `atom.global.cas.b32` |
| `atomicExch(ptr, val)` | `u32`, `f32` | `atom.global.exch.b32` |
| `atomicAnd(ptr, val)` | `u32` | `atom.global.and.b32` |
| `atomicOr(ptr, val)` | `u32` | `atom.global.or.b32` |
| `atomicXor(ptr, val)` | `u32` | `atom.global.xor.b32` |
| `atomicInc(ptr, val)` | `u32` | `atom.global.inc.u32` — wraps to 0 when `*ptr >= val` |
| `atomicDec(ptr, val)` | `u32` | `atom.global.dec.u32` — wraps to val when `*ptr == 0` |

`ptr` accepts `anytype` — supports `*f32`, `*u32`, `*i32`.

---

## Warp Primitives

### Shuffle

```zig
cuda.__shfl_sync(mask, val, src_lane, width) -> u32      // shfl.sync.idx  — broadcast from lane
cuda.__shfl_down_sync(mask, val, delta, width) -> u32    // shfl.sync.down — reduce pattern
cuda.__shfl_up_sync(mask, val, delta, width) -> u32      // shfl.sync.up   — scan pattern
cuda.__shfl_xor_sync(mask, val, lane_mask, width) -> u32 // shfl.sync.bfly — butterfly pattern
```

All four work on `u32`. For `f32`, bitcast: `@bitCast(__shfl_sync(mask, @bitCast(val), src, w))`.

### Vote

```zig
cuda.__ballot_sync(mask, pred: bool) -> u32   // vote.sync.ballot — 1 bit per thread
cuda.__all_sync(mask, pred: bool) -> bool     // vote.sync.all    — all threads true?
cuda.__any_sync(mask, pred: bool) -> bool     // vote.sync.any    — any thread true?
cuda.__activemask() -> u32                    // activemask.b32   — currently active threads
```

### Match (sm_70+)

```zig
cuda.__match_any_sync(mask, val: u32) -> u32
// Returns mask of threads with the same `val`

cuda.__match_all_sync(mask, val: u32) -> struct { mask: u32, pred: bool }
// Returns `mask` if all threads agree, 0 otherwise; pred indicates agreement
```

### Warp-Level Reduce (sm_80+)

All SM-guarded at comptime — compile error if target SM < sm_80.

```zig
cuda.__reduce_add_sync(mask, val: u32) -> u32  // redux.sync.add.u32
cuda.__reduce_min_sync(mask, val: u32) -> u32  // redux.sync.min.u32
cuda.__reduce_max_sync(mask, val: u32) -> u32  // redux.sync.max.u32
cuda.__reduce_and_sync(mask, val: u32) -> u32  // redux.sync.and.b32
cuda.__reduce_or_sync(mask, val: u32) -> u32   // redux.sync.or.b32
cuda.__reduce_xor_sync(mask, val: u32) -> u32  // redux.sync.xor.b32
```

---

## Fast Math

| Function | PTX | Notes |
|----------|-----|-------|
| `rsqrtf(x: f32) -> f32` | `rsqrt.approx.f32` | Fast reciprocal sqrt |
| `sqrtf(x: f32) -> f32` | `sqrt.rn.f32` | Correctly-rounded sqrt |
| `fabsf(x: f32) -> f32` | `abs.f32` | Absolute value |
| `fminf(a, b: f32) -> f32` | `min.f32` | Minimum |
| `fmaxf(a, b: f32) -> f32` | `max.f32` | Maximum |
| `__sinf(x: f32) -> f32` | `sin.approx.f32` | |
| `__cosf(x: f32) -> f32` | `cos.approx.f32` | |
| `__tanf(x: f32) -> f32` | `sin/cos/div` | sin(x)/cos(x) |
| `__expf(x: f32) -> f32` | via `ex2.approx` | exp2(x·log₂e) |
| `__exp2f(x: f32) -> f32` | `ex2.approx.f32` | |
| `__logf(x: f32) -> f32` | via `lg2.approx` | log2(x)·ln2 |
| `__log2f(x: f32) -> f32` | `lg2.approx.f32` | |
| `__log10f(x: f32) -> f32` | via `lg2.approx` | log2(x)·log10(2) |
| `__powf(x, y: f32) -> f32` | `lg2 + ex2` | exp2(y·log₂x) |
| `__fmaf_rn(a, b, c: f32) -> f32` | `fma.rn.f32` | Fused multiply-add |
| `__fdividef(a, b: f32) -> f32` | `div.approx.f32` | Fast approximate division |
| `__saturatef(x: f32) -> f32` | `min(max(x,0),1)` | Clamp to [0, 1] |

---

## Integer Intrinsics

```zig
cuda.__clz(x: u32)        -> u32   // clz.b32   — count leading zeros
cuda.__clzll(x: u64)      -> u32   // clz.b64   — 64-bit CLZ
cuda.__popc(x: u32)       -> u32   // popc.b32  — population count (popcount)
cuda.__popcll(x: u64)     -> u32   // popc.b64  — 64-bit popcount
cuda.__brev(x: u32)       -> u32   // brev.b32  — bit reverse
cuda.__brevll(x: u64)     -> u64   // brev.b64  — 64-bit bit reverse
cuda.__ffs(x: u32)        -> u32   // bfind + brev — find first set (1-indexed, 0 if none)
cuda.__byte_perm(a, b, s: u32) -> u32  // prmt.b32 — select 4 bytes from {b,a} by selector s
```

---

## Dot Product (sm_75+)

```zig
cuda.__dp4a(a, b, c: u32) -> u32      // dp4a.u32.u32 — 4×u8 dot product + u32 accumulate
cuda.__dp4a_s32(a, b, c: i32) -> i32  // dp4a.s32.s32 — signed variant
cuda.__dp2a_lo(a, b, c: u32) -> u32   // dp2a.lo.u32  — 2×u16 dot (low halfwords)
cuda.__dp2a_hi(a, b, c: u32) -> u32   // dp2a.hi.u32  — 2×u16 dot (high halfwords)
```

---

## Cache Load/Store Hints

All operate on `f32` (or `u32` for `__ldg_u32`).

```zig
// Read-only cache (L1 texture cache)
cuda.__ldg(ptr: *const f32) -> f32      // ld.global.nc.f32  — __ldg()
cuda.__ldg_u32(ptr: *const u32) -> u32  // ld.global.nc.u32

// Load with cache policy
cuda.__ldca(ptr: *const f32) -> f32     // ld.global.ca.f32  — cache-all
cuda.__ldcs(ptr: *const f32) -> f32     // ld.global.cs.f32  — cache-streaming
cuda.__ldcg(ptr: *const f32) -> f32     // ld.global.cg.f32  — cache-global

// Store with cache policy
cuda.__stcg(ptr: *f32, val: f32)        // st.global.cg.f32  — cache-global store
cuda.__stcs(ptr: *f32, val: f32)        // st.global.cs.f32  — streaming store
cuda.__stwb(ptr: *f32, val: f32)        // st.global.wb.f32  — write-back (bypass L1)
```

---

## Address Space Predicates

```zig
cuda.__isGlobal(ptr: *const anyopaque) -> bool    // isspacep.global
cuda.__isShared(ptr: *const anyopaque) -> bool    // isspacep.shared
cuda.__isConstant(ptr: *const anyopaque) -> bool  // isspacep.const
cuda.__isLocal(ptr: *const anyopaque) -> bool     // isspacep.local
```

---

## Type Conversion Intrinsics

```zig
cuda.__float2int_rn(x: f32) -> i32    // cvt.rni.s32.f32 — round-to-nearest
cuda.__float2int_rz(x: f32) -> i32    // cvt.rzi.s32.f32 — truncate
cuda.__float2uint_rn(x: f32) -> u32   // cvt.rni.u32.f32
cuda.__float2uint_rz(x: f32) -> u32   // cvt.rzi.u32.f32
cuda.__int2float_rn(x: i32) -> f32    // cvt.rn.f32.s32
cuda.__uint2float_rn(x: u32) -> f32   // cvt.rn.f32.u32

// Bitcast (reinterpret bits, not numeric conversion)
cuda.__float_as_int(x: f32) -> i32    // @bitCast(x)
cuda.__int_as_float(x: i32) -> f32    // @bitCast(x)
cuda.__float_as_uint(x: f32) -> u32   // @bitCast(x)
cuda.__uint_as_float(x: u32) -> f32   // @bitCast(x)

// f64 component extraction
cuda.__double2hiint(x: f64) -> i32    // high 32 bits of f64
cuda.__double2loint(x: f64) -> i32    // low 32 bits of f64
cuda.__hiloint2double(hi, lo: i32) -> f64  // pack two i32 into f64
```

---

## Clock & Timer

```zig
cuda.clock()       -> u32   // %clock   — SM cycle counter (low 32 bits)
cuda.clock64()     -> u64   // %clock64 — full 64-bit cycle counter
cuda.globaltimer() -> u64   // %globaltimer — nanosecond timer
```

---

## Misc

```zig
cuda.__nanosleep(ns: u32)    // nanosleep.u32 (sm_70+) — suspend ~ns nanoseconds
```

---

## `cuda.types` — Device-Side Data Structures

### Grid-Stride Loop

```zig
// Standard pattern (replaces 1D: i = blockIdx.x * blockDim.x + threadIdx.x)
var iter = cuda.types.gridStrideLoop(n);
while (iter.next()) |i| {
    output[i] = process(input[i]);
}

// Iterator struct (manually if needed)
const GridStrideIterator = cuda.types.GridStrideIterator;
iter.reset();  // reset to initial position
```

### `DeviceSlice(T)` — Typed Device Array

```zig
// In kernel signature
export fn myKernel(data: types.DeviceSlice(f32), n: u32) callconv(.Kernel) void {
    const i = cuda.types.globalThreadIdx();
    const val = data.get(i);        // data.ptr[i]
    data.set(i, val * 2.0);         // data.ptr[i] = ...
    _ = data.len;                   // number of elements
}

// Methods
DeviceSlice(T).get(self, idx: u32) -> T
DeviceSlice(T).set(self, idx: u32, val: T)
DeviceSlice(T).init(ptr: [*]T, len: u32) -> Self
// Fields: .ptr: [*]T, .len: u32
```

> `DeviceSlice(T)` is an `extern struct` — safe to pass from host via `stream.launch()`.

### `DevicePtr(T)` — Single-Value Output Pointer

```zig
// For reduction output, error flags, etc.
export fn reduce(data: [*]f32, n: u32, result: types.DevicePtr(f32)) callconv(.Kernel) void {
    // ... compute sum ...
    _ = result.atomicAdd(partial_sum);  // atomic add, returns old value
}

DevicePtr(T).load(self) -> T          // read
DevicePtr(T).store(self, val: T)      // write
DevicePtr(T).atomicAdd(self, val: T) -> T   // atomicAdd on .ptr
// Field: .ptr: *T
```

---

## `cuda.shared_mem` — Shared Memory

### Static Shared Memory

```zig
const smem = cuda.shared_mem;

// 256-element f32 tile — addrspace(3) → PTX .shared
const tile = smem.SharedArray(f32, 256);
const p = tile.ptr();         // [*]f32
const s = tile.slice();       // []f32 = p[0..256]
_ = tile.len();               // 256
_ = tile.sizeBytes();         // 256 * 4 = 1024

p[cuda.threadIdx().x] = some_value;
cuda.__syncthreads();
const loaded = p[cuda.threadIdx().x ^ 1];
```

> **Warning:** Two `SharedArray(f32, N)` with same `(T, N)` share storage.
> Use different sizes or a combined array + manual split for independent tiles.

### Dynamic Shared Memory

Size set at launch via `LaunchConfig.shared_mem_bytes` / `stream.launch()`.

```zig
// Single array
const dyn = smem.dynamicShared(f32);  // [*]f32
dyn[cuda.threadIdx().x] = val;

// Multiple arrays in same region
const base = smem.dynamicSharedBytes();  // [*]u8
const arr_a: [*]f32 = @ptrCast(@alignCast(base));
const arr_b: [*]f32 = @ptrCast(@alignCast(base + 1024));
```

### Cooperative Utilities

All require `__syncthreads()` after (except `reduceSum` which calls it internally):

```zig
smem.clearShared(f32, sptr: [*]f32, num_elements: u32)         // zero-fill cooperatively
smem.loadToShared(f32, dst: [*]f32, src: [*]const f32, n: u32) // global → shared
smem.storeFromShared(f32, dst: [*]f32, src: [*]const f32, n: u32) // shared → global
smem.reduceSum(f32, sdata: [*]f32, tid: u32, n: u32)           // tree reduction in smem
```

`n` must be a power of 2 for `reduceSum`. All threads in the block participate.

---

## `cuda.shared` — Host-Device Shared Types

All types use `extern struct` — identical layout on CPU and GPU.

### Vector Types

| Type | Fields | Compatible with |
|------|--------|-----------------|
| `Vec2` | `x, y: f32` | CUDA `float2` |
| `Vec3` | `x, y, z: f32` | CUDA `float3` |
| `Vec4` | `x, y, z, w: f32` | CUDA `float4` |
| `Int2` | `x, y: i32` | CUDA `int2` |
| `Int3` | `x, y, z: i32` | CUDA `int3` / `dim3` |

**Vec2 methods:** `init`, `add`, `scale`, `dot`
**Vec3 methods:** `init`, `add`, `sub`, `scale`, `dot`, `cross`
**Vec4 methods:** `init`, `add`, `dot`
**Int2/Int3:** `init`

### Matrix Types

```zig
// 3×3 row-major float matrix
const m = cuda.shared.Matrix3x3.identity();
const v = m.get(row, col);       // f32
m.set(row, col, val);            // void

// 4×4 row-major float matrix (for transforms)
const m4 = cuda.shared.Matrix4x4.identity();
// .data: [16]f32, row-major
```

### `LaunchConfig`

Host-side kernel launch parameters. Pass to `stream.launch()` / `ctx.launch()`.

```zig
// 1D launch
const cfg = cuda.shared.LaunchConfig.init1D(grid_size, block_size);

// 2D launch
const cfg = cuda.shared.LaunchConfig.init2D(gx, gy, bx, by);

// Auto-compute grid size from element count
const cfg = cuda.shared.LaunchConfig.forElementCount(n, block_size);
// → grid_size = ceil(n / block_size)

// Fields:
cfg.grid_dim_x = 1;   cfg.grid_dim_y = 1;   cfg.grid_dim_z = 1;
cfg.block_dim_x = 256; cfg.block_dim_y = 1;  cfg.block_dim_z = 1;
cfg.shared_mem_bytes = 0;   // set for dynamic shared memory
```

---

## `cuda.debug` — Device-Side Debugging

```zig
// Assertions
cuda.debug.assertf(condition: bool)           // trap if false
cuda.debug.assertInBounds(idx: u32, bound: u32) // trap if idx >= bound
cuda.debug.safeGet(ptr: [*]const f32, idx, len: u32, default: f32) -> f32  // OOB-safe read

// Halt / breakpoint
cuda.debug.__trap()    // noreturn — halt warp, host gets CUDA error
cuda.debug.__brkpt()   // debugger breakpoint

// GPU-side printf (backed by CUDA vprintf)
cuda.debug.printf("Thread %u val %f\n", .{ tid, val });
// f32 is promoted to f64 automatically (CUDA convention)
```

### `ErrorFlag` — Device-to-Host Error Reporting

Allocate on device, write from kernel, copy to host after launch.

```zig
// Error codes
ErrorFlag.NO_ERROR        = 0
ErrorFlag.OUT_OF_BOUNDS   = 1
ErrorFlag.NAN_DETECTED    = 2
ErrorFlag.INF_DETECTED    = 3
ErrorFlag.ASSERTION_FAILED = 4
ErrorFlag.CUSTOM_ERROR    = 0x100  // user-defined start

// Usage in kernel
export fn myKernel(data: [*]f32, n: u32, err: *debug.ErrorFlag) callconv(.Kernel) void {
    const i = cuda.types.globalThreadIdx();
    if (i >= n) {
        cuda.debug.setError(err, debug.ErrorFlag.OUT_OF_BOUNDS);
        return;
    }
    cuda.debug.checkNaN(data[i], err);  // auto-sets NAN_DETECTED
}

cuda.debug.setError(flag: *ErrorFlag, code: u32)   // atomicCAS — first error wins
cuda.debug.checkNaN(val: f32, flag: *ErrorFlag)    // set NAN_DETECTED if val is NaN
```

### `CycleTimer` — Cycle-Level Profiling

```zig
const timer = cuda.debug.CycleTimer.begin();
// ... work to profile ...
const cycles = timer.elapsed();   // u32, wrapping subtraction
```

For longer intervals use raw `cuda.clock64()` or `cuda.globaltimer()`.

---

## `cuda.arch` — Architecture Guards

```zig
// Comptime SM guard — emits clear compile error if target doesn't meet requirement
cuda.arch.requireSM(cuda.SM, .sm_80, "myFeature()");
// error: myFeature() requires sm_80+, but target is sm_70

// SmVersion enum
pub const SmVersion = enum(u32) {
    sm_52 = 52,  // Maxwell
    sm_60 = 60,  // Pascal
    sm_70 = 70,  // Volta
    sm_75 = 75,  // Turing
    sm_80 = 80,  // Ampere
    sm_86 = 86,  // Ampere (consumer)
    sm_89 = 89,  // Ada Lovelace
    sm_90 = 90,  // Hopper
    sm_100 = 100, // Blackwell
    _,           // forward-compatible
};

// Methods
cuda.SM.atLeast(.sm_80)          // -> bool, runtime check
cuda.SM.asInt()                  // -> u32
cuda.SM.codename()               // -> []const u8, e.g. "Volta"
```

---

## `cuda.tensor_core` — Tensor Core Operations

> All Tensor Core functions are SM-guarded at comptime.

### WMMA — Warp Matrix Multiply-Accumulate (sm_70+)

Uses NVIDIA's higher-level WMMA API. Each warp holds a `m16n16k16` tile.

```zig
const tc = cuda.tensor_core;

// Fragment types (arrays of register values)
const a = tc.wmma_load_a_f16(ptr_a, stride);   // WmmaFragA_f16
const b = tc.wmma_load_b_f16(ptr_b, stride);   // WmmaFragB_f16
const c = tc.wmma_load_c_f32(ptr_c, stride);   // WmmaFragC_f32

const d = tc.wmma_mma_f16_f32(a, b, c);        // D = A*B + C (f16 in, f32 acc)
tc.wmma_store_d_f32(ptr_d, d, stride);

// Integer variants (sm_75+)
const d_i = tc.wmma_mma_s8_s32(a_s8, b_s8, c_s32);  // m8n8k16 s8→s32
const d_u = tc.wmma_mma_u8_s32(a, b, c);             // u8→s32
const d_s4 = tc.wmma_mma_s4_s32(a, b, c);            // m8n8k32 s4→s32
const d_b1 = tc.wmma_mma_b1_s32(a, b, c);            // m8n8k128 b1→s32
```

### MMA PTX — Inline PTX Tensor Core (sm_80+)

Direct `mma.sync` PTX instructions. More control, less abstraction.

```zig
tc.mma_f16_f32(a, b, c)     // m16n8k16 f16 → f32 accumulate
tc.mma_bf16_f32(a, b, c)    // m16n8k16 bf16 → f32
tc.mma_tf32_f32(a, b, c)    // m16n8k8  tf32 → f32
tc.mma_f64_f64(a, b, c)     // m8n8k4   f64 → f64
tc.mma_s8_s32(a, b, c)      // m16n8k16 s8 → s32

// FP8 variants (sm_89+ / Ada Lovelace)
tc.mma_e4m3_f32(a, b, c)        // e4m3 FP8 → f32
tc.mma_e5m2_f32(a, b, c)        // e5m2 FP8 → f32
tc.mma_e4m3_e5m2_f32(a, b, c)   // mixed e4m3×e5m2 → f32
```

### cp.async — Async Data Pipeline (sm_80+)

```zig
tc.memcpy_async(dst_shared, src_global, 16)  // async 16B global→shared copy
tc.cp_async_commit_group()                   // end of a copy group
tc.cp_async_wait_group(0)                    // wait for ≤0 pending groups
tc.cp_async_wait_all()                       // wait for all pending copies
```

### wgmma — Warp Group MMA (sm_90+ / Hopper, 128-thread warpgroup)

```zig
tc.wgmma_fence();
const d = tc.wgmma_f16_f32(desc_a, desc_b);   // m64n16k16 — desc_a/desc_b are matrix descriptors
// Variants:
// tc.wgmma_bf16_f32, tc.wgmma_tf32_f32
// tc.wgmma_e4m3_f32, tc.wgmma_e5m2_f32
tc.wgmma_commit_group();
tc.wgmma_wait_group(0);
```

### TMA — Tensor Memory Accelerator (sm_90+ / Hopper)

```zig
tc.tma_load(smem_ptr, desc, coord0, coord1)   // global → shared async
tc.tma_store(desc, smem_ptr, coord0, coord1)  // shared → global async
tc.bulk_copy_g2s(dst, src, size)              // bulk copy global → shared
tc.bulk_commit_group()
tc.bulk_wait_group(0)
```

### Cluster — Cross-SM Cooperation (sm_90+ / Hopper)

```zig
tc.cluster_sync()                               // barrier across all blocks in cluster
const cdim = tc.clusterDim()                    // Dim3 — cluster dimensions
const cidx = tc.clusterIdx()                    // Dim3 — this block's position in cluster
const remote = tc.map_shared_cluster(ptr, rank) // distributed smem pointer
const val = tc.dsmem_load(remote_addr)          // load from remote block's shared mem
```

### tcgen05 — 5th Gen Tensor Core (sm_100+ / Blackwell)

```zig
// MMA: fp4 / fp6 / fp8 / fp16 / bf16 / tf32
const d = tc.tcgen05_mma_fp4(desc_a, desc_b)  // matrix multiply with tcgen05
// Variants: tcgen05_mma_fp6, tcgen05_mma_fp8, tcgen05_mma_f16, tcgen05_mma_bf16

// Tensor Memory management
const addr = tc.tcgen05_alloc(size)   // allocate tensor memory
const data = tc.tcgen05_ld(addr)      // load
tc.tcgen05_st(addr, data)             // store
tc.tcgen05_cp(dst, src)               // copy
tc.tcgen05_fence()                    // ordering fence
tc.tcgen05_wait()                     // wait for completion
tc.tcgen05_dealloc(addr)              // free
```

---

## `cuda_bridge` — Type-Safe Kernel Bridge (Way 5)

Located in `src/kernel/bridge_gen.zig`. Generates a type-safe host-side struct for loading and calling PTX kernels.

### Setup in `build.zig`

```zig
const zcuda_bridge = b.dependency("zcuda", .{}).module("zcuda_bridge");

const wf = b.addWriteFiles();
const bridge_src = wf.add("bridge_my_kernel.zig",
    \\pub usingnamespace @import("zcuda_bridge").init(.{
    \\    .name = "my_kernel",
    \\    .ptx_path = "zig-out/bin/kernel/my_kernel.ptx",
    \\    .fn_names = &.{ "vectorAdd", "reduce" },
    \\});
);
const bridge_mod = b.addModule("bridge_my_kernel", .{ .root_source_file = bridge_src });
bridge_mod.addImport("zcuda_bridge", zcuda_bridge);
bridge_mod.addImport("zcuda", zcuda_mod);
exe.root_module.addImport("bridge_my_kernel", bridge_mod);
```

### Usage in Application

```zig
const kernel = @import("bridge_my_kernel");

const module = try kernel.load(ctx, allocator);   // disk or embedded PTX
defer module.deinit();

const func = try kernel.getFunction(module, .vectorAdd);  // compile-time checked!
try stream.launch(func, grid, block, .{ d_input, d_output, n });

// Escape hatch (runtime string)
const f2 = try kernel.getFunctionByName(module, "vectorAdd");

// Embedded PTX (no file I/O at runtime)
const func2 = try kernel.loadFromPtx(ctx, @embedFile("my_kernel.ptx"));
```

**`kernel.Function`** is a compile-time enum of all function names — typos become build errors.

### `Config` struct

| Field | Type | Description |
|-------|------|-------------|
| `name` | `[]const u8` | Kernel name for logging |
| `ptx_path` | `[]const u8` | Path to `.ptx` file (disk mode) |
| `fn_names` | `[]const []const u8` | Exported function names |
| `source_path` | `?[]const u8` | Optional source file path (docs) |
| `ptx_data` | `?[]const u8` | Embedded PTX data (embedded mode) |

---

## Kernel Calling Convention

Kernels must use `callconv(.Kernel)` and be `export`-ed:

```zig
export fn myKernel(ptr: [*]f32, n: u32) callconv(.Kernel) void {
    // ...
}
```

**Argument type rules:**

| Zig type | Host sends |
|----------|------------|
| `[*]f32` / `[*]u32` | device pointer (from `CudaSlice.ptr`) |
| `u32`, `i32`, `f32` | by value |
| `types.DeviceSlice(T)` | `extern struct` passed by value |
| `types.DevicePtr(T)` | `extern struct` passed by value |
| `*debug.ErrorFlag` | device pointer |

---

## Build Options

```bash
zig build compile-kernels                    # all kernels, default sm_80
zig build compile-kernels -Dgpu-arch=sm_86  # RTX 30xx
zig build compile-kernels -Dgpu-arch=sm_90  # Hopper
zig build example-kernel-0-basic-kernel_vector_add -Dgpu-arch=sm_86  # single example
```

Available `-Dgpu-arch` values: `sm_52`, `sm_60`, `sm_70`, `sm_75`, `sm_80`, `sm_86`, `sm_89`, `sm_90`, `sm_100`.
