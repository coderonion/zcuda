# CUDA C++ → Zig Kernel Migration Guide

> Migrate your CUDA C++ kernels to Zig with minimal effort. zcuda follows CUDA naming conventions exactly.

## Quick Comparison

```diff
- // CUDA C++
- __global__ void vectorAdd(float* A, float* B, float* C, int n) {
-     int i = blockIdx.x * blockDim.x + threadIdx.x;
-     if (i < n) C[i] = A[i] + B[i];
- }

+ // Zig
+ const cuda = @import("zcuda_kernel");
+
+ export fn vectorAdd(A: [*]const f32, B: [*]const f32, C: [*]f32, n: u32) callconv(.Kernel) void {
+     const i = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
+     if (i < n) C[i] = A[i] + B[i];
+ }
```

> **Note:** Zig uses `.Kernel` (capital K) for the calling convention.

---

## Key Differences

| Feature | CUDA C++ | Zig zcuda |
|---------|----------|-----------|
| Kernel declaration | `__global__ void fn()` | `export fn fn() callconv(.Kernel) void` |
| Thread index | `threadIdx.x` | `cuda.threadIdx().x` |
| Pointer types | `float*`, `const float*` | `[*]f32`, `[*]const f32` |
| Integer types | `int`, `unsigned` | `i32`, `u32` |
| Sync barrier | `__syncthreads()` | `cuda.__syncthreads()` |
| Atomic add | `atomicAdd(&x, 1)` | `cuda.atomicAdd(&x, 1)` |
| Atomic sub | `atomicSub(&x, 1)` | `cuda.atomicSub(&x, 1)` |
| Warp shuffle down | `__shfl_down_sync(m,v,d)` | `cuda.__shfl_down_sync(m,v,d,32)` |
| Fast sine | `__sinf(x)` | `cuda.__sinf(x)` |
| FMA | `__fmaf_rn(a,b,c)` | `cuda.__fmaf_rn(a,b,c)` |
| Saturate | `__saturatef(x)` | `cuda.__saturatef(x)` |
| Type cast | `(float)x` | `@floatFromInt(x)` |
| Bitcast | `__float_as_int(x)` | `@bitCast(x)` or `cuda.__float_as_int(x)` |
| Shared mem | `__shared__ float arr[256]` | `const T = SharedArray(f32, 256); T.ptr()` |
| Dynamic smem | `extern __shared__ float arr[]` | `smem.dynamicShared(f32)` |
| printf | `printf("i=%d\n", i)` | `cuda.debug.printf("i=%u\n", .{i})` |
| Assert | `assert(cond)` | `cuda.debug.assertf(cond)` |

---

## Migration Cheatsheet

### 1. Kernel Signature

```diff
- __global__ void myKernel(float* data, int n)
+ export fn myKernel(data: [*]f32, n: u32) callconv(.Kernel) void
```

### 2. Thread Indexing

```diff
- int tid = threadIdx.x;
- int bid = blockIdx.x;
- int i = bid * blockDim.x + tid;
+ const tid = cuda.threadIdx().x;
+ const bid = cuda.blockIdx().x;
+ const i = bid * cuda.blockDim().x + tid;
```

Or use the helper:
```zig
const i = cuda.types.globalThreadIdx();
```

### 3. Grid-Stride Loop

```diff
- // CUDA C++
- for (int i = blockIdx.x * blockDim.x + threadIdx.x;
-      i < n;
-      i += blockDim.x * gridDim.x) {
-     output[i] = process(input[i]);
- }

+ // Zig — cleaner with iterator
+ var iter = cuda.types.gridStrideLoop(n);
+ while (iter.next()) |i| {
+     output[i] = process(input[i]);
+ }
```

### 4. Warp Shuffle Reduction

```diff
- // CUDA C++ — f32 warp reduce
- float sum = val;
- for (int offset = 16; offset > 0; offset /= 2)
-     sum += __shfl_down_sync(0xffffffff, sum, offset);

+ // Zig — explicit unroll (same pattern, compiler optimizes)
+ var sum = val;
+ // shfl_down_sync works on u32; bitcast f32 ↔ u32
+ sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(sum), 16, 32));
+ sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(sum), 8,  32));
+ sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(sum), 4,  32));
+ sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(sum), 2,  32));
+ sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(sum), 1,  32));
```

Or on sm_80+ use the single-instruction reduce:
```zig
const sum = cuda.__reduce_add_sync(cuda.FULL_MASK, @bitCast(val)); // u32
```

### 5. Shared Memory — Static

```diff
- // CUDA C++
- __shared__ float tile[256];
- tile[threadIdx.x] = val;
- __syncthreads();

+ // Zig
+ const smem = cuda.shared_mem;
+ const tile = smem.SharedArray(f32, 256);   // addrspace(3) — .shared
+ tile.ptr()[cuda.threadIdx().x] = val;
+ cuda.__syncthreads();
```

### 6. Shared Memory — Dynamic

```diff
- // CUDA C++
- extern __shared__ float arr[];
- arr[threadIdx.x] = val;

+ // Zig
+ const dyn = cuda.shared_mem.dynamicShared(f32);
+ dyn[cuda.threadIdx().x] = val;
```

Multiple arrays in the same dynamic region:
```diff
- extern __shared__ char base[];
- float* a = (float*)base;
- float* b = (float*)(base + 1024);

+ const base = cuda.shared_mem.dynamicSharedBytes();
+ const a: [*]f32 = @ptrCast(@alignCast(base));
+ const b: [*]f32 = @ptrCast(@alignCast(base + 1024));
```

### 7. FMA and Matrix Multiply

```diff
- // CUDA C++
- float sum = 0.0f;
- for (int k = 0; k < K; ++k)
-     sum = __fmaf_rn(A[row*K+k], B[k*N+col], sum);

+ // Zig
+ var sum: f32 = 0.0;
+ for (0..K) |k| {
+     sum = cuda.__fmaf_rn(A[row * K + k], B[k * N + col], sum);
+ }
```

### 8. printf (Device-Side)

```diff
- // CUDA C++
- printf("Thread %d: val = %f\n", tid, val);

+ // Zig — Zig tuple syntax; f32 auto-promoted to f64 (CUDA convention)
+ cuda.debug.printf("Thread %u: val = %f\n", .{ tid, val });
```

### 9. Assertions and Error Detection

```diff
- // CUDA C++ — no built-in device assert (relies on <assert.h>)
- assert(i < n);

+ // Zig — traps all threads in warp on failure
+ cuda.debug.assertf(i < n);
+ cuda.debug.assertInBounds(i, n);  // equivalent
```

Error flag pattern (no direct CUDA C++ equivalent — cleaner than `__trap`):
```zig
// Kernel
export fn myKernel(data: [*]f32, n: u32, err: *cuda.debug.ErrorFlag) callconv(.Kernel) void {
    const i = cuda.types.globalThreadIdx();
    if (i >= n) { cuda.debug.setError(err, cuda.debug.ErrorFlag.OUT_OF_BOUNDS); return; }
    cuda.debug.checkNaN(data[i], err);  // auto-sets NAN_DETECTED
}

// Host: allocate err on device, copy back after launch to check
```

### 10. Host-Device Shared Structs

```diff
- // CUDA C++ — must manually ensure layout matches on both sides
- struct Vec3 { float x, y, z; };  // host
- // kernel also declares: struct Vec3 { float x, y, z; };  // device — must match!

+ // Zig — same extern struct imported on both sides, guaranteed identical layout
+ const Vec3 = cuda.shared.Vec3;     // in kernel
+ const Vec3 = @import("zcuda").shared_types.Vec3;  // on host
```

Available shared types: `Vec2`, `Vec3`, `Vec4`, `Int2`, `Int3`, `Matrix3x3`, `Matrix4x4`, `LaunchConfig`.

### 11. Architecture Guards

```diff
- // CUDA C++ — no compile-time check, runtime failure
- // (just hope the hardware supports redux.sync)

+ // Zig — compile error if SM too low
+ cuda.arch.requireSM(cuda.SM, .sm_80, "myFeature()");
+ // error: myFeature() requires sm_80+, but target is sm_70
```

Or use conditional compilation:
```zig
if (cuda.SM.atLeast(.sm_80)) {
    return cuda.__reduce_add_sync(mask, val);
} else {
    // fallback path
}
```

---

## Build & Target

```bash
# Compile all kernels to PTX
zig build compile-kernels

# Target specific SM architecture
zig build compile-kernels -Dgpu-arch=sm_80   # Ampere (A100, RTX 30xx data-center)
zig build compile-kernels -Dgpu-arch=sm_86   # Ampere consumer (RTX 3080, 3090)
zig build compile-kernels -Dgpu-arch=sm_89   # Ada Lovelace (RTX 40xx)
zig build compile-kernels -Dgpu-arch=sm_90   # Hopper (H100)

# Compile single kernel example
zig build example-kernel-0-basic-kernel_vector_add -Dgpu-arch=sm_86
```

---

## Why Zig over CUDA C++

| Advantage | Description |
|-----------|-------------|
| **Compile-time SM guard** | `requireSM()` catches unsupported intrinsics at compile time instead of silent runtime failures |
| **Type safety** | Comptime-generic types (`DeviceSlice(T)`, `SharedArray(T,N)`) replace `void*` casts |
| **No nvcc** | Zig compiles kernels via LLVM NVPTX backend — no separate CUDA toolkit install needed |
| **No headers** | No `cuda_runtime.h` — all intrinsics are pure Zig inline assembly |
| **Cross-platform build** | `zig build` works on macOS/Linux/Windows |
| **Shared types** | Same `extern struct` layout on host and device — no manual layout matching |
| **printf** | Full `debug.printf` with Zig tuple syntax and auto-promotion |
| **Build integration** | PTX compilation, bridge generation, and host code all in `build.zig` |
