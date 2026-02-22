# Kernel DSL Examples

Pure-Zig GPU kernels compiled to PTX via Zig's built-in LLVM NVPTX backend.
**No CUDA C++ required** — all kernels are written in Zig.

## Building

```bash
# Build a single kernel example
zig build example-kernel-0-basic-kernel_vector_add -Dgpu-arch=sm_86

# Build an entire category
zig build example-kernel-0-basic -Dgpu-arch=sm_86

# Run (after build, binary is in zig-out/bin/)
./zig-out/bin/0-basic-kernel_vector_add
```

Required flag: `-Dgpu-arch=<arch>` (e.g., `sm_80`, `sm_86`, `sm_89`, `sm_90`).

---

## 0_Basic — Core Kernel Patterns (8 examples)

Essential kernel primitives: element-wise ops, grid-stride loops, vector ops.

| Example | Description |
|---------|-------------|
| [kernel_vector_add](0_Basic/kernel_vector_add.zig) | Element-wise vector addition |
| [kernel_saxpy](0_Basic/kernel_saxpy.zig) | SAXPY: y = α·x + y |
| [kernel_relu](0_Basic/kernel_relu.zig) | ReLU activation in-place |
| [kernel_scale_bias](0_Basic/kernel_scale_bias.zig) | Scale + bias: y = α·x + β |
| [kernel_grid_stride](0_Basic/kernel_grid_stride.zig) | Grid-stride loop pattern |
| [kernel_dot_product](0_Basic/kernel_dot_product.zig) | Two-phase parallel dot product |
| [kernel_residual_norm](0_Basic/kernel_residual_norm.zig) | Residual norm computation |
| [kernel_vec3_normalize](0_Basic/kernel_vec3_normalize.zig) | Batch 3D vector normalization |

---

## 1_Reduction — Parallel Reductions (5 examples)

Warp-level and block-level parallel reduction patterns.

| Example | Description |
|---------|-------------|
| [kernel_reduce_warp](1_Reduction/kernel_reduce_warp.zig) | Warp shuffle reduction |
| [kernel_reduce_sum](1_Reduction/kernel_reduce_sum.zig) | Block-level sum reduction |
| [kernel_reduce_multiblock](1_Reduction/kernel_reduce_multiblock.zig) | Multi-block two-phase reduction |
| [kernel_prefix_sum](1_Reduction/kernel_prefix_sum.zig) | Exclusive prefix scan (Blelloch) |
| [kernel_scalar_product](1_Reduction/kernel_scalar_product.zig) | Scalar product via dual reduction |

---

## 2_Matrix — Matrix Operations (6 examples)

Matrix multiplication, transpose, extraction, and padding on GPU.

| Example | Description |
|---------|-------------|
| [kernel_matmul_naive](2_Matrix/kernel_matmul_naive.zig) | Naive O(N³) matrix multiply |
| [kernel_matmul_tiled](2_Matrix/kernel_matmul_tiled.zig) | Tiled (shared-memory) matrix multiply |
| [kernel_matvec](2_Matrix/kernel_matvec.zig) | Matrix-vector product |
| [kernel_transpose](2_Matrix/kernel_transpose.zig) | Coalesced matrix transpose |
| [kernel_extract_diag](2_Matrix/kernel_extract_diag.zig) | Extract diagonal elements |
| [kernel_pad_2d](2_Matrix/kernel_pad_2d.zig) | 2D zero-padding |

---

## 3_Atomics — Atomic Operations (5 examples)

Atomic arithmetic, histograms, and warp-aggregated patterns.

| Example | Description |
|---------|-------------|
| [kernel_atomic_ops](3_Atomics/kernel_atomic_ops.zig) | `atomicAdd`, `atomicMin`, `atomicCAS` |
| [kernel_histogram](3_Atomics/kernel_histogram.zig) | Basic histogram with atomics |
| [kernel_histogram_256bin](3_Atomics/kernel_histogram_256bin.zig) | 256-bin histogram (shared mem opt) |
| [kernel_warp_aggregated_atomics](3_Atomics/kernel_warp_aggregated_atomics.zig) | Warp-aggregated atomics (1 CAS per warp) |
| [kernel_system_atomics](3_Atomics/kernel_system_atomics.zig) | System-scope (cross-device) atomics |

---

## 4_SharedMemory — Shared Memory (3 examples)

Static and dynamic shared memory usage patterns.

| Example | Description |
|---------|-------------|
| [kernel_shared_mem_demo](4_SharedMemory/kernel_shared_mem_demo.zig) | Static shared memory bank access patterns |
| [kernel_stencil_1d](4_SharedMemory/kernel_stencil_1d.zig) | 1D stencil with shared memory caching |
| [kernel_dynamic_smem](4_SharedMemory/kernel_dynamic_smem.zig) | Dynamic shared memory allocation |

---

## 5_Warp — Warp Intrinsics (5 examples)

Ballot, broadcast, match, and scan using warp shuffle instructions.

| Example | Description |
|---------|-------------|
| [kernel_warp_reduce](5_Warp/kernel_warp_reduce.zig) | Warp shuffle reduction (`__shfl_down_sync`) |
| [kernel_warp_broadcast](5_Warp/kernel_warp_broadcast.zig) | Warp broadcast (`__shfl_sync`) |
| [kernel_warp_scan](5_Warp/kernel_warp_scan.zig) | Warp-level inclusive prefix scan |
| [kernel_ballot_vote](5_Warp/kernel_ballot_vote.zig) | Ballot vote: `__ballot_sync`, `__all_sync` |
| [kernel_warp_match](5_Warp/kernel_warp_match.zig) | Match lanes with equal values (`__match_any_sync`) |

---

## 6_MathAndTypes — Math & Type Operations (9 examples)

FP16, complex numbers, type conversion, and math intrinsics.

| Example | Description |
|---------|-------------|
| [kernel_half_precision](6_MathAndTypes/kernel_half_precision.zig) | FP16 arithmetic and conversion |
| [kernel_complex_mul](6_MathAndTypes/kernel_complex_mul.zig) | Complex multiply on GPU |
| [kernel_fast_math](6_MathAndTypes/kernel_fast_math.zig) | Fast math approximations (`__fmaf_rn`, `rsqrtf`) |
| [kernel_integer_intrinsics](6_MathAndTypes/kernel_integer_intrinsics.zig) | `__popc`, `__clz`, `__brev`, `__ffs` |
| [kernel_type_conversion](6_MathAndTypes/kernel_type_conversion.zig) | i8/u8/f16/bf16/f32/f64 conversions |
| [kernel_math_test](6_MathAndTypes/kernel_math_test.zig) | Full math function coverage |
| [kernel_sigmoid](6_MathAndTypes/kernel_sigmoid.zig) | Sigmoid activation |
| [kernel_freq_filter](6_MathAndTypes/kernel_freq_filter.zig) | Frequency-domain filter (complex multiply) |
| [kernel_signal_gen](6_MathAndTypes/kernel_signal_gen.zig) | Waveform signal generation (sin, cos) |

---

## 7_Debug — Debug Utilities (2 examples)

Error checking and GPU-side `printf` debugging.

| Example | Description |
|---------|-------------|
| [kernel_error_check](7_Debug/kernel_error_check.zig) | CUDA error detection and reporting |
| [kernel_printf_debug](7_Debug/kernel_printf_debug.zig) | GPU-side `printf` for thread-level debugging |

---

## 8_TensorCore — Tensor Core Operations (11 examples)

WMMA and MMA matrix fragments: f16, bf16, int8, tf32, fp8.

| Example | Architecture | Description |
|---------|-------------|-------------|
| [kernel_wmma_gemm_f16](8_TensorCore/kernel_wmma_gemm_f16.zig) | sm_70+ | WMMA FP16 GEMM |
| [kernel_wmma_gemm_bf16](8_TensorCore/kernel_wmma_gemm_bf16.zig) | sm_80+ | WMMA BF16 GEMM |
| [kernel_wmma_gemm_int8](8_TensorCore/kernel_wmma_gemm_int8.zig) | sm_72+ | WMMA INT8 GEMM |
| [kernel_wmma_gemm_tf32](8_TensorCore/kernel_wmma_gemm_tf32.zig) | sm_80+ | WMMA TF32 GEMM |
| [kernel_mma_gemm_f16](8_TensorCore/kernel_mma_gemm_f16.zig) | sm_70+ | MMA m16n8k16 FP16 inline PTX |
| [kernel_mma_gemm_fp8](8_TensorCore/kernel_mma_gemm_fp8.zig) | sm_89+ | MMA FP8 (Ada/Hopper) |
| + 5 PTX variants | — | Pre-compiled PTX references (.ptx files) |

> WMMA = Warp Matrix Multiply-Accumulate (WMMA API).
> MMA = Inline PTX `mma.sync` — exact hardware instructions.

---

## 9_Advanced — Advanced Patterns (8 examples)

Async copy pipelines, cooperative groups, and complex algorithms.

| Example | Description |
|---------|-------------|
| [kernel_async_copy_pipeline](9_Advanced/kernel_async_copy_pipeline.zig) | `cp.async` pipelining (sm_80+ cp.async) |
| [kernel_cooperative_groups](9_Advanced/kernel_cooperative_groups.zig) | Cooperative groups: grid sync |
| [kernel_softmax](9_Advanced/kernel_softmax.zig) | Online softmax (numerically stable) |
| [kernel_thread_fence](9_Advanced/kernel_thread_fence.zig) | `__threadfence` memory ordering |
| [kernel_particle_init](9_Advanced/kernel_particle_init.zig) | Particle system initialization |
| [kernel_particle_step](9_Advanced/kernel_particle_step.zig) | Particle physics step integration |
| [kernel_gbm_paths](9_Advanced/kernel_gbm_paths.zig) | Geometric Brownian Motion paths |
| [kernel_intrinsics_coverage](9_Advanced/kernel_intrinsics_coverage.zig) | Comprehensive intrinsics coverage test |

---

## 10_Integration — End-to-End Pipelines (24 examples)

Multi-library pipelines and benchmarks. Compiled by: `zig build example-integration`.

See [../README.md#integration-examples](../README.md#integration-examples-24-examples) for the full list.
