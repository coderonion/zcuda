// src/kernel/device.zig — Root module for zcuda device-side GPU programming
//
// This module re-exports all device intrinsics for use in GPU kernels.
// Import as: const cuda = @import("zcuda_kernel");
//
// Example kernel:
//   const cuda = @import("zcuda_kernel");
//
//   export fn vectorAdd(A: [*]const f32, B: [*]const f32, C: [*]f32, n: u32) callconv(.kernel) void {
//       const i = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
//       if (i < n) C[i] = A[i] + B[i];
//   }

const intrinsics = @import("intrinsics.zig");
pub const arch = @import("arch.zig");
pub const types = @import("types.zig");
pub const shared = @import("shared_types.zig");
pub const debug = @import("debug.zig");
pub const tensor_core = @import("tensor_core.zig");
pub const shared_mem = @import("shared_mem.zig");

// -- SM Architecture --
// SM version is passed from build.zig via -Dgpu-arch option (default: sm_80)
const build_options = @import("build_options");
/// The target SM architecture version, set at compile time via `-Dgpu-arch=sm_XX`.
pub const SM: arch.SmVersion = @enumFromInt(build_options.sm_version);

// -- Types --
pub const Dim3 = intrinsics.Dim3;

// -- Thread Indexing --
pub const threadIdx = intrinsics.threadIdx;
pub const blockIdx = intrinsics.blockIdx;
pub const blockDim = intrinsics.blockDim;
pub const gridDim = intrinsics.gridDim;
pub const warpSize = intrinsics.warpSize;
pub const FULL_MASK = intrinsics.FULL_MASK;

// -- Synchronization --
pub const __syncthreads = intrinsics.__syncthreads;
pub const __syncthreads_count = intrinsics.__syncthreads_count;
pub const __syncthreads_and = intrinsics.__syncthreads_and;
pub const __syncthreads_or = intrinsics.__syncthreads_or;
pub const __threadfence = intrinsics.__threadfence;
pub const __threadfence_block = intrinsics.__threadfence_block;
pub const __threadfence_system = intrinsics.__threadfence_system;
pub const __syncwarp = intrinsics.__syncwarp;

// -- Atomic Operations --
pub const atomicAdd = intrinsics.atomicAdd;
pub const atomicAdd_f64 = intrinsics.atomicAdd_f64;
pub const atomicSub = intrinsics.atomicSub;
pub const atomicMax = intrinsics.atomicMax;
pub const atomicMin = intrinsics.atomicMin;
pub const atomicCAS = intrinsics.atomicCAS;
pub const atomicExch = intrinsics.atomicExch;
pub const atomicAnd = intrinsics.atomicAnd;
pub const atomicOr = intrinsics.atomicOr;
pub const atomicXor = intrinsics.atomicXor;
pub const atomicInc = intrinsics.atomicInc;
pub const atomicDec = intrinsics.atomicDec;

// -- Warp Primitives --
pub const __shfl_sync = intrinsics.__shfl_sync;
pub const __shfl_down_sync = intrinsics.__shfl_down_sync;
pub const __shfl_up_sync = intrinsics.__shfl_up_sync;
pub const __shfl_xor_sync = intrinsics.__shfl_xor_sync;
pub const __ballot_sync = intrinsics.__ballot_sync;
pub const __all_sync = intrinsics.__all_sync;
pub const __any_sync = intrinsics.__any_sync;
pub const __activemask = intrinsics.__activemask;

// -- Fast Math --
pub const __sinf = intrinsics.__sinf;
pub const __cosf = intrinsics.__cosf;
pub const __tanf = intrinsics.__tanf;
pub const __exp2f = intrinsics.__exp2f;
pub const __expf = intrinsics.__expf;
pub const __log2f = intrinsics.__log2f;
pub const __logf = intrinsics.__logf;
pub const __log10f = intrinsics.__log10f;
pub const rsqrtf = intrinsics.rsqrtf;
pub const sqrtf = intrinsics.sqrtf;
pub const fabsf = intrinsics.fabsf;
pub const fminf = intrinsics.fminf;
pub const fmaxf = intrinsics.fmaxf;
pub const __fmaf_rn = intrinsics.__fmaf_rn;
pub const __fdividef = intrinsics.__fdividef;
pub const __powf = intrinsics.__powf;
pub const __saturatef = intrinsics.__saturatef;

// -- Warp Match (sm_70+) --
pub const __match_any_sync = intrinsics.__match_any_sync;
pub const __match_all_sync = intrinsics.__match_all_sync;

// -- Warp Reduce (sm_80+, SM-guarded) --
pub const __reduce_add_sync = intrinsics.__reduce_add_sync;
pub const __reduce_min_sync = intrinsics.__reduce_min_sync;
pub const __reduce_max_sync = intrinsics.__reduce_max_sync;
pub const __reduce_and_sync = intrinsics.__reduce_and_sync;
pub const __reduce_or_sync = intrinsics.__reduce_or_sync;
pub const __reduce_xor_sync = intrinsics.__reduce_xor_sync;

// -- Integer Intrinsics --
pub const __clz = intrinsics.__clz;
pub const __clzll = intrinsics.__clzll;
pub const __popc = intrinsics.__popc;
pub const __popcll = intrinsics.__popcll;
pub const __brev = intrinsics.__brev;
pub const __brevll = intrinsics.__brevll;
pub const __ffs = intrinsics.__ffs;
pub const __byte_perm = intrinsics.__byte_perm;

// -- Dot Product (sm_75+) --
pub const __dp4a = intrinsics.__dp4a;
pub const __dp4a_s32 = intrinsics.__dp4a_s32;
pub const __dp2a_lo = intrinsics.__dp2a_lo;
pub const __dp2a_hi = intrinsics.__dp2a_hi;

// -- Cache Load/Store Hints --
pub const __ldg = intrinsics.__ldg;
pub const __ldg_u32 = intrinsics.__ldg_u32;
pub const __ldca = intrinsics.__ldca;
pub const __ldcs = intrinsics.__ldcs;
pub const __ldcg = intrinsics.__ldcg;
pub const __stcg = intrinsics.__stcg;
pub const __stcs = intrinsics.__stcs;
pub const __stwb = intrinsics.__stwb;

// -- Address Space Predicates --
pub const __isGlobal = intrinsics.__isGlobal;
pub const __isShared = intrinsics.__isShared;
pub const __isConstant = intrinsics.__isConstant;
pub const __isLocal = intrinsics.__isLocal;

// -- Type Conversion --
pub const __float2int_rn = intrinsics.__float2int_rn;
pub const __float2int_rz = intrinsics.__float2int_rz;
pub const __float2uint_rn = intrinsics.__float2uint_rn;
pub const __float2uint_rz = intrinsics.__float2uint_rz;
pub const __int2float_rn = intrinsics.__int2float_rn;
pub const __uint2float_rn = intrinsics.__uint2float_rn;
pub const __float_as_int = intrinsics.__float_as_int;
pub const __int_as_float = intrinsics.__int_as_float;
pub const __float_as_uint = intrinsics.__float_as_uint;
pub const __uint_as_float = intrinsics.__uint_as_float;
pub const __double2hiint = intrinsics.__double2hiint;
pub const __double2loint = intrinsics.__double2loint;
pub const __hiloint2double = intrinsics.__hiloint2double;

// -- Clock --
pub const clock = intrinsics.clock;
pub const clock64 = intrinsics.clock64;
pub const globaltimer = intrinsics.globaltimer;

// -- Misc --
pub const __nanosleep = intrinsics.__nanosleep;
