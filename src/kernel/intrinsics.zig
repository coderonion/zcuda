// src/kernel/intrinsics.zig — CUDA Device Intrinsics for Zig
//
// Naming convention: matches CUDA C++ exactly for seamless migration.
//
// Usage in kernel files:
//   const cuda = @import("zcuda_kernel");
//   const i = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;

// ============================================================================
// Types
// ============================================================================

/// Dim3 — mirrors CUDA C++ dim3, returned by threadIdx/blockIdx/blockDim/gridDim
pub const Dim3 = struct { x: u32, y: u32, z: u32 };

// ============================================================================
// Thread Indexing — matches CUDA C++ threadIdx.x / blockIdx.x / etc.
// ============================================================================

/// threadIdx — equivalent to CUDA C++ `threadIdx.x/y/z`
pub inline fn threadIdx() Dim3 {
    return .{
        .x = asm volatile ("mov.u32 %[ret], %tid.x;"
            : [ret] "=r" (-> u32),
        ),
        .y = asm volatile ("mov.u32 %[ret], %tid.y;"
            : [ret] "=r" (-> u32),
        ),
        .z = asm volatile ("mov.u32 %[ret], %tid.z;"
            : [ret] "=r" (-> u32),
        ),
    };
}

/// blockIdx — equivalent to CUDA C++ `blockIdx.x/y/z`
pub inline fn blockIdx() Dim3 {
    return .{
        .x = asm volatile ("mov.u32 %[ret], %ctaid.x;"
            : [ret] "=r" (-> u32),
        ),
        .y = asm volatile ("mov.u32 %[ret], %ctaid.y;"
            : [ret] "=r" (-> u32),
        ),
        .z = asm volatile ("mov.u32 %[ret], %ctaid.z;"
            : [ret] "=r" (-> u32),
        ),
    };
}

/// blockDim — equivalent to CUDA C++ `blockDim.x/y/z`
pub inline fn blockDim() Dim3 {
    return .{
        .x = asm volatile ("mov.u32 %[ret], %ntid.x;"
            : [ret] "=r" (-> u32),
        ),
        .y = asm volatile ("mov.u32 %[ret], %ntid.y;"
            : [ret] "=r" (-> u32),
        ),
        .z = asm volatile ("mov.u32 %[ret], %ntid.z;"
            : [ret] "=r" (-> u32),
        ),
    };
}

/// gridDim — equivalent to CUDA C++ `gridDim.x/y/z`
pub inline fn gridDim() Dim3 {
    return .{
        .x = asm volatile ("mov.u32 %[ret], %nctaid.x;"
            : [ret] "=r" (-> u32),
        ),
        .y = asm volatile ("mov.u32 %[ret], %nctaid.y;"
            : [ret] "=r" (-> u32),
        ),
        .z = asm volatile ("mov.u32 %[ret], %nctaid.z;"
            : [ret] "=r" (-> u32),
        ),
    };
}

/// warpSize — equivalent to CUDA C++ `warpSize` (always 32)
pub const warpSize: u32 = 32;

/// FULL_MASK — all 32 threads active, commonly used with warp intrinsics
pub const FULL_MASK: u32 = 0xffffffff;

// ============================================================================
// Synchronization — matches CUDA C++ __syncthreads() / __threadfence() / etc.
// ============================================================================

/// __syncthreads — equivalent to CUDA C++ `__syncthreads()`
pub inline fn __syncthreads() void {
    asm volatile ("bar.sync 0;");
}

/// __syncthreads_count — equivalent to CUDA C++ `__syncthreads_count()`
/// Barrier + returns the number of threads for which `predicate` is true.
pub inline fn __syncthreads_count(predicate: bool) u32 {
    // bar.red.popc requires a predicate register (%p), not a GPR.
    // LLVM NVPTX doesn't support predicate clobbers, so use hardcoded %p0.
    return asm volatile (
        \\setp.ne.b32 %p0, %[pred], 0;
        \\bar.red.popc.u32 %[ret], 0, %p0;
        : [ret] "=r" (-> u32),
        : [pred] "r" (@as(u32, @intFromBool(predicate))),
    );
}

/// __syncthreads_and — equivalent to CUDA C++ `__syncthreads_and()`
/// Barrier + returns non-zero if `predicate` is true for ALL threads.
pub inline fn __syncthreads_and(predicate: bool) u32 {
    return asm volatile (
        \\setp.ne.b32 %p0, %[pred], 0;
        \\bar.red.and.pred.u32 %[ret], 0, %p0;
        : [ret] "=r" (-> u32),
        : [pred] "r" (@as(u32, @intFromBool(predicate))),
    );
}

/// __syncthreads_or — equivalent to CUDA C++ `__syncthreads_or()`
/// Barrier + returns non-zero if `predicate` is true for ANY thread.
pub inline fn __syncthreads_or(predicate: bool) u32 {
    return asm volatile (
        \\setp.ne.b32 %p0, %[pred], 0;
        \\bar.red.or.pred.u32 %[ret], 0, %p0;
        : [ret] "=r" (-> u32),
        : [pred] "r" (@as(u32, @intFromBool(predicate))),
    );
}

/// __threadfence — equivalent to CUDA C++ `__threadfence()`
pub inline fn __threadfence() void {
    asm volatile ("membar.gl;");
}

/// __threadfence_block — equivalent to CUDA C++ `__threadfence_block()`
pub inline fn __threadfence_block() void {
    asm volatile ("membar.cta;");
}

/// __threadfence_system — equivalent to CUDA C++ `__threadfence_system()`
pub inline fn __threadfence_system() void {
    asm volatile ("membar.sys;");
}

/// __syncwarp — equivalent to CUDA C++ `__syncwarp()` (sm_70+)
/// Synchronizes threads within a warp according to the given mask.
pub inline fn __syncwarp(mask: u32) void {
    asm volatile ("bar.warp.sync %[mask];"
        :
        : [mask] "r" (mask),
    );
}

// ============================================================================
// Atomic Operations — matches CUDA C++ atomicAdd() / atomicCAS() / etc.
// ============================================================================

/// atomicAdd — equivalent to CUDA C++ `atomicAdd()`
/// Supports f32, u32, i32 via comptime dispatch.
pub inline fn atomicAdd(ptr: anytype, val: anytype) @TypeOf(val) {
    const T = @TypeOf(val);
    return switch (T) {
        f32 => asm volatile ("atom.global.add.f32 %[ret], [%[ptr]], %[val];"
            : [ret] "=f" (-> f32),
            : [ptr] "l" (ptr),
              [val] "f" (val),
        ),
        u32 => asm volatile ("atom.global.add.u32 %[ret], [%[ptr]], %[val];"
            : [ret] "=r" (-> u32),
            : [ptr] "l" (ptr),
              [val] "r" (val),
        ),
        i32 => asm volatile ("atom.global.add.s32 %[ret], [%[ptr]], %[val];"
            : [ret] "=r" (-> i32),
            : [ptr] "l" (ptr),
              [val] "r" (val),
        ),
        else => @compileError("atomicAdd: unsupported type " ++ @typeName(T)),
    };
}

/// atomicMax — equivalent to CUDA C++ `atomicMax()`
pub inline fn atomicMax(ptr: anytype, val: anytype) @TypeOf(val) {
    const T = @TypeOf(val);
    return switch (T) {
        u32 => asm volatile ("atom.global.max.u32 %[ret], [%[ptr]], %[val];"
            : [ret] "=r" (-> u32),
            : [ptr] "l" (ptr),
              [val] "r" (val),
        ),
        i32 => asm volatile ("atom.global.max.s32 %[ret], [%[ptr]], %[val];"
            : [ret] "=r" (-> i32),
            : [ptr] "l" (ptr),
              [val] "r" (val),
        ),
        else => @compileError("atomicMax: unsupported type " ++ @typeName(T)),
    };
}

/// atomicMin — equivalent to CUDA C++ `atomicMin()`
pub inline fn atomicMin(ptr: anytype, val: anytype) @TypeOf(val) {
    const T = @TypeOf(val);
    return switch (T) {
        u32 => asm volatile ("atom.global.min.u32 %[ret], [%[ptr]], %[val];"
            : [ret] "=r" (-> u32),
            : [ptr] "l" (ptr),
              [val] "r" (val),
        ),
        i32 => asm volatile ("atom.global.min.s32 %[ret], [%[ptr]], %[val];"
            : [ret] "=r" (-> i32),
            : [ptr] "l" (ptr),
              [val] "r" (val),
        ),
        else => @compileError("atomicMin: unsupported type " ++ @typeName(T)),
    };
}

/// atomicCAS — equivalent to CUDA C++ `atomicCAS()`
pub inline fn atomicCAS(ptr: *u32, compare: u32, val: u32) u32 {
    return asm volatile ("atom.global.cas.b32 %[ret], [%[ptr]], %[cmp], %[val];"
        : [ret] "=r" (-> u32),
        : [ptr] "l" (ptr),
          [cmp] "r" (compare),
          [val] "r" (val),
    );
}

/// atomicExch — equivalent to CUDA C++ `atomicExch()`
pub inline fn atomicExch(ptr: anytype, val: anytype) @TypeOf(val) {
    const T = @TypeOf(val);
    return switch (T) {
        u32 => asm volatile ("atom.global.exch.b32 %[ret], [%[ptr]], %[val];"
            : [ret] "=r" (-> u32),
            : [ptr] "l" (ptr),
              [val] "r" (val),
        ),
        f32 => asm volatile ("atom.global.exch.b32 %[ret], [%[ptr]], %[val];"
            : [ret] "=f" (-> f32),
            : [ptr] "l" (ptr),
              [val] "f" (val),
        ),
        else => @compileError("atomicExch: unsupported type " ++ @typeName(T)),
    };
}

/// atomicSub — equivalent to CUDA C++ `atomicSub()`
/// Implemented as atomicAdd with negated value.
pub inline fn atomicSub(ptr: anytype, val: anytype) @TypeOf(val) {
    const T = @TypeOf(val);
    return switch (T) {
        u32 => atomicAdd(ptr, 0 -% val),
        i32 => atomicAdd(ptr, -val),
        f32 => atomicAdd(ptr, -val),
        else => @compileError("atomicSub: unsupported type " ++ @typeName(T)),
    };
}

/// atomicAnd — equivalent to CUDA C++ `atomicAnd()`
pub inline fn atomicAnd(ptr: *u32, val: u32) u32 {
    return asm volatile ("atom.global.and.b32 %[ret], [%[ptr]], %[val];"
        : [ret] "=r" (-> u32),
        : [ptr] "l" (ptr),
          [val] "r" (val),
    );
}

/// atomicOr — equivalent to CUDA C++ `atomicOr()`
pub inline fn atomicOr(ptr: *u32, val: u32) u32 {
    return asm volatile ("atom.global.or.b32 %[ret], [%[ptr]], %[val];"
        : [ret] "=r" (-> u32),
        : [ptr] "l" (ptr),
          [val] "r" (val),
    );
}

/// atomicXor — equivalent to CUDA C++ `atomicXor()`
pub inline fn atomicXor(ptr: *u32, val: u32) u32 {
    return asm volatile ("atom.global.xor.b32 %[ret], [%[ptr]], %[val];"
        : [ret] "=r" (-> u32),
        : [ptr] "l" (ptr),
          [val] "r" (val),
    );
}

/// atomicInc — equivalent to CUDA C++ `atomicInc()`
/// Increments *ptr, wrapping to 0 when *ptr >= val.
pub inline fn atomicInc(ptr: *u32, val: u32) u32 {
    return asm volatile ("atom.global.inc.u32 %[ret], [%[ptr]], %[val];"
        : [ret] "=r" (-> u32),
        : [ptr] "l" (ptr),
          [val] "r" (val),
    );
}

/// atomicDec — equivalent to CUDA C++ `atomicDec()`
/// Decrements *ptr, wrapping to val when *ptr == 0 or *ptr > val.
pub inline fn atomicDec(ptr: *u32, val: u32) u32 {
    return asm volatile ("atom.global.dec.u32 %[ret], [%[ptr]], %[val];"
        : [ret] "=r" (-> u32),
        : [ptr] "l" (ptr),
          [val] "r" (val),
    );
}

/// atomicAdd for f64 — equivalent to CUDA C++ `atomicAdd(double*, double)` (sm_60+)
pub inline fn atomicAdd_f64(ptr: *f64, val: f64) f64 {
    const arch_mod = @import("arch.zig");
    const build_options = @import("build_options");
    const sm: arch_mod.SmVersion = @enumFromInt(build_options.sm_version);
    arch_mod.requireSM(sm, .sm_60, "atomicAdd_f64()");
    return asm volatile ("atom.global.add.f64 %[ret], [%[ptr]], %[val];"
        : [ret] "=d" (-> f64),
        : [ptr] "l" (ptr),
          [val] "d" (val),
    );
}

// ============================================================================
// Warp-level Primitives — matches CUDA C++ __shfl_sync() / __ballot_sync() / etc.
// ============================================================================

/// __shfl_sync — equivalent to CUDA C++ `__shfl_sync()`
///
/// PTX `shfl.sync.idx` uses a packed `c` operand:
///   bits [4:0]  = maxLane = width - 1
///   bits [12:8] = segmentMask = warpSize - width
pub inline fn __shfl_sync(mask: u32, val: u32, src_lane: u32, width: u32) u32 {
    const c = ((32 - width) << 8) | (width - 1);
    return asm volatile ("shfl.sync.idx.b32 %[ret], %[val], %[src], %[c], %[mask];"
        : [ret] "=r" (-> u32),
        : [val] "r" (val),
          [src] "r" (src_lane),
          [c] "r" (c),
          [mask] "r" (mask),
    );
}

/// __shfl_down_sync — equivalent to CUDA C++ `__shfl_down_sync()`
pub inline fn __shfl_down_sync(mask: u32, val: u32, delta: u32, width: u32) u32 {
    const c = ((32 - width) << 8) | (width - 1);
    return asm volatile ("shfl.sync.down.b32 %[ret], %[val], %[delta], %[c], %[mask];"
        : [ret] "=r" (-> u32),
        : [val] "r" (val),
          [delta] "r" (delta),
          [c] "r" (c),
          [mask] "r" (mask),
    );
}

/// __shfl_up_sync — equivalent to CUDA C++ `__shfl_up_sync()`
/// For `up`, maxLane = 0 (clamped at lane 0 within segment).
pub inline fn __shfl_up_sync(mask: u32, val: u32, delta: u32, width: u32) u32 {
    const c = (32 - width) << 8; // maxLane = 0 for up
    return asm volatile ("shfl.sync.up.b32 %[ret], %[val], %[delta], %[c], %[mask];"
        : [ret] "=r" (-> u32),
        : [val] "r" (val),
          [delta] "r" (delta),
          [c] "r" (c),
          [mask] "r" (mask),
    );
}

/// __shfl_xor_sync — equivalent to CUDA C++ `__shfl_xor_sync()`
pub inline fn __shfl_xor_sync(mask: u32, val: u32, lane_mask: u32, width: u32) u32 {
    const c = ((32 - width) << 8) | (width - 1);
    return asm volatile ("shfl.sync.bfly.b32 %[ret], %[val], %[lane_mask], %[c], %[mask];"
        : [ret] "=r" (-> u32),
        : [val] "r" (val),
          [lane_mask] "r" (lane_mask),
          [c] "r" (c),
          [mask] "r" (mask),
    );
}

/// __ballot_sync — equivalent to CUDA C++ `__ballot_sync()`
/// PTX vote.sync.ballot.b32 requires a predicate register (%p) for the input.
/// LLVM NVPTX doesn't support predicate clobbers, so use hardcoded %p0.
pub inline fn __ballot_sync(mask: u32, predicate: bool) u32 {
    return asm volatile (
        \\setp.ne.b32 %p0, %[pred], 0;
        \\vote.sync.ballot.b32 %[ret], %p0, %[mask];
        : [ret] "=r" (-> u32),
        : [pred] "r" (@as(u32, @intFromBool(predicate))),
          [mask] "r" (mask),
    );
}

/// __all_sync — equivalent to CUDA C++ `__all_sync()`
pub inline fn __all_sync(mask: u32, predicate: bool) bool {
    const result = asm volatile (
        \\setp.ne.b32 %p0, %[pred], 0;
        \\vote.sync.all.pred %p1, %p0, %[mask];
        \\selp.u32 %[ret], 1, 0, %p1;
        : [ret] "=r" (-> u32),
        : [pred] "r" (@as(u32, @intFromBool(predicate))),
          [mask] "r" (mask),
    );
    return result != 0;
}

/// __any_sync — equivalent to CUDA C++ `__any_sync()`
pub inline fn __any_sync(mask: u32, predicate: bool) bool {
    const result = asm volatile (
        \\setp.ne.b32 %p0, %[pred], 0;
        \\vote.sync.any.pred %p1, %p0, %[mask];
        \\selp.u32 %[ret], 1, 0, %p1;
        : [ret] "=r" (-> u32),
        : [pred] "r" (@as(u32, @intFromBool(predicate))),
          [mask] "r" (mask),
    );
    return result != 0;
}

/// __activemask — equivalent to CUDA C++ `__activemask()`
pub inline fn __activemask() u32 {
    return asm volatile ("activemask.b32 %[ret];"
        : [ret] "=r" (-> u32),
    );
}

// ============================================================================
// Fast Math Intrinsics — matches CUDA C++ __sinf() / __cosf() / etc.
// P1: __logf, rsqrtf   P2: __sinf, __cosf, __expf, __exp2f, __log2f,
//     sqrtf, fabsf, fminf, fmaxf, __fmaf_rn, __fdividef
// ============================================================================

/// __sinf — fast sine approximation, equivalent to CUDA C++ `__sinf()`
pub inline fn __sinf(x: f32) f32 {
    return asm volatile ("sin.approx.f32 %[ret], %[x];"
        : [ret] "=f" (-> f32),
        : [x] "f" (x),
    );
}

/// __cosf — fast cosine approximation, equivalent to CUDA C++ `__cosf()`
pub inline fn __cosf(x: f32) f32 {
    return asm volatile ("cos.approx.f32 %[ret], %[x];"
        : [ret] "=f" (-> f32),
        : [x] "f" (x),
    );
}

/// __exp2f — fast base-2 exponential, equivalent to CUDA C++ `__exp2f()`
pub inline fn __exp2f(x: f32) f32 {
    return asm volatile ("ex2.approx.f32 %[ret], %[x];"
        : [ret] "=f" (-> f32),
        : [x] "f" (x),
    );
}

/// __expf — fast exponential (e^x), equivalent to CUDA C++ `__expf()`
/// Computed as: exp(x) = exp2(x * log2(e))
pub inline fn __expf(x: f32) f32 {
    const log2e: f32 = 1.4426950408889634;
    return __exp2f(x * log2e);
}

/// __log2f — fast base-2 logarithm, equivalent to CUDA C++ `__log2f()`
pub inline fn __log2f(x: f32) f32 {
    return asm volatile ("lg2.approx.f32 %[ret], %[x];"
        : [ret] "=f" (-> f32),
        : [x] "f" (x),
    );
}

/// __logf — fast natural log (P1), equivalent to CUDA C++ `__logf()`
/// Computed as: ln(x) = log2(x) * ln(2)
pub inline fn __logf(x: f32) f32 {
    const ln2: f32 = 0.6931471805599453;
    return __log2f(x) * ln2;
}

/// __log10f — fast base-10 logarithm, equivalent to CUDA C++ `__log10f()`
/// Computed as: log10(x) = log2(x) * log10(2)
pub inline fn __log10f(x: f32) f32 {
    const log10_2: f32 = 0.3010299957316530;
    return __log2f(x) * log10_2;
}

/// rsqrtf — fast reciprocal square root (P1), equivalent to CUDA C++ `rsqrtf()`
pub inline fn rsqrtf(x: f32) f32 {
    return asm volatile ("rsqrt.approx.f32 %[ret], %[x];"
        : [ret] "=f" (-> f32),
        : [x] "f" (x),
    );
}

/// sqrtf — square root, equivalent to CUDA C++ `sqrtf()`
pub inline fn sqrtf(x: f32) f32 {
    return asm volatile ("sqrt.rn.f32 %[ret], %[x];"
        : [ret] "=f" (-> f32),
        : [x] "f" (x),
    );
}

/// fabsf — absolute value, equivalent to CUDA C++ `fabsf()`
pub inline fn fabsf(x: f32) f32 {
    return asm volatile ("abs.f32 %[ret], %[x];"
        : [ret] "=f" (-> f32),
        : [x] "f" (x),
    );
}

/// fminf — minimum of two floats, equivalent to CUDA C++ `fminf()`
pub inline fn fminf(a: f32, b: f32) f32 {
    return asm volatile ("min.f32 %[ret], %[a], %[b];"
        : [ret] "=f" (-> f32),
        : [a] "f" (a),
          [b] "f" (b),
    );
}

/// fmaxf — maximum of two floats, equivalent to CUDA C++ `fmaxf()`
pub inline fn fmaxf(a: f32, b: f32) f32 {
    return asm volatile ("max.f32 %[ret], %[a], %[b];"
        : [ret] "=f" (-> f32),
        : [a] "f" (a),
          [b] "f" (b),
    );
}

/// __fmaf_rn — fused multiply-add (round-to-nearest), equivalent to CUDA C++ `__fmaf_rn()`
pub inline fn __fmaf_rn(a: f32, b: f32, c: f32) f32 {
    return asm volatile ("fma.rn.f32 %[ret], %[a], %[b], %[c];"
        : [ret] "=f" (-> f32),
        : [a] "f" (a),
          [b] "f" (b),
          [c] "f" (c),
    );
}

/// __fdividef — fast approximate division, equivalent to CUDA C++ `__fdividef()`
pub inline fn __fdividef(a: f32, b: f32) f32 {
    return asm volatile ("div.approx.f32 %[ret], %[a], %[b];"
        : [ret] "=f" (-> f32),
        : [a] "f" (a),
          [b] "f" (b),
    );
}

/// __powf — fast power function, equivalent to CUDA C++ `__powf()`
/// Computed as: pow(x, y) = exp2(y * log2(x))
pub inline fn __powf(x: f32, y: f32) f32 {
    return __exp2f(y * __log2f(x));
}

// ============================================================================
// Warp Match Functions — sm_70+ (require .sync variants)
// ============================================================================

/// __match_any_sync — equivalent to CUDA C++ `__match_any_sync()`
/// Returns a mask of threads that have the same value.
pub inline fn __match_any_sync(mask: u32, val: u32) u32 {
    return asm volatile ("match.any.sync.b32 %[ret], %[val], %[mask];"
        : [ret] "=r" (-> u32),
        : [val] "r" (val),
          [mask] "r" (mask),
    );
}

/// __match_all_sync — equivalent to CUDA C++ `__match_all_sync()`
/// Returns mask if all threads have the same value (pred set to true), 0 otherwise.
pub inline fn __match_all_sync(mask: u32, val: u32) struct { mask: u32, pred: bool } {
    var result_mask: u32 = undefined;
    var pred_val: u32 = undefined;
    asm volatile ("match.all.sync.b32 %[ret], %[val], %[mask], %[pred];"
        : [ret] "=r" (result_mask),
          [pred] "=r" (pred_val),
        : [val] "r" (val),
          [mask] "r" (mask),
    );
    return .{ .mask = result_mask, .pred = pred_val != 0 };
}

// ============================================================================
// Warp Reduce Functions — sm_80+ (redux.sync)
// ============================================================================

const arch = @import("arch.zig");
const device = @import("device.zig");

/// __reduce_add_sync — warp-level add reduction (sm_80+)
pub inline fn __reduce_add_sync(mask: u32, val: u32) u32 {
    arch.requireSM(device.SM, .sm_80, "__reduce_add_sync()");
    return asm volatile ("redux.sync.add.u32 %[ret], %[val], %[mask];"
        : [ret] "=r" (-> u32),
        : [val] "r" (val),
          [mask] "r" (mask),
    );
}

/// __reduce_min_sync — warp-level min reduction (sm_80+)
pub inline fn __reduce_min_sync(mask: u32, val: u32) u32 {
    arch.requireSM(device.SM, .sm_80, "__reduce_min_sync()");
    return asm volatile ("redux.sync.min.u32 %[ret], %[val], %[mask];"
        : [ret] "=r" (-> u32),
        : [val] "r" (val),
          [mask] "r" (mask),
    );
}

/// __reduce_max_sync — warp-level max reduction (sm_80+)
pub inline fn __reduce_max_sync(mask: u32, val: u32) u32 {
    arch.requireSM(device.SM, .sm_80, "__reduce_max_sync()");
    return asm volatile ("redux.sync.max.u32 %[ret], %[val], %[mask];"
        : [ret] "=r" (-> u32),
        : [val] "r" (val),
          [mask] "r" (mask),
    );
}

/// __reduce_and_sync — warp-level bitwise AND reduction (sm_80+)
pub inline fn __reduce_and_sync(mask: u32, val: u32) u32 {
    arch.requireSM(device.SM, .sm_80, "__reduce_and_sync()");
    return asm volatile ("redux.sync.and.b32 %[ret], %[val], %[mask];"
        : [ret] "=r" (-> u32),
        : [val] "r" (val),
          [mask] "r" (mask),
    );
}

/// __reduce_or_sync — warp-level bitwise OR reduction (sm_80+)
pub inline fn __reduce_or_sync(mask: u32, val: u32) u32 {
    arch.requireSM(device.SM, .sm_80, "__reduce_or_sync()");
    return asm volatile ("redux.sync.or.b32 %[ret], %[val], %[mask];"
        : [ret] "=r" (-> u32),
        : [val] "r" (val),
          [mask] "r" (mask),
    );
}

/// __reduce_xor_sync — warp-level bitwise XOR reduction (sm_80+)
pub inline fn __reduce_xor_sync(mask: u32, val: u32) u32 {
    arch.requireSM(device.SM, .sm_80, "__reduce_xor_sync()");
    return asm volatile ("redux.sync.xor.b32 %[ret], %[val], %[mask];"
        : [ret] "=r" (-> u32),
        : [val] "r" (val),
          [mask] "r" (mask),
    );
}

// ============================================================================
// Integer Intrinsics — bit manipulation
// ============================================================================

/// __clz — count leading zeros (32-bit), equivalent to CUDA C++ `__clz()`
pub inline fn __clz(x: u32) u32 {
    return asm volatile ("clz.b32 %[ret], %[x];"
        : [ret] "=r" (-> u32),
        : [x] "r" (x),
    );
}

/// __clzll — count leading zeros (64-bit), equivalent to CUDA C++ `__clzll()`
pub inline fn __clzll(x: u64) u32 {
    return asm volatile ("clz.b64 %[ret], %[x];"
        : [ret] "=r" (-> u32),
        : [x] "l" (x),
    );
}

/// __popc — population count (32-bit), equivalent to CUDA C++ `__popc()`
pub inline fn __popc(x: u32) u32 {
    return asm volatile ("popc.b32 %[ret], %[x];"
        : [ret] "=r" (-> u32),
        : [x] "r" (x),
    );
}

/// __popcll — population count (64-bit), equivalent to CUDA C++ `__popcll()`
pub inline fn __popcll(x: u64) u32 {
    return asm volatile ("popc.b64 %[ret], %[x];"
        : [ret] "=r" (-> u32),
        : [x] "l" (x),
    );
}

/// __brev — bit reversal (32-bit), equivalent to CUDA C++ `__brev()`
pub inline fn __brev(x: u32) u32 {
    return asm volatile ("brev.b32 %[ret], %[x];"
        : [ret] "=r" (-> u32),
        : [x] "r" (x),
    );
}

/// __brevll — bit reversal (64-bit), equivalent to CUDA C++ `__brevll()`
pub inline fn __brevll(x: u64) u64 {
    return asm volatile ("brev.b64 %[ret], %[x];"
        : [ret] "=l" (-> u64),
        : [x] "l" (x),
    );
}

/// __ffs — find first set bit (32-bit), equivalent to CUDA C++ `__ffs()`
/// Returns position of the least significant bit set (1-indexed), or 0 if no bits set.
pub inline fn __ffs(x: u32) u32 {
    if (x == 0) return 0;
    // bfind finds the most significant set bit; we use brev + bfind to get least significant
    const reversed = __brev(x);
    return asm volatile ("bfind.u32 %[ret], %[x];"
        : [ret] "=r" (-> u32),
        : [x] "r" (reversed),
    ) +% 1;
}

// ============================================================================
// Read-Only Data Cache Load
// ============================================================================

/// __ldg — read-only texture cache load, equivalent to CUDA C++ `__ldg()`
/// Loads through the read-only data cache (L1 texture cache), useful for read-only data.
pub inline fn __ldg(ptr: *const f32) f32 {
    return asm volatile ("ld.global.nc.f32 %[ret], [%[ptr]];"
        : [ret] "=f" (-> f32),
        : [ptr] "l" (ptr),
    );
}

/// __ldg_u32 — read-only cache load for u32
pub inline fn __ldg_u32(ptr: *const u32) u32 {
    return asm volatile ("ld.global.nc.u32 %[ret], [%[ptr]];"
        : [ret] "=r" (-> u32),
        : [ptr] "l" (ptr),
    );
}

// ============================================================================
// Clock Functions — cycle counters for profiling
// ============================================================================

/// clock — equivalent to CUDA C++ `clock()`, returns low 32 bits of cycle counter
pub inline fn clock() u32 {
    return asm volatile ("mov.u32 %[ret], %clock;"
        : [ret] "=r" (-> u32),
    );
}

/// clock64 — equivalent to CUDA C++ `clock64()`, returns full 64-bit cycle counter
pub inline fn clock64() u64 {
    return asm volatile ("mov.u64 %[ret], %clock64;"
        : [ret] "=l" (-> u64),
    );
}

/// globaltimer — equivalent to CUDA C++ `globaltimer`, returns nanosecond timer
pub inline fn globaltimer() u64 {
    return asm volatile ("mov.u64 %[ret], %globaltimer;"
        : [ret] "=l" (-> u64),
    );
}

// ============================================================================
// Dot Product Intrinsics (sm_75+ / Turing) — P2
// ============================================================================

/// __dp4a — 4-element dot product of packed bytes with 32-bit accumulate
/// Computes: result = a.byte[0]*b.byte[0] + a.byte[1]*b.byte[1] +
///                    a.byte[2]*b.byte[2] + a.byte[3]*b.byte[3] + c
pub inline fn __dp4a(a: u32, b: u32, c: u32) u32 {
    return asm volatile ("dp4a.u32.u32 %[ret], %[a], %[b], %[c];"
        : [ret] "=r" (-> u32),
        : [a] "r" (a),
          [b] "r" (b),
          [c] "r" (c),
    );
}

/// __dp4a_s32 — signed variant
pub inline fn __dp4a_s32(a: i32, b: i32, c: i32) i32 {
    return asm volatile ("dp4a.s32.s32 %[ret], %[a], %[b], %[c];"
        : [ret] "=r" (-> i32),
        : [a] "r" (a),
          [b] "r" (b),
          [c] "r" (c),
    );
}

/// __dp2a_lo — 2-element dot product (low halfwords) with accumulate
pub inline fn __dp2a_lo(a: u32, b: u32, c: u32) u32 {
    return asm volatile ("dp2a.lo.u32.u32 %[ret], %[a], %[b], %[c];"
        : [ret] "=r" (-> u32),
        : [a] "r" (a),
          [b] "r" (b),
          [c] "r" (c),
    );
}

/// __dp2a_hi — 2-element dot product (high halfwords) with accumulate
pub inline fn __dp2a_hi(a: u32, b: u32, c: u32) u32 {
    return asm volatile ("dp2a.hi.u32.u32 %[ret], %[a], %[b], %[c];"
        : [ret] "=r" (-> u32),
        : [a] "r" (a),
          [b] "r" (b),
          [c] "r" (c),
    );
}

// ============================================================================
// Cache Hint Load/Store — P3 (10.11/10.12)
// ============================================================================

/// __ldca — cache-all load, equivalent to CUDA C++ `__ldca()`
pub inline fn __ldca(ptr: *const f32) f32 {
    return asm volatile ("ld.global.ca.f32 %[ret], [%[ptr]];"
        : [ret] "=f" (-> f32),
        : [ptr] "l" (ptr),
    );
}

/// __ldcs — cache-streaming load
pub inline fn __ldcs(ptr: *const f32) f32 {
    return asm volatile ("ld.global.cs.f32 %[ret], [%[ptr]];"
        : [ret] "=f" (-> f32),
        : [ptr] "l" (ptr),
    );
}

/// __ldcg — cache-global load
pub inline fn __ldcg(ptr: *const f32) f32 {
    return asm volatile ("ld.global.cg.f32 %[ret], [%[ptr]];"
        : [ret] "=f" (-> f32),
        : [ptr] "l" (ptr),
    );
}

/// __stcg — cache-global store
pub inline fn __stcg(ptr: *f32, val: f32) void {
    asm volatile ("st.global.cg.f32 [%[ptr]], %[val];"
        :
        : [ptr] "l" (ptr),
          [val] "f" (val),
    );
}

/// __stcs — cache-streaming store
pub inline fn __stcs(ptr: *f32, val: f32) void {
    asm volatile ("st.global.cs.f32 [%[ptr]], %[val];"
        :
        : [ptr] "l" (ptr),
          [val] "f" (val),
    );
}

/// __stwb — write-back store (bypass L1)
pub inline fn __stwb(ptr: *f32, val: f32) void {
    asm volatile ("st.global.wb.f32 [%[ptr]], %[val];"
        :
        : [ptr] "l" (ptr),
          [val] "f" (val),
    );
}

// ============================================================================
// Address Space Predicate Functions — P3 (10.15)
// ============================================================================

/// __isGlobal — returns true if pointer is in global memory
pub inline fn __isGlobal(ptr: *const anyopaque) bool {
    const result = asm volatile ("isspacep.global %[ret], %[ptr];"
        : [ret] "=r" (-> u32),
        : [ptr] "l" (ptr),
    );
    return result != 0;
}

/// __isShared — returns true if pointer is in shared memory
pub inline fn __isShared(ptr: *const anyopaque) bool {
    const result = asm volatile ("isspacep.shared %[ret], %[ptr];"
        : [ret] "=r" (-> u32),
        : [ptr] "l" (ptr),
    );
    return result != 0;
}

/// __isConstant — returns true if pointer is in constant memory
pub inline fn __isConstant(ptr: *const anyopaque) bool {
    const result = asm volatile ("isspacep.const %[ret], %[ptr];"
        : [ret] "=r" (-> u32),
        : [ptr] "l" (ptr),
    );
    return result != 0;
}

/// __isLocal — returns true if pointer is in local memory
pub inline fn __isLocal(ptr: *const anyopaque) bool {
    const result = asm volatile ("isspacep.local %[ret], %[ptr];"
        : [ret] "=r" (-> u32),
        : [ptr] "l" (ptr),
    );
    return result != 0;
}

// ============================================================================
// Nanosleep — P3 (10.23, sm_70+)
// ============================================================================

/// __nanosleep — equivalent to CUDA C++ `__nanosleep()` (sm_70+)
/// Suspends the thread for approximately `ns` nanoseconds.
pub inline fn __nanosleep(ns: u32) void {
    asm volatile ("nanosleep.u32 %[ns];"
        :
        : [ns] "r" (ns),
    );
}

// ============================================================================
// Byte Permute — P3
// ============================================================================

/// __byte_perm — equivalent to CUDA C++ `__byte_perm()`
/// Selects 4 bytes from the 8-byte value {b, a} using selector `s`.
pub inline fn __byte_perm(a: u32, b: u32, s: u32) u32 {
    return asm volatile ("prmt.b32 %[ret], %[a], %[b], %[s];"
        : [ret] "=r" (-> u32),
        : [a] "r" (a),
          [b] "r" (b),
          [s] "r" (s),
    );
}

// ============================================================================
// Additional Math — P3
// ============================================================================

/// __saturatef — clamp float to [0.0, 1.0], equivalent to CUDA C++ `__saturatef()`
pub inline fn __saturatef(x: f32) f32 {
    return fminf(fmaxf(x, 0.0), 1.0);
}

/// __tanf — fast tangent approximation, equivalent to CUDA C++ `__tanf()`
/// Computed as sin(x)/cos(x) using hardware approximations.
pub inline fn __tanf(x: f32) f32 {
    return __fdividef(__sinf(x), __cosf(x));
}

// ============================================================================
// Type Conversion Intrinsics — P3 (10.8)
// ============================================================================

/// __float2int_rn — convert f32 to i32 with round-to-nearest
pub inline fn __float2int_rn(x: f32) i32 {
    return asm volatile ("cvt.rni.s32.f32 %[ret], %[x];"
        : [ret] "=r" (-> i32),
        : [x] "f" (x),
    );
}

/// __float2int_rz — convert f32 to i32 with round-toward-zero (truncate)
pub inline fn __float2int_rz(x: f32) i32 {
    return asm volatile ("cvt.rzi.s32.f32 %[ret], %[x];"
        : [ret] "=r" (-> i32),
        : [x] "f" (x),
    );
}

/// __float2uint_rn — convert f32 to u32 with round-to-nearest
pub inline fn __float2uint_rn(x: f32) u32 {
    return asm volatile ("cvt.rni.u32.f32 %[ret], %[x];"
        : [ret] "=r" (-> u32),
        : [x] "f" (x),
    );
}

/// __float2uint_rz — convert f32 to u32 with round-toward-zero
pub inline fn __float2uint_rz(x: f32) u32 {
    return asm volatile ("cvt.rzi.u32.f32 %[ret], %[x];"
        : [ret] "=r" (-> u32),
        : [x] "f" (x),
    );
}

/// __int2float_rn — convert i32 to f32 with round-to-nearest
pub inline fn __int2float_rn(x: i32) f32 {
    return asm volatile ("cvt.rn.f32.s32 %[ret], %[x];"
        : [ret] "=f" (-> f32),
        : [x] "r" (x),
    );
}

/// __uint2float_rn — convert u32 to f32 with round-to-nearest
pub inline fn __uint2float_rn(x: u32) f32 {
    return asm volatile ("cvt.rn.f32.u32 %[ret], %[x];"
        : [ret] "=f" (-> f32),
        : [x] "r" (x),
    );
}

/// __float_as_int — reinterpret f32 bits as i32
pub inline fn __float_as_int(x: f32) i32 {
    return @bitCast(x);
}

/// __int_as_float — reinterpret i32 bits as f32
pub inline fn __int_as_float(x: i32) f32 {
    return @bitCast(x);
}

/// __float_as_uint — reinterpret f32 bits as u32
pub inline fn __float_as_uint(x: f32) u32 {
    return @bitCast(x);
}

/// __uint_as_float — reinterpret u32 bits as f32
pub inline fn __uint_as_float(x: u32) f32 {
    return @bitCast(x);
}

/// __double2hiint — extract high 32 bits of a f64
pub inline fn __double2hiint(x: f64) i32 {
    const bits: u64 = @bitCast(x);
    return @bitCast(@as(u32, @truncate(bits >> 32)));
}

/// __double2loint — extract low 32 bits of a f64
pub inline fn __double2loint(x: f64) i32 {
    const bits: u64 = @bitCast(x);
    return @bitCast(@as(u32, @truncate(bits)));
}

/// __hiloint2double — pack two i32 into a f64
pub inline fn __hiloint2double(hi: i32, lo: i32) f64 {
    const hi_u: u64 = @as(u32, @bitCast(hi));
    const lo_u: u64 = @as(u32, @bitCast(lo));
    return @bitCast((hi_u << 32) | lo_u);
}
