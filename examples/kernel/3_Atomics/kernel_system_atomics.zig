// examples/kernel/3_Atomics/kernel_system_atomics.zig — System-scope atomics
//
// Reference: cuda-samples/0_Introduction/simpleAtomicIntrinsics (extended)
// API exercised: atomicAdd_f64, atomicAdd (f32, u32, i32), atomicCAS

const cuda = @import("zcuda_kernel");

/// f64 atomic accumulation (requires sm_60+)
/// Useful for high-precision reductions where f32 is insufficient.
export fn atomicAccumulateF64(
    input: [*]const f64,
    result: *f64,
    n: u32,
) callconv(.kernel) void {
    var sum: f64 = 0.0;
    var iter = cuda.types.gridStrideLoop(n);
    while (iter.next()) |i| {
        sum += input[i];
    }

    // Warp-level f64 reduction via bitcast shuffle
    const sum_bits: u64 = @bitCast(sum);
    const lo: u32 = @truncate(sum_bits);
    const hi: u32 = @truncate(sum_bits >> 32);

    // Shuffle each 32-bit half independently
    var lo_r = lo;
    var hi_r = hi;
    var offset: u32 = 16;
    while (offset > 0) : (offset >>= 1) {
        lo_r +%= cuda.__shfl_down_sync(cuda.FULL_MASK, lo_r, offset, 32);
        hi_r +%= cuda.__shfl_down_sync(cuda.FULL_MASK, hi_r, offset, 32);
    }

    if (cuda.threadIdx().x % cuda.warpSize == 0) {
        // Reconstruct f64 from shuffled halves
        const combined: u64 = @as(u64, hi_r) << 32 | @as(u64, lo_r);
        _ = cuda.atomicAdd_f64(result, @bitCast(combined));
    }
}

/// CAS-based atomic max for f32 (CUDA doesn't have native atomicMax for floats)
export fn atomicMaxF32(
    input: [*]const f32,
    result: *f32,
    n: u32,
) callconv(.kernel) void {
    const gid = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (gid >= n) return;

    const val = input[gid];
    // CAS loop for f32 max — standard CUDA pattern
    const result_u32: *u32 = @ptrCast(result);
    var old: u32 = result_u32.*;
    while (true) {
        const old_f: f32 = @bitCast(old);
        if (old_f >= val) break;
        const assumed = old;
        old = cuda.atomicCAS(result_u32, assumed, @bitCast(val));
        if (old == assumed) break;
    }
}
