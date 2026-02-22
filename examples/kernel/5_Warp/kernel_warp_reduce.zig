// examples/kernel/5_Warp/kernel_warp_reduce.zig — Warp-level shuffle reductions
//
// Reference: cuda-samples/6_Advanced/reduction (warp-shuffle variants)
// API exercised: __shfl_down_sync, __shfl_xor_sync, FULL_MASK, warpSize

const cuda = @import("zcuda_kernel");

/// Warp-level sum reduction via shuffle-down.
/// Returns the sum in lane 0 of each warp.
inline fn warpReduceSum(val: f32) f32 {
    var v = val;
    v += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(v), 16, 32));
    v += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(v), 8, 32));
    v += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(v), 4, 32));
    v += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(v), 2, 32));
    v += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(v), 1, 32));
    return v;
}

/// Warp-level max reduction
inline fn warpReduceMax(val: f32) f32 {
    var v = val;
    v = cuda.fmaxf(v, @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(v), 16, 32)));
    v = cuda.fmaxf(v, @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(v), 8, 32)));
    v = cuda.fmaxf(v, @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(v), 4, 32)));
    v = cuda.fmaxf(v, @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(v), 2, 32)));
    v = cuda.fmaxf(v, @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(v), 1, 32)));
    return v;
}

/// Kernel: per-warp sum reduction, one result per warp written to output
export fn warpReduceSumKernel(
    input: [*]const f32,
    output: [*]f32,
    n: u32,
) callconv(.kernel) void {
    const gid = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    const val = if (gid < n) input[gid] else 0.0;

    const sum = warpReduceSum(val);

    // Lane 0 of each warp writes
    if (cuda.threadIdx().x % cuda.warpSize == 0) {
        const warp_idx = gid / cuda.warpSize;
        output[warp_idx] = sum;
    }
}

/// Kernel: per-warp max reduction
export fn warpReduceMaxKernel(
    input: [*]const f32,
    output: [*]f32,
    n: u32,
) callconv(.kernel) void {
    const gid = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    const val = if (gid < n) input[gid] else -3.40282347e+38;

    const max_val = warpReduceMax(val);

    if (cuda.threadIdx().x % cuda.warpSize == 0) {
        const warp_idx = gid / cuda.warpSize;
        output[warp_idx] = max_val;
    }
}
