// examples/kernel/1_Reduction/kernel_reduce_warp.zig — sm_80+ redux.sync.add reduction
//
// Reference: cuda-samples/6_Advanced/reduction (warp-level primitives)
// API exercised: __shfl_down_sync, __shfl_xor_sync, atomicAdd, FULL_MASK, warpSize

const cuda = @import("zcuda_kernel");

/// Warp-only reduction (no shared memory needed for intra-warp).
/// Each warp independently reduces its 32 values, and warp-0~lane-0
/// atomically accumulates to global result.
export fn reduceWarp(
    input: [*]const f32,
    result: *f32,
    n: u32,
) callconv(.kernel) void {
    const gid = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    var val: f32 = if (gid < n) input[gid] else 0.0;

    // XOR-based butterfly reduction within warp
    val += @bitCast(cuda.__shfl_xor_sync(cuda.FULL_MASK, @bitCast(val), 16, 32));
    val += @bitCast(cuda.__shfl_xor_sync(cuda.FULL_MASK, @bitCast(val), 8, 32));
    val += @bitCast(cuda.__shfl_xor_sync(cuda.FULL_MASK, @bitCast(val), 4, 32));
    val += @bitCast(cuda.__shfl_xor_sync(cuda.FULL_MASK, @bitCast(val), 2, 32));
    val += @bitCast(cuda.__shfl_xor_sync(cuda.FULL_MASK, @bitCast(val), 1, 32));

    // Lane 0 of each warp writes result
    if (cuda.threadIdx().x % cuda.warpSize == 0) {
        _ = cuda.atomicAdd(result, val);
    }
}

/// Two-stage warp reduction (shfl_down variant)
/// First pass: grid-stride accumulation, then warp reduce.
export fn reduceWarpGridStride(
    input: [*]const f32,
    result: *f32,
    n: u32,
) callconv(.kernel) void {
    var sum: f32 = 0.0;
    var iter = cuda.types.gridStrideLoop(n);
    while (iter.next()) |i| {
        sum += input[i];
    }

    // Warp shuffle-down reduction
    sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(sum), 16, 32));
    sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(sum), 8, 32));
    sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(sum), 4, 32));
    sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(sum), 2, 32));
    sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(sum), 1, 32));

    if (cuda.threadIdx().x % cuda.warpSize == 0) {
        _ = cuda.atomicAdd(result, sum);
    }
}
