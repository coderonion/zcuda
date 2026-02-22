// examples/kernel/5_Warp/kernel_warp_broadcast.zig — Warp broadcast via __shfl_sync
//
// Reference: cuda-samples/0_Introduction/simpleVoteIntrinsics (warp communication)
// API exercised: __shfl_sync, __shfl_xor_sync, warpSize

const cuda = @import("zcuda_kernel");

/// Broadcast lane 0's value to all lanes in the warp.
export fn warpBroadcast(
    input: [*]const f32,
    output: [*]f32,
    n: u32,
) callconv(.kernel) void {
    const gid = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (gid >= n) return;

    const val = input[gid];

    // Broadcast: all lanes get lane 0's value
    const broadcast: f32 = @bitCast(cuda.__shfl_sync(cuda.FULL_MASK, @bitCast(val), 0, 32));
    output[gid] = broadcast;
}

/// Warp-level swap: each thread exchanges value with its XOR partner.
/// lane i swaps with lane i^1 (adjacent pair swap).
export fn warpSwapAdjacent(
    data: [*]f32,
    n: u32,
) callconv(.kernel) void {
    const gid = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (gid >= n) return;

    const val = data[gid];
    // XOR with 1: swaps lanes 0↔1, 2↔3, 4↔5, ...
    const partner: f32 = @bitCast(cuda.__shfl_xor_sync(cuda.FULL_MASK, @bitCast(val), 1, 32));
    data[gid] = partner;
}

/// Butterfly pattern: each thread receives value from its XOR partner.
/// Useful for FFT-like communication patterns.
export fn warpButterfly(
    data: [*]f32,
    n: u32,
    xor_mask: u32,
) callconv(.kernel) void {
    const gid = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (gid >= n) return;

    const val = data[gid];
    const partner: f32 = @bitCast(cuda.__shfl_xor_sync(cuda.FULL_MASK, @bitCast(val), xor_mask, 32));

    // Sum with partner (butterfly reduction step)
    data[gid] = val + partner;
}
