// examples/kernel/5_Warp/kernel_ballot_vote.zig — Ballot and vote intrinsics
//
// Reference: cuda-samples/0_Introduction/simpleVoteIntrinsics
// API exercised: __ballot_sync, __all_sync, __any_sync, __popc, FULL_MASK

const cuda = @import("zcuda_kernel");

/// Count elements matching a predicate using ballot + popc.
/// More efficient than atomicAdd per thread.
export fn ballotCount(
    data: [*]const f32,
    threshold: f32,
    block_counts: [*]u32,
    n: u32,
) callconv(.kernel) void {
    const gid = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;

    // Each thread evaluates predicate
    const pred = gid < n and data[gid] > threshold;

    // Get warp-level ballot mask
    const ballot = cuda.__ballot_sync(cuda.FULL_MASK, pred);

    // Lane 0 computes popcount and atomically adds
    if (cuda.threadIdx().x % cuda.warpSize == 0) {
        const count = cuda.__popc(ballot);
        _ = cuda.atomicAdd(&block_counts[cuda.blockIdx().x], count);
    }
}

/// All-sync: check if all threads in warp satisfy predicate
export fn allSyncCheck(
    data: [*]const f32,
    lower: f32,
    upper: f32,
    warp_results: [*]u32,
    n: u32,
) callconv(.kernel) void {
    const gid = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;

    // Predicate: value in range [lower, upper]
    const pred = gid < n and data[gid] >= lower and data[gid] <= upper;

    // all_sync: true only if ALL threads in warp satisfy predicate
    const all_in_range = cuda.__all_sync(cuda.FULL_MASK, pred);

    // any_sync: true if ANY thread in warp satisfies predicate
    const any_in_range = cuda.__any_sync(cuda.FULL_MASK, pred);

    if (cuda.threadIdx().x % cuda.warpSize == 0) {
        const warp_idx = gid / cuda.warpSize;
        // Pack both results: bit 0 = all, bit 1 = any
        warp_results[warp_idx] = @as(u32, @intFromBool(all_in_range)) | (@as(u32, @intFromBool(any_in_range)) << 1);
    }
}
