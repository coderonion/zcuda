// examples/kernel/3_Atomics/kernel_warp_aggregated_atomics.zig — Warp-aggregated atomicAdd
//
// Reference: cuda-samples/6_Advanced/warpAggregatedAtomicsCG
// API exercised: __ballot_sync, __popc, __shfl_sync, atomicAdd, FULL_MASK
//
// Optimization: Instead of N threads each doing atomicAdd(1), use ballot+popc
// to count active threads in the warp, then have the leader do a single atomicAdd.
// Reduces atomic traffic by up to 32×.

const cuda = @import("zcuda_kernel");

/// Warp-aggregated atomic increment.
/// Only the elected warp leader performs the atomic operation.
export fn warpAggregatedCount(
    data: [*]const u32,
    predicate_threshold: u32,
    result: *u32,
    n: u32,
) callconv(.kernel) void {
    const gid = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (gid >= n) return;

    // Each thread evaluates its predicate
    const pred = data[gid] > predicate_threshold;

    // Ballot: build a bitmask of threads with true predicate
    const ballot = cuda.__ballot_sync(cuda.FULL_MASK, pred);

    // Count set bits = number of threads that matched
    const count = cuda.__popc(ballot);

    // Elect the first active lane as leader
    // ffs (find first set) equivalent: lane 0 of the ballot mask
    const lane = cuda.threadIdx().x % cuda.warpSize;
    const first_lane = @ctz(ballot);

    // Only the leader does the atomic
    if (lane == first_lane and count > 0) {
        _ = cuda.atomicAdd(result, count);
    }
}
