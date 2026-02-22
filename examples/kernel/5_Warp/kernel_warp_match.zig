// examples/kernel/5_Warp/kernel_warp_match.zig — Warp match/segmented operations
//
// Reference: cuda-samples/3_CUDA_Features/warpAggregatedAtomicsCG
// API exercised: __ballot_sync, __shfl_sync, __popc, __ffs equivalent

const cuda = @import("zcuda_kernel");

/// Warp-level histogram using ballot to identify matching values.
/// Each unique value in the warp gets counted via ballot+popc.
export fn warpHistogram(
    data: [*]const u32,
    histogram: [*]u32,
    n: u32,
) callconv(.kernel) void {
    const gid = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (gid >= n) return;

    const my_val = data[gid];
    const lane = cuda.threadIdx().x % cuda.warpSize;

    // Find all lanes with the same value
    // Check each lane's value using broadcast
    var match_mask: u32 = 0;
    var l: u32 = 0;
    while (l < 32) : (l += 1) {
        const other_val = cuda.__shfl_sync(cuda.FULL_MASK, my_val, l, 32);
        if (other_val == my_val) {
            match_mask |= (@as(u32, 1) << @intCast(l));
        }
    }

    // Only the first matching lane (lowest set bit) does the atomic
    const first_match = @ctz(match_mask);
    if (lane == first_match) {
        const count = cuda.__popc(match_mask);
        _ = cuda.atomicAdd(&histogram[my_val], count);
    }
}

/// Segmented warp reduction: reduce within segments defined by a flag array.
/// flag[i] = 1 marks the start of a new segment.
export fn segmentedWarpReduce(
    data: [*]const f32,
    flags: [*]const u32,
    output: [*]f32,
    n: u32,
) callconv(.kernel) void {
    const gid = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (gid >= n) return;

    var val = data[gid];
    const is_head = flags[gid] != 0;

    // Build segment mask using ballot of head flags
    const head_mask = cuda.__ballot_sync(cuda.FULL_MASK, is_head);

    // Each thread finds its segment leader
    const lane = cuda.threadIdx().x % cuda.warpSize;
    // Find the rightmost head at or before this lane
    const mask_up_to_lane = head_mask & ((@as(u32, 2) << @intCast(lane)) - 1);
    const segment_start = 31 - @clz(mask_up_to_lane);

    // Segmented reduction: only add values from the same segment
    var offset: u32 = 1;
    while (offset < 32) : (offset *= 2) {
        const received: f32 = @bitCast(cuda.__shfl_up_sync(cuda.FULL_MASK, @bitCast(val), offset, 32));
        if (lane >= offset and lane - offset >= segment_start) {
            val += received;
        }
    }

    // Segment tails write output
    // A lane is a tail if the next lane is a head (or it's lane 31)
    const is_tail = (lane == 31) or ((head_mask & (@as(u32, 1) << @intCast(lane + 1))) != 0);
    if (is_tail) {
        output[gid] = val;
    }
}
