// examples/kernel/9_Advanced/kernel_cooperative_groups.zig — Cooperative groups reduce
//
// Reference: cuda-samples/3_CUDA_Features/binaryPartitionCG
// API exercised: __syncthreads, __ballot_sync, __shfl_down_sync, SharedArray
//
// Note: Full CG API requires runtime support. This emulates CG patterns
// using existing warp/block primitives.

const cuda = @import("zcuda_kernel");
const smem = cuda.shared_mem;

const BLOCK_SIZE = 256;
const WARPS_PER_BLOCK = BLOCK_SIZE / 32;

/// Binary partition: partition warp into subgroups and reduce within each.
/// Emulates cooperative_groups::binary_partition().
export fn binaryPartitionReduce(
    input: [*]const f32,
    output: [*]f32,
    predicate_val: f32,
    n: u32,
) callconv(.kernel) void {
    const gid = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (gid >= n) return;

    const val = input[gid];
    const pred = val > predicate_val;

    // Partition the warp: get mask of threads with same predicate
    const all_mask = cuda.__ballot_sync(cuda.FULL_MASK, pred);
    const true_mask = all_mask;
    const false_mask = ~all_mask;

    // Each thread determines its partition mask
    const my_mask = if (pred) true_mask else false_mask;
    const partition_size = cuda.__popc(my_mask);

    // Reduce within partition using only matching threads
    var sum = val;
    var offset: u32 = 1;
    while (offset < 32) : (offset *= 2) {
        const received: f32 = @bitCast(cuda.__shfl_down_sync(my_mask, @bitCast(sum), offset, 32));
        sum += received;
    }

    // Normalize by partition size
    output[gid] = sum / @as(f32, @floatFromInt(partition_size));
}

/// Block-level cooperative reduce: all warps cooperate via shared memory
export fn cooperativeBlockReduce(
    input: [*]const f32,
    output: *f32,
    n: u32,
) callconv(.kernel) void {
    const warp_sums = smem.SharedArray(f32, WARPS_PER_BLOCK);
    const ws = warp_sums.ptr();
    const tid = cuda.threadIdx().x;
    const lane = tid % 32;
    const warp_id = tid / 32;

    var sum: f32 = 0.0;
    var iter = cuda.types.gridStrideLoop(n);
    while (iter.next()) |i| {
        sum += input[i];
    }

    // Warp reduce
    sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(sum), 16, 32));
    sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(sum), 8, 32));
    sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(sum), 4, 32));
    sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(sum), 2, 32));
    sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(sum), 1, 32));

    if (lane == 0) ws[warp_id] = sum;
    cuda.__syncthreads();

    // Final reduction by first warp
    if (tid < WARPS_PER_BLOCK) sum = ws[tid] else sum = 0.0;
    if (warp_id == 0) {
        sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(sum), 4, 32));
        sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(sum), 2, 32));
        sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(sum), 1, 32));
    }

    if (tid == 0) _ = cuda.atomicAdd(output, sum);
}
