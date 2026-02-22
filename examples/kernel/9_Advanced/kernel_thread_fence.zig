// examples/kernel/9_Advanced/kernel_thread_fence.zig — Single-pass threadfence reduction
//
// Reference: cuda-samples/2_Concepts_and_Techniques/threadFenceReduction
// API exercised: __threadfence, atomicAdd, atomicInc, SharedArray, __syncthreads
//
// Algorithm: Each block reduces locally, then uses threadfence + global atomic counter
// to detect when it is the "last block", which then performs the final reduction.

const cuda = @import("zcuda_kernel");
const smem = cuda.shared_mem;

const BLOCK_SIZE = 256;

/// Single-pass reduction using __threadfence for inter-block synchronization.
/// Avoids a second kernel launch by having the last block do final reduction.
export fn threadFenceReduce(
    input: [*]const f32,
    block_results: [*]f32,
    final_result: *f32,
    retired_count: *u32,
    num_blocks: u32,
    n: u32,
) callconv(.kernel) void {
    const tile = smem.SharedArray(f32, BLOCK_SIZE);
    const sdata = tile.ptr();
    const tid = cuda.threadIdx().x;
    const gid = cuda.blockIdx().x * BLOCK_SIZE + tid;

    // Phase 1: Block-local reduction
    sdata[tid] = if (gid < n) input[gid] else 0.0;
    cuda.__syncthreads();
    smem.reduceSum(f32, sdata, tid, BLOCK_SIZE);

    if (tid == 0) {
        // Write block result
        block_results[cuda.blockIdx().x] = sdata[0];

        // Ensure the write is visible to all blocks
        cuda.__threadfence();

        // Atomically increment retired counter
        const ticket = cuda.atomicInc(retired_count, num_blocks);

        // Last block to finish does the final reduction
        if (ticket == num_blocks - 1) {
            // This block is the last one — reduce all block results
            var sum: f32 = 0.0;
            var i: u32 = 0;
            while (i < num_blocks) : (i += 1) {
                sum += block_results[i];
            }
            final_result.* = sum;
        }
    }
}
