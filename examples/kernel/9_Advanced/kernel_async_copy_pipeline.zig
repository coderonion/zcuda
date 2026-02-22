// examples/kernel/9_Advanced/kernel_async_copy_pipeline.zig — Multi-stage async pipeline
//
// Reference: cuda-samples/3_CUDA_Features/globalToShmemAsyncCopy
// API exercised: SharedArray, __syncthreads, gridStrideLoop
//
// Note: True cp.async requires sm_80+ PTX. This demonstrates the
// multi-stage pipeline pattern using synchronous copies as a baseline,
// which will be upgraded to cp.async when building for sm_80+.

const cuda = @import("zcuda_kernel");
const smem = cuda.shared_mem;

const BLOCK_SIZE = 128;
const STAGES = 2; // double-buffered

/// Double-buffered pipeline: overlap global memory loads with computation.
/// Stage S loads data while stage (S^1) computes, maximizing memory/compute overlap.
export fn pipelinedTransform(
    input: [*]const f32,
    output: [*]f32,
    scale: f32,
    bias: f32,
    n: u32,
) callconv(.kernel) void {
    // Combined double buffer in shared memory — avoids Zig comptime type aliasing
    const combined = smem.SharedArray(f32, BLOCK_SIZE * 2);
    const buffers: [2][*]f32 = .{ combined.ptr(), combined.ptr() + BLOCK_SIZE };

    const tid = cuda.threadIdx().x;
    const blocks_total = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    var block_iter: u32 = cuda.blockIdx().x;

    // Pipeline prologue: fill stage 0
    if (block_iter < blocks_total) {
        const gid = block_iter * BLOCK_SIZE + tid;
        buffers[0][tid] = if (gid < n) input[gid] else 0.0;
    }
    cuda.__syncthreads();

    var stage: u32 = 0;
    while (block_iter < blocks_total) : (block_iter += cuda.gridDim().x) {
        const cur = stage & 1;
        const nxt = cur ^ 1;

        // Prefetch next block into alternate buffer
        const next_block = block_iter + cuda.gridDim().x;
        if (next_block < blocks_total) {
            const next_gid = next_block * BLOCK_SIZE + tid;
            buffers[nxt][tid] = if (next_gid < n) input[next_gid] else 0.0;
        }

        // Compute on current buffer: y = scale * x + bias
        const gid = block_iter * BLOCK_SIZE + tid;
        if (gid < n) {
            output[gid] = cuda.__fmaf_rn(scale, buffers[cur][tid], bias);
        }

        cuda.__syncthreads();
        stage += 1;
    }
}
