// examples/kernel/1_Reduction/kernel_reduce_multiblock.zig — Multi-block reduction
//
// Reference: cuda-samples/6_Advanced/reduction (kernel 7: multi-block CG)
// API exercised: SharedArray, __shfl_down_sync, atomicAdd, __syncthreads

const cuda = @import("zcuda_kernel");
const smem = cuda.shared_mem;

const BLOCK_SIZE = 256;
const WARPS_PER_BLOCK = BLOCK_SIZE / 32;

/// Multi-block reduction with two-level hierarchy:
///   1. Grid-stride per-thread accumulation
///   2. Warp-level shuffle-down reduction
///   3. Block-level shared memory reduction (warp leaders)
///   4. Block leader atomicAdd to global result
export fn reduceMultiBlock(
    input: [*]const f32,
    result: *f32,
    n: u32,
) callconv(.kernel) void {
    const warp_sums = smem.SharedArray(f32, WARPS_PER_BLOCK);
    const ws = warp_sums.ptr();
    const tid = cuda.threadIdx().x;
    const lane = tid % cuda.warpSize;
    const warp_id = tid / cuda.warpSize;

    // Phase 1: Grid-stride accumulation
    var sum: f32 = 0.0;
    var iter = cuda.types.gridStrideLoop(n);
    while (iter.next()) |i| {
        sum += input[i];
    }

    // Phase 2: Warp-level shuffle reduction
    sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(sum), 16, 32));
    sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(sum), 8, 32));
    sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(sum), 4, 32));
    sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(sum), 2, 32));
    sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(sum), 1, 32));

    // Phase 3: Warp leaders write to shared memory
    if (lane == 0) {
        ws[warp_id] = sum;
    }
    cuda.__syncthreads();

    // Phase 4: First warp reduces warp-level partials
    if (tid < WARPS_PER_BLOCK) {
        sum = ws[tid];
    } else {
        sum = 0.0;
    }

    if (warp_id == 0) {
        sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(sum), 4, 32));
        sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(sum), 2, 32));
        sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(sum), 1, 32));
    }

    // Phase 5: Block leader atomicAdd
    if (tid == 0) {
        _ = cuda.atomicAdd(result, sum);
    }
}
