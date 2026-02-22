// examples/kernel/1_Reduction/kernel_prefix_sum.zig — Inclusive/exclusive prefix sum (scan)
//
// Reference: cuda-samples/2_Concepts_and_Techniques/scan
// API exercised: SharedArray, __syncthreads, __shfl_up_sync

const cuda = @import("zcuda_kernel");
const smem = cuda.shared_mem;

const BLOCK_SIZE = 256;

/// Inclusive prefix sum (Hillis-Steele style within a block).
/// Each block independently scans its segment.
export fn inclusiveScanBlock(
    data: [*]f32,
    n: u32,
) callconv(.kernel) void {
    const tile = smem.SharedArray(f32, BLOCK_SIZE);
    const s = tile.ptr();
    const tid = cuda.threadIdx().x;
    const gid = cuda.blockIdx().x * cuda.blockDim().x + tid;

    // Load
    s[tid] = if (gid < n) data[gid] else 0.0;
    cuda.__syncthreads();

    // Up-sweep (Hillis-Steele)
    var offset: u32 = 1;
    while (offset < BLOCK_SIZE) : (offset *= 2) {
        const val = if (tid >= offset) s[tid - offset] else 0.0;
        cuda.__syncthreads();
        s[tid] += val;
        cuda.__syncthreads();
    }

    // Store
    if (gid < n) {
        data[gid] = s[tid];
    }
}

/// Warp-level inclusive scan using shfl_up (no shared memory needed)
export fn inclusiveScanWarp(
    data: [*]f32,
    n: u32,
) callconv(.kernel) void {
    const gid = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (gid >= n) return;

    var val = data[gid];

    // Warp-level inclusive scan via shfl_up
    var offset: u32 = 1;
    while (offset < 32) : (offset *= 2) {
        const received: f32 = @bitCast(cuda.__shfl_up_sync(cuda.FULL_MASK, @bitCast(val), offset, 32));
        if (cuda.threadIdx().x % 32 >= offset) {
            val += received;
        }
    }

    data[gid] = val;
}
