// examples/kernel/4_SharedMemory/kernel_stencil_1d.zig — 1D stencil with halo exchange
//
// Reference: cuda-samples/3_CUDA_Features/cudaCompressibleMemory (stencil pattern)
// API exercised: SharedArray, __syncthreads, loadToShared, blockIdx/threadIdx

const cuda = @import("zcuda_kernel");
const smem = cuda.shared_mem;

const BLOCK_SIZE = 256;
const HALO = 1; // stencil radius

/// 1D 3-point stencil: output[i] = 0.25*input[i-1] + 0.5*input[i] + 0.25*input[i+1]
/// Uses shared memory with halo regions to avoid redundant global loads.
export fn stencil1D(
    input: [*]const f32,
    output: [*]f32,
    n: u32,
) callconv(.kernel) void {
    // Shared memory with halo on both sides
    const tile = smem.SharedArray(f32, BLOCK_SIZE + 2 * HALO);
    const s = tile.ptr();
    const tid = cuda.threadIdx().x;
    const gid = cuda.blockIdx().x * BLOCK_SIZE + tid;

    // Load center region
    s[tid + HALO] = if (gid < n) input[gid] else 0.0;

    // Load left halo
    if (tid < HALO) {
        const left_idx = cuda.blockIdx().x * BLOCK_SIZE - HALO + tid;
        s[tid] = if (cuda.blockIdx().x > 0) input[left_idx] else 0.0;
    }

    // Load right halo
    if (tid >= BLOCK_SIZE - HALO) {
        const right_idx = gid + HALO;
        s[tid + 2 * HALO] = if (right_idx < n) input[right_idx] else 0.0;
    }

    cuda.__syncthreads();

    // Apply stencil
    if (gid < n) {
        output[gid] = 0.25 * s[tid + HALO - 1] + 0.5 * s[tid + HALO] + 0.25 * s[tid + HALO + 1];
    }
}

/// 1D 5-point stencil (radius=2)
export fn stencil1D_5pt(
    input: [*]const f32,
    output: [*]f32,
    n: u32,
) callconv(.kernel) void {
    const R = 2;
    const tile = smem.SharedArray(f32, BLOCK_SIZE + 2 * R);
    const s = tile.ptr();
    const tid = cuda.threadIdx().x;
    const gid = cuda.blockIdx().x * BLOCK_SIZE + tid;

    // Load center
    s[tid + R] = if (gid < n) input[gid] else 0.0;

    // Load halos
    if (tid < R) {
        const left = cuda.blockIdx().x * BLOCK_SIZE - R + tid;
        s[tid] = if (cuda.blockIdx().x > 0 and left < n) input[left] else 0.0;
    }
    if (tid >= BLOCK_SIZE - R) {
        const right = gid + R;
        s[tid + 2 * R] = if (right < n) input[right] else 0.0;
    }

    cuda.__syncthreads();

    if (gid < n) {
        const idx = tid + R;
        output[gid] = 0.0625 * s[idx - 2] + 0.25 * s[idx - 1] + 0.375 * s[idx] + 0.25 * s[idx + 1] + 0.0625 * s[idx + 2];
    }
}
