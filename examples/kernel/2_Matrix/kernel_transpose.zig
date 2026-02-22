// examples/kernel/2_Matrix/kernel_transpose.zig — Bank-conflict-free matrix transpose
//
// Reference: cuda-samples/0_Introduction/matrixMul, 6_Advanced/transpose
// API exercised: SharedArray, __syncthreads, 2D thread/block indexing

const cuda = @import("zcuda_kernel");
const smem = cuda.shared_mem;

const TILE_DIM = 16;

/// Naive transpose: output[j][i] = input[i][j]
export fn transposeNaive(
    input: [*]const f32,
    output: [*]f32,
    width: u32,
    height: u32,
) callconv(.kernel) void {
    const x = cuda.blockIdx().x * TILE_DIM + cuda.threadIdx().x;
    const y = cuda.blockIdx().y * TILE_DIM + cuda.threadIdx().y;

    if (x < width and y < height) {
        output[x * height + y] = input[y * width + x];
    }
}

/// Shared-memory transpose with padding to avoid bank conflicts.
/// Uses TILE_DIM+1 stride to eliminate 32-bank conflicts.
export fn transposeCoalesced(
    input: [*]const f32,
    output: [*]f32,
    width: u32,
    height: u32,
) callconv(.kernel) void {
    // +1 padding to avoid bank conflicts
    const tile = smem.SharedArray(f32, TILE_DIM * (TILE_DIM + 1));
    const s = tile.ptr();

    const bx = cuda.blockIdx().x;
    const by = cuda.blockIdx().y;
    const tx = cuda.threadIdx().x;
    const ty = cuda.threadIdx().y;

    // Read tile from input (coalesced)
    const in_x = bx * TILE_DIM + tx;
    const in_y = by * TILE_DIM + ty;
    if (in_x < width and in_y < height) {
        s[ty * (TILE_DIM + 1) + tx] = input[in_y * width + in_x];
    }
    cuda.__syncthreads();

    // Write transposed tile to output (coalesced, bank-conflict-free)
    const out_x = by * TILE_DIM + tx;
    const out_y = bx * TILE_DIM + ty;
    if (out_x < height and out_y < width) {
        output[out_y * height + out_x] = s[tx * (TILE_DIM + 1) + ty];
    }
}
