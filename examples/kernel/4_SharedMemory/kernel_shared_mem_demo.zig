// kernels/shared_mem_test.zig — Test kernel exercising shared memory features
//
// Features: SharedArray (static smem), cooperative load, syncthreads, tree reduction
//
// This kernel computes a block-level reduction using shared memory,
// demonstrating:
//   1. Static shared memory allocation via SharedArray
//   2. Cooperative global→shared loading
//   3. Tree-based reduction in shared memory
//   4. Atomic accumulation of block results

const cuda = @import("zcuda_kernel");
const smem = cuda.shared_mem;

const BLOCK_SIZE = 256;

/// Block-level sum reduction using shared memory
export fn smem_reduce(
    input: [*]const f32,
    output: *f32,
    n: u32,
) callconv(.kernel) void {
    // 1. Declare static shared memory — per-block, 256 floats
    const tile = smem.SharedArray(f32, BLOCK_SIZE);
    const sdata = tile.ptr();

    const tid = cuda.threadIdx().x;
    const gid = cuda.blockIdx().x * cuda.blockDim().x + tid;

    // 2. Load from global to shared (or zero if out of bounds)
    if (gid < n) {
        sdata[tid] = input[gid];
    } else {
        sdata[tid] = 0.0;
    }
    cuda.__syncthreads();

    // 3. Tree reduction in shared memory
    smem.reduceSum(f32, sdata, tid, BLOCK_SIZE);

    // 4. Thread 0 of each block atomically adds to output
    if (tid == 0) {
        _ = cuda.atomicAdd(output, sdata[0]);
    }
}

/// Shared memory transpose — demonstrates 2D shared memory tiling
export fn smem_transpose(
    input: [*]const f32,
    output: [*]f32,
    width: u32,
    height: u32,
) callconv(.kernel) void {
    const TILE = 16;
    const tile = smem.SharedArray(f32, TILE * TILE);
    const sdata = tile.ptr();

    const tx = cuda.threadIdx().x;
    const ty = cuda.threadIdx().y;
    const bx = cuda.blockIdx().x;
    const by = cuda.blockIdx().y;

    // Read input tile into shared memory
    const in_x = bx * TILE + tx;
    const in_y = by * TILE + ty;
    if (in_x < width and in_y < height) {
        sdata[ty * TILE + tx] = input[in_y * width + in_x];
    }
    cuda.__syncthreads();

    // Write transposed tile to output
    const out_x = by * TILE + tx;
    const out_y = bx * TILE + ty;
    if (out_x < height and out_y < width) {
        output[out_y * height + out_x] = sdata[tx * TILE + ty];
    }
}
