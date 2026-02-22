// examples/kernel/0_Basic/kernel_dot_product.zig — Dot product via smem + atomicAdd
//
// Reference: cuda-samples/0_Introduction/fp16ScalarProduct
// API exercised: SharedArray, __fmaf_rn, atomicAdd, __syncthreads, gridStrideLoop

const cuda = @import("zcuda_kernel");
const smem = cuda.shared_mem;

const BLOCK_SIZE = 256;

/// Dot product: result = sum(a[i] * b[i])
/// Uses shared memory for block-level reduction + atomicAdd for cross-block.
export fn dotProduct(
    a: [*]const f32,
    b: [*]const f32,
    result: *f32,
    n: u32,
) callconv(.kernel) void {
    const tile = smem.SharedArray(f32, BLOCK_SIZE);
    const sdata = tile.ptr();
    const tid = cuda.threadIdx().x;

    // Grid-stride FMA accumulation
    var sum: f32 = 0.0;
    var iter = cuda.types.gridStrideLoop(n);
    while (iter.next()) |i| {
        sum = cuda.__fmaf_rn(a[i], b[i], sum);
    }

    sdata[tid] = sum;
    cuda.__syncthreads();

    // Tree reduction in shared memory
    smem.reduceSum(f32, sdata, tid, BLOCK_SIZE);

    // Block leader atomically accumulates
    if (tid == 0) {
        _ = cuda.atomicAdd(result, sdata[0]);
    }
}
