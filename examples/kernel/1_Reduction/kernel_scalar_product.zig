// examples/kernel/1_Reduction/kernel_scalar_product.zig — Batched dot products
//
// Reference: cuda-samples/0_Introduction/fp16ScalarProduct
// API exercised: SharedArray, __fmaf_rn, __syncthreads, reduceSum

const cuda = @import("zcuda_kernel");
const smem = cuda.shared_mem;

const BLOCK_SIZE = 256;

/// Batched scalar (dot) product: result[b] = dot(A[b], B[b])
/// Each block computes one dot product of length `vec_len`.
export fn batchedDotProduct(
    A: [*]const f32,
    B: [*]const f32,
    results: [*]f32,
    vec_len: u32,
    num_vectors: u32,
) callconv(.kernel) void {
    const batch = cuda.blockIdx().x;
    if (batch >= num_vectors) return;

    const tile = smem.SharedArray(f32, BLOCK_SIZE);
    const sdata = tile.ptr();
    const tid = cuda.threadIdx().x;
    const offset = batch * vec_len;

    // Each thread accumulates partial dot product
    var sum: f32 = 0.0;
    var i = tid;
    while (i < vec_len) : (i += BLOCK_SIZE) {
        sum = cuda.__fmaf_rn(A[offset + i], B[offset + i], sum);
    }

    sdata[tid] = sum;
    cuda.__syncthreads();

    // Block-level reduction
    smem.reduceSum(f32, sdata, tid, BLOCK_SIZE);

    if (tid == 0) {
        results[batch] = sdata[0];
    }
}
