// examples/kernel/4_SharedMemory/kernel_dynamic_smem.zig — Dynamic shared memory demo
//
// Reference: cuda-samples/0_Introduction/simpleTemplates (dynamic smem partition)
// API exercised: dynamicShared, dynamicSharedBytes, __syncthreads

const cuda = @import("zcuda_kernel");
const smem = cuda.shared_mem;

/// Reduction using dynamically-sized shared memory.
/// The shared memory size is set via LaunchConfig.shared_mem_bytes on the host.
export fn dynamicReduceSum(
    input: [*]const f32,
    output: *f32,
    n: u32,
) callconv(.kernel) void {
    const sdata = smem.dynamicShared(f32);
    const tid = cuda.threadIdx().x;
    const gid = cuda.blockIdx().x * cuda.blockDim().x + tid;

    sdata[tid] = if (gid < n) input[gid] else 0.0;
    cuda.__syncthreads();

    // Tree reduction
    var stride = cuda.blockDim().x >> 1;
    while (stride > 0) : (stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        cuda.__syncthreads();
    }

    if (tid == 0) {
        _ = cuda.atomicAdd(output, sdata[0]);
    }
}

/// Multiple dynamic shared arrays in one allocation.
/// Partitions dynamic smem bytes into two f32 arrays.
export fn dynamicDualBuffer(
    a: [*]const f32,
    b: [*]const f32,
    output: [*]f32,
    n: u32,
) callconv(.kernel) void {
    const base = smem.dynamicSharedBytes();
    const block_size = cuda.blockDim().x;
    const tid = cuda.threadIdx().x;
    const gid = cuda.blockIdx().x * block_size + tid;

    // Partition: first block_size floats for A, next block_size for B
    const sa: [*]f32 = @ptrCast(@alignCast(base));
    const sb: [*]f32 = @ptrCast(@alignCast(base + block_size * @sizeOf(f32)));

    sa[tid] = if (gid < n) a[gid] else 0.0;
    sb[tid] = if (gid < n) b[gid] else 0.0;
    cuda.__syncthreads();

    // Element-wise operation using both shared buffers
    if (gid < n) {
        output[gid] = cuda.__fmaf_rn(sa[tid], sb[tid], sa[tid] + sb[tid]);
    }
}
