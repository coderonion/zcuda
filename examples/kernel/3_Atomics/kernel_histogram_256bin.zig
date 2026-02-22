// examples/kernel/3_Atomics/kernel_histogram_256bin.zig — 256-bin histogram with shared+global
//
// Reference: cuda-samples/2_Concepts_and_Techniques/histogram
// API exercised: SharedArray, atomicAdd (shared+global), __syncthreads, clearShared

const cuda = @import("zcuda_kernel");
const smem = cuda.shared_mem;

/// Privatized histogram: each block uses shared memory bins,
/// then atomically merges into global bins.
/// This avoids global atomic contention.
export fn histogram256Privatized(
    data: [*]const u8,
    global_bins: [*]u32,
    n: u32,
) callconv(.kernel) void {
    const local_bins = smem.SharedArray(u32, 256);
    const lb = local_bins.ptr();
    const tid = cuda.threadIdx().x;

    // Zero shared bins cooperatively
    smem.clearShared(u32, lb, 256);
    cuda.__syncthreads();

    // Accumulate into shared bins
    var iter = cuda.types.gridStrideLoop(n);
    while (iter.next()) |i| {
        const bin = @as(u32, data[i]);
        _ = cuda.atomicAdd(&lb[bin], @as(u32, 1));
    }
    cuda.__syncthreads();

    // Merge shared → global (one thread per bin)
    if (tid < 256) {
        _ = cuda.atomicAdd(&global_bins[tid], lb[tid]);
    }
}
