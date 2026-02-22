// examples/kernel/5_Warp/kernel_warp_scan.zig — Warp-level inclusive scan via shuffle
//
// Reference: cuda-samples/2_Concepts_and_Techniques/scan (warp primitive path)
// API exercised: __shfl_up_sync, FULL_MASK, warpSize

const cuda = @import("zcuda_kernel");

/// Warp-level inclusive scan (prefix sum) using shfl_up.
inline fn warpInclusiveScanSum(val: f32) f32 {
    var v = val;
    const lane = cuda.threadIdx().x % cuda.warpSize;

    var offset: u32 = 1;
    while (offset < 32) : (offset *= 2) {
        const received: f32 = @bitCast(cuda.__shfl_up_sync(cuda.FULL_MASK, @bitCast(v), offset, 32));
        if (lane >= offset) {
            v += received;
        }
    }
    return v;
}

/// Kernel: warp-level inclusive scan, output written in-place
export fn warpScanKernel(
    data: [*]f32,
    n: u32,
) callconv(.kernel) void {
    const gid = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (gid >= n) return;

    var val = data[gid];
    val = warpInclusiveScanSum(val);
    data[gid] = val;
}
