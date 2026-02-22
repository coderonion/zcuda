// kernels/histogram.zig — GPU histogram kernel using atomics
//
// Features: atomicAdd, grid-stride loop, integer operations

const cuda = @import("zcuda_kernel");

/// Compute histogram of byte values (256 bins).
/// Each thread processes multiple elements via grid-stride loop and
/// atomically increments the corresponding bin.
export fn histogram256(
    data: [*]const u32,
    bins: [*]u32,
    n: u32,
) callconv(.kernel) void {
    var iter = cuda.types.gridStrideLoop(n);
    while (iter.next()) |i| {
        // Extract each byte from the u32 and increment its bin
        const val = data[i];
        _ = cuda.atomicAdd(&bins[val & 0xFF], @as(u32, 1));
        _ = cuda.atomicAdd(&bins[(val >> 8) & 0xFF], @as(u32, 1));
        _ = cuda.atomicAdd(&bins[(val >> 16) & 0xFF], @as(u32, 1));
        _ = cuda.atomicAdd(&bins[(val >> 24) & 0xFF], @as(u32, 1));
    }
}

/// Simple histogram — each element is treated as a bin index directly
export fn histogramSimple(
    indices: [*]const u32,
    bins: [*]u32,
    n: u32,
) callconv(.kernel) void {
    var iter = cuda.types.gridStrideLoop(n);
    while (iter.next()) |i| {
        _ = cuda.atomicAdd(&bins[indices[i]], @as(u32, 1));
    }
}
