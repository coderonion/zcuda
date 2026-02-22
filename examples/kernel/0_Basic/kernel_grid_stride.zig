// kernels/grid_stride_demo.zig — Demo kernel showcasing Phase 3 high-level abstractions
//
// Demonstrates:
//   1. gridStrideLoop() — automatic work distribution
//   2. shared Vec3 types — host/device compatible
//   3. Fast math intrinsics — __fmaf_rn, rsqrtf

const cuda = @import("zcuda_kernel");

// ── Grid-Stride Vector Scale ──
// Scales every element in an array by a constant factor.
// Uses gridStrideLoop so a single kernel launch handles any array size.
export fn vectorScale(
    data: [*]f32,
    scale: f32,
    n: u32,
) callconv(.kernel) void {
    var iter = cuda.types.gridStrideLoop(n);
    while (iter.next()) |i| {
        data[i] = data[i] * scale;
    }
}

// ── SAXPY via FMA ──
// y[i] = a * x[i] + y[i] — the classic BLAS Level-1 operation.
// Uses __fmaf_rn for fused multiply-add (single PTX instruction).
export fn saxpy(
    x: [*]const f32,
    y: [*]f32,
    a: f32,
    n: u32,
) callconv(.kernel) void {
    var iter = cuda.types.gridStrideLoop(n);
    while (iter.next()) |i| {
        y[i] = cuda.__fmaf_rn(a, x[i], y[i]);
    }
}

// ── Vec3 Normalize ──
// Normalizes an array of Vec3 vectors to unit length.
// Uses rsqrtf for fast reciprocal square root.
export fn vec3Normalize(
    vectors: [*]cuda.shared.Vec3,
    n: u32,
) callconv(.kernel) void {
    var iter = cuda.types.gridStrideLoop(n);
    while (iter.next()) |i| {
        const v = vectors[i];
        const len_sq = cuda.shared.Vec3.dot(v, v);
        const inv_len = cuda.rsqrtf(len_sq);
        vectors[i] = cuda.shared.Vec3.scale(v, inv_len);
    }
}

// ── Dot Product Reduction (Warp-level) ──
// Computes dot product of two arrays using warp shuffle reduction.
export fn dotProduct(
    a: [*]const f32,
    b: [*]const f32,
    result: *f32,
    n: u32,
) callconv(.kernel) void {
    var sum: f32 = 0.0;

    // Grid-stride accumulation
    var iter = cuda.types.gridStrideLoop(n);
    while (iter.next()) |i| {
        sum = cuda.__fmaf_rn(a[i], b[i], sum);
    }

    // Warp-level reduction via shuffle
    sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(sum), 16, 32));
    sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(sum), 8, 32));
    sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(sum), 4, 32));
    sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(sum), 2, 32));
    sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(sum), 1, 32));

    // Thread 0 of each warp atomically adds to the result
    if (cuda.threadIdx().x % 32 == 0) {
        _ = cuda.atomicAdd(result, sum);
    }
}
