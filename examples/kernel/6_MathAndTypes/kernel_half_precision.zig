// examples/kernel/6_MathAndTypes/kernel_half_precision.zig — Half-precision (f16) operations
//
// Reference: cuda-samples/0_Introduction/fp16ScalarProduct
// API exercised: f16 type, type conversions, grid-stride patterns
//
// Note: f16 compute requires sm_53+, f16 storage is always available.

const cuda = @import("zcuda_kernel");

/// Convert f32 array to f16 for storage
export fn f32ToF16(
    input: [*]const f32,
    output: [*]f16,
    n: u32,
) callconv(.kernel) void {
    var iter = cuda.types.gridStrideLoop(n);
    while (iter.next()) |i| {
        output[i] = @floatCast(input[i]);
    }
}

/// Convert f16 array to f32 for compute
export fn f16ToF32(
    input: [*]const f16,
    output: [*]f32,
    n: u32,
) callconv(.kernel) void {
    var iter = cuda.types.gridStrideLoop(n);
    while (iter.next()) |i| {
        output[i] = @floatCast(input[i]);
    }
}

/// f16 SAXPY: y = a*x + y in half precision
export fn f16Saxpy(
    x: [*]const f16,
    y: [*]f16,
    a: f16,
    n: u32,
) callconv(.kernel) void {
    var iter = cuda.types.gridStrideLoop(n);
    while (iter.next()) |i| {
        // Promote to f32 for FMA, then truncate back
        const xf: f32 = @floatCast(x[i]);
        const yf: f32 = @floatCast(y[i]);
        const af: f32 = @floatCast(a);
        y[i] = @floatCast(cuda.__fmaf_rn(af, xf, yf));
    }
}

/// Mixed-precision dot product: accumulate f16 inputs into f32
export fn f16DotProduct(
    a: [*]const f16,
    b: [*]const f16,
    result: *f32,
    n: u32,
) callconv(.kernel) void {
    var sum: f32 = 0.0;
    var iter = cuda.types.gridStrideLoop(n);
    while (iter.next()) |i| {
        const af: f32 = @floatCast(a[i]);
        const bf: f32 = @floatCast(b[i]);
        sum = cuda.__fmaf_rn(af, bf, sum);
    }

    // Warp reduction
    sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(sum), 16, 32));
    sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(sum), 8, 32));
    sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(sum), 4, 32));
    sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(sum), 2, 32));
    sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(sum), 1, 32));

    if (cuda.threadIdx().x % cuda.warpSize == 0) {
        _ = cuda.atomicAdd(result, sum);
    }
}
