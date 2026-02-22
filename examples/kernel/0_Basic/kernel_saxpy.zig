// examples/kernel/0_Basic/kernel_saxpy.zig — SAXPY: y[i] = a*x[i] + y[i]
//
// Reference: cuda-samples/0_Introduction/vectorAdd (BLAS Level-1 variant)
// API exercised: gridStrideLoop, __fmaf_rn
//
// Separated from grid_stride_demo.zig for standalone SAXPY demonstration.

const cuda = @import("zcuda_kernel");

/// SAXPY — Single-Precision A*X Plus Y
/// Uses fused multiply-add for maximum precision and throughput.
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

/// DAXPY — Double-Precision A*X Plus Y (sm_60+ for native f64)
export fn daxpy(
    x: [*]const f64,
    y: [*]f64,
    a: f64,
    n: u32,
) callconv(.kernel) void {
    var iter = cuda.types.gridStrideLoop(n);
    while (iter.next()) |i| {
        y[i] = @mulAdd(f64, a, x[i], y[i]);
    }
}
