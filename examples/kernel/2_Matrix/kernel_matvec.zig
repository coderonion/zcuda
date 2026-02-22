// examples/kernel/2_Matrix/kernel_matvec.zig — Matrix-vector multiplication y = A*x
//
// Reference: cuda-samples/0_Introduction/matrixMul (adapted for Mx1 output)
// API exercised: globalThreadIdx, __fmaf_rn, gridStrideLoop

const cuda = @import("zcuda_kernel");

/// Matrix-vector multiplication: y = A × x
/// A is M×N (row-major), x is N×1, y is M×1
/// One thread per output row.
export fn matvec(
    A: [*]const f32,
    x: [*]const f32,
    y: [*]f32,
    M: u32,
    N: u32,
) callconv(.kernel) void {
    var iter = cuda.types.gridStrideLoop(M);
    while (iter.next()) |row| {
        var sum: f32 = 0.0;
        var j: u32 = 0;
        while (j < N) : (j += 1) {
            sum = cuda.__fmaf_rn(A[row * N + j], x[j], sum);
        }
        y[row] = sum;
    }
}

/// Sparse-like matvec with stride (for banded matrices)
/// A is M×bandwidth (row-major), col_idx has bandwidth entries per row
export fn matvecBanded(
    A_vals: [*]const f32,
    col_idx: [*]const u32,
    x: [*]const f32,
    y: [*]f32,
    M: u32,
    bandwidth: u32,
) callconv(.kernel) void {
    var iter = cuda.types.gridStrideLoop(M);
    while (iter.next()) |row| {
        var sum: f32 = 0.0;
        var j: u32 = 0;
        while (j < bandwidth) : (j += 1) {
            const col = col_idx[row * bandwidth + j];
            sum = cuda.__fmaf_rn(A_vals[row * bandwidth + j], x[col], sum);
        }
        y[row] = sum;
    }
}
