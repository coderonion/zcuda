// kernels/matmul.zig — Naive matrix multiplication kernel
//
// Features: 2D thread indexing, FMA, grid-stride concepts
//
// C[i][j] = sum(A[i][k] * B[k][j]) for k = 0..K

const cuda = @import("zcuda_kernel");

/// Naive matrix multiplication: C = A × B
/// A is M×K, B is K×N, C is M×N (all row-major)
export fn matmulNaive(
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    M: u32,
    N: u32,
    K: u32,
) callconv(.kernel) void {
    const row = cuda.blockIdx().y * cuda.blockDim().y + cuda.threadIdx().y;
    const col = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;

    if (row >= M or col >= N) return;

    var sum: f32 = 0.0;
    var k: u32 = 0;
    while (k < K) : (k += 1) {
        // C[row][col] += A[row][k] * B[k][col]
        sum = cuda.__fmaf_rn(A[row * K + k], B[k * N + col], sum);
    }

    C[row * N + col] = sum;
}

/// Matrix-vector multiplication: y = A × x
/// A is M×N (row-major), x is N×1, y is M×1
export fn matvecMul(
    A: [*]const f32,
    x: [*]const f32,
    y: [*]f32,
    M: u32,
    N: u32,
) callconv(.kernel) void {
    const row = cuda.types.globalThreadIdx();
    if (row >= M) return;

    var sum: f32 = 0.0;
    var j: u32 = 0;
    while (j < N) : (j += 1) {
        sum = cuda.__fmaf_rn(A[row * N + j], x[j], sum);
    }

    y[row] = sum;
}
