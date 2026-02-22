// kernel_extract_diag.zig — Extract diagonal from square matrix
const cuda = @import("zcuda_kernel");

export fn extractDiagonal(matrix: [*]const f32, diagonal: [*]f32, n: u32, stride: u32) callconv(.kernel) void {
    const idx = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (idx < n) {
        diagonal[idx] = matrix[idx * stride + idx];
    }
}
