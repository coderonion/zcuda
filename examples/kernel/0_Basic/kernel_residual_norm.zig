// kernel_residual_norm.zig — Compute residual: r = x - y, in-place
const cuda = @import("zcuda_kernel");

export fn residualNorm(x: [*]f32, y: [*]const f32, n: u32) callconv(.kernel) void {
    const idx = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (idx < n) {
        x[idx] = x[idx] - y[idx];
    }
}
