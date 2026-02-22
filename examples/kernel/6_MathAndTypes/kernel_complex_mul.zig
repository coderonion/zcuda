// kernel_complex_mul.zig — Element-wise complex multiplication for FFT convolution
const cuda = @import("zcuda_kernel");

export fn complexMul(
    a_re: [*]const f32,
    a_im: [*]const f32,
    b_re: [*]const f32,
    b_im: [*]const f32,
    out_re: [*]f32,
    out_im: [*]f32,
    n: u32,
) callconv(.kernel) void {
    const idx = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (idx < n) {
        // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        const ar = a_re[idx];
        const ai = a_im[idx];
        const br = b_re[idx];
        const bi = b_im[idx];
        out_re[idx] = ar * br - ai * bi;
        out_im[idx] = ar * bi + ai * br;
    }
}
