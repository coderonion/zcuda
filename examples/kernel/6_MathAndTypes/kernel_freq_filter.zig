// kernel_freq_filter.zig — Zero out frequency bins outside a band
const cuda = @import("zcuda_kernel");

export fn bandpassFilter(data_re: [*]f32, data_im: [*]f32, low_bin: u32, high_bin: u32, n: u32) callconv(.kernel) void {
    const idx = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (idx < n) {
        if (idx < low_bin or idx > high_bin) {
            data_re[idx] = 0.0;
            data_im[idx] = 0.0;
        }
    }
}
