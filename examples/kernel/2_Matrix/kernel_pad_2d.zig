// kernel_pad_2d.zig — Zero-pad a 2D matrix for FFT convolution
const cuda = @import("zcuda_kernel");

export fn zeroPad2d(src: [*]const f32, dst: [*]f32, src_rows: u32, src_cols: u32, dst_cols: u32) callconv(.kernel) void {
    const idx = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    const row = idx / dst_cols;
    const col = idx % dst_cols;
    if (row < src_rows and col < src_cols) {
        dst[idx] = src[row * src_cols + col];
    } else {
        dst[idx] = 0.0;
    }
}
