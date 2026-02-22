// kernel_signal_gen.zig — Generate sine wave signal for FFT pipeline
const cuda = @import("zcuda_kernel");

export fn generateSineWave(output: [*]f32, frequency: f32, sample_rate: f32, n: u32) callconv(.kernel) void {
    const idx = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (idx < n) {
        const t = @as(f32, @floatFromInt(idx)) / sample_rate;
        output[idx] = cuda.__sinf(2.0 * 3.14159265 * frequency * t);
    }
}
