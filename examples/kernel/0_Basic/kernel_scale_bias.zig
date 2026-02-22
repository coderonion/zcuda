// kernel_scale_bias.zig — Scale + Bias: data[i] = data[i] * alpha + bias
const cuda = @import("zcuda_kernel");

export fn scaleBias(data: [*]f32, alpha: f32, bias: f32, n: u32) callconv(.kernel) void {
    const idx = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (idx < n) {
        data[idx] = data[idx] * alpha + bias;
    }
}
