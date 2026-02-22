// kernel_sigmoid.zig — In-place sigmoid transform: x = 1 / (1 + exp(-x))
const cuda = @import("zcuda_kernel");

export fn sigmoidTransform(data: [*]f32, n: u32) callconv(.kernel) void {
    const idx = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (idx < n) {
        const v = data[idx];
        data[idx] = 1.0 / (1.0 + cuda.__expf(-v));
    }
}
