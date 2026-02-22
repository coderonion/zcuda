// kernel_relu.zig — ReLU activation kernel: x = max(x, 0)
const cuda = @import("zcuda_kernel");

export fn relu(data: [*]f32, n: u32) callconv(.kernel) void {
    const idx = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (idx < n) {
        const v = data[idx];
        data[idx] = if (v > 0.0) v else 0.0;
    }
}
