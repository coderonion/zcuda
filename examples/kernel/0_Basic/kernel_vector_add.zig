// kernels/vector_add.zig — GPU kernel using zcuda device intrinsics
//
// Demonstrates seamless migration from CUDA C++:
//
//   CUDA C++:  int i = blockIdx.x * blockDim.x + threadIdx.x;
//   Zig:       const i = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;

const cuda = @import("zcuda_kernel");

/// vectorAdd — element-wise vector addition (matches CUDA C++ naming)
export fn vectorAdd(
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    n: u32,
) callconv(.kernel) void {
    const i = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}
