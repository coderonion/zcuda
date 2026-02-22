/// kernel_math_test.zig — Test kernels for math intrinsics verification
///
/// Functions: sincos_identity, explog_roundtrip, atomic_sum, test_clz, test_popc, test_brev_roundtrip
const cuda = @import("zcuda_kernel");

// sin²(x) + cos²(x) → output[i]  (should ≈ 1.0)
export fn sincos_identity(input: [*]const f32, output: [*]f32, n: u32) callconv(.kernel) void {
    const tid = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (tid >= n) return;
    const x = input[tid];
    const s = cuda.__sinf(x);
    const c = cuda.__cosf(x);
    output[tid] = s * s + c * c;
}

// log(exp(x)) → output[i]  (should ≈ x)
export fn explog_roundtrip(input: [*]const f32, output: [*]f32, n: u32) callconv(.kernel) void {
    const tid = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (tid >= n) return;
    output[tid] = cuda.__logf(cuda.__expf(input[tid]));
}

// Each thread atomically adds 1.0 to output[0]
export fn atomic_sum(output: [*]f32, n: u32) callconv(.kernel) void {
    const tid = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (tid >= n) return;
    _ = cuda.atomicAdd(&output[0], @as(f32, 1.0));
}

// Count leading zeros
export fn test_clz(input: [*]const u32, output: [*]u32, n: u32) callconv(.kernel) void {
    const tid = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (tid >= n) return;
    output[tid] = cuda.__clz(input[tid]);
}

// Population count (number of set bits)
export fn test_popc(input: [*]const u32, output: [*]u32, n: u32) callconv(.kernel) void {
    const tid = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (tid >= n) return;
    output[tid] = cuda.__popc(input[tid]);
}

// Bit reverse involution: brev(brev(x)) == x
export fn test_brev_roundtrip(input: [*]const u32, output: [*]u32, n: u32) callconv(.kernel) void {
    const tid = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (tid >= n) return;
    output[tid] = cuda.__brev(cuda.__brev(input[tid]));
}
