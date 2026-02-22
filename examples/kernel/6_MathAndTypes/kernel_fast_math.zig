// kernels/math_test.zig — Test kernel exercising new P1/P2 math & warp intrinsics
const cuda = @import("zcuda_kernel");

export fn math_kernel(
    input: [*]const f32,
    output: [*]f32,
    n: u32,
) callconv(.kernel) void {
    const i = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (i >= n) return;

    const x = input[i];

    // Fast math chain: sin(x) + cos(x) + log(x) + rsqrt(x)
    const result = cuda.__sinf(x) + cuda.__cosf(x) + cuda.__logf(x) + cuda.rsqrtf(x);

    // FMA: a * b + c
    const fma_result = cuda.__fmaf_rn(x, x, result);

    // Min/max clamping
    const clamped = cuda.fmaxf(cuda.fminf(fma_result, 100.0), 0.0);

    output[i] = clamped;
}

export fn bitops_kernel(
    input: [*]const u32,
    output: [*]u32,
    n: u32,
) callconv(.kernel) void {
    const i = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (i >= n) return;

    const x = input[i];

    // Integer intrinsics: clz + popc + brev
    const clz = cuda.__clz(x);
    const popc = cuda.__popc(x);
    const rev = cuda.__brev(x);

    output[i] = clz +% popc +% rev;
}
