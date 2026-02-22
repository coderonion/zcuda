// kernels/intrinsics_test.zig — Comprehensive intrinsic coverage test kernel
//
// Features: Tests P2/P3 intrinsics — sync variants, cache hints, type conversions,
//           address space predicates, nanosleep, byte_perm, saturatef
//
// Each sub-kernel exercises a different intrinsic category.

const cuda = @import("zcuda_kernel");

/// Test sync barrier variants
export fn test_sync_variants(
    output: [*]u32,
    n: u32,
) callconv(.kernel) void {
    const tid = cuda.threadIdx().x;
    if (tid >= n) return;

    // __syncthreads_count: count threads where predicate is true
    const count = cuda.__syncthreads_count(tid < 128);
    output[tid] = count;
}

/// Test type conversion intrinsics
export fn test_type_conversions(
    input_f: [*]const f32,
    out_int: [*]i32,
    out_uint: [*]u32,
    out_float: [*]f32,
    n: u32,
) callconv(.kernel) void {
    const i = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (i >= n) return;

    const x = input_f[i];

    // f32 → i32 conversions
    out_int[i] = cuda.__float2int_rn(x);

    // i32 → f32 conversion
    out_float[i] = cuda.__int2float_rn(out_int[i]);

    // Bit reinterpretation
    out_uint[i] = cuda.__float_as_uint(x);
}

/// Test cache hint load/store
export fn test_cache_hints(
    input: [*]const f32,
    output: [*]f32,
    n: u32,
) callconv(.kernel) void {
    const i = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (i >= n) return;

    // Load with cache-all hint
    const val = cuda.__ldca(&input[i]);

    // Saturate to [0.0, 1.0]
    const sat = cuda.__saturatef(val);

    // Store with write-back hint
    cuda.__stwb(@constCast(&output[i]), sat);
}

/// Test dp4a integer dot product
export fn test_dp4a(
    a: [*]const u32,
    b: [*]const u32,
    c: [*]u32,
    n: u32,
) callconv(.kernel) void {
    const i = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (i >= n) return;

    // 4-element dot product of packed bytes + accumulator
    c[i] = cuda.__dp4a(a[i], b[i], c[i]);
}

/// Test miscellaneous intrinsics
export fn test_misc(
    output: [*]u32,
    n: u32,
) callconv(.kernel) void {
    const tid = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (tid >= n) return;

    // Byte permutation
    const perm = cuda.__byte_perm(0x12345678, 0xAABBCCDD, 0x0123);

    // Clock timer
    const t = cuda.clock();

    // Combine results
    output[tid] = perm ^ t;
}
