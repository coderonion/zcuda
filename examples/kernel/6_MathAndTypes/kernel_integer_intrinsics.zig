// examples/kernel/6_MathAndTypes/kernel_integer_intrinsics.zig — Integer intrinsics
//
// Reference: cuda-samples/0_Introduction/simpleAtomicIntrinsics (integer ops)
// API exercised: __clz, __popc, __brev, __ffs, __byte_perm, __dp4a

const cuda = @import("zcuda_kernel");

/// Count leading zeros, population count, and bit reverse
export fn integerBitOps(
    input: [*]const u32,
    clz_out: [*]u32,
    popc_out: [*]u32,
    brev_out: [*]u32,
    n: u32,
) callconv(.kernel) void {
    var iter = cuda.types.gridStrideLoop(n);
    while (iter.next()) |i| {
        const val = input[i];
        clz_out[i] = cuda.__clz(val);
        popc_out[i] = cuda.__popc(val);
        brev_out[i] = cuda.__brev(val);
    }
}

/// dp4a: 4-element dot product of packed bytes + accumulator
/// Useful for int8 quantized inference
export fn dp4aAccumulate(
    a: [*]const u32,
    b: [*]const u32,
    c: [*]u32,
    n: u32,
) callconv(.kernel) void {
    var iter = cuda.types.gridStrideLoop(n);
    while (iter.next()) |i| {
        c[i] = cuda.__dp4a(a[i], b[i], c[i]);
    }
}

/// Byte permutation demo
export fn bytePerm(
    input: [*]const u32,
    output: [*]u32,
    selector: u32,
    n: u32,
) callconv(.kernel) void {
    var iter = cuda.types.gridStrideLoop(n);
    while (iter.next()) |i| {
        output[i] = cuda.__byte_perm(input[i], input[i], selector);
    }
}
