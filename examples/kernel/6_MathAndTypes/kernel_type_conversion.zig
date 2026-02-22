// examples/kernel/6_MathAndTypes/kernel_type_conversion.zig — Type conversion intrinsics
//
// Reference: cuda-samples/0_Introduction/fp16ScalarProduct (type conversion)
// API exercised: __float2int_rn, __int2float_rn, __float_as_uint, __uint_as_float, __saturatef

const cuda = @import("zcuda_kernel");

/// Convert f32 array to i32 (round-to-nearest)
export fn f32ToI32(
    input: [*]const f32,
    output: [*]i32,
    n: u32,
) callconv(.kernel) void {
    var iter = cuda.types.gridStrideLoop(n);
    while (iter.next()) |i| {
        output[i] = cuda.__float2int_rn(input[i]);
    }
}

/// Convert i32 array to f32
export fn i32ToF32(
    input: [*]const i32,
    output: [*]f32,
    n: u32,
) callconv(.kernel) void {
    var iter = cuda.types.gridStrideLoop(n);
    while (iter.next()) |i| {
        output[i] = cuda.__int2float_rn(input[i]);
    }
}

/// Saturate f32 values to [0.0, 1.0] range
export fn saturate(
    data: [*]f32,
    n: u32,
) callconv(.kernel) void {
    var iter = cuda.types.gridStrideLoop(n);
    while (iter.next()) |i| {
        data[i] = cuda.__saturatef(data[i]);
    }
}

/// Bit manipulation: reinterpret float bits as uint and back
export fn bitManipulation(
    input: [*]const f32,
    output: [*]f32,
    n: u32,
) callconv(.kernel) void {
    var iter = cuda.types.gridStrideLoop(n);
    while (iter.next()) |i| {
        // Extract float bits, clear sign bit, convert back
        const bits = cuda.__float_as_uint(input[i]);
        const abs_bits = bits & 0x7FFFFFFF; // clear sign bit = abs()
        output[i] = cuda.__uint_as_float(abs_bits);
    }
}
