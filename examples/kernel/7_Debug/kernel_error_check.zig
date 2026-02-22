// examples/kernel/7_Debug/kernel_error_check.zig — Error checking patterns
//
// Reference: cuda-samples/0_Introduction/simpleAssert
// API exercised: debug.printf, debug.assert, atomicAdd, clock

const cuda = @import("zcuda_kernel");

/// Validate input data with error reporting.
/// Writes error indices to an output buffer for host inspection.
export fn validateData(
    data: [*]const f32,
    error_indices: [*]u32,
    error_count: *u32,
    max_errors: u32,
    n: u32,
) callconv(.kernel) void {
    var iter = cuda.types.gridStrideLoop(n);
    while (iter.next()) |i| {
        const val = data[i];

        // Check for NaN or Inf
        const bits = cuda.__float_as_uint(val);
        const exponent = (bits >> 23) & 0xFF;
        const is_special = exponent == 0xFF; // NaN or Inf

        if (is_special) {
            const idx = cuda.atomicAdd(error_count, @as(u32, 1));
            if (idx < max_errors) {
                error_indices[idx] = @intCast(i);
            }
        }
    }
}

/// Performance timing kernel: measure clock cycles per operation
export fn timingKernel(
    input: [*]const f32,
    output: [*]f32,
    cycle_counts: [*]u32,
    n: u32,
) callconv(.kernel) void {
    const gid = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (gid >= n) return;

    const start = cuda.clock();

    // Compute-intensive operation to time
    var val = input[gid];
    val = cuda.__sinf(val);
    val = cuda.__cosf(val);
    val = cuda.__expf(val);
    val = cuda.rsqrtf(cuda.fmaxf(val * val, 1e-8));
    output[gid] = val;

    const end = cuda.clock();
    cycle_counts[gid] = end -% start;
}
