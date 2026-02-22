// examples/kernel/7_Debug/kernel_printf_debug.zig — Device-side printf debugging
//
// Reference: cuda-samples/0_Introduction/simplePrintf
// API exercised: debug.printf, threadIdx, blockIdx

const cuda = @import("zcuda_kernel");

/// Debug kernel that prints thread/block info and computed values.
/// Useful for verifying kernel launch configuration and data flow.
export fn printfDebug(
    data: [*]const f32,
    n: u32,
) callconv(.kernel) void {
    const tid = cuda.threadIdx().x;
    const bid = cuda.blockIdx().x;
    const gid = bid * cuda.blockDim().x + tid;

    if (gid >= n) return;

    // Only print from first few threads to avoid output flood
    if (gid < 4) {
        cuda.debug.printf("Thread [%u,%u] gid=%u val=%f\n", .{ bid, tid, gid, data[gid] });
    }
}

/// Assertion-style debug kernel: checks invariants
export fn assertInvariants(
    data: [*]const f32,
    expected_min: f32,
    expected_max: f32,
    error_count: *u32,
    n: u32,
) callconv(.kernel) void {
    const gid = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (gid >= n) return;

    const val = data[gid];
    if (val < expected_min or val > expected_max) {
        _ = cuda.atomicAdd(error_count, @as(u32, 1));
        if (cuda.atomicAdd(error_count, @as(u32, 0)) < 10) { // limit output
            cuda.debug.printf("ASSERT FAIL: data[%u] = %f not in [%f, %f]\n", .{ gid, val, expected_min, expected_max });
        }
    }
}
