// examples/kernel/3_Atomics/kernel_atomic_ops.zig — All atomic operations demo
//
// Reference: cuda-samples/0_Introduction/simpleAtomicIntrinsics
// API exercised: atomicAdd, atomicMax, atomicMin, atomicExch, atomicCAS, atomicAnd, atomicOr, atomicXor, atomicInc, atomicDec

const cuda = @import("zcuda_kernel");

/// Demonstrate all atomic operations on a set of counters.
/// Each thread applies operations to shared counters.
export fn atomicOpsDemo(
    counters: [*]u32,
    n: u32,
) callconv(.kernel) void {
    const gid = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (gid >= n) return;

    // counters[0]: atomicAdd — total thread count
    _ = cuda.atomicAdd(&counters[0], @as(u32, 1));

    // counters[1]: atomicMax — max thread ID
    _ = cuda.atomicMax(&counters[1], gid);

    // counters[2]: atomicMin — min thread ID (init to 0xFFFFFFFF)
    _ = cuda.atomicMin(&counters[2], gid);

    // counters[3]: atomicAnd — AND all thread IDs
    _ = cuda.atomicAnd(&counters[3], gid);

    // counters[4]: atomicOr — OR all thread IDs
    _ = cuda.atomicOr(&counters[4], gid);

    // counters[5]: atomicXor — XOR all thread IDs
    _ = cuda.atomicXor(&counters[5], gid);

    // counters[6]: atomicInc — modular increment
    _ = cuda.atomicInc(&counters[6], n - 1);

    // counters[7]: atomicDec — modular decrement
    _ = cuda.atomicDec(&counters[7], n - 1);

    // counters[8]: atomicExch — last writer wins
    _ = cuda.atomicExch(&counters[8], gid);

    // counters[9]: atomicCAS — compare-and-swap (set to gid if currently 0)
    _ = cuda.atomicCAS(&counters[9], 0, gid);
}
